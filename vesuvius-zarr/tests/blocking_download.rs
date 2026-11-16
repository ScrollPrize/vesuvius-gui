use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};
use vesuvius_zarr::ZarrArray;

const CT: &[u8] = include_bytes!("data/PHerc0800-download/0.0.0");
const CHILD: &str = "VESUVIUS_BLOCKING_DOWNLOAD_CHILD";

fn read_request(stream: &mut TcpStream) {
    stream.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
    let mut request = Vec::new();
    let mut buffer = [0; 1024];
    while !request.ends_with(b"\r\n\r\n") {
        let count = stream.read(&mut buffer).unwrap();
        assert_ne!(count, 0);
        request.extend_from_slice(&buffer[..count]);
    }
    assert!(request.starts_with(b"GET /0.0.0 HTTP/1.1\r\n"));
}

fn concurrent_read(status: &str) {
    let cache = tempfile::tempdir().unwrap();
    std::fs::write(
        cache.path().join(".zarray"),
        include_str!("data/PHerc0800-download/.zarray"),
    )
    .unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (received, started) = mpsc::channel();
    let (release, response) = mpsc::channel();
    let status = status.to_string();
    let successful = status == "200 OK";
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        read_request(&mut stream);
        received.send(()).unwrap();
        response.recv_timeout(Duration::from_secs(3)).unwrap();
        let payload = if successful { CT } else { &[] };
        write!(
            stream,
            "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            payload.len()
        )
        .unwrap();
        stream.write_all(payload).unwrap();
    });
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .unwrap();
    let array = ZarrArray::<3, u8>::from_url_blocking(&url, cache.path().to_str().unwrap(), client);
    let first = array.clone();
    let first = thread::spawn(move || first.load_chunk([0, 0, 0]));
    started.recv_timeout(Duration::from_secs(3)).unwrap();
    let second = array.clone();
    let (waiting, ready) = mpsc::channel();
    let second = thread::spawn(move || {
        waiting.send(()).unwrap();
        second.load_chunk([0, 0, 0])
    });
    ready.recv_timeout(Duration::from_secs(3)).unwrap();
    thread::sleep(Duration::from_millis(200));
    release.send(()).unwrap();
    server.join().unwrap();
    for result in [first.join().unwrap(), second.join().unwrap()] {
        if successful {
            let chunk = result.expect("both blocking readers should receive the CT chunk");
            let bytes = (0..CT.len()).map(|index| chunk.get(index)).collect::<Vec<_>>();
            assert_eq!(bytes, CT);
        } else {
            assert!(result.is_none());
        }
    }
    assert_eq!(array.load_chunk([0, 0, 0]).is_some(), successful);
    assert_eq!(cache.path().join("0/0/0.missing").exists(), !successful);
}

fn run_guarded(name: &str, test: fn()) {
    if std::env::var(CHILD).ok().as_deref() == Some(name) {
        test();
        return;
    }
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", name, "--nocapture"])
        .env(CHILD, name)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if child.try_wait().unwrap().is_some() {
            let result = child.wait_with_output().unwrap();
            assert!(
                result.status.success(),
                "{}{}",
                String::from_utf8_lossy(&result.stdout),
                String::from_utf8_lossy(&result.stderr)
            );
            break;
        }
        if Instant::now() >= deadline {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("{name} did not finish within 10 seconds");
        }
        thread::sleep(Duration::from_millis(25));
    }
}

#[test]
fn concurrent_blocking_reads_finish_and_share_the_chunk() {
    run_guarded("concurrent_blocking_reads_finish_and_share_the_chunk", || {
        concurrent_read("200 OK");
        concurrent_read("404 Not Found");
    });
}

#[test]
fn blocking_http_errors_do_not_become_missing_chunks() {
    run_guarded("blocking_http_errors_do_not_become_missing_chunks", || {
        for failure in [
            "403 Forbidden",
            "429 Too Many Requests",
            "500 Internal Server Error",
            "503 Service Unavailable",
        ] {
            let cache = tempfile::tempdir().unwrap();
            std::fs::write(
                cache.path().join(".zarray"),
                include_str!("data/PHerc0800-download/.zarray"),
            )
            .unwrap();
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let url = format!("http://{}", listener.local_addr().unwrap());
            let server = thread::spawn(move || {
                for status in [failure, "200 OK"] {
                    let (mut stream, _) = listener.accept().unwrap();
                    read_request(&mut stream);
                    let payload = if status == "200 OK" { CT } else { &[] };
                    write!(
                        stream,
                        "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        payload.len()
                    )
                    .unwrap();
                    stream.write_all(payload).unwrap();
                }
            });
            let client = reqwest::blocking::Client::builder()
                .timeout(Duration::from_secs(3))
                .build()
                .unwrap();
            let ctx = ZarrArray::<3, u8>::from_url_blocking(&url, cache.path().to_str().unwrap(), client)
                .into_ctx()
                .into_ctx();
            let first = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ctx.get([0, 0, 0])));
            assert!(first.is_err(), "{failure} should be reported as an HTTP error");
            let message = first.unwrap_err().downcast::<String>().unwrap();
            assert!(message.contains(failure.split_once(' ').unwrap().0));
            assert!(!cache.path().join("0/0/0.missing").exists());
            assert!(!cache.path().join("0/0/0").exists());
            for z in 0..3 {
                for y in 0..32 {
                    for x in 0..32 {
                        assert_eq!(
                            ctx.get([z, y, x]),
                            Some(CT[(z * 32 + y) * 32 + x]),
                            "{failure} at {z},{y},{x}"
                        );
                    }
                }
            }
            server.join().unwrap();
        }
    });
}
