use std::fs::File;
use std::io;
use std::os::windows::io::AsRawHandle;
use windows_sys::Win32::System::Ioctl::{FILE_ZERO_DATA_INFORMATION, FSCTL_SET_SPARSE, FSCTL_SET_ZERO_DATA};
use windows_sys::Win32::System::IO::DeviceIoControl;

fn control(file: &File, code: u32, input: *const std::ffi::c_void, len: u32) -> io::Result<()> {
    let mut returned = 0;
    let result = unsafe {
        DeviceIoControl(
            file.as_raw_handle(),
            code,
            input,
            len,
            std::ptr::null_mut(),
            0,
            &mut returned,
            std::ptr::null_mut(),
        )
    };
    if result == 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

pub(super) fn mark_sparse(file: &File) -> io::Result<()> {
    control(file, FSCTL_SET_SPARSE, std::ptr::null(), 0)
}

pub(super) fn punch_hole(file: &File, offset: u64, len: u64) -> io::Result<()> {
    let end = offset
        .checked_add(len)
        .and_then(|end| i64::try_from(end).ok())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "sparse range exceeds i64"))?;
    let data = FILE_ZERO_DATA_INFORMATION {
        FileOffset: offset as i64,
        BeyondFinalZero: end,
    };
    control(
        file,
        FSCTL_SET_ZERO_DATA,
        std::ptr::from_ref(&data).cast(),
        std::mem::size_of_val(&data) as u32,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Seek, SeekFrom, Write};
    use windows_sys::Win32::System::Ioctl::{FILE_ALLOCATED_RANGE_BUFFER, FSCTL_QUERY_ALLOCATED_RANGES};

    fn allocated_bytes(file: &File, length: u64) -> u64 {
        let range = FILE_ALLOCATED_RANGE_BUFFER {
            FileOffset: 0,
            Length: length as i64,
        };
        let mut output = [FILE_ALLOCATED_RANGE_BUFFER::default(); 16];
        let mut returned = 0;
        let result = unsafe {
            DeviceIoControl(
                file.as_raw_handle(),
                FSCTL_QUERY_ALLOCATED_RANGES,
                std::ptr::from_ref(&range).cast(),
                std::mem::size_of_val(&range) as u32,
                output.as_mut_ptr().cast(),
                std::mem::size_of_val(&output) as u32,
                &mut returned,
                std::ptr::null_mut(),
            )
        };
        assert_ne!(result, 0, "{}", io::Error::last_os_error());
        let count = returned as usize / std::mem::size_of::<FILE_ALLOCATED_RANGE_BUFFER>();
        output[..count].iter().map(|range| range.Length as u64).sum()
    }

    #[test]
    fn sparse_ct_data_can_be_written_and_physically_reclaimed() {
        let temporary = tempfile::NamedTempFile::new().unwrap();
        let mut file = temporary.reopen().unwrap();
        let logical_size = 512 * 1024 * 1024 * 1024;
        let data = include_bytes!("../../tests/data/PHerc0800-cache.raw");
        mark_sparse(&file).unwrap();
        file.set_len(logical_size).unwrap();
        assert_eq!(allocated_bytes(&file, logical_size), 0);
        for offset in [1024 * 1024, 2 * 1024 * 1024] {
            file.seek(SeekFrom::Start(offset)).unwrap();
            file.write_all(data).unwrap();
        }
        file.sync_all().unwrap();
        let mapped = unsafe {
            memmap::MmapOptions::new()
                .len(logical_size as usize)
                .map(&File::open(temporary.path()).unwrap())
                .unwrap()
        };
        assert_eq!(&mapped[1024 * 1024..1024 * 1024 + data.len()], data);
        file.sync_all().unwrap();
        let before = allocated_bytes(&file, logical_size);
        assert!(before > 0 && before < 1024 * 1024);
        punch_hole(&file, 1024 * 1024, 262144).unwrap();
        file.sync_all().unwrap();
        let after = allocated_bytes(&file, logical_size);
        assert!(after < before && after > 0, "before={before}, after={after}");
        assert_eq!(file.metadata().unwrap().len(), logical_size);
        assert!(mapped[1024 * 1024..1024 * 1024 + data.len()]
            .iter()
            .all(|&byte| byte == 0));
        let mut readback = vec![0; data.len()];
        file.seek(SeekFrom::Start(1024 * 1024)).unwrap();
        file.read_exact(&mut readback).unwrap();
        assert!(readback.iter().all(|&byte| byte == 0));
        file.seek(SeekFrom::Start(2 * 1024 * 1024)).unwrap();
        file.read_exact(&mut readback).unwrap();
        assert_eq!(readback, data);
    }
}
