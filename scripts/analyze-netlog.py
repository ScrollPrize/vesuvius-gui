#!/usr/bin/env python3
"""Summarize a VESUVIUS_NET_LOG JSONL capture.

Usage: VESUVIUS_NET_LOG=/tmp/netlog.jsonl cargo run --release --bin vesuvius-gui
       python3 scripts/analyze-netlog.py /tmp/netlog.jsonl

Breaks each download into queue wait / TTFB / body transfer, aggregates per
host (and per CloudFront hit/miss), reconstructs the downlink-utilization and
concurrency timeline, and contrasts the wait of viewport-critical chunks
(touches > 0) against background ones — the head-of-line-blocking signal.
"""
import json
import sys
from collections import defaultdict


def pct(xs, p):
    if not xs:
        return 0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p / 100))]


def fmt_ms(v):
    return f"{v:>6.0f}ms"


def summarize(label, rows):
    if not rows:
        return
    waits = [r["wait_ms"] for r in rows]
    ttfbs = [r["ttfb_ms"] for r in rows if r["ttfb_ms"]]
    bodys = [r["body_ms"] for r in rows]
    bytes_ = sum(r["bytes"] for r in rows)
    span_s = (max(r["t"] for r in rows) - min(r["t"] for r in rows)) / 1000 or 1
    print(f"\n{label}: n={len(rows)} bytes={bytes_/1e6:.1f}MB avg_rate_over_span={bytes_/span_s/1e6:.2f}MB/s")
    for name, xs in [("queue_wait", waits), ("ttfb", ttfbs), ("body", bodys)]:
        print(
            f"  {name:>10}: p50={fmt_ms(pct(xs,50))} p90={fmt_ms(pct(xs,90))} "
            f"p99={fmt_ms(pct(xs,99))} max={fmt_ms(max(xs) if xs else 0)}"
        )
    per_req = [r["bytes"] / ((r["ttfb_ms"] + r["body_ms"]) / 1000) / 1e6 for r in rows if r["ttfb_ms"] + r["body_ms"] > 0 and r["bytes"]]
    if per_req:
        print(f"  per-request rate (incl ttfb): p50={pct(per_req,50):.2f}MB/s p90={pct(per_req,90):.2f}MB/s")


def main(path):
    downloads, aged = [], []
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        (downloads if r.get("event") == "download" else aged).append(r)

    if not downloads:
        print("no download events")
        return

    print(f"=== {len(downloads)} downloads, {len(aged)} aged-out cancellations ===")

    by_host = defaultdict(list)
    for r in downloads:
        by_host[r["host"]].append(r)
    for host, rows in sorted(by_host.items()):
        ok = [r for r in rows if r["ok"] and r["status"] in (200, 206)]
        errs = defaultdict(int)
        for r in rows:
            if r["status"] not in (200, 206):
                errs[r["status"]] += 1
        summarize(f"host {host}", ok)
        if errs:
            print(f"  non-2xx: {dict(errs)}")
        hits = [r for r in ok if (r.get("x_cache") or "").startswith("Hit")]
        misses = [r for r in ok if (r.get("x_cache") or "").startswith("Miss")]
        if hits or misses:
            summarize(f"  └ CF hit", hits)
            summarize(f"  └ CF miss", misses)

    # head-of-line signal: chunks the paint loop kept re-touching vs the rest
    touched = [r for r in downloads if r["touches"] > 0]
    fresh = [r for r in downloads if r["touches"] == 0]
    print("\n=== priority / head-of-line ===")
    print(f"touched>0 (viewport kept asking): n={len(touched)} "
          f"wait p50={pct([r['wait_ms'] for r in touched],50)}ms p90={pct([r['wait_ms'] for r in touched],90)}ms "
          f"wait_total p90={pct([r['wait_total_ms'] for r in touched],90)}ms")
    print(f"touches=0:                        n={len(fresh)} "
          f"wait p50={pct([r['wait_ms'] for r in fresh],50)}ms p90={pct([r['wait_ms'] for r in fresh],90)}ms")
    if aged:
        qm = [r["queued_ms"] for r in aged]
        print(f"aged out: n={len(aged)} queued p50={pct(qm,50)}ms touched={sum(1 for r in aged if r['touches']>0)}")

    # timeline: per-second downlink utilization + concurrency
    buckets = defaultdict(lambda: [0.0, []])  # sec -> [bytes, [in_flight..]]
    for r in downloads:
        if not r["bytes"]:
            continue
        end = r["t"]
        start = end - max(r["body_ms"], 1)
        rate = r["bytes"] / max(r["body_ms"], 1)  # bytes per ms, assumed uniform
        s = start
        while s < end:
            nxt = min((s // 1000 + 1) * 1000, end)
            buckets[s // 1000][0] += rate * (nxt - s)
            s = nxt
        buckets[end // 1000][1].append(r["in_flight"])
    if buckets:
        rates = [b[0] / 1e6 for b in buckets.values()]
        print("\n=== downlink utilization (per-second, body transfer only) ===")
        print(f"seconds active: {len(rates)}  p50={pct(rates,50):.2f}MB/s p90={pct(rates,90):.2f}MB/s max={max(rates):.2f}MB/s")
        infl = [r["in_flight"] for r in downloads]
        qd = [r["q_depth"] for r in downloads]
        print(f"in_flight at completion: p50={pct(infl,50)} p90={pct(infl,90)} max={max(infl)}")
        print(f"queue depth at pop: p50={pct(qd,50)} p90={pct(qd,90)} max={max(qd)}")

    slow = sorted(downloads, key=lambda r: r["wait_ms"] + r["ttfb_ms"] + r["body_ms"], reverse=True)[:10]
    print("\n=== slowest 10 (wait+ttfb+body) ===")
    for r in slow:
        print(f"  wait={r['wait_ms']:>5} ttfb={r['ttfb_ms']:>5} body={r['body_ms']:>5} "
              f"bytes={r['bytes']:>8} touches={r['touches']} inflight={r['in_flight']} {r['url'][-60:]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/netlog.jsonl")
