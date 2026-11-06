"""Time the real Idol decoders: float32 against quantized tiers, both quantized kernels, both attention paths.

Each configuration runs idol_decode --bench (one warm-up, then the median of --repeat runs of each decoder on the first
--rows anchors of a real latent), several times, interleaved across configurations so a drifting machine does not
favour one of them. Reports the median time of each decoder, the weight bytes the runtime holds, the peak process
memory and process CPU time. The profile shares cover matrix multiplication and attention across both decoders,
including warm-ups; they exclude other operations and are not a breakdown of Gaussian decoder wall time.

Usage:
    python tools/bench_idol.py --export DIR --case DIR --run idol_decode --work DIR
           --out bench.json [--tiers 8:64,6:64,4:32,3:32,2:32] [--threads 0,8] [--rows 1024] [--repeats 3] [--load N]

--threads 0 means all hardware threads. --load N keeps N busy-loop processes running during the measurements.
The tier exports use round-to-nearest and are written into --work. Use a fresh work directory: existing tier exports
are not reused, because their source model and quantization settings cannot be inferred from the directory name.
The integer kernel is requested, but the runtime can fall back to floating-point arithmetic on unsupported hardware.
"""

import argparse
import json
import multiprocessing
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quant_idol as qi  # noqa: E402


def spin(stop):
    while not stop.is_set():
        sum(i * i for i in range(20000))


def run_once(run, export, case, rows, repeat, threads, kernel, attention):
    cmd = [str(run), str(export), "--check", "--case", str(case), "--level", "8", "--bench", str(rows), "--repeat", str(repeat)]
    if threads:
        cmd += ["--threads", str(threads)]
    env = dict(os.environ, IDOL_PROFILE="1", IDOL_QUANT_KERNEL=kernel or "float", IDOL_ATTENTION=attention)
    done = subprocess.run(cmd, capture_output=True, text=True, env=env)
    text = done.stdout + done.stderr
    bench = re.search(r"BENCH rows=(\d+) tokens=(\d+) oct_s=(\S+) gs_s=(\S+) resident_mb=(\S+) peak_mb=(\S+) cpu_s=(\S+)", text)
    profile = re.search(r"profile: gemm ([\d.]+) ms \((\d+) calls\), attention ([\d.]+) ms", text)
    if done.returncode or not (bench and profile):
        raise RuntimeError(text)
    oct_s, gs_s, resident, peak, cpu = (float(bench.group(i)) for i in (3, 4, 5, 6, 7))
    gemm, attn = float(profile.group(1)) / 1e3, float(profile.group(3)) / 1e3
    return {"rows": int(bench.group(1)), "tokens": int(bench.group(2)),
            "oct_s": oct_s, "gs_s": gs_s, "resident_mb": resident, "peak_mb": peak, "cpu_s": cpu,
            "gemm_share": gemm / (gemm + attn), "attention_share": attn / (gemm + attn)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("--export", "--case", "--run", "--work", "--out"):
        parser.add_argument(flag, type=Path, required=True)
    parser.add_argument("--tiers", default="8:64,6:64,4:32,3:32,2:32")
    parser.add_argument("--threads", default="0,8")
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--repeat", type=int, default=3, help="runs per decoder inside one process")
    parser.add_argument("--repeats", type=int, default=3, help="processes per configuration")
    parser.add_argument("--load", type=int, default=0)
    args = parser.parse_args()
    if args.rows < 1 or args.repeat < 1 or args.repeats < 1 or args.load < 0:
        parser.error("rows, repeat and repeats must be positive; load must be nonnegative")

    keep, tensors_meta = qi.parse_manifest(args.export)
    exports = [("float32", args.export)]
    for spec in args.tiers.split(","):
        bits, group = (int(x) for x in spec.split(":"))
        folder = args.work / f"bench_{bits}_{group}"
        if folder.exists():
            parser.error(f"{folder} already exists; choose a fresh --work directory")
        plan = {n: (bits, group) for n, s, _, _ in tensors_meta if qi.eligible(n, s, group)}
        qi.write_tier(args.export, folder, keep, tensors_meta, plan, {}, "rtn")
        exports.append((f"{bits}-bit g{group}", folder))

    threads = [int(t) for t in args.threads.split(",")]
    configs = []
    for label, folder in exports:
        for t in threads:
            if label == "float32":
                configs += [(label, folder, t, None, "blocked"), (label, folder, t, None, "dot")]
            else:
                configs += [(label, folder, t, "float", "blocked"), (label, folder, t, "int8", "blocked")]

    stop = multiprocessing.Event()
    hogs = [multiprocessing.Process(target=spin, args=(stop,), daemon=True) for _ in range(args.load)]
    for h in hogs:
        h.start()
    time.sleep(1.0 if hogs else 0.0)
    samples = {c: [] for c in configs}
    try:
        for r in range(args.repeats):
            for c in configs:
                label, folder, t, kernel, attention = c
                samples[c].append(run_once(args.run, folder, args.case, args.rows, args.repeat, t, kernel, attention))
            print(f"repeat {r + 1}/{args.repeats} done", flush=True)
    finally:
        stop.set()
        for h in hogs:
            h.join(timeout=5)

    results = []
    for (label, folder, t, kernel, attention), rows in samples.items():
        def med(key):
            return statistics.median(r[key] for r in rows)
        results.append({"label": label, "threads": t, "kernel": kernel, "attention": attention, "load": args.load,
                        "rows": rows[0]["rows"], "tokens": rows[0]["tokens"],
                        "process_samples": len(rows), "repeat_per_process": args.repeat,
                        "oct_s": med("oct_s"), "gs_s": med("gs_s"),
                        "gs_s_min": min(r["gs_s"] for r in rows), "gs_s_max": max(r["gs_s"] for r in rows),
                        "resident_mb": med("resident_mb"), "peak_mb": max(r["peak_mb"] for r in rows),
                        "cpu_s": med("cpu_s"), "gemm_share": med("gemm_share"), "attention_share": med("attention_share")})
    args.out.write_text(json.dumps(results, indent=1))
    for r in results:
        print(f"{r['label']:10s} thr {r['threads']:2d} {str(r['kernel']):5s} {r['attention']:7s} oct {r['oct_s']:6.2f}s "
              f"gs {r['gs_s']:6.2f}s resident {r['resident_mb']:7.1f} peak {r['peak_mb']:7.1f} "
              f"gemm {r['gemm_share']:.0%} attn {r['attention_share']:.0%}")


if __name__ == "__main__":
    sys.exit(main())
