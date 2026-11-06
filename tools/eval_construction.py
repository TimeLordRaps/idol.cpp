"""Decode the same latent with several exports and compare generated Gaussian centers.

The alternate reference seed estimates finite-sampling variation; comparisons are not image renders or ground-truth
surface distances. A failure leaves a partial report and generated outputs for inspection, without overwriting them.
"""
import argparse
import ctypes
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

from compare_splats import compare, digest, load


def available_bytes():
    if sys.platform.startswith("linux"):
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                parts = line.split()
                if len(parts) == 3 and parts[2] == "kB":
                    return int(parts[1]) * 1024
        raise RuntimeError("cannot read available memory from /proc/meminfo")
    if sys.platform != "win32":
        raise RuntimeError("available-memory guard supports Windows and Linux")
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                    ("available", ctypes.c_ulonglong), ("total_page", ctypes.c_ulonglong),
                    ("available_page", ctypes.c_ulonglong), ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong), ("available_extended", ctypes.c_ulonglong)]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("cannot read available memory")
    return status.available


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True, metavar="LABEL=DIR")
    parser.add_argument("--latent", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gaussians", type=int, default=16384)
    parser.add_argument("--level", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--min-available-gb", type=float, default=6.0)
    args = parser.parse_args()
    candidates = []
    for spec in args.candidate:
        label, path = spec.split("=", 1)
        if not label.isidentifier() or label in ("reference", "reference_other_seed"):
            parser.error("candidate labels must be unique identifiers and not reserved")
        candidates.append((label, Path(path), 0))
    if (len({x[0] for x in candidates}) != len(candidates) or args.gaussians < 1 or
            not 1 <= args.level <= 20 or args.threads < 1 or not math.isfinite(args.min_available_gb) or args.min_available_gb < 0):
        parser.error("invalid candidates or parameters")
    plans = [("reference", args.reference, 0), ("reference_other_seed", args.reference, 1)] + candidates
    inputs = {"runner": digest(args.runner), "latent": digest(args.latent),
              "tool": digest(Path(__file__)), "comparison_tool": digest(Path(__file__).with_name("compare_splats.py")),
              "exports": {name: {f: digest(path / f) for f in ("manifest.txt", "weights.bin")}
                          for name, path, _ in plans}}
    args.out_dir.mkdir(parents=True, exist_ok=False)
    report_path = args.out_dir / "report.json"
    report = {"status": "running", "inputs": inputs, "gaussians": args.gaussians, "level": args.level,
              "threads": args.threads, "activation_kernel": "float", "outputs": {}, "comparisons": {}}
    def save():
        temp = args.out_dir / "report.tmp"
        with temp.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        os.replace(temp, report_path)
    save()
    try:
        for name, model, seed in plans:
            if available_bytes() < args.min_available_gb * 1_000_000_000:
                raise RuntimeError("available system memory fell below guard before " + name)
            if shutil.disk_usage(args.out_dir).free < 1_000_000_000 + args.gaussians * 68:
                raise RuntimeError("insufficient disk space before " + name)
            output = args.out_dir / (name + ".ply")
            command = [str(args.runner), str(model), "--latent", str(args.latent), "--out", str(output),
                       "--gaussians", str(args.gaussians), "--level", str(args.level), "--seed", str(seed),
                       "--threads", str(args.threads)]
            done = subprocess.run(command, capture_output=True, text=True, env=dict(os.environ, IDOL_QUANT_KERNEL="float"))
            if done.returncode:
                raise RuntimeError(name + " failed: " + done.stderr)
            report["outputs"][name] = {"sha256": digest(output), "seed": seed, "runner_output": done.stdout}
            save()
            print("decoded", name, flush=True)
        reference = load(args.out_dir / "reference.ply")
        for name, _, _ in plans[1:]:
            report["comparisons"][name] = compare(reference, load(args.out_dir / (name + ".ply")))
            save()
        after = {"runner": digest(args.runner), "latent": digest(args.latent),
                 "tool": digest(Path(__file__)), "comparison_tool": digest(Path(__file__).with_name("compare_splats.py")),
                 "exports": {name: {f: digest(path / f) for f in ("manifest.txt", "weights.bin")}
                             for name, path, _ in plans}}
        if inputs != after:
            raise RuntimeError("inputs changed during experiment")
        report["status"] = "complete"
        report["hashes_unchanged_after_run"] = True
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        save()
        raise
    save()


if __name__ == "__main__":
    main()
