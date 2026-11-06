"""Calibration must reject duplicate or overlapping latent bytes before loading a model."""
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

if any(importlib.util.find_spec(name) is None for name in ("numpy", "torch", "scipy", "plyfile")):
    print("skipped: needs numpy, torch, scipy and plyfile")
    sys.exit(77)
tool = Path(__file__).resolve().parents[2] / "tools" / "calibrate_gaussian.py"
with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    weights, sdk, latents, out = (root / n for n in ("weights", "sdk", "latents", "out"))
    weights.mkdir()
    latents.mkdir()
    (sdk / "idol").mkdir(parents=True)
    for name in ("config.json", "decoder.safetensors", "decoder_gs.safetensors"):
        (weights / name).write_bytes(b"test")
    (sdk / "idol" / "module.py").write_text("# test\n")
    for name in ("a.f32", "b.f32"):
        (latents / name).write_bytes(b"same latent")
    command = [sys.executable, str(tool), "--weights", str(weights), "--sdk", str(sdk), "--latents", str(latents),
               "--out-dir", str(out), "--holdout", "1", "--calib", "1"]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0 and "byte-disjoint" in result.stderr and not out.exists()
    result = subprocess.run(command + ["--min-available-gb", "nan"], capture_output=True, text=True)
    assert result.returncode != 0 and "invalid parameters" in result.stderr and not out.exists()
print("calibration preflight rejects overlap and invalid memory limits")
