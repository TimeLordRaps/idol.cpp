"""Compare the native decoder with the same small SDK models loaded from a genuine GGUF export."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

if any(importlib.util.find_spec(name) is None for name in ("numpy", "torch", "gguf", "safetensors", "plyfile")):
    print("skipped: needs numpy, torch, gguf, safetensors and plyfile")
    sys.exit(77)
sdk = Path(os.environ.get("IDOL_SDK", ""))
if not (sdk / "idol/models/decoder.py").is_file():
    print("skipped: set IDOL_SDK to the upstream SDK source")
    sys.exit(77)
sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "tools"), str(sdk)]
import numpy as np
import torch
from safetensors.numpy import save_file
from idol.models.decoder import OctreeProbabilityFixedlenDecoder, GaussianFixedlenDecoder
from export_full_gguf import export
from gguf_to_decoder import materialize
from idol_gguf import IdolGGUF

rep = {"num_gaussians": 4, "lr": {"_xyz": 1.0, "_features_dc": 1.0, "_opacity": 1.0,
                                   "_scaling": 1.0, "_rotation": 1.0},
       "perturbe_size": 1.5, "3d_filter_kernel_size": 0.0009, "scaling_bias": 0.004, "opacity_bias": 0.1}
oct_args = {"model_channels": 64, "cond_channels": 16, "num_blocks": 2, "num_heads": 4, "mlp_ratio": 4}
gs_args = {"in_channels": 3, **oct_args, "representation_config": rep}
torch.manual_seed(17)
octree, gaussian = OctreeProbabilityFixedlenDecoder(**oct_args).float(), GaussianFixedlenDecoder(**gs_args).float()
octree.dtype = gaussian.dtype = torch.float32
for model in (octree, gaussian):
    for param in model.parameters():
        torch.nn.init.normal_(param, std=0.02)
with tempfile.TemporaryDirectory(prefix="idol-gguf-test-") as scratch:
    root = Path(scratch)
    config = {"decoder": {"args": oct_args}, "decoder_gs": {"args": gs_args}, "denoiser": {}, "image_encoders": {}}
    (root / "config.json").write_text(json.dumps(config))
    save_file({k: v.detach().numpy() for k, v in octree.state_dict().items()}, str(root / "decoder.safetensors"))
    save_file({k: v.detach().numpy() for k, v in gaussian.state_dict().items()}, str(root / "decoder_gs.safetensors"))
    for name in ("denoiser.safetensors", "birefnet.safetensors"):
        save_file({"dummy.weight": np.zeros((2, 32), dtype=np.float32)}, str(root / name))
    for tier in ("Q8_0", "Q4_0"):
        path = root / (tier + ".gguf")
        export(root, path, tier, "synthetic")
        packed = IdolGGUF(path)
        model_o = packed.load_module("octree", OctreeProbabilityFixedlenDecoder(**oct_args).float()).eval()
        model_g = packed.load_module("gaussian", GaussianFixedlenDecoder(**gs_args).float()).eval()
        model_o.dtype = model_g.dtype = torch.float32
        native = root / (tier + "-native")
        materialize(packed, native)
        del packed
        case = root / (tier + "-case")
        case.mkdir()
        anchors, cond = torch.rand(1, 32, 3), torch.randn(1, 96, 16)
        with torch.no_grad():
            logits = model_o(anchors, torch.tensor([32.0]), cond)
            features = model_g(anchors, cond)
        for name, value in (("anchors", anchors), ("cond", cond), ("ref_oct", logits), ("ref_gs", features)):
            np.ascontiguousarray(value.detach().numpy(), dtype="<f4").tofile(case / (name + ".bin"))
        result = subprocess.run([sys.argv[1], str(native), "--check", "--case", str(case), "--level", "5",
                                 "--threads", "1", "--tolerance", "2e-5"],
                                env=dict(os.environ, IDOL_QUANT_KERNEL="float"), capture_output=True, text=True)
        assert result.returncode == 0, (tier, result.stdout, result.stderr)
print("native decoder matches small SDK models loaded from GGUF")
