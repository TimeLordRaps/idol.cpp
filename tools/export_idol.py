"""Export the real Idol decoders for idol::Decoders: weights, a manifest and a PyTorch reference case.

Reads decoder.safetensors (the octree decoder) and decoder_gs.safetensors (the Gaussian decoder) from a download of
speridlabs/idol, loads them strictly into the official SDK modules to prove the tensor names and shapes, and writes
both into one export directory. Tensor names are prefixed "oct." and "gs.". Weights stay float32.

The reference case is a seeded latent, a set of anchor points and an octree level. ref_oct.bin and ref_gs.bin are what
the SDK modules produce for them, computed in float64 (the SDK runs its blocks in float16 on a GPU, so float64 is the
yardstick for the runtime's float32, not a copy of the GPU's rounding).

Usage:
    python tools/export_idol.py --weights DIR --sdk idol-sdk/ --out build/idol-real [--anchors 256 --latent 1024]
    python tools/export_idol.py --random-dim 64 --sdk idol-sdk/ --out build/idol-tiny     # small random decoders, for tests

DIR holds config.json, decoder.safetensors and decoder_gs.safetensors. --sdk is a checkout of the SDK source (the
folder that contains the "idol" package). Needs torch, safetensors and plyfile.
"""

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file

ALIGN = 64  # each tensor starts on a 64-byte boundary in weights.bin


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", help="folder with config.json and the two decoder .safetensors files")
    parser.add_argument("--random-dim", type=int, default=0,
                        help="instead of --weights: small seeded random decoders of this width, for tests")
    parser.add_argument("--sdk", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--anchors", type=int, default=256, help="anchor points in the reference case")
    parser.add_argument("--latent", type=int, default=1024, help="latent tokens in the reference case (the model uses 8192)")
    parser.add_argument("--level", type=int, default=5, help="octree level of the reference case (resolution 2^level)")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, args.sdk)
    from idol.models.decoder import GaussianFixedlenDecoder, OctreeProbabilityFixedlenDecoder

    if args.random_dim:
        d = args.random_dim
        oct_args = dict(model_channels=d, cond_channels=16, num_blocks=2, num_heads=4, mlp_ratio=4)
        gs_args = dict(in_channels=3, model_channels=d, cond_channels=16, num_blocks=2, num_heads=4, mlp_ratio=4,
                       representation_config={"num_gaussians": 4, "lr": {"_xyz": 1.0, "_features_dc": 1.0, "_opacity": 1.0,
                                                                          "_scaling": 1.0, "_rotation": 0.1},
                                              "perturbe_size": 1.5, "3d_filter_kernel_size": 0.0009,
                                              "scaling_bias": 0.004, "opacity_bias": 0.1})
    elif args.weights:
        config = json.load(open(os.path.join(args.weights, "config.json")))
        oct_args, gs_args = config["decoder"]["args"], config["decoder_gs"]["args"]
    else:
        raise SystemExit("give --weights or --random-dim")
    # The SDK's constructors convert every block Linear to float16 (it runs them that way on a GPU), so loading the
    # float32 file into one rounds the weights to half. Convert back to float32 first so the reference sees the file.
    oct_model = OctreeProbabilityFixedlenDecoder(**oct_args).float()
    gs_model = GaussianFixedlenDecoder(**gs_args).float()
    if args.random_dim:
        torch.manual_seed(args.seed)
        for model in (oct_model, gs_model):
            for p in model.parameters():
                torch.nn.init.normal_(p, std=0.5 / p.shape[-1] ** 0.5 if p.ndim > 1 else 0.05)
        gs_model.points_offset_perturbation.normal_(std=0.5)
        gs_model.base_offset_scale.fill_(-3.0)
        oct_state, gs_state = oct_model.state_dict(), gs_model.state_dict()
    else:
        oct_state = load_file(os.path.join(args.weights, "decoder.safetensors"))
        gs_state = load_file(os.path.join(args.weights, "decoder_gs.safetensors"))
        oct_model.load_state_dict(oct_state, strict=True)
        gs_model.load_state_dict(gs_state, strict=True)

    os.makedirs(args.out, exist_ok=True)
    dim, heads = oct_args["model_channels"], oct_args["num_heads"]
    per_anchor = gs_args["representation_config"]["num_gaussians"]
    manifest = [f"decoders {dim} {heads} {oct_args['num_blocks']} {gs_args['num_blocks']} {oct_args['cond_channels']} "
                f"{per_anchor}"]
    offset = 0
    with open(os.path.join(args.out, "weights.bin"), "wb") as stream:
        # The embedders' frequency tables are not in the state dicts, and the exact float32 values matter: the
        # position angles reach ~3000 rad, so a 1-ulp difference in a frequency moves a feature by ~1e-3.
        oct_state = dict(oct_state)
        gs_state = dict(gs_state)
        oct_state["pos_embedder.freqs"] = oct_model.pos_embedder.freqs.float()
        gs_state["pos_embedder.freqs"] = gs_model.pos_embedder.freqs.float()
        half = oct_model.l_embedder.frequency_embedding_size // 2
        oct_state["l_embedder.freqs"] = torch.exp(
            -np.log(oct_model.l_embedder.max_period) * torch.arange(0, half, dtype=torch.float32) / half).float()
        for prefix, state in (("oct.", oct_state), ("gs.", gs_state)):
            for name, tensor in state.items():
                pad = (-offset) % ALIGN
                stream.write(b"\0" * pad)
                offset += pad
                data = tensor.detach().cpu().float().contiguous().numpy().tobytes()
                dims = " ".join(str(d) for d in tensor.shape)
                manifest.append(f"{prefix}{name} {tensor.dim()} {dims} {offset} {len(data)}")
                stream.write(data)
                offset += len(data)

    rep = gs_args["representation_config"]
    manifest.append("rep " + " ".join(str(v) for v in (
        rep["perturbe_size"], rep["3d_filter_kernel_size"], rep["scaling_bias"], rep["opacity_bias"],
        rep["lr"]["_xyz"], rep["lr"]["_features_dc"], rep["lr"]["_opacity"], rep["lr"]["_scaling"],
        rep["lr"]["_rotation"])))

    # Reference case, float64.
    torch.manual_seed(args.seed)
    cond_ch = oct_args["cond_channels"]
    anchors = torch.rand(1, args.anchors, 3)
    cond = torch.randn(1, args.latent, cond_ch)
    level = torch.tensor([float(1 << args.level)])
    with torch.no_grad():
        om, gm = copy.deepcopy(oct_model).double(), copy.deepcopy(gs_model).double()
        om.dtype = gm.dtype = torch.float64
        # The SDK's norms cast to float32 internally; the float64 yardstick runs them in the tensor's own dtype.
        from idol.models import layers
        layers.LayerNorm32.forward = lambda self, x: torch.nn.LayerNorm.forward(self, x)
        layers.MultiHeadRMSNorm.forward = lambda self, x: torch.nn.functional.normalize(x, dim=-1) * self.gamma * self.scale
        # The position embedding's angles reach about 3000 rad, so float32 rounds them to 2e-4 rad. Deployment runs
        # this layer in float32, so the yardstick does too; everything after it is float64.
        for model in (om, gm):
            embedder = model.pos_embedder
            embedder.forward = lambda x, embedder=embedder: embedder.__class__.forward(embedder, x.float()).double()
        # The SDK builds the level embedding in float32 whatever the module's dtype; feed each Linear what it holds.
        for module in list(om.modules()) + list(gm.modules()):
            if isinstance(module, torch.nn.Linear):
                module.register_forward_pre_hook(lambda m, inp: (inp[0].to(m.weight.dtype),))
        ref_oct = om(anchors.double(), level.double(), cond.double())[0].float().numpy()
        ref_gs = gm(anchors.double(), cond.double())[0].float().numpy()
    manifest.append(f"case {args.anchors} {args.latent} {args.level}")
    with open(os.path.join(args.out, "manifest.txt"), "w", encoding="ascii", newline="\n") as stream:
        stream.write("\n".join(manifest) + "\n")
    for name, array in (("anchors", anchors[0].numpy()), ("cond", cond[0].numpy()), ("ref_oct", ref_oct),
                        ("ref_gs", ref_gs)):
        np.ascontiguousarray(array, dtype=np.float32).tofile(os.path.join(args.out, name + ".bin"))
    print(f"wrote {args.out}: {len(oct_state) + len(gs_state)} tensors, reference {ref_oct.shape} and {ref_gs.shape}")


if __name__ == "__main__":
    main()
