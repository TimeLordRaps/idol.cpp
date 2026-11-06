"""Checks on tools/idolq.py that need no runtime: the rounding methods and the packed blob size.

Usage: test_quant_methods.py <path to the tools directory>
Exits 77 (skipped) when torch or numpy is missing.
"""

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

SKIPPED = 77


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    for module in ("torch", "numpy"):
        if importlib.util.find_spec(module) is None:
            print(f"{module} is not installed, so the quantization method test is skipped")
            return SKIPPED
    sys.path.insert(0, sys.argv[1])
    import torch
    import idolq
    import quant_idol
    import quant_idol_search
    import octree_probability_sensitivity as probability_sensitivity

    flat = torch.zeros(1, 8)
    assert float(probability_sensitivity.squared_child_probability_variation(flat, flat + 100)) == 0
    left, right = flat.clone(), flat.clone()
    left[0, 0], right[0, 1] = 100, 100
    assert float(probability_sensitivity.squared_child_probability_variation(left, right)) > 0.999
    with tempfile.TemporaryDirectory(prefix="idol_probability_cache_") as scratch:
        cache_path = Path(scratch) / "cache.json"
        cache = probability_sensitivity.load_cache(cache_path, "expected")
        cache["values"]["oct.test.weight"] = {"2,32": 0.25}
        probability_sensitivity.save_cache(cache_path, cache)
        assert probability_sensitivity.load_cache(cache_path, "expected")["values"] == cache["values"]
        try:
            probability_sensitivity.load_cache(cache_path, "different")
        except ValueError:
            pass
        else:
            raise AssertionError("mismatched sensitivity cache accepted")

    tensors = [("oct.blocks.0.weight", (64, 32), 0, 0), ("gs.blocks.0.weight", (64, 32), 0, 0),
               ("gs.bias", (64,), 0, 0)]
    assert set(quant_idol.uniform_plan(tensors, 4, 32)) == {"oct.blocks.0.weight", "gs.blocks.0.weight"}
    assert quant_idol.uniform_plan(tensors, 4, 32, True) == {"gs.blocks.0.weight": (4, 32)}
    assert quant_idol.uniform_plan(tensors, 16, 0, True) == {"gs.blocks.0.weight": (16, 0)}

    with tempfile.TemporaryDirectory(prefix="idol_existing_export_") as scratch:
        existing = Path(scratch) / "existing"
        existing.mkdir()
        sentinel = existing / "weights.bin"
        sentinel.write_bytes(b"preserve")
        try:
            quant_idol.write_tier(existing, existing, [], [], {}, {}, "rtn")
        except FileExistsError:
            assert sentinel.read_bytes() == b"preserve"
        else:
            raise AssertionError("existing export was overwritten")

    # A target below the cheapest packed representation is infeasible, not permission to exceed it.
    shapes = {"layer": (16, 32)}
    cheap, rich = {"layer": (2, 32)}, {"layer": (4, 32)}
    front = [(0, cheap), (0, rich)]
    cheap_bytes = idolq.blob_bytes(16, 32, 2, 32)
    rich_bytes = idolq.blob_bytes(16, 32, 4, 32)
    assert quant_idol_search.select_within_budget(front, shapes, cheap_bytes - 1) == (None, None)
    assert quant_idol_search.select_within_budget(front, shapes, rich_bytes - 1) == (cheap, cheap_bytes)
    assert quant_idol_search.select_within_budget(front, shapes, rich_bytes) == (rich, rich_bytes)

    # Octree priority affects allocation, not the saved sensitivities or the packed-byte budget.
    import numpy as np
    import alloc
    tiny = [("oct.a.weight", np.zeros((16, 32), dtype=np.float32)),
            ("gs.b.weight", np.zeros((16, 32), dtype=np.float32))]
    sens = {name: {"2,32": 1.0, "4,32": 0.1} for name, _ in tiny}
    candidates = lambda _a: [(2, 32), (4, 32)]
    oct_front = alloc.allocation_front(tiny, quant_idol_search.weight_octree_sensitivity(sens, 4), candidates)
    gs_front = alloc.allocation_front(tiny, quant_idol_search.weight_octree_sensitivity(sens, 0.25), candidates)
    assert oct_front[1][1]["oct.a.weight"] == (4, 32)
    assert gs_front[1][1]["gs.b.weight"] == (4, 32)
    assert sens["oct.a.weight"]["2,32"] == 1.0
    for invalid in (0.0, -1.0, float("nan"), float("inf")):
        try:
            quant_idol_search.weight_octree_sensitivity(sens, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid octree weight accepted")

    # Failed child processes must report their error even when they printed a plausible result line.
    output = "RESULT oct_rel=0 gs_rel=0 oct_s=1 gs_s=1 resident_mb=1 peak_mb=1 pos=0 col=0 opa=0 lsc=0"
    failed = subprocess.CompletedProcess([], 7, stdout=output, stderr="decoder failure detail")
    with patch.object(quant_idol.subprocess, "run", return_value=failed):
        try:
            quant_idol.score("runner", "export", ["case"], 1)
        except RuntimeError as exc:
            assert "exit 7" in str(exc) and "decoder failure detail" in str(exc)
        else:
            raise AssertionError("failed decoder process accepted")
    bad_metrics = subprocess.CompletedProcess([], 0, stdout=output.replace("oct_rel=0", "oct_rel=nan"), stderr="")
    with patch.object(quant_idol.subprocess, "run", return_value=bad_metrics):
        try:
            quant_idol.score("runner", "export", ["case"], 1)
        except RuntimeError as exc:
            assert "non-finite" in str(exc)
        else:
            raise AssertionError("non-finite metrics accepted")
    first = subprocess.CompletedProcess([], 0, stdout=output.replace("oct_rel=0", "oct_rel=0.1"), stderr="")
    second = subprocess.CompletedProcess([], 0, stdout=output.replace("oct_rel=0", "oct_rel=0.3"), stderr="")
    with patch.object(quant_idol.subprocess, "run", side_effect=[first, second]):
        measured = quant_idol.score("runner", "export", ["one", "two"], 1)
        assert measured["cases_count"] == 2 and abs(measured["oct_rel"] - 0.2) < 1e-9
        assert measured["oct_rel_max"] == 0.3
    probabilities = subprocess.CompletedProcess([], 0, stdout=output + " oct_prob_mean=0.1 oct_prob_max=0.4", stderr="")
    with patch.object(quant_idol.subprocess, "run", return_value=probabilities):
        measured = quant_idol.score("runner", "export", ["one"], 1)
        assert measured["oct_prob_tv_mean_median"] == 0.1 and measured["oct_prob_tv_max"] == 0.4
    with patch.object(quant_idol.subprocess, "run", side_effect=[probabilities, first]):
        try:
            quant_idol.score("runner", "export", ["one", "two"], 1)
        except RuntimeError as exc:
            assert "only some cases" in str(exc)
        else:
            raise AssertionError("partial probability metrics accepted")

    torch.manual_seed(0)
    out, inp = 48, 256
    weight = torch.randn(out, inp)
    # Correlated inputs: a few directions carry most of the energy, which is where GPTQ's error feedback pays off.
    basis = torch.randn(16, inp)
    x = torch.randn(4000, 16) @ basis + 0.1 * torch.randn(4000, inp)
    hessian = (x.T @ x) / x.shape[0]
    failures = 0
    for bits, group in [(2, 32), (3, 64), (4, 64)]:
        c, s, m = idolq.quantize_rtn(weight, bits, group, torch.diag(hessian))
        rtn = idolq.dequantize(c, s, m, group)
        c, s, m = idolq.quantize_gptq(weight, hessian, bits, group)
        gptq = idolq.dequantize(c, s, m, group)
        plain_c, plain_s, plain_m = idolq.quantize_rtn(weight, bits, group)
        plain = idolq.dequantize(plain_c, plain_s, plain_m, group)
        err = lambda q: float(((x @ (weight - q).T) ** 2).mean())  # noqa: E731
        print(f"{bits} bits, group {group}: plain {err(plain):.4g}, weighted rtn {err(rtn):.4g}, gptq {err(gptq):.4g}")
        if not err(gptq) < err(rtn):
            print("FAIL: gptq should beat weighted rtn on the layer output")
            failures += 1
        if not err(rtn) <= err(plain) * 1.001:
            print("FAIL: importance-weighted fitting should not lose to plain min/max")
            failures += 1
        if not (c.dtype == torch.uint8 and int(c.max()) <= (1 << bits) - 1):
            print("FAIL: codes out of range")
            failures += 1
        blob = idolq.pack_blob(c, s, m, bits, group)
        if len(blob) != idolq.blob_bytes(out, inp, bits, group):
            print("FAIL: blob size does not match the format")
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
