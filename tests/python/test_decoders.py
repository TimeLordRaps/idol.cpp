"""Parity test: the C++ Idol decoders against the official SDK modules, on small random decoders.

Usage: test_decoders.py <path to idol_decode> <path to the tools directory> [tolerance [decoder_probe [octree_compare]]]

Exports seeded random octree and Gaussian decoders (tools/export_idol.py --random-dim) built from the SDK's own classes,
checks the C++ outputs against the SDK's float64 run, then decodes a latent to a PLY and checks the file. When the test
probe is supplied, also compares packed decoder weights against their dense dequantized equivalents. The SDK source
(a checkout of https://github.com/speridlabs/idol-sdk, the folder that contains the "idol" package) is found through the
IDOL_SDK environment variable. Exits 77, which ctest reads as "skipped", when torch, safetensors, plyfile or the SDK is
missing. The tolerance defaults to 2e-6 (float32 weights measure about 2e-7).
"""

import importlib.util
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

SKIPPED = 77


def main():
    if len(sys.argv) not in (3, 4, 5, 6):
        print(__doc__)
        return 2
    runner, tools = Path(sys.argv[1]), Path(sys.argv[2])
    tolerance = sys.argv[3] if len(sys.argv) >= 4 else "2e-6"
    probe = Path(sys.argv[4]) if len(sys.argv) >= 5 else None
    compare = Path(sys.argv[5]) if len(sys.argv) == 6 else None
    sdk = os.environ.get("IDOL_SDK", "")
    missing = [m for m in ("torch", "safetensors", "plyfile") if importlib.util.find_spec(m) is None]
    if missing or not (Path(sdk) / "idol" / "models" / "decoder.py").is_file():
        print("skipped: needs", ", ".join(missing) or "the SDK source in IDOL_SDK")
        return SKIPPED

    with tempfile.TemporaryDirectory(prefix="idol_decoders_") as scratch:
        model = Path(scratch) / "model"
        subprocess.run([sys.executable, str(tools / "export_idol.py"), "--random-dim", "64", "--sdk", sdk, "--out", str(model),
                        "--anchors", "64", "--latent", "96"], check=True)
        checked = subprocess.run([str(runner), str(model), "--check", "--tolerance", tolerance])
        if checked.returncode:
            return checked.returncode
        octree = subprocess.run([str(runner), str(model), "--check", "--octree-only", "--threads", "1", "--tolerance", tolerance],
                                check=True, capture_output=True, text=True)
        assert "OCT_RESULT" in octree.stdout and "oct_prob_mean=" in octree.stdout and "gaussian decoder" not in octree.stdout
        probability_report = Path(scratch) / "probability-report.json"
        subprocess.run([sys.executable, str(tools / "eval_octree_probabilities.py"), "--runner", str(runner),
                        "--candidate", f"same={model}", "--case", str(model), "--out", str(probability_report),
                        "--level", "5", "--threads", "1", "--reference-label", "same"], check=True)
        probability_summary = json.loads(probability_report.read_text())
        assert probability_summary["status"] == "complete" and probability_summary["hashes_unchanged_after_run"]
        assert probability_summary['reference_validation']['status'] == 'passed'
        assert probability_summary["summary"]["same"]["median_case_mean_total_variation"] < 1e-4
        wrong_level = Path(scratch) / 'wrong-level.json'
        failure = subprocess.run([sys.executable, str(tools / 'eval_octree_probabilities.py'), '--runner', str(runner),
                                  '--candidate', f'same={model}', '--case', str(model), '--out', str(wrong_level),
                                  '--level', '8', '--threads', '1', '--reference-label', 'same'],
                                 capture_output=True, text=True)
        assert failure.returncode != 0
        bad_report = json.loads(wrong_level.read_text())
        assert bad_report['status'] == 'failed' and bad_report['reference_validation']['status'] == 'failed'

        if compare:
            result = subprocess.run([str(compare), str(model), str(model), str(model / "cond.bin"), "--anchors", "64",
                                     "--level", "4", "--seed", "3", "--threads", "1", "--candidate", str(model),
                                     "--compare-levels", "1,4", "--noise-seed", "3"],
                                    check=True, capture_output=True, text=True)
            assert result.stdout.count("cell_total_variation=0.000000000") == 6, result.stdout
            assert result.stdout.count("candidate=-1") == 2, result.stdout
            result = subprocess.run([str(compare), str(model), str(model), str(model / "cond.bin"), "--anchors", "64",
                                     "--level", "4", "--seed", "3", "--threads", "1", "--probability-only",
                                     "--probability-levels", "1,2,4"], check=True, capture_output=True, text=True)
            assert result.stdout.count("PROB_RESULT") == 3 and result.stdout.count("weighted_mean_tv=0.000000000") == 3
            result = subprocess.run([str(compare), str(model), str(model), str(model / "cond.bin"), "--level", "4",
                                     "--compare-levels", "5"], capture_output=True, text=True)
            assert result.returncode != 0 and "comparison level" in result.stderr
            invalid = Path(scratch) / "invalid-latent.bin"
            invalid.write_bytes(struct.pack("<f", float("nan")))
            result = subprocess.run([str(compare), str(model), str(model), str(invalid), "--threads", "1"],
                                    capture_output=True, text=True)
            assert result.returncode != 0 and "non-finite latent" in result.stderr
            report = Path(scratch) / "octree-report.json"
            command = [sys.executable, str(tools / "eval_octree.py"), "--runner", str(compare), "--reference", str(model),
                       "--candidate", f"same={model}", "--latent", str(model / "cond.bin"), "--out", str(report),
                       "--anchors", "64", "--level", "4", "--compare-levels", "1,4", "--threads", "1"]
            subprocess.run(command, check=True)
            contents = report.read_bytes()
            summary = json.loads(contents)
            assert summary["status"] == "complete" and summary["hashes_unchanged_after_run"]
            assert all(summary["summary"][level]["same"]["median_cell_total_variation"] == 0 for level in ("1", "4"))
            assert subprocess.run(command, capture_output=True).returncode != 0 and report.read_bytes() == contents

        if probe:
            # Compare the actual quantized decoder path with the same decoded weights in dense form.
            # This tests the loader-to-kernel connection, independently of whether quantization is accurate.
            sys.path.insert(0, str(tools))
            import numpy as np
            import idolq
            import quant_idol
            import audit_idol_exports
            import audit_octree_preservation

            keep, tensors = quant_idol.parse_manifest(model)
            plan = {name: (4, 32) for name, shape, _, _ in tensors if quant_idol.eligible(name, shape, 32)}
            assert plan, "test did not select any decoder weights"
            quantized, dense = Path(scratch) / "quantized", Path(scratch) / "dense"
            quant_idol.write_tier(model, quantized, keep, tensors, plan, {}, "rtn")
            dense.mkdir()
            source = np.memmap(model / "weights.bin", dtype=np.uint8, mode="r")
            manifest = list(keep)
            with (dense / "weights.bin").open("wb") as stream:
                for name, shape, off, nbytes in tensors:
                    array = np.frombuffer(source[off:off + nbytes], dtype="<f4").reshape(shape)
                    if name in plan:
                        codes, scale, minimum = idolq.quantize_tensor(array, 4, 32, "rtn")
                        array = idolq.dequantize(codes, scale, minimum, 32).numpy()
                    padding = (-stream.tell()) % idolq.ALIGN
                    stream.write(b"\0" * padding)
                    offset = stream.tell()
                    stream.write(np.ascontiguousarray(array, dtype="<f4").tobytes())
                    manifest.append(f"{name} {len(shape)} {' '.join(str(d) for d in shape)} {offset} {nbytes}")
            del array, source
            (dense / "manifest.txt").write_text("\n".join(manifest) + "\n", encoding="ascii")
            env = dict(os.environ, IDOL_QUANT_KERNEL="float", OMP_NUM_THREADS="2")
            for export, prefix in ((quantized, "q"), (dense, "d")):
                subprocess.run([str(probe), str(export), str(model), str(Path(scratch) / prefix)], check=True, env=env)
            for suffix in ("oct", "gs"):
                q = np.fromfile(Path(scratch) / f"q.{suffix}", dtype="<f4")
                d = np.fromfile(Path(scratch) / f"d.{suffix}", dtype="<f4")
                assert q.size and q.shape == d.shape and np.isfinite(q).all() and np.isfinite(d).all()
                np.testing.assert_allclose(q, d, rtol=0, atol=float(tolerance))
            print("quantized decoder matches its dense dequantized weights")

            preserved = Path(scratch) / "preserved"
            built = Path(scratch) / "built-nearest"
            build_report = Path(scratch) / "build-report.json"
            subprocess.run([sys.executable, str(tools / "build_quant_candidate.py"), "--export", str(model),
                            "--out", str(built), "--report", str(build_report), "--bits", "8", "--group", "32",
                            "--method", "rtn"], check=True, env=env)
            build_info = json.loads(build_report.read_text())
            assert build_info["calibration_source_provenance"] == "not applicable"
            assert audit_octree_preservation.audit(model, built)["quantized_gaussian_matrices"] > 0
            # The selector must preserve the highest-sensitivity eligible Gaussian matrix and bind the input export.
            eight_plan = quant_idol.uniform_plan(tensors, 8, 32, preserve_octree=True)
            chosen = sorted(eight_plan)[0]
            sensitivity_report = Path(scratch) / "sensitivity.json"
            sensitivity_report.write_text(json.dumps({"hashes_unchanged_after_run": True, "bits": 8, "group": 32,
                "input_sha256": {"manifest": audit_idol_exports.digest(model / "manifest.txt"),
                                 "export_weights": audit_idol_exports.digest(model / "weights.bin")},
                "matrices": {name: {"max_relative_squared_error": float(name == chosen)} for name in eight_plan}}))
            selected_out, selected_report = Path(scratch) / "selected", Path(scratch) / "selected.json"
            subprocess.run([sys.executable, str(tools / "build_quant_candidate.py"), "--export", str(model),
                            "--out", str(selected_out), "--report", str(selected_report), "--bits", "8", "--group", "32",
                            "--method", "rtn", "--sensitivity-report", str(sensitivity_report), "--keep-top-sensitive", "1"],
                           check=True, env=env)
            assert json.loads(selected_report.read_text())["additional_float32_matrices"] == [chosen]
            _, selected_tensors = audit_idol_exports.manifest(selected_out / "manifest.txt")
            assert "bits" not in selected_tensors[chosen]
            preserve_plan = quant_idol.uniform_plan(tensors, 4, 32, preserve_octree=True)
            quant_idol.write_tier(model, preserved, keep, tensors, preserve_plan, {}, "rtn")
            preservation = audit_octree_preservation.audit(model, preserved)
            assert preservation["octree_tensors"] > 0 and preservation["quantized_gaussian_matrices"] > 0
            _, preserved_tensors = audit_idol_exports.manifest(preserved / "manifest.txt")
            first_octree = next(v for n, v in preserved_tensors.items() if n.startswith("oct.") and v["size"])
            with (preserved / "weights.bin").open("r+b") as stream:
                stream.seek(first_octree["offset"])
                byte = stream.read(1)
                stream.seek(first_octree["offset"])
                stream.write(bytes([byte[0] ^ 1]))
            try:
                audit_octree_preservation.audit(model, preserved)
            except ValueError as exc:
                assert "octree tensor bytes differ" in str(exc)
            else:
                raise AssertionError("changed octree tensor accepted")

        # Invalid cases must fail rather than report a misleading successful comparison.
        case = Path(scratch) / "bad-case"
        case.mkdir()
        for name in ("anchors.bin", "cond.bin", "ref_oct.bin", "ref_gs.bin"):
            shutil.copyfile(model / name, case / name)
        def must_fail(message, *extra):
            result = subprocess.run([str(runner), str(model), "--check", "--case", str(case), "--threads", "1", *extra],
                                    capture_output=True, text=True)
            assert result.returncode != 0 and message in result.stderr, (extra, result.stdout, result.stderr)

        must_fail("repeat must be positive", "--bench", "1", "--repeat", "0")
        must_fail("tolerance must be finite", "--tolerance", "nan")
        oct_ref = (case / "ref_oct.bin").read_bytes()
        (case / "ref_oct.bin").write_bytes(oct_ref[:-4])
        must_fail("reference size mismatch")
        (case / "ref_oct.bin").write_bytes(struct.pack("<f", float("nan")) + oct_ref[4:])
        must_fail("non-finite decoder output or reference")
        (case / "ref_oct.bin").write_bytes(oct_ref + b"x")
        must_fail("invalid float file size")

        ply = Path(scratch) / "out.ply"
        count = 512
        subprocess.run([str(runner), str(model), "--random-latent", "96", "--gaussians", str(count), "--level", "4",
                        "--seed", "3", "--out", str(ply)], check=True)
        data = ply.read_bytes()
        header, _, body = data.partition(b"end_header\n")
        lines = header.decode("ascii").splitlines()
        assert lines[0] == "ply" and "format binary_little_endian 1.0" in lines, "not a binary PLY"
        assert f"element vertex {count}" in lines, "wrong vertex count"
        properties = [l.split()[2] for l in lines if l.startswith("property")]
        assert properties[:3] == ["x", "y", "z"] and len(properties) == 17, properties
        assert len(body) == count * 17 * 4, "wrong body size"
        values = struct.unpack(f"<{count * 17}f", body)
        assert all(math.isfinite(v) for v in values), "non-finite value in the PLY"
        # z-up -> y-up: x stays in [-0.5 - spread, 0.5 + spread], and each quaternion is unit length.
        for i in range(count):
            row = values[i * 17:(i + 1) * 17]
            assert abs(math.sqrt(sum(q * q for q in row[13:17])) - 1.0) < 1e-4, "rotation is not a unit quaternion"
        print("PLY ok:", count, "Gaussians")
    return 0


if __name__ == "__main__":
    sys.exit(main())
