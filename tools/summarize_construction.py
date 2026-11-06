"""Aggregate construction pilots after checking that preview inputs match the decoded point clouds."""
import argparse
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or len(set(args.candidate)) != len(args.candidate):
        parser.error("output exists or candidate names repeat")
    values = {name: [] for name in args.candidate}
    seen, common, common_preview = set(), None, None
    for case in args.case:
        report = json.loads((case / "report.json").read_text())
        if report["status"] != "complete" or not report.get("hashes_unchanged_after_run"):
            raise ValueError("incomplete construction report")
        latent_hash = report["inputs"]["latent"]
        if latent_hash in seen:
            raise ValueError("duplicate latent")
        seen.add(latent_hash)
        config = (report["gaussians"], report["level"], report["activation_kernel"],
                  report["inputs"]["runner"], report["inputs"]["exports"])
        if common is None:
            common = config
        if config != common:
            raise ValueError("decode settings or model inputs differ")
        for name in args.candidate:
            preview = json.loads((case / ("preview-" + name) / "report.json").read_text())
            if (not preview.get("hashes_unchanged_after_run") or len(preview["views"]) != 6 or
                    preview["input_sha256"]["reference"] != report["outputs"]["reference"]["sha256"] or
                    preview["input_sha256"]["candidate"] != report["outputs"][name]["sha256"]):
                raise ValueError("preview does not match decoded inputs")
            preview_config = (preview["input_sha256"]["tool"], preview["size"], preview["extent"])
            if common_preview is None:
                common_preview = preview_config
            if preview_config != common_preview:
                raise ValueError("preview tool or settings differ")
            views = list(preview["views"].values())
            comparison = report["comparisons"][name]
            values[name].append({"foreground_rgb_difference": statistics.mean(v["foreground_rgb_mean_absolute_difference"] for v in views),
                                 "worst_view_foreground_rgb_difference": max(v["foreground_rgb_mean_absolute_difference"] for v in views),
                                 "silhouette_overlap": statistics.mean(v["alpha_threshold_0_5_iou"] for v in views),
                                 "mean_reference_center_distance": comparison["reference_to_candidate"]["mean_center_distance"],
                                 "voxel_overlap_32": comparison["center_voxel_iou"]["32"]})
    summary = {}
    for name, cases in values.items():
        summary[name] = {"median_case_mean_foreground_rgb_difference": statistics.median(c["foreground_rgb_difference"] for c in cases),
                         "worst_view_foreground_rgb_difference": max(c["worst_view_foreground_rgb_difference"] for c in cases),
                         "median_case_mean_silhouette_overlap": statistics.median(c["silhouette_overlap"] for c in cases),
                         "median_mean_reference_center_distance": statistics.median(c["mean_reference_center_distance"] for c in cases),
                         "median_voxel_overlap_32": statistics.median(c["voxel_overlap_32"] for c in cases)}
    result = {"scope": "Finite sampled Gaussian centers and six approximate orthographic preview views per latent. "
              "The preview is not the upstream renderer or a ground-truth visual-quality evaluation.",
              "case_count": len(seen), "gaussians": common[0], "level": common[1], "activation_kernel": common[2],
              "summary": summary}
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
