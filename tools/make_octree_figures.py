"""Plot fixed-parent probability and finite-sample cell comparisons for the octree experiments."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from eval_octree import digest  # noqa: E402


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probabilities", type=Path, required=True)
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-audit", type=Path, required=True)
    parser.add_argument("--priority4", type=Path, required=True)
    parser.add_argument("--priority4-audit", type=Path, required=True)
    parser.add_argument("--priority16", type=Path, required=True)
    parser.add_argument("--priority16-audit", type=Path, required=True)
    parser.add_argument("--octree-float", type=Path, required=True)
    parser.add_argument("--octree-float-audit", type=Path, required=True)
    parser.add_argument("--octree-float-probabilities", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    prob, cells = read(args.probabilities), read(args.cells)
    pfloat, preserved = read(args.octree_float_probabilities), read(args.octree_float)
    plans = {"baseline": read(args.baseline), "priority4": read(args.priority4), "priority16": read(args.priority16)}
    plan_paths = {"baseline": args.baseline, "priority4": args.priority4, "priority16": args.priority16}
    audits = {"baseline": read(args.baseline_audit), "priority4": read(args.priority4_audit),
              "priority16": read(args.priority16_audit)}
    if any(x.get("status") != "complete" or not x.get("hashes_unchanged_after_run") for x in (prob, cells, pfloat)):
        raise ValueError("probability and cell evaluations must be complete with unchanged input snapshots")
    if not prob["cases"] or len(prob["cases"]) != len(cells["cases"]) or len(prob["cases"]) != len(pfloat["cases"]):
        raise ValueError("case counts differ")
    for pc, cc, fc in zip(prob["cases"], cells["cases"], pfloat["cases"]):
        if pc["case_sha256"]["cond.bin"] != cc["latent_sha256"] or pc["case_sha256"] != fc["case_sha256"]:
            raise ValueError("probability cases and sampled latents do not match")
    for label in ("baseline", "priority4", "priority16"):
        if prob["input_sha256"]["candidates"][label + "_4_5"] != cells["input_sha256"]["candidates"][label]:
            raise ValueError(f"candidate hashes differ: {label}")
        if audits[label]["results_sha256"] != digest(plan_paths[label]):
            raise ValueError(f"scored results do not match their audit: {label}")
        for budget, suffix in ((4.5, "_4_5"), (5.0, "_5")):
            audited = next(row for row in audits[label]["candidates"] if row["target_bits_per_quantized_weight"] == budget)
            if prob["input_sha256"]["candidates"][label + suffix] != {
                    "manifest.txt": audited["manifest_sha256"], "weights.bin": audited["weights_sha256"]}:
                raise ValueError(f"candidate does not match audited files: {label} {budget}")
    float_audit = read(args.octree_float_audit)
    if pfloat["input_sha256"]["candidates"]["octree_float"] != {
            "manifest.txt": float_audit["candidate_manifest_sha256"], "weights.bin": float_audit["candidate_weights_sha256"]}:
        raise ValueError("float-octree candidate does not match its audit")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.axisbelow": True})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.grid(color="#e6e5e1", linewidth=0.7)
    for budget, marker, color in ((4.5, "o", "#2a78d6"), (5.0, "s", "#eb6834")):
        for label, caption in (("baseline", "baseline"), ("priority4", "priority 4"), ("priority16", "priority 16")):
            row = next(r for r in plans[label]["points"] if r["target_bpw"] == budget)
            key = label + ("_4_5" if budget == 4.5 else "_5")
            x = 100 * prob["summary"][key]["median_case_mean_total_variation"]
            y = 100 * row["gs_rel"]
            ax.scatter([x], [y], s=65, marker=marker, color=color, edgecolors="#fcfcfb", zorder=3,
                       label=f"{budget:g}-bit budget" if caption == "baseline" else None)
            ax.annotate(f"{caption} · {row['resident_mb']:.0f} MB", (x, y), xytext=(6, 6),
                        textcoords="offset points", fontsize=8, color="#52514e")
    frow = preserved["tiers"][0]
    fx = 100 * pfloat["summary"]["octree_float"]["median_case_mean_total_variation"]
    fy = 100 * frow["gs_rel"]
    ax.scatter([fx], [fy], s=75, marker="D", color="#1b8f69", edgecolors="#fcfcfb", zorder=3,
               label="float octree, 4-bit Gaussian")
    ax.annotate(f"float octree · {frow['resident_mb']:.0f} MB", (fx, fy), xytext=(6, 6),
                textcoords="offset points", fontsize=8, color="#52514e")
    ax.set_xlabel("Median case mean child-probability total variation (%)")
    ax.set_ylabel("Median Gaussian output relative difference (%)")
    ax.set_xlim(left=-0.08, right=max(2.25, ax.get_xlim()[1]))
    ax.set_ylim(bottom=0, top=max(9.0, ax.get_ylim()[1] * 1.1))
    ax.set_title("Octree fidelity and Gaussian tradeoff", loc="left", weight="bold", fontsize=14)
    ax.legend(frameon=False, loc="lower right")
    fig.text(0.1, 0.02, "Six saved fixed-anchor cases; anchors are not necessarily visited parent-cell centers.\n"
             "Labels show resident weight memory in decimal megabytes (MB).\n"
             "Budgets are packed bits per quantized weight, including quantization metadata.\n"
             "Cached calibration provenance is incomplete; this is not visual quality or a full-tree guarantee.",
             fontsize=8, color="#52514e")
    fig.subplots_adjust(left=0.1, right=0.97, top=0.88, bottom=0.25)
    fig.savefig(args.out_dir / "idol-octree-probability-tradeoff.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.3))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.grid(color="#e6e5e1", linewidth=0.7)
    levels = [int(x) for x in cells["comparison_levels"]]
    series = (("baseline", "baseline", "#2a78d6", "o"), ("priority4", "octree priority 4", "#eb6834", "s"),
              ("priority16", "octree priority 16", "#1b8f69", "D"),
              ("reference_other_seed", "reference, different seed", "#777777", "^"))
    for key, label, color, marker in series:
        values = [100 * cells["summary"][str(level)][key]["median_cell_total_variation"] for level in levels]
        if any(value <= 0 for value in values):
            raise ValueError("log-scale chart requires positive median values")
        ax.plot(levels, values, label=label, color=color, marker=marker, linewidth=1.7, markersize=5)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.set_xticks(levels)
    ax.set_xlabel("Spatial cell level (larger is finer)")
    ax.set_ylabel("Median cell-count total variation (%)")
    ax.set_title("Sampled octree counts across spatial scales", loc="left", weight="bold", fontsize=14)
    ax.legend(frameon=False, loc="upper left")
    fig.text(0.1, 0.02, "Vertical axis is logarithmic. Six saved latents; 8,192 sampled anchors per model and seed.\n"
             "Candidates share the reference seed; the grey line uses another reference seed.\n"
             "Finite samples and random-number paths can dominate fine cells; this is not exact distribution distance or visual quality.",
             fontsize=8, color="#52514e")
    fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.25)
    fig.savefig(args.out_dir / "idol-octree-sampled-scales.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
