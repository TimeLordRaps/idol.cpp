"""Plot exploratory real-decoder measurements from the evaluation and benchmark JSON files.

These plots describe fixed-anchor decoder outputs and decoder timings, not end-to-end image generation or visual
quality. Input files must come from comparable, separately audited runs. The script does not validate their provenance.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounding", type=Path, required=True)
    parser.add_argument("--calibrated", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mixed", type=Path, help="optional budgeted mixed-plan results")
    parser.add_argument("--front-out", type=Path, help="write an exploratory size/error comparison")
    parser.add_argument("--busy", type=Path, help="optional loaded-machine timing results")
    parser.add_argument("--busy-out", type=Path, help="write a loaded-machine timing comparison")
    args = parser.parse_args()
    if bool(args.mixed) != bool(args.front_out):
        parser.error("--mixed and --front-out must be supplied together")
    if bool(args.busy) != bool(args.busy_out):
        parser.error("--busy and --busy-out must be supplied together")
    raw, calibrated, timing = read(args.rounding), read(args.calibrated), read(args.timing)
    rows = []
    for bits in (2, 3, 4):
        paired = []
        for title, source, method in (("Nearest-code", raw, "rtn"), ("Feedback-assisted", calibrated, "gptq")):
            matches = [r for r in source["tiers"] if r["bits"] == bits and r["group"] == 32 and r["method"] == method]
            if len(matches) > 1:
                raise ValueError(f"multiple {bits}-bit group-32 {title} results")
            if matches:
                paired.append((f"{bits}-bit\n{title}", matches[0]))
        if len(paired) == 2:
            rows.extend(paired)
    if not rows:
        raise ValueError("no paired group-32 results")
    def timed(label, kernel):
        matches = [r for r in timing if r["label"] == label and r["kernel"] == kernel and r["attention"] == "blocked"]
        if len(matches) != 1:
            raise ValueError(f"expected one timing for {label}, {kernel}")
        return matches[0]
    speed = [("32-bit float", timed("float32", None)), ("4-bit float", timed("4-bit g32", "float")),
             ("4-bit integer requested", timed("4-bit g32", "int8"))]
    if len({(r["rows"], r["threads"], r["load"]) for _, r in speed}) != 1:
        raise ValueError("timing rows, thread settings and background load must match")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.axisbelow": True})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8), gridspec_kw={"width_ratios": [1.35, 1, 1]})
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes:
        ax.set_facecolor("#fcfcfb")
        ax.grid(axis="x", color="#e6e5e1", linewidth=0.7)
        ax.tick_params(length=0, pad=6)
    names = [name for name, _ in rows]
    y = list(range(len(rows)))
    axes[0].barh([i - 0.18 for i in y], [100 * r["oct_rel"] for _, r in rows], 0.34, color="#2a78d6", label="Octree")
    axes[0].barh([i + 0.18 for i in y], [100 * r["gs_rel"] for _, r in rows], 0.34, color="#eb6834", label="Gaussian")
    axes[0].set_yticks(y, names)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Median relative output difference (%)")
    axes[0].set_title("Fixed-anchor output difference", loc="left", weight="bold")
    axes[0].legend(frameon=False, loc="lower right")
    for i, (_, row) in enumerate(rows):
        for offset, key in ((-0.18, "oct_rel"), (0.18, "gs_rel")):
            value = 100 * row[key]
            axes[0].annotate(f"{value:.1f}%", (value, i + offset), xytext=(4, 0), textcoords="offset points",
                             va="center", fontsize=8, color="#333333")
    axes[0].set_xlim(right=max(100 * max(r["oct_rel"], r["gs_rel"]) for _, r in rows) * 1.15)

    labels = [label for label, _ in speed]
    ys = list(range(len(speed)))
    axes[1].barh(ys, [r["gs_s"] for _, r in speed], 0.45, color="#2a78d6")
    axes[1].set_yticks(ys, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Seconds per Gaussian decoder call")
    axes[1].set_title("Decoder time", loc="left", weight="bold")
    for i, (_, row) in enumerate(speed):
        axes[1].annotate(f"{row['gs_s']:.2f} s", (row["gs_s"], i), xytext=(4, 0), textcoords="offset points",
                         va="center", fontsize=8, color="#333333")
    axes[1].set_xlim(right=max(r["gs_s"] for _, r in speed) * 1.2)
    axes[2].barh(ys, [r["peak_mb"] for _, r in speed], 0.45, color="#1baf7a")
    axes[2].set_yticks(ys, labels)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Peak process memory (decimal megabytes)")
    axes[2].set_title("Process memory", loc="left", weight="bold")
    for i, (_, row) in enumerate(speed):
        axes[2].annotate(f"{row['peak_mb']:.0f}", (row["peak_mb"], i), xytext=(4, 0), textcoords="offset points",
                         va="center", fontsize=8, color="#333333")
    axes[2].set_xlim(right=max(r["peak_mb"] for _, r in speed) * 1.2)
    fig.suptitle("Real Idol decoders: exploratory measurements", x=0.06, y=0.98, ha="left", fontsize=16, weight="bold")
    fig.text(0.06, 0.02, "Accuracy: nominal code bits, groups of 32 weights; packed storage includes metadata. "
             "Median across fixed-anchor cases with float activations.\n"
             f"Timing: {speed[0][1]['rows']:,} anchors; timing weights were quantized separately. Integer requests can fall back.\n"
             "Feedback-assisted weights can fall back to nearest-code rounding. No image encoder, denoiser, or visual quality is measured. "
             "Runs are separate; provenance needs audit.", fontsize=8, color="#52514e")
    fig.subplots_adjust(left=0.1, right=0.98, top=0.84, bottom=0.24, wspace=0.8)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)

    if args.mixed:
        mixed = read(args.mixed)["points"]
        if not mixed:
            raise ValueError("no mixed-plan results")
        methods = (("Nearest-code", [r for r in raw["tiers"] if r["bits"] in (2, 3, 4) and r["group"] == 32], "#2a78d6", "o"),
                   ("Feedback-assisted", [r for r in calibrated["tiers"] if r["bits"] in (2, 3, 4) and r["group"] == 32], "#eb6834", "s"),
                   ("Mixed plans", mixed, "#1b8f69", "D"))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
        fig.patch.set_facecolor("#fcfcfb")
        for ax, metric, title in ((axes[0], "oct_rel", "Octree decoder"), (axes[1], "gs_rel", "Gaussian decoder")):
            ax.set_facecolor("#fcfcfb")
            ax.grid(color="#e6e5e1", linewidth=0.7)
            ax.set_title(title, loc="left", weight="bold")
            for name, values, color, marker in methods:
                ax.scatter([r["resident_mb"] for r in values], [100 * r[metric] for r in values],
                           label=name, color=color, marker=marker, s=50, edgecolors="#fcfcfb", linewidths=0.6, zorder=3)
                for r in values:
                    label = f"{r['actual_bpw']:.2f}" if name == "Mixed plans" else str(r["bits"])
                    offset = (5, -12) if name == "Feedback-assisted" else (5, 5)
                    ax.annotate(label, (r["resident_mb"], 100 * r[metric]), xytext=offset,
                                textcoords="offset points", fontsize=7, color="#52514e")
            ax.set_xlabel("Resident weight memory (decimal megabytes)")
            ax.set_ylabel("Median relative output difference (%)")
            ax.set_ylim(bottom=0)
        axes[1].legend(frameon=False, loc="upper right")
        fig.suptitle("Real Idol decoders: exploratory size/error trade-offs", x=0.07, y=0.97,
                     ha="left", fontsize=16, weight="bold")
        fig.text(0.07, 0.03, "Labels: nominal code bits for uniform tiers; packed bits per quantized weight for mixed plans. "
                 "Saved fixed-anchor cases.\nRuns are separate and caches are not fully provenance-bound. These are candidates, "
                 "not proof of a globally optimal frontier or visual quality.",
                 fontsize=8, color="#52514e")
        fig.subplots_adjust(left=0.09, right=0.97, top=0.83, bottom=0.24, wspace=0.3)
        args.front_out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.front_out, dpi=180, facecolor=fig.get_facecolor())
        plt.close(fig)

    if args.busy:
        busy = read(args.busy)["results"]
        keys = ("float32", "4-bit float activations", "4-bit integer activations")
        labels = ("32-bit float weights", "4-bit weights, float activations", "4-bit weights, integer activations")
        fig, ax = plt.subplots(figsize=(10, 4.2))
        fig.patch.set_facecolor("#fcfcfb")
        ax.set_facecolor("#fcfcfb")
        ax.grid(axis="x", color="#e6e5e1", linewidth=0.7)
        for i, key in enumerate(keys):
            values = [r["gaussian_s"] for r in busy[key]["samples"]]
            if not values:
                raise ValueError(f"no samples for {key}")
            median = busy[key]["median_gaussian_seconds"]
            ax.plot([min(values), max(values)], [i, i], color="#2a78d6", linewidth=2)
            ax.scatter(values, [i] * len(values), color="#86b6ef", s=36, zorder=3)
            ax.scatter([median], [i], color="#1c5cab", marker="D", s=55, zorder=4)
            ax.annotate(f"{median:.2f} s", (median, i), xytext=(7, 9), textcoords="offset points", fontsize=9)
        ax.set_yticks(range(len(labels)), labels)
        ax.invert_yaxis()
        ax.set_xlabel("Seconds per Gaussian decoder call")
        ax.set_title("Loaded-machine timing: median and observed range", loc="left", weight="bold", fontsize=14)
        fig.text(0.1, 0.02, "Points are process-level medians; diamonds are medians across processes. "
                 "Ranges are not confidence intervals.\nOther work was active. "
                 "See the accompanying timing data for the exact conditions and file hashes.", fontsize=8, color="#52514e")
        fig.subplots_adjust(left=0.29, right=0.97, top=0.84, bottom=0.28)
        args.busy_out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.busy_out, dpi=180, facecolor=fig.get_facecolor())
        plt.close(fig)


if __name__ == "__main__":
    main()
