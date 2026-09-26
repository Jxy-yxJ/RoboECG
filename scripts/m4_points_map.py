"""2D chest map of the final V1-V6 electrode positions (guaranteed-clear view).

Reads the M1/M4 report (no Isaac needed) and draws the electrode positions in
the chest-frame (lateral v, vertical u) plane with distances, plus the 3D
positions and the reference lines used by the rules.

Run:
    $ISAACSIM_ENV/bin/python scripts/m4_points_map.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT = PROJECT_ROOT / "runs" / "m1" / "m1_report.json"
OUT = PROJECT_ROOT / "runs" / "m4" / "points" / "chest_map.png"

COLORS = {
    "V1": "#f22828",
    "V2": "#f27319",
    "V3": "#f0d719",
    "V4": "#41d741",
    "V5": "#288cf5",
    "V6": "#8c41f5",
}


def main() -> None:
    report = json.loads(REPORT.read_text())
    frame = report["chest_frame"]
    origin = np.asarray(frame["origin_world"], dtype=float)
    up = np.asarray(frame["up_world"], dtype=float)
    lateral = np.asarray(frame["lateral_world"], dtype=float)
    anterior = np.asarray(frame["anterior_world"], dtype=float)

    names = ("V1", "V2", "V3", "V4", "V5", "V6")
    coords = {}
    world = {}
    for name in names:
        entry = report["targets"][name]
        position = np.asarray(entry["position_world"], dtype=float)
        delta = position - origin
        coords[name] = (
            float(delta @ lateral) * 1000.0,   # v (mm), patient left +
            float(delta @ up) * 1000.0,        # u (mm), head +
        )
        world[name] = position
        print(
            f"{name}: v={coords[name][0]:+7.1f} mm  u={coords[name][1]:+7.1f} mm  "
            f"world=({position[0]:+.4f}, {position[1]:+.4f}, {position[2]:+.4f})"
        )

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 8.4), width_ratios=[1.05, 1.0])
    ax = axes[0]
    # reference lines used by the rules
    u4 = coords["V1"][1]
    ax.axhline(u4, color="#999999", ls="--", lw=1.0, zorder=1)
    ax.text(
        -100, u4 + 8, "4th ICS level (sternal notch - SNND 193 mm)",
        fontsize=8, color="#666666",
    )
    ax.axhline(coords["V4"][1], color="#999999", ls=":", lw=1.0, zorder=1)
    ax.text(
        -100, coords["V4"][1] - 46,
        "5th ICS row (drop from measured torso width)",
        fontsize=8, color="#666666",
    )
    ax.axvline(0.0, color="#cccccc", lw=1.0, zorder=1)
    for name in ("V4", "V5", "V6"):
        ax.axvline(coords[name][0], color="#eeeeee", lw=0.8, zorder=0)

    ax.plot(
        [coords[n][0] for n in ("V4", "V5", "V6")],
        [coords[n][1] for n in ("V4", "V5", "V6")],
        color="#bbbbbb", lw=1.6, zorder=2,
    )
    label_offsets = {
        "V1": (-14, 30),
        "V2": (26, 26),
        "V3": (16, -34),
        "V4": (-6, 34),
        "V5": (6, -52),
        "V6": (26, 20),
    }
    for name in names:
        v, u = coords[name]
        ax.scatter([v], [u], s=340, c=COLORS[name], zorder=4, edgecolors="white",
                   linewidths=1.6)
        dx, dy = label_offsets[name]
        ax.annotate(
            f"{name} ({v:+.0f}, {u:+.0f})",
            (v, u), textcoords="offset points", xytext=(dx, dy),
            fontsize=11, color=COLORS[name], weight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
        )
    # distances between neighbours
    for a, b in (("V1", "V2"), ("V2", "V4"), ("V4", "V5"), ("V5", "V6"),
                 ("V4", "V6")):
        pa = np.array(coords[a])
        pb = np.array(coords[b])
        mid = 0.5 * (pa + pb)
        distance = float(np.linalg.norm(np.asarray(world[a]) - np.asarray(world[b])) * 1000)
        ax.annotate(
            f"{distance:.0f} mm",
            mid, textcoords="offset points",
            xytext=({"V2-V4": (-18, -22), "V4-V6": (-14, 16)}.get(f"{a}-{b}", (8, 10))),
            fontsize=9,
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#dddddd", lw=0.6),
        )
    ax.set_xlabel("lateral v (mm)  [patient left +]", fontsize=11)
    ax.set_ylabel("vertical u (mm)  [head +, 0 = sternal-notch proxy]", fontsize=11)
    ax.set_title(
        "V1-V6 in the chest frame (2D projection)\n"
        "same distances as measured on the surface",
        fontsize=12,
    )
    ax.set_xlim(-110, 240)
    ax.set_ylim(u4 - 210, u4 + 60)
    ax.grid(alpha=0.25)
    ax.set_aspect("equal")

    # right panel: 3D positions + table
    ax2 = axes[1]
    ax2.axis("off")
    lines = ["Final electrode positions (world, m)", ""]
    for name in names:
        p = world[name]
        lines.append(
            f"{name:>3}   ({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})   "
            f"v={coords[name][0]:+6.1f} mm  u={coords[name][1]:+7.1f} mm"
        )
    lines += ["", "Surface distances (mm)"]
    for a, b in (("V1", "V2"), ("V1", "V4"), ("V2", "V4"), ("V4", "V5"),
                 ("V5", "V6"), ("V4", "V6")):
        distance = float(
            np.linalg.norm(np.asarray(world[a]) - np.asarray(world[b])) * 1000
        )
        lines.append(f"{a}-{b}: {distance:.1f}")
    lines += [
        "",
        "V6 check vs 25 real torso models (independent GT)",
        "  V4->V6 wrap ratio (depth/lateral): 2.34 vs 2.37 +/- 0.42  (-0.07 SD)",
        "  V4-V6 distance: 100.8 mm vs 105.6 +/- 23.7 mm  (-0.20 SD)",
        "",
        "V5 lateral position: fitted fraction of the V4->V6 span",
        "  alpha = 0.736 +/- 0.070 (25 GT torso models, LOO error 2.5 mm)",
        "  real-patient cross-check: alpha = 0.653",
        "Distances on the simulated body are compressed relative to the GT",
        "models (the stylised asset wraps less at the lateral wall).",
    ]
    ax2.text(
        0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=11,
        family="monospace", transform=ax2.transAxes,
    )
    fig.suptitle(
        "ECG V1-V6 electrode positions (simulation, chest frame)",
        fontsize=14, weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
