"""P0b v2: robust digitisation + WCT-free matching for the PhysioNet leads.

Two fixes over v1:
  * tracking extraction: each column's dark pixel is chosen nearest to the
    previous column's position, so tall R peaks are followed instead of being
    clipped at the panel band edge;
  * difference matching: the precordial leads share the Wilson central terminal
    (a common temporal component), so V_i - V_1 = node_i - node_1.  Matching the
    *differences* cancels the WCT and makes the node assignment identifiable.

Run:
    $ISAACSIM_ENV/bin/python scripts/physionet_lead_matching_v2.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from physionet_lead_matching import (  # noqa: E402
    EXTERNAL_DIR,
    LEAD_ORDER,
    MAT_URL,
    PDF_URL,
    WORK_DIR,
    bandpass,
    download,
)

V_LEADS = ("V1", "V2", "V3", "V4", "V5", "V6")


def digitise_tracking(pdf_path: Path) -> dict:
    """Extract one waveform per panel by tracking the nearest dark pixel."""
    from PIL import Image

    png = WORK_DIR / "plot-1.png"
    if not png.is_file():
        subprocess.run(
            ["pdftoppm", "-r", "200", "-png", "-f", "1", "-l", "1",
             str(pdf_path), str(WORK_DIR / "plot")],
            check=True,
        )
    image = np.array(Image.open(png).convert("L"))
    dark = image < 128
    height, width = image.shape

    label_cols = dark[:, : int(width * 0.13)]
    rows = np.where(label_cols.sum(axis=1) > 3)[0]
    bands = []
    start = prev = rows[0]
    for row in rows[1:]:
        if row - prev > 10:
            bands.append((start, prev))
            start = row
        prev = row
    bands.append((start, prev))
    centres = [(b[0] + b[1]) // 2 for b in bands][: len(LEAD_ORDER)]

    x0, x1 = 214, 1177
    signals = {}
    for name, centre in zip(LEAD_ORDER, centres):
        previous = None
        values = np.full(x1 - x0, np.nan)
        for column in range(x0, x1):
            candidates = np.where(dark[:, column])[0]
            if candidates.size == 0:
                continue
            if previous is None:
                near = candidates[np.argmin(np.abs(candidates - centre))]
            else:
                near = candidates[np.argmin(np.abs(candidates - previous))]
            # reject jumps larger than half a panel (tracking lost)
            if previous is not None and abs(near - previous) > 60:
                continue
            values[column - x0] = float(near)
            previous = near
        good = np.isfinite(values)
        if good.sum() < values.size * 0.5:
            continue
        values = np.interp(np.arange(values.size), np.where(good)[0], values[good])
        signals[name] = -values
    return signals


def main() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = download(PDF_URL, WORK_DIR / "case0003_15.pdf")
    mat_path = download(MAT_URL, WORK_DIR / "case0003_dat.mat")

    signals = digitise_tracking(pdf_path)
    print(f"digitised: {sorted(signals)}", flush=True)

    import scipy.io as sio

    data = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    potvals = np.asarray(data["bspmdata"].potvals, dtype=float)
    n_nodes, n_samples = potvals.shape

    nodes = np.loadtxt(EXTERNAL_DIR / "case0003_b352.pts") / 1000.0
    electrodes = np.loadtxt(EXTERNAL_DIR / "case0003_b120.pts") / 1000.0
    electrode_lookup = {
        tuple(np.round(point, 6)): index for index, point in enumerate(electrodes)
    }

    # node potentials: zero mean, unit variance per node
    pot = potvals - potvals.mean(axis=1, keepdims=True)
    pot = pot / (pot.std(axis=1, keepdims=True) + 1e-12)

    # digitised leads resampled to the matrix time base
    leads = {}
    for name in V_LEADS:
        signal = signals[name]
        samples = np.interp(
            np.linspace(0.0, 1.0, n_samples),
            np.linspace(0.0, 1.0, signal.size),
            signal,
        )
        samples = bandpass(samples)
        samples = samples - samples.mean()
        leads[name] = samples / (samples.std() + 1e-12)

    # WCT-free matching: d_i = V_i - V_1 = node_i - node_1
    gram = pot @ pot.T / n_samples  # node-node correlations
    results = {}
    for name in V_LEADS[1:]:
        difference = leads[name] - leads["V1"]
        difference = difference - difference.mean()
        difference = difference / (difference.std() + 1e-12)
        g = pot @ difference / n_samples
        # score(a, b) = corr(pot_a - pot_b, difference)
        numerator = g[:, None] - g[None, :]
        denominator = np.sqrt(np.maximum(2.0 - 2.0 * gram, 1e-9))
        score = numerator / denominator
        score[np.arange(n_nodes), np.arange(n_nodes)] = -2.0
        best_a, best_b = np.unravel_index(np.argmax(score), score.shape)
        results[name] = {
            "best_a": int(best_a),
            "best_b": int(best_b),
            "score": float(score[best_a, best_b]),
            "a_position": nodes[best_a].tolist(),
            "b_position": nodes[best_b].tolist(),
            "b_is_electrode": electrode_lookup.get(
                tuple(np.round(nodes[best_b], 6))
            ),
        }
        print(
            f"{name}-V1: a={best_a} b={best_b} score={score[best_a, best_b]:.3f} "
            f"a_pos={np.round(nodes[best_a], 3).tolist()} "
            f"b_pos={np.round(nodes[best_b], 3).tolist()} "
            f"b_electrode={results[name]['b_is_electrode']}",
            flush=True,
        )

    # consistency: V1's node should be the same in every difference
    a_votes = [results[name]["best_a"] for name in V_LEADS[1:]]
    report = {
        "v1_candidate_votes": a_votes,
        "consistent": len(set(a_votes)) == 1,
        "differences": results,
        "provenance": (
            "PhysioNet/CinC 2007 case0003: tracking-digitised 15-lead plot, "
            "WCT-free difference matching against the 352-node BSPM matrix "
            "(ODC-By 1.0)"
        ),
    }
    (WORK_DIR / "lead_matching_v2.json").write_text(json.dumps(report, indent=2) + "\n")
    print("P0b v2: wrote lead_matching_v2.json", flush=True)
    print("V1 candidate votes:", a_votes, flush=True)


if __name__ == "__main__":
    main()
