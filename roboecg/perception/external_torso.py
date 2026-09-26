"""External validation data: PhysioNet/CinC 2007 Dalhousie torso with 120 leads.

Real human torso surface (352 nodes) with measured electrode positions (120),
released under ODC-By 1.0.  Used offline to validate electrode localization and
torso-surface measurements; see assets/external/physionet_cinc2007/README.md
for provenance, license and limitations.

Pure numpy (no Isaac dependency).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "external"
    / "physionet_cinc2007"
)

PROVENANCE = (
    "PhysioNet/CinC Challenge 2007 (Dalhousie torso, 120 leads), "
    "https://physionet.org/content/challenge-2007/1.0.0/, ODC-By 1.0"
)


@dataclass(frozen=True)
class ExternalTorso:
    nodes_m: np.ndarray
    electrodes_m: np.ndarray
    provenance: str = PROVENANCE

    @property
    def electrode_to_surface_distance_m(self) -> np.ndarray:
        """Distance from each electrode to the nearest torso node (metres)."""
        diff = self.electrodes_m[:, None, :] - self.nodes_m[None, :, :]
        return np.linalg.norm(diff, axis=2).min(axis=1)


def load_physionet_torso(data_dir=None) -> ExternalTorso:
    """Load the case-3 torso nodes and electrode positions (mm -> metres)."""
    directory = Path(data_dir) if data_dir else DATA_DIR
    nodes = np.loadtxt(directory / "case0003_b352.pts", dtype=float) / 1000.0
    electrodes = np.loadtxt(directory / "case0003_b120.pts", dtype=float) / 1000.0
    if nodes.shape != (352, 3):
        raise ValueError(f"unexpected torso node shape: {nodes.shape}")
    if electrodes.shape != (120, 3):
        raise ValueError(f"unexpected electrode shape: {electrodes.shape}")
    return ExternalTorso(nodes_m=nodes, electrodes_m=electrodes)


def geometry_report(torso: ExternalTorso) -> dict:
    """Basic measured geometry for the findings document."""
    return {
        "provenance": torso.provenance,
        "nodes": int(torso.nodes_m.shape[0]),
        "electrodes": int(torso.electrodes_m.shape[0]),
        "node_extent_m": (torso.nodes_m.max(axis=0) - torso.nodes_m.min(axis=0)).tolist(),
        "electrode_extent_m": (
            torso.electrodes_m.max(axis=0) - torso.electrodes_m.min(axis=0)
        ).tolist(),
        "electrode_to_surface_m": {
            "mean": float(torso.electrode_to_surface_distance_m.mean()),
            "max": float(torso.electrode_to_surface_distance_m.max()),
        },
        "note": (
            "Raw external measurements (mm -> m). Axis convention is the "
            "Dalhousie one and must be verified on import; the standard 12 "
            "leads are a subset of the 120 electrodes but the V1-V6 index "
            "mapping is not included in the released files."
        ),
    }
