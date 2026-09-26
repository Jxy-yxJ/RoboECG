"""Tests for the external PhysioNet/CinC 2007 torso validation data."""
from __future__ import annotations

import numpy as np

from roboecg.perception.external_torso import (
    geometry_report,
    load_physionet_torso,
)


def test_shapes_and_units():
    torso = load_physionet_torso()
    assert torso.nodes_m.shape == (352, 3)
    assert torso.electrodes_m.shape == (120, 3)
    # A human torso is on the order of tens of centimetres, not metres.
    extent = torso.nodes_m.max(axis=0) - torso.nodes_m.min(axis=0)
    assert np.all(extent > 0.15) and np.all(extent < 0.8)


def test_every_electrode_lies_on_the_torso_surface():
    torso = load_physionet_torso()
    distances = torso.electrode_to_surface_distance_m
    assert distances.max() < 1e-9


def test_geometry_report_is_json_friendly():
    report = geometry_report(load_physionet_torso())
    assert report["nodes"] == 352
    assert report["electrodes"] == 120
    assert report["electrode_to_surface_m"]["max"] < 1e-9
