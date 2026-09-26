"""Provenance-policy tests for the ECG rule config."""
from __future__ import annotations

import pytest

yaml = pytest.importorskip("yaml")

from roboecg.target_localization.ecg_rules import (  # noqa: E402
    ProvenanceError,
    force_target_n,
    load_ecg_rules,
    validate_ecg_rules,
)


def minimal_config():
    return {
        "anatomy": {
            "example": {
                "value": None,
                "landmark": "measured_or_learned",
                "provenance": "measured_or_learned",
            }
        }
    }


def test_project_config_is_valid_and_press_force_is_derived():
    config = load_ecg_rules()
    force = force_target_n(config)
    pressure = config["contact"]["target_pressure_pa"]["value"]
    area = config["contact"]["contact_area_m2"]["value"]
    assert force == pytest.approx(pressure * area)


def test_missing_provenance_is_rejected():
    config = minimal_config()
    del config["anatomy"]["example"]["provenance"]
    with pytest.raises(ProvenanceError):
        validate_ecg_rules(config)


def test_unknown_provenance_is_rejected():
    config = minimal_config()
    config["anatomy"]["example"]["provenance"] = "vibes"
    with pytest.raises(ProvenanceError):
        validate_ecg_rules(config)


def test_measured_or_learned_must_not_carry_a_fixed_value():
    config = minimal_config()
    config["anatomy"]["example"]["value"] = 0.02
    with pytest.raises(ProvenanceError):
        validate_ecg_rules(config)


def test_published_statistic_requires_citation():
    config = minimal_config()
    config["anatomy"]["example"]["provenance"] = "published_statistic"
    with pytest.raises(ProvenanceError):
        validate_ecg_rules(config)
