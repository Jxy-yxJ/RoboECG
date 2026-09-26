"""Load and validate the ECG rule config with provenance enforcement.

Every anatomical quantity in `configs/ecg_rules.yaml` must carry a provenance
tag from a fixed vocabulary.  The loader refuses configurations that contain
an un-sourced anatomical constant, which is the project rule: no hand-written
medical numbers.

Pure-python except for PyYAML; safe to unit test without Isaac Sim.
"""
from __future__ import annotations

from pathlib import Path

import yaml

PROVENANCE_KINDS = {
    "clinical_definition",
    "asset_measurement",
    "published_regression",
    "published_statistic",
    "learned_model",
    "measured_or_learned",
    "engineering",
    "simulation_assumption",
}

# Provenance kinds that may carry a fixed numeric value in the config.
VALUE_BEARING_KINDS = {
    "clinical_definition",
    "asset_measurement",
    "published_regression",
    "published_statistic",
    "learned_model",
    "engineering",
    "simulation_assumption",
}

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "ecg_rules.yaml"
)


class ProvenanceError(ValueError):
    """Raised when the rule config violates the provenance policy."""


def load_ecg_rules(path=None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ProvenanceError(f"{path}: top level must be a mapping")
    validate_ecg_rules(config, source=str(path))
    return config


def validate_ecg_rules(config: dict, source: str = "<dict>") -> None:
    anatomy = config.get("anatomy")
    if not isinstance(anatomy, dict) or not anatomy:
        raise ProvenanceError(f"{source}: 'anatomy' section is missing or empty")

    for name, entry in anatomy.items():
        if not isinstance(entry, dict):
            raise ProvenanceError(f"{source}: anatomy.{name} must be a mapping")
        provenance = entry.get("provenance")
        if provenance not in PROVENANCE_KINDS:
            raise ProvenanceError(
                f"{source}: anatomy.{name} has invalid provenance {provenance!r}; "
                f"allowed: {sorted(PROVENANCE_KINDS)}"
            )
        if provenance == "measured_or_learned" and entry.get("value") is not None:
            raise ProvenanceError(
                f"{source}: anatomy.{name} is 'measured_or_learned' but carries a "
                f"fixed value {entry['value']!r}; the value must be null"
            )
        if provenance == "published_statistic" and not entry.get("citation"):
            raise ProvenanceError(
                f"{source}: anatomy.{name} is 'published_statistic' without a citation"
            )
        if provenance == "published_regression" and not entry.get("citation"):
            raise ProvenanceError(
                f"{source}: anatomy.{name} is 'published_regression' without a citation"
            )

    contact = config.get("contact") or {}
    for name, entry in contact.items():
        if not isinstance(entry, dict):
            raise ProvenanceError(f"{source}: contact.{name} must be a mapping")
        provenance = entry.get("provenance")
        if provenance not in PROVENANCE_KINDS:
            raise ProvenanceError(
                f"{source}: contact.{name} has invalid provenance {provenance!r}"
            )
        if provenance in {"published_statistic", "published_regression"} and not entry.get(
            "citation"
        ):
            raise ProvenanceError(
                f"{source}: contact.{name} is '{provenance}' without a citation"
            )


def force_target_n(config: dict) -> float | None:
    """Press force target = published pressure x engineering contact area."""
    contact = config.get("contact") or {}
    pressure = (contact.get("target_pressure_pa") or {}).get("value")
    area = (contact.get("contact_area_m2") or {}).get("value")
    if pressure is None or area is None:
        return None
    return float(pressure) * float(area)


def provenance_report(config: dict) -> list[dict]:
    """Flat, printable provenance table for audit and findings documents."""
    rows = []
    for section in ("anatomy", "contact"):
        for name, entry in (config.get(section) or {}).items():
            rows.append(
                {
                    "path": f"{section}.{name}",
                    "provenance": entry.get("provenance"),
                    "value": entry.get("value"),
                    "derived": entry.get("derived"),
                    "landmark": entry.get("landmark"),
                    "citation": entry.get("citation"),
                }
            )
    return rows
