# Torso models with standard 12-lead electrode positions (independent GT)

Source: Bender et al., "Spatial distribution of Wilson's central terminal on the
body surface", Zenodo, DOI 10.5281/zenodo.20086105, license CC-BY-4.0.

Contents (copied here for reproducible offline validation):
- `T_01..T_25_torso_coarse_surface.vtk` — torso surface point clouds (mm,
  shared shape-model coordinate space)
- `T_01..T_25_electrodes.csv` — 10 electrode positions per model (4 limb
  leads + V1-V6). Column order is unlabelled; rows 4-9 are V1-V6, established
  geometrically (parasternal -> midclavicular -> midaxillary, with the expected
  4th/5th intercostal vertical pattern; see docs/ECG_GT_VALIDATION.md).
- `meantorso_*` — mean torso surface and electrode positions

Use: calibrates and validates the V1-V6 rules in `configs/ecg_rules.yaml`
(the 4th->5th ICS vertical drop regression) and exposes rule errors that a
self-consistent ground truth cannot. Redistribution requires attribution.

Validation entry point: `scripts/validate_rules_vs_torso_models.py`
Report: `runs/gt_validation/torso_models_report.json`
