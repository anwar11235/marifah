"""Config-consistency tests.

These tests exist to catch architecture mismatches between training configs and
checkpoint baselines before they cause silent failures at launch.
"""

from pathlib import Path

import yaml

CONFIG_ROOT = Path(__file__).parent.parent / "configs"

# Sudoku Phase 3c canonical architecture — extracted from checkpoint dump 2026-05-16.
# Update these constants only when switching to a different warm-start checkpoint.
SUDOKU_H_LAYERS = 4
SUDOKU_L_LAYERS = 4
SUDOKU_D_MODEL = 512


def test_warmstart_configs_match_sudoku_checkpoint_architecture():
    """
    The warmstart configs must match the Sudoku Phase 3c checkpoint's
    H/L layer count, or load_state_dict(strict=False) will silently drop
    weights and make the warm-vs-cold comparison meaningless.

    If you change the Sudoku checkpoint to a different architecture, update
    this test deliberately — do not just bump the numbers.
    """
    for cfg_name in ["warmstart_cold.yaml", "warmstart_warm.yaml"]:
        cfg = yaml.safe_load((CONFIG_ROOT / cfg_name).read_text())
        model = cfg["model"]
        assert model["H_layers"] == SUDOKU_H_LAYERS, (
            f"{cfg_name}: H_layers={model['H_layers']} != Sudoku checkpoint "
            f"H_layers={SUDOKU_H_LAYERS}. load_state_dict will silently truncate."
        )
        assert model["L_layers"] == SUDOKU_L_LAYERS, (
            f"{cfg_name}: L_layers={model['L_layers']} != Sudoku checkpoint "
            f"L_layers={SUDOKU_L_LAYERS}."
        )
        assert model["d_model"] == SUDOKU_D_MODEL, (
            f"{cfg_name}: d_model={model['d_model']} != Sudoku "
            f"d_model={SUDOKU_D_MODEL}."
        )


def test_warmstart_cold_and_warm_have_identical_model_architecture():
    """
    Cold vs warm must isolate exactly one variable: checkpoint init.
    Any other model-config difference contaminates the comparison.
    """
    cold = yaml.safe_load((CONFIG_ROOT / "warmstart_cold.yaml").read_text())
    warm = yaml.safe_load((CONFIG_ROOT / "warmstart_warm.yaml").read_text())
    assert cold["model"] == warm["model"], (
        "warmstart_cold and warmstart_warm have differing model configs. "
        f"Cold: {cold['model']}\nWarm: {warm['model']}"
    )
