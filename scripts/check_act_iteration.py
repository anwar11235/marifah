"""Sanity check: compare single-step vs ACT-iterated z_H statistics on a checkpoint.

Run this BEFORE running ACT probes to confirm:
  1. ACT iterates exactly halt_max_steps times in eval mode
  2. ACT-iterated z_H differs from single-step z_H (ACT is doing work)
  3. Checkpoint loads correctly into act_model.inner

Usage:
    python scripts/check_act_iteration.py \\
        --checkpoint checkpoints/warmstart_cold/final.pt \\
        --config configs/warmstart_cold.yaml \\
        [--device cuda] \\
        [--n_tokens 32]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ACT iteration sanity check")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--config", required=True, help="TrainingConfig YAML")
    p.add_argument("--device", default="cuda", help="Device: cuda or cpu (default: cuda)")
    p.add_argument("--n_tokens", type=int, default=32,
                   help="Synthetic sequence length for the check (default: 32)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from marifah.training.config import load_config
    from marifah.training.trainer import build_model
    from marifah.training.checkpointing import load_checkpoint
    from marifah.models.coral_base import CoralConfig, InnerCarry
    from delta_probe_act import _build_act_model, _load_checkpoint_into_act

    config = load_config(args.config)
    device = torch.device(args.device)

    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"halt_max_steps: {config.model.halt_max_steps}")
    print()

    # ---- Single-step model (CoralV3Inner) -----------------------------------
    print("Loading single-step model (CoralV3Inner) ...")
    single_model = build_model(config, device)
    load_checkpoint(args.checkpoint, single_model)
    single_model.eval()

    # ---- ACT-wrapped model --------------------------------------------------
    print("Building ACT-wrapped model (CoralV3ACT) ...")
    act_model = _build_act_model(config, device)
    _load_checkpoint_into_act(act_model, args.checkpoint, device)

    # ---- Synthetic batch ----------------------------------------------------
    B = 1
    N = min(args.n_tokens, config.model.max_nodes)
    inputs = torch.randint(0, config.model.vocab_size, (B, N), device=device)
    batch = {"inputs": inputs}

    # ---- Single-step forward ------------------------------------------------
    with torch.no_grad():
        coral_cfg = CoralConfig(
            batch_size=B, seq_len=config.model.max_nodes,
            vocab_size=config.model.vocab_size,
            H_cycles=config.model.H_cycles, L_cycles=config.model.L_cycles,
            H_layers=config.model.H_layers, L_layers=config.model.L_layers,
            hidden_size=config.model.d_model, num_heads=config.model.num_heads,
            use_predictive_coding=True, use_hmsc=False,
            forward_dtype=config.model.forward_dtype,
            halt_max_steps=config.model.halt_max_steps,
            halt_exploration_prob=config.model.halt_exploration_prob,
        )
        carry_single = InnerCarry.zeros(B, coral_cfg, device)
        result_single = single_model(carry_single, batch, is_last_segment=True)
        z_H_single = result_single[0].z_H.float().cpu()

    # ---- ACT-iterated forward -----------------------------------------------
    with torch.no_grad():
        act_carry = act_model.initial_carry(batch)
        steps_run = 0
        for step in range(config.model.halt_max_steps):
            act_carry, _ = act_model(act_carry, batch)
            steps_run += 1
            if act_carry.halted.all():
                break
        z_H_act = act_carry.inner_carry.z_H.float().cpu()
        act_steps_tensor = act_carry.steps.cpu()
        all_halted = bool(act_carry.halted.all().item())

    # ---- Report -------------------------------------------------------------
    max_diff = (z_H_act - z_H_single).abs().max().item()
    are_different = not torch.allclose(z_H_single, z_H_act)

    print("=== ACT Iteration Check ===")
    print()
    print(f"  halt_max_steps config:  {config.model.halt_max_steps}")
    print(f"  steps_run (actual):     {steps_run}")
    print(f"  act_carry.steps:        {act_steps_tensor.tolist()}")
    print(f"  all_halted:             {all_halted}")
    print()
    print(f"  z_H single-step :  mean={z_H_single.mean():+.6f}  std={z_H_single.std():.6f}  "
          f"norm={z_H_single.norm():.4f}")
    print(f"  z_H ACT {steps_run:d}-step  :  mean={z_H_act.mean():+.6f}  std={z_H_act.std():.6f}  "
          f"norm={z_H_act.norm():.4f}")
    print(f"  max |diff|:             {max_diff:.6f}")
    print(f"  are_different:          {are_different}")
    print()

    if steps_run == config.model.halt_max_steps and all_halted:
        print("[OK] ACT ran exactly halt_max_steps iterations and halted correctly.")
    else:
        print(f"[WARN] ACT ran {steps_run}/{config.model.halt_max_steps} steps; all_halted={all_halted}")

    if are_different:
        print("[OK] ACT and single-step produce DIFFERENT z_H — ACT is doing work.")
    else:
        print("[WARN] ACT and single-step produce IDENTICAL z_H — ACT may not be functioning.")

    print()
    print("If both [OK] lines appear, proceed to run delta_probe_act.py and shuffled_probe_act.py.")


if __name__ == "__main__":
    main()
