"""CLI entry point for graph DAG training.

Usage:
    python -m marifah.training.cli train  --config configs/phase0.yaml [--resume CKPT] [--device cuda]
    python -m marifah.training.cli eval   --config configs/phase0.yaml --checkpoint CKPT [--split val]
    python -m marifah.training.cli smoke  --config configs/smoke.yaml  [--device cpu]

The `smoke` subcommand auto-generates a tiny dataset if it does not exist.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import os
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("marifah.cli")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_config(path: str, overrides: dict | None = None):
    from marifah.training.config import load_config
    return load_config(path, overrides)


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _ensure_tiny_dataset(dataset_root: str) -> None:
    """Generate the tiny dataset if the train split does not exist."""
    train_dir = Path(dataset_root) / "train"
    if train_dir.exists() and any(train_dir.glob("shard_*.parquet")):
        logger.info("Tiny dataset found at %s", dataset_root)
        return

    logger.info("Generating tiny dataset at %s ...", dataset_root)
    from marifah.data.synthetic.generator import DagGenerator
    from marifah.data.synthetic.storage import write_split
    from marifah.data.synthetic.vertical_config import (
        GeneratorConfig, SplitSizes, _hash_config
    )

    cfg = GeneratorConfig(seed=42)
    cfg.split_sizes = SplitSizes(
        train=200, val=50, test_id=30, test_ood_size=20, test_ood_composition=20
    )
    cfg.config_hash = _hash_config(cfg)
    gen = DagGenerator(cfg)

    root = Path(dataset_root)
    splits = {
        "train": (cfg.split_sizes.train, 0),
        "val": (cfg.split_sizes.val, 10_000),
        "test_id": (cfg.split_sizes.test_id, 20_000),
        "test_ood_size": (cfg.split_sizes.test_ood_size, 30_000),
        "test_ood_composition": (cfg.split_sizes.test_ood_composition, 40_000),
    }
    for split_name, (n, seed_off) in splits.items():
        records = gen.generate_split(split_name, n, seed_offset=seed_off)
        write_split(records, root, split_name)
        logger.info("Generated %d records for split=%s", len(records), split_name)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> None:
    """Run full training with a config file."""
    config = _load_config(args.config)
    device = _resolve_device(args.device)

    from marifah.training.trainer import build_model, Trainer
    from marifah.training.data_pipeline import build_data_loaders
    from marifah.training.logging import TrainingLogger

    torch.manual_seed(config.experiment.seed)

    model = build_model(config, device)
    data_loaders = build_data_loaders(config)
    logger_obj = TrainingLogger(config)

    trainer = Trainer(model, config, data_loaders, logger_obj, device)

    if args.resume:
        trainer.resume(args.resume)
    elif (config.warm_start is not None
          and config.warm_start.checkpoint is not None):
        if config.warm_start.load_optimizer:
            trainer.resume(config.warm_start.checkpoint)
        else:
            trainer.warmstart(
                config.warm_start.checkpoint,
                remap=config.warm_start.remap,
            )

    trainer.train()


def cmd_eval(args: argparse.Namespace) -> None:
    """Evaluate a saved checkpoint on a dataset split."""
    config = _load_config(args.config)
    device = _resolve_device(args.device)

    from marifah.training.trainer import build_model
    from marifah.training.data_pipeline import build_data_loaders
    from marifah.training.checkpointing import load_checkpoint
    from marifah.training.eval_loop import evaluate

    model = build_model(config, device)
    load_checkpoint(args.checkpoint, model)

    data_loaders = build_data_loaders(config)
    split = args.split or "val"
    loader = data_loaders.get(split)
    if loader is None:
        logger.error("Split '%s' not available.", split)
        sys.exit(1)

    metrics = evaluate(model, loader, config, device)
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.6f}")


def cmd_smoke(args: argparse.Namespace) -> None:
    """Run a tiny smoke test to verify pipeline plumbing end-to-end."""
    config = _load_config(args.config)
    device = _resolve_device(getattr(args, "device", "cpu"))

    # Auto-generate tiny dataset if missing
    _ensure_tiny_dataset(config.data.dataset_root)

    from marifah.training.trainer import build_model, Trainer
    from marifah.training.data_pipeline import build_data_loaders
    from marifah.training.logging import TrainingLogger

    torch.manual_seed(config.experiment.seed)

    model = build_model(config, device)
    data_loaders = build_data_loaders(config)

    if data_loaders.get("train") is None:
        logger.error("Train split unavailable after dataset generation. Aborting.")
        sys.exit(1)

    logger_obj = TrainingLogger(config)
    trainer = Trainer(model, config, data_loaders, logger_obj, device)
    trainer.train()

    # Verification: confirm checkpoint exists
    ckpt = os.path.join(config.logging.checkpoint_dir, "final.pt")
    if not os.path.exists(ckpt):
        logger.error("Final checkpoint not found at %s — smoke FAILED.", ckpt)
        sys.exit(1)

    # Verification: reload checkpoint into fresh model
    from marifah.training.checkpointing import load_checkpoint
    model2 = build_model(config, device)
    meta = load_checkpoint(ckpt, model2)
    logger.info(
        "Smoke test PASSED. Checkpoint reload OK. step=%d epoch=%d",
        meta.get("step"), meta.get("epoch"),
    )

    # Verification: check JSONL log is parseable
    import json
    jsonl = logger_obj.jsonl_path
    if os.path.exists(jsonl):
        records = []
        with open(jsonl) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        logger.info("JSONL log OK: %d records at %s", len(records), jsonl)
        if any(r.get("_type") == "codebook_stats" for r in records):
            logger.info("HMSC codebook utilisation stats present in log.")
        if any(r.get("train/aux_total", 0) != 0 for r in records if r.get("_type") == "step"):
            logger.info("Non-zero aux losses confirmed in log.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="marifah training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = sub.add_parser("train", help="Run training")
    p_train.add_argument("--config", required=True, help="Path to YAML config")
    p_train.add_argument("--resume", default=None, help="Resume from checkpoint")
    p_train.add_argument("--device", default="auto", help="cuda / cpu / auto")

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate a checkpoint")
    p_eval.add_argument("--config", required=True)
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--split", default="val")
    p_eval.add_argument("--device", default="auto")

    # smoke
    p_smoke = sub.add_parser("smoke", help="Smoke test (tiny dataset, fast)")
    p_smoke.add_argument("--config", required=True)
    p_smoke.add_argument("--device", default="cpu")

    args = parser.parse_args()
    dispatch = {"train": cmd_train, "eval": cmd_eval, "smoke": cmd_smoke}
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
