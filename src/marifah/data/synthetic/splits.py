"""§7 — Split generation: train/val/test_id/test_ood_size/test_ood_composition.

Splits use disjoint seed ranges so no DAG can appear in more than one split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterator, List, Optional, Tuple

from marifah.data.synthetic.generator import DagGenerator, GenerationTask
from marifah.data.synthetic.labels import DAGRecord
from marifah.data.synthetic.vertical_config import GeneratorConfig, SplitSizes


# ---------------------------------------------------------------------------
# Seed range allocation (disjoint per split)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SeedRange:
    start: int
    end: int   # exclusive

    def __len__(self) -> int:
        return self.end - self.start


def allocate_seed_ranges(base_seed: int, sizes: SplitSizes) -> Dict[str, SeedRange]:
    """Return disjoint seed ranges for each split."""
    MULTIPLIER = 10  # room for retries / workflow expansion
    ranges: Dict[str, SeedRange] = {}
    cursor = base_seed

    for split, n in [
        ("train", sizes.train),
        ("val", sizes.val),
        ("test_id", sizes.test_id),
        ("test_ood_size", sizes.test_ood_size),
        ("test_ood_composition", sizes.test_ood_composition),
    ]:
        budget = n * MULTIPLIER
        ranges[split] = SeedRange(cursor, cursor + budget)
        cursor += budget

    return ranges


# ---------------------------------------------------------------------------
# Split generation
# ---------------------------------------------------------------------------

class SplitGenerator:
    def __init__(self, config: GeneratorConfig, num_workers: int = 1) -> None:
        self.config = config
        self.num_workers = num_workers
        self.generator = DagGenerator(config)
        self.seed_ranges = allocate_seed_ranges(config.seed, config.split_sizes)

    def generate_train(self) -> List[DAGRecord]:
        sr = self.seed_ranges["train"]
        n = self.config.split_sizes.train
        return self.generator.generate_split(
            "train", n, sr.start, num_workers=self.num_workers
        )

    def generate_val(self) -> List[DAGRecord]:
        sr = self.seed_ranges["val"]
        n = self.config.split_sizes.val
        return self.generator.generate_split(
            "val", n, sr.start, num_workers=self.num_workers
        )

    def generate_test_id(self) -> List[DAGRecord]:
        sr = self.seed_ranges["test_id"]
        n = self.config.split_sizes.test_id
        return self.generator.generate_split(
            "test_id", n, sr.start, num_workers=self.num_workers
        )

    def generate_test_ood_size(self) -> List[DAGRecord]:
        """§7.3 — Same workflows, but 2–5× larger DAGs."""
        sr = self.seed_ranges["test_ood_size"]
        n = self.config.split_sizes.test_ood_size
        return self.generator.generate_split(
            "test_ood_size", n, sr.start,
            ood_size=True,
            num_workers=self.num_workers,
        )

    def generate_test_ood_composition(self) -> List[DAGRecord]:
        """§7.4 — Novel primitive compositions not seen in training."""
        sr = self.seed_ranges["test_ood_composition"]
        n = self.config.split_sizes.test_ood_composition
        return self.generator.generate_split(
            "test_ood_composition", n, sr.start,
            require_reserved_pair=True,
            num_workers=self.num_workers,
        )

    def generate_split_streaming(
        self,
        split_name: str,
        batch_size: int = 10_000,
    ) -> Iterator[List[DAGRecord]]:
        """Yield batches of records for split_name, streaming shards to disk progressively."""
        sr = self.seed_ranges[split_name]
        n = getattr(self.config.split_sizes, split_name)
        extra_kwargs: Dict[str, bool] = {}
        if split_name == "test_ood_size":
            extra_kwargs["ood_size"] = True
        elif split_name == "test_ood_composition":
            extra_kwargs["require_reserved_pair"] = True
        yield from self.generator.generate_split_streaming(
            split_name, n, sr.start,
            num_workers=self.num_workers,
            batch_size=batch_size,
            **extra_kwargs,
        )

    def generate_all(self) -> Dict[str, List[DAGRecord]]:
        return {
            "train":                self.generate_train(),
            "val":                  self.generate_val(),
            "test_id":              self.generate_test_id(),
            "test_ood_size":        self.generate_test_ood_size(),
            "test_ood_composition": self.generate_test_ood_composition(),
        }
