# CORAL Synthetic DAG Benchmark Specification

**Date:** 2026-05-13
**Status:** Pre-implementation specification. Decision menu in §14; commitments in §2–§8. Generator implementation gated on this doc + vertical decision (game plan §3.3, Week 2).
**Companion docs:** `Aktuator_Game_Plan_2026-05.md` (strategic source-of-truth). `CORAL_DAG_Codebook_Design.md` (architectural companion; depends on this doc for vocabulary sizing and label format).
**Successor artifact:** Build plan (separate doc, drafted immediately after this).

---

## §0 — Why this document exists

The synthetic DAG benchmark is the gating artifact for the validation track. It produces the training data for Phase 0 substrate validation, the structural labels for codebook auxiliary supervision, and the test splits that operationalize all four Layer A claims from the game plan (A1 faithfulness, A2 halt precision, A3 compositional generalization, A4 training-time compounding).

Without this benchmark spec, the codebook implementation can't proceed (vocabulary sizing depends on it), training can't start (no data), and the probes have nothing to measure. This is the load-bearing technical artifact for the next 6 weeks of work.

The spec is *vertical-agnostic in structure* but *parameterizable per vertical*. The same generator machinery produces different benchmarks for different verticals by changing config — what's invariant is the architecture of the generator; what varies is the topology statistics, primitive augmentations, and workflow library specific to the vertical.

---

## §1 — Scope and non-scope

**In scope:**
- DAG generation pipeline (generator architecture, deterministic given seed)
- Primitive vocabulary (the per-position-codebook foundation)
- Sub-DAG pattern library (the regional-codebook foundation)
- Workflow signature library (the global-codebook foundation, plus the recurrence dimension)
- Training distribution design (log-spaced workflow frequencies — load-bearing for A4)
- Splits: train, val, test-ID, test-OOD-size, test-OOD-composition
- Label scheme at all three scales plus per-step execution traces
- Vertical parameterization mechanism (config-driven, single generator)

**Not in scope:**
- Graph adapter (DAG → CORAL input encoding, attention mask, positional encoding) — separate adapter design doc
- Training loop changes — handled in adapter doc + CC prompt
- Real customer data (Layer C uses customer DAGs directly, not synthetic)
- SOP-to-DAG encoder pipeline (LLM-based, upstream product layer)
- Hyperparameters of CORAL training itself (covered in CC prompt)

---

## §2 — Primitive vocabulary (the per-position scale)

### 2.1 Generic primitive set

The default vocabulary contains 10 primitives covering the operations enterprise workflows actually do. Vertical-specific augmentations (§9.3) extend but don't replace this set.

| # | Primitive | Inputs | Output | Operational semantics |
|---|---|---|---|---|
| 1 | `conditional` | state, condition | branch_id ∈ {0, 1} | Evaluate boolean on state; route to one of two child nodes |
| 2 | `aggregate` | list of states | single state | Combine inputs (sum, count, concat, max, min — selected by node attribute) |
| 3 | `lookup` | key, table | value | Retrieve value from a reference table given a key |
| 4 | `compare` | state_a, state_b | ordered ∈ {<, =, >} | Compare two values, produce ordering |
| 5 | `transform` | state, transform_fn | new_state | Apply a deterministic function to state (selected by node attribute) |
| 6 | `validate` | state, constraint | bool | Check whether state satisfies a constraint |
| 7 | `route` | state, routing_table | branch_id ∈ {0, ..., K} | Multi-way branching; route to one of K children based on state |
| 8 | `terminate` | state | (halt) | Workflow terminal node; halts execution |
| 9 | `accumulate` | state, running_total | new_total | Update a running aggregation across the path |
| 10 | `nop` | state | state | Identity / passthrough; useful for synthetic complexity tuning |

**Coverage argument.** This set covers: branching (conditional, route), aggregation (aggregate, accumulate), retrieval (lookup), constraint checking (validate, compare), state transformation (transform), and control flow (terminate, nop). Vertical-specific operations are mostly compositions of these — adding a vertical-specific primitive is the exception, not the rule.

### 2.2 Type signatures and the executor

Each primitive has a Python implementation in the reference executor (§6). The executor is the ground-truth oracle — CORAL's predicted traces are compared against the executor's deterministic output.

Each node in a generated DAG carries:
- A primitive type (one of the 10 above + vertical augmentations)
- Primitive-specific attributes (e.g., the constraint for `validate`, the function for `transform`)
- An input state coming from predecessor nodes
- An output state produced by the primitive

States are typed (currently: int, float, bool, tuple, dict — the executor handles type coercion at edges). Type compatibility is enforced at generation time so every DAG is well-formed.

### 2.3 Why this list and not more

Smaller vocabularies are easier to crystallize cleanly. Larger vocabularies dilute per-primitive sample count and make probe analysis harder. Ten primitives × ~50 workflow types means each primitive appears in many contexts but each has high sample count. Adding primitives is reversible (config change); removing them mid-experiment is not.

If the vertical decision (Week 2) reveals that important operations are missing, the augmentation slot (§9.3) handles that without restructuring the base set.

---

## §3 — Sub-DAG patterns (the regional scale)

### 3.1 Pattern library

The default library contains 12 sub-DAG patterns. Each is a parametric template — instantiation fills in primitives, edges, and attribute values.

| # | Pattern | Size (nodes) | Description |
|---|---|---|---|
| 1 | `linear_chain` | 3–8 | Sequential application of transforms/lookups; no branching |
| 2 | `conditional_fork` | 4–8 | Conditional split, separate processing per branch, no rejoin |
| 3 | `fork_and_join` | 5–12 | Conditional split, parallel processing, aggregate at join |
| 4 | `sequential_validation` | 4–10 | Chain of validate-then-route nodes; failure terminates |
| 5 | `hierarchical_aggregate` | 6–15 | Multiple lookups feeding into one aggregate, then transform |
| 6 | `lookup_and_compare` | 4–7 | Lookup + compare + conditional routing |
| 7 | `multi_way_route` | 4–10 | Single route node with K children, each terminal or short |
| 8 | `validate_then_route` | 5–10 | Validate node controls multi-way routing decision |
| 9 | `accumulate_path` | 5–12 | Accumulate across a chain of transforms; final terminate |
| 10 | `branch_merge_resolve` | 8–18 | Two branches with different transforms, merge, conditional resolution |
| 11 | `lookup_aggregate_validate` | 6–14 | Multi-source lookups, aggregate, validate result against constraint |
| 12 | `constrained_terminate` | 5–10 | Validate-on-entry, transform-if-valid, terminate-with-state |

### 3.2 Pattern composition

A workflow signature (§4) specifies which patterns appear and how they connect. Patterns nest within workflows but are themselves built from primitives (not other patterns) — keeps the hierarchy strictly two-deep (workflow → pattern → primitive).

**Pattern boundaries.** Each pattern has a defined entry node and one or more exit nodes. Workflow assembly connects exit nodes of one pattern to entry nodes of the next. Region segmentation (§8) emits the pattern boundaries as labels — the regional codebook learns to recognize these boundaries.

### 3.3 Pattern parameterization

Each pattern template has:
- A size range (min/max node count) — generator samples a specific size
- Topology constraints (DAG shape — fan-in, fan-out limits per primitive)
- Primitive distribution (which primitives appear; e.g., `sequential_validation` is heavily `validate` + `route`)
- Variation points (where random parameters change behavior without changing identity)

Two instances of the same pattern look different (different sizes, different attributes, different state types) but share the same pattern ID label.

---

## §4 — Workflow signatures (the global scale + recurrence)

### 4.1 Default workflow library

The default library contains 50 workflow signatures. Each is a composition of 2–8 sub-DAG patterns with topology constraints.

Workflow signatures span a complexity spectrum:
- **Simple (workflows 1–10):** 2–3 patterns, 15–40 nodes total
- **Medium (workflows 11–35):** 3–5 patterns, 40–150 nodes
- **Complex (workflows 36–50):** 5–8 patterns, 150–500 nodes

Each workflow has:
- A workflow type ID (1–50)
- A composition spec (which patterns, in what topology)
- A pattern-instance budget (how patterns connect)
- A semantic identity (the workflow's intended "meaning" — only relevant for vertical tuning)

### 4.2 Frequency distribution (load-bearing for A4)

The training set frequency distribution is the most consequential design decision in this doc. Without varied frequencies across workflow types, the A4 compounding probe (game plan §2.1 / codebook design §6.4) is unprovable.

**Default frequency distribution:** log-spaced across 5 orders of magnitude.

For 50 workflow types in a ~1M DAG training set:

| Tier | Workflows | Instances per workflow | Total instances | % of training set |
|---|---|---|---|---|
| Very high | 5 | 100,000 | 500,000 | 50% |
| High | 10 | 20,000 | 200,000 | 20% |
| Medium | 15 | 5,000 | 75,000 | 7.5% |
| Low | 15 | 500 | 7,500 | 0.75% |
| Very low | 5 | 50 | 250 | 0.025% |

Total: ~782,750 instances at this exact split; rounded to ~800K, with the remaining 200K used for OOD splits (§7).

Spans frequencies from 50 to 100,000 — three orders of magnitude in raw count, sufficient for clean A4 regression. The log-spacing ensures the regression has data points across the range, not clustered at one extreme.

**Why this matters operationally.** Halt-step regression for A4 needs multiple frequency points to fit a line. Five tiers gives five log-frequency-anchored cohorts of test instances. If the slope is negative and significant across these tiers, the compounding claim passes.

### 4.3 Workflow-pattern coverage

Each pattern (§3) appears in multiple workflows; each workflow uses multiple patterns. The coverage matrix is designed so:
- Every pattern appears in ≥ 5 different workflows (sufficient sample for regional probe)
- Every workflow uses ≥ 2 distinct patterns (regional structure exists at workflow level)
- Pattern co-occurrence is varied (no two workflows have identical pattern sets)

This is generator-config-validated at generation start; an invalid coverage matrix aborts generation before producing data.

---

## §5 — Generator pipeline

### 5.1 Pipeline stages

```
generator_config (vertical-tuned)
       │
       ▼
[1] sample workflow type (weighted by frequency distribution)
       │
       ▼
[2] retrieve workflow composition spec
       │
       ▼
[3] for each pattern in composition: sample pattern instance (size, primitive distribution)
       │
       ▼
[4] instantiate primitives within each pattern (attributes, types)
       │
       ▼
[5] connect patterns per workflow topology spec
       │
       ▼
[6] randomize state types and values within type constraints
       │
       ▼
[7] reference executor runs the DAG; records ground-truth trace
       │
       ▼
[8] validate: well-formed DAG, executable, unique solution, halt occurs
       │  (fail → discard and resample)
       │
       ▼
[9] emit DAG + labels (workflow type, region/pattern assignments, primitive per node, trace)
```

### 5.2 Determinism and seeding

Generation is fully deterministic given (config_hash, seed). Same config + same seed → byte-identical dataset. Different seeds give different instances within the same distribution.

The generator emits a `manifest.json` per dataset containing: config hash, generator version, seeds used, all per-tier instance counts, and statistical summaries of the produced data (mean node count, primitive distribution, pattern coverage). The manifest is checked in alongside the dataset for reproducibility.

### 5.3 Speed target

Target: 100–1000 DAGs/sec on a single CPU. For 1M training DAGs:
- At 100 DAGs/sec: ~3 hours
- At 1000 DAGs/sec: ~17 minutes

Realistic estimate at ~500 DAGs/sec for 1M training: ~33 minutes. Plus validation passes and label emission: total ~1 hour for a full dataset generation.

The execution step ([7]) is the bottleneck. Reference executor is Python (clarity over speed); if generation is too slow we batch DAGs across workers. Generation is embarrassingly parallel — run with `multiprocessing.Pool` across CPU cores.

### 5.4 Generator output structure

```
dataset_root/
├── manifest.json
├── train/
│   ├── shard_0000.parquet  (~10K DAGs per shard)
│   ├── shard_0001.parquet
│   └── ...
├── val/
├── test_id/
├── test_ood_size/
├── test_ood_composition/
└── generator_config.json
```

Parquet for compactness and HuggingFace Datasets compatibility. Sharded for parallel data loading.

---

## §6 — Execution semantics

### 6.1 What "executing" a DAG means

DAG execution is deterministic state propagation:

1. Initialize state at root nodes (input values per generator)
2. For each non-terminal node in topological order: apply the node's primitive to incoming state, produce outgoing state
3. Continue until terminate node is reached
4. Output: (final state, full execution trace)

### 6.2 Execution trace format

The trace is a sequence of records, one per executed node:

```python
TraceStep = {
    "step": int,           # execution order (0-indexed)
    "node_id": int,        # which node was executed
    "primitive": str,      # primitive type at this node
    "input_state": Any,    # state arriving at this node
    "output_state": Any,   # state leaving this node
    "branch_taken": Optional[int],  # for conditional/route nodes
}
```

The trace is what Claim A1 (execution faithfulness) is computed against: edit distance between CORAL's predicted trace and the ground-truth trace.

### 6.3 Reference executor

Python implementation, deterministic, ~300 LOC for the 10 primitives + traversal logic. Comprehensive unit tests on each primitive (input/output correctness) before integration. The executor is the *oracle* — disagreement between executor and any other reference means executor is right by definition.

### 6.4 Halt semantics

Halt occurs when execution reaches a `terminate` primitive. Every valid generated DAG has at least one path leading to a terminate node. Generator step [8] verifies this; DAGs without a reachable terminate are discarded.

CORAL's halt mechanism (Q-halt) is supervised against this ground-truth halt step. A2 (halt precision) measures: did CORAL halt at the right step?

---

## §7 — Splits

### 7.1 Split design

| Split | Size | Source | Purpose |
|---|---|---|---|
| `train` | ~800K DAGs | Per §4.2 frequency distribution | Substrate training, codebook training |
| `val` | ~10K DAGs | Same distribution as train | Early stopping, hyperparameter selection |
| `test_id` | ~10K DAGs | Same workflows + patterns as train, held-out instances | Claims A1, A2, A4 measurement |
| `test_ood_size` | ~5K DAGs | Same workflows + patterns, but DAG sizes 2–5× larger than training max | Claim B1 (OOD generalization, CLRS-style) |
| `test_ood_composition` | ~5K DAGs | Novel primitive compositions not seen in training | Claim A3 (compositional generalization) |

Total: ~830K DAGs across all splits. Training is the dominant cost.

### 7.2 Test-ID split

In-distribution test. Same workflow type IDs, same pattern library, same primitive vocabulary. New instances (different seeds, different parameter values). Held out from training by seed-range partitioning.

A4 measurement on test_id: for each test DAG, look up its workflow type's frequency in training, regress halt step against log(frequency).

### 7.3 Test-OOD-size split

OOD by scale. Workflows from the same library, but generated at larger sizes (2–5× max node count of training). Tests whether the architecture generalizes execution to longer reasoning chains — the LLM weakness that CLRS-30 captures.

If training maxes at 500-node DAGs, test_ood_size includes 1000-node and 2500-node DAGs. Same workflow types; longer instances.

### 7.4 Test-OOD-composition split

OOD by composition. Specifically engineered for Claim A3. During generator config, a subset of primitive pairs is *reserved* — combinations of primitives that never appear adjacent in training. The test_ood_composition split contains DAGs that use those reserved combinations.

Example: training has primitive pairs (validate→route), (validate→transform), (route→aggregate) but never (validate→aggregate). Test_ood_composition contains DAGs with (validate→aggregate) edges. CORAL has seen both primitives, has seen them in other compositions, but has never seen this specific composition.

If CORAL correctly executes test_ood_composition DAGs, the architecture composes primitives. If not, it memorized training compositions. The pre-registered threshold (codebook §6.3) is ≥ 80% accuracy on this split.

**Holdout strategy decision (OD4-S):** how many primitive pairs are reserved? Default: ~15% of all possible primitive-pair combinations. Final number set during generator config (Week 3).

### 7.5 Split independence

Splits use disjoint random seeds. No DAG appears in more than one split. Generator manifest tracks seed ranges per split for audit.

---

## §8 — Labels and ground truth

### 8.1 Label schema

Per DAG:

```python
DAGLabels = {
    "workflow_type_id": int,           # 1–50, per §4
    "region_assignments": List[Tuple], # (node_id, pattern_id, pattern_instance_id)
    "primitive_assignments": List[int], # primitive type per node, indexed by node_id
    "execution_trace": List[TraceStep], # per §6.2
    "halt_step": int,                  # ground-truth halt step
    "ood_flags": Dict[str, bool],      # which OOD axes this DAG is on (for test splits)
}
```

### 8.2 Region segmentation

Region labels are emitted by the generator (it knows the pattern boundaries at construction time — they're the units of composition). Each node belongs to exactly one region; each region is exactly one pattern instance.

This is cleaner than post-hoc region inference (which would require an unsupervised clustering step). Generator-emitted regions are the ground truth.

The regional codebook supervision (codebook design §5.2) reads `region_assignments` directly.

### 8.3 Label format on disk

Labels are stored alongside DAG structure in the same parquet records. Columns:
- `dag_structure` — serialized graph (node features, edge list)
- `workflow_type_id` — int
- `region_assignments` — list of dicts
- `primitive_assignments` — list of ints
- `execution_trace` — list of dicts (serialized as JSON within parquet for nested-struct flexibility)
- `halt_step` — int
- `ood_flags` — dict (JSON-serialized)

HuggingFace Datasets handles parquet + nested-JSON columns natively.

### 8.4 Label completeness audit

Generator step [9] verifies, before emitting any DAG, that all label fields are populated and self-consistent:
- workflow_type_id ∈ [1, 50]
- region_assignments covers every node exactly once
- primitive_assignments has one entry per node
- execution_trace's halt step matches `halt_step` field
- trace length matches expected for the DAG

Any inconsistency aborts dataset generation. No partial-label data ships.

---

## §9 — Vertical parameterization

### 9.1 The mechanism

Same generator code, different config per vertical. The config controls:

1. Topology statistics (fan-out distribution, depth distribution, node count distribution)
2. Workflow library shape (how many of each tier, which patterns combine)
3. Primitive distribution within workflows
4. Vertical-specific primitive augmentations (§9.3)
5. State type bias (which value types dominate — e.g., financial workflows are float-heavy; coding workflows are string-heavy)

### 9.2 Vertical tuning process

Once a vertical is selected (game plan §3.3, Week 2):

1. Sample 10–20 real workflows from the vertical. Sources, in order of preference:
   - Customer-provided workflows (anonymized) — requires NDA, depends on design partner timing
   - Public examples (BPMN libraries, regulatory documentation, published process diagrams)
   - Vendor documentation (insurance carrier workflow templates, healthcare coding guidelines)
   - Domain SME interviews (when public sources are sparse)

2. Extract structural statistics:
   - Mean and tail of node count
   - Branching factor distribution
   - Depth distribution
   - Pattern frequencies (which sub-DAG motifs appear how often)
   - Cycle prevalence (resolves OD8 cyclic-or-acyclic)

3. Fit generator config:
   - Adjust §4 frequency tier sizes if vertical has more/fewer recurring workflow types than default
   - Adjust §3 pattern weights to match vertical's pattern frequencies
   - Adjust §2 primitive weights to match vertical's operation distribution
   - Add 2–3 vertical-specific primitives if real workflows reveal operations not covered

4. Validate generator output by generating ~1K DAGs and comparing distribution statistics to real-workflow statistics. Iterate config until distributions match within tolerance.

### 9.3 Vertical-specific primitive augmentations

Vertical examples (illustrative, not committed — depends on actual vertical pick):

- **Insurance claims:** `eligibility_check`, `coverage_lookup`, `deductible_apply`
- **Financial reconciliation:** `balance_check`, `transaction_match`, `reconcile`
- **Healthcare ICD-10:** `code_lookup`, `hierarchy_navigate`, `exclusion_check`
- **Regulatory compliance (KYC):** `identity_verify`, `sanction_check`, `risk_score`

Each augmentation is a primitive (semantics + Python implementation in the executor + label) added to the vocabulary for that vertical's config. Vocabulary growth: 10 → 12–13. K_P sizing in the codebook (codebook §3.1) adjusts upward correspondingly.

### 9.4 Vertical config artifact

Per-vertical config is checked in as `configs/vertical_{name}.yaml`. Same generator, different config produces vertical-tuned dataset. Multiple verticals can be generated in parallel if needed (e.g., synthetic for design partner vertical + a second vertical for cross-vertical generalization probes later).

---

## §10 — Cyclic handling

### 10.1 Default and opt-in modes

Default config: **acyclic-only** (cycles forbidden in generated DAGs).

Opt-in: **cyclic mode**, enabled by config flag `allow_cycles: true`. Cyclic mode introduces back-edges between specific node pairs with explicit cycle annotations.

### 10.2 Cyclic semantics (when enabled)

Cycles handled per codebook design §2.3 Option 2: ACT-loop captures the cycle. The DAG is presented as a graph with annotated cycle edges; CORAL's ACT mechanism iterates over the cycle until a halt condition is met. No unrolling — the cycle exists as a structural property of the graph.

Cycle annotations in labels: each cycle edge carries a `cycle_id` and a `halt_condition` (a constraint that, when satisfied, exits the cycle).

### 10.3 Vertical-dependence

Cyclic mode is enabled or disabled per vertical, based on §9 tuning:

- Insurance claims routing, healthcare coding, KYC/AML: predominantly acyclic in real workflows. Default acyclic.
- Financial reconciliation with re-opened accounts, audit/review workflows: cyclic structure is fundamental. Cyclic mode enabled.
- Compliance with iteration loops: depends on subdomain.

The vertical's tuning step (§9.2) makes this decision. The generator supports both; the choice is config-time, not architecture-time.

### 10.4 Codebook coupling

Cyclic mode resolution feeds back to codebook design OD8 (cyclic workflow handling). If cyclic mode is enabled, the codebook design's §2.3 Option 2 (ACT-based cycle handling) is committed; if disabled, the codebook simplifies (no cycle-specific architecture needed).

---

## §11 — Data formats and storage

### 11.1 On-disk format

Per §5.4 directory structure. Parquet files, ~10K DAGs per shard.

Estimated storage:
- Mean DAG: ~100 nodes × ~50 bytes per node (features + labels) = ~5 KB per DAG
- 1M DAGs × 5 KB = 5 GB total
- With parquet compression: ~2–3 GB on disk
- Plus labels (traces, etc.): add 50% = ~4–5 GB total per dataset

Trivial storage on a Vast.ai instance. Generation, transfer, and load are not storage-bound.

### 11.2 Loading

HuggingFace Datasets API:

```python
from datasets import load_from_disk
ds = load_from_disk("dataset_root/train")
# Streaming-compatible; doesn't load all 1M into memory
```

Compatible with HRM-derived training loop; the graph adapter handles the DAG-structure column at batch construction time.

### 11.3 Tokenization

Tokenization is the graph adapter's job, not the generator's. Generator outputs structured DAGs; adapter converts to model input (token embeddings + edge-attention mask + positional encoding) at batch load time.

This separation keeps the generator vertical-agnostic about model details and keeps the adapter focused on encoding.

### 11.4 Determinism on load

Generator output is deterministic. Adapter tokenization is deterministic. Combined: same dataset + same adapter + same seed → byte-identical training batches. Reproducibility is end-to-end.

---

## §12 — Implementation pathway

### 12.1 Components

Implementation breakdown for the CC prompt:

| Component | LOC estimate | Description |
|---|---|---|
| Primitive implementations | ~250 | 10 primitive functions + augmentation slots, type system, attribute schemas |
| Pattern templates | ~300 | 12 pattern classes, instantiation logic, validation |
| Workflow assembler | ~200 | Workflow signature spec, pattern composition, topology assembly |
| Generator pipeline | ~300 | Steps [1]–[9], seeding, manifest generation |
| Reference executor | ~250 | Traversal, state propagation, trace recording, halt detection |
| Validation logic | ~150 | Well-formedness checks, completeness audit, distribution audit |
| Split generation | ~150 | Seed partitioning, OOD-size scaling, OOD-composition holdout |
| Vertical config loader | ~100 | YAML parsing, config validation, primitive augmentation registration |
| Storage/manifest | ~100 | Parquet writing, manifest emission, dataset directory structure |
| Unit tests | ~400 | Per-primitive correctness, per-pattern correctness, executor oracle tests, generator end-to-end smoke test |

Total: ~2200 LOC. Estimated 4–5 days CC implementation.

### 12.2 Implementation order

1. Primitives + executor (foundation; everything downstream depends on these)
2. Pattern templates (test against executor)
3. Workflow assembler (test against pattern templates)
4. Generator pipeline (orchestrates everything)
5. Validation logic
6. Split generation
7. Vertical config loader (deferred until vertical picked)
8. Storage/manifest
9. Unit tests (interleaved with each component, not at end)

### 12.3 Tiny-version first

Before generating the full 1M training set, generate a tiny version (~1K DAGs total across all splits) for:
- Smoke testing (does training loop accept this data?)
- Warm-start comparison (codebook OD7: Sudoku-warm vs from-scratch on tiny DAG benchmark)
- Probe script development (do probes run on this data shape?)

Tiny version generation: ~1 minute. Used heavily in the first few days of integration testing.

### 12.4 Full generation cadence

Full 1M dataset generated once per vertical config. Re-generation only on:
- Vertical config change
- Generator code change that affects output (versioned in manifest)
- Major synthetic-spec revision (new doc version)

Stable training proceeds against the generated dataset for the rest of Phase 0 / Phase 1.

---

## §13 — Coupling with codebook design

### 13.1 Vocabulary sizing (resolves codebook OD1)

- K_P (per-position codebook) = primitive vocabulary size + small margin. Default: 10 primitives + 2–3 augmentations = 12–13. Codebook K_P = 16 (rounded up, allows for vocabulary growth).
- K_R (regional codebook) = pattern library size + small margin. Default: 12 patterns. Codebook K_R = 16.
- K_G (global codebook) = workflow library size. Default: 50 workflows. Codebook K_G = 64 (rounded up, allows for vertical-specific workflow types added to library).

These sizes are pre-committed values for the codebook design; they assume the default workflow library size of 50. If vertical tuning expands the library (some verticals may have ~100 distinct workflow types), K_G adjusts upward at config time.

### 13.2 Cyclic decision (resolves codebook OD8)

Cyclic mode in this doc (§10) is opt-in. Codebook OD8 resolves to:
- Acyclic-only mode → codebook design defers cyclic handling, simplification permitted
- Cyclic mode enabled → codebook design §2.3 Option 2 (ACT-based cycle handling) is committed

Decision propagates from vertical pick → this doc's config → codebook design.

### 13.3 Label format feeds auxiliary loss heads

Codebook §5 auxiliary loss heads consume exactly the labels emitted by this generator:
- Global head consumes `workflow_type_id`
- Regional head consumes `region_assignments` (specifically the pattern_id field)
- Per-position head consumes `primitive_assignments`

No format translation needed. Generator output drops straight into codebook training pipeline.

### 13.4 Probe scripts consume splits

Codebook §6 probes operate against splits defined here:
- A1 (faithfulness): computed on `test_id` against `execution_trace` ground truth
- A2 (halt precision): computed on `test_id` against `halt_step` ground truth
- A3 (compositional generalization): computed on `test_ood_composition`
- A4 (training-time compounding): regression on `test_id`, frequency from `train` distribution per workflow type
- B1 (OOD-size generalization): computed on `test_ood_size`

Probe scripts read directly from the parquet splits and label columns; no separate label files needed.

---

## §14 — Open decisions

| # | Decision | Resolution path | Deadline |
|---|---|---|---|
| OD1-S | Final workflow library size (50 default vs. vertical-expanded) | Set during vertical tuning per §9.2 | Week 3 |
| OD2-S | Exact frequency distribution for A4 (5-tier default; refine count or spacing?) | Sanity-check default during tiny-version generation; adjust if A4 regression is underpowered | Week 4 (during impl) |
| OD3-S | OOD scale multiplier (2× default, 5× option) | Default 2–5× range; if compute-bound at 5×, drop to 2–3× | Week 4 |
| OD4-S | Novel-composition holdout strategy (15% pair holdout default) | Default 15%; expand to 25% if A3 needs harder test, contract if too sparse | Week 3 |
| OD5-S | Vertical-specific primitive augmentations | Depends on vertical pick (game plan §3.3) | Week 3 (after vertical) |
| OD6-S | Cyclic-or-acyclic default for chosen vertical | Resolved by vertical tuning per §10.3 | Week 3 |

### 14.1 OD-S vs. OD coupling

OD1-S, OD6-S resolve directly to codebook OD1, OD8. OD5-S extends primitive vocabulary which affects codebook K_P (codebook OD1 again). The two docs' open decisions are linked; resolving one resolves the other.

### 14.2 No decisions block initial implementation

Generator implementation (CC prompt drafting + ~5 days build) can start with default config values for all OD-S decisions. Tuning happens before the *full* dataset generation, not before *generator implementation*.

This matters for the build plan (next doc) — implementation work isn't gated on vertical decision, just full-dataset generation is.

---

## §15 — Document maintenance

This doc is technical source-of-truth for the synthetic benchmark during the 16-week validation window.

Update protocol:
- Decisions in §14 resolved in-line as they close
- Generator config files versioned alongside this doc (configs/ directory)
- Generated dataset releases versioned (v1.0 first, vN for subsequent generations)
- Major scope changes (new vertical, new primitive class, format change) trigger new doc version
- Updates every two weeks during active generation/use period (Weeks 3–8)

Generator code, generator config, and this doc are versioned together. Reproducing a dataset requires the matched triple.

---

*End of synthetic benchmark specification. Status: active, gated on vertical decision for vertical-specific config. Generator implementation can start in parallel with default config.*
