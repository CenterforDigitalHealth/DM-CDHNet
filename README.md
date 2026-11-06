# JMDC clinical trajectory pipeline

This branch provides selected DM-CDHNet model, training, evaluation and generation
source code for research inspection. It is not a complete runnable distribution.
Patient data, checkpoints, results, figures, and selected internal runtime files,
dependencies, provenance and tests are intentionally excluded from Git.

The usage instructions below describe the complete local project. They require
files that are not published on this branch; cloning this branch alone is not
sufficient to run the examples or reproduce the reported results.

## Local results bundle

The local copy also contains `checkpoints/final.pt`, saved research results under
`results/`, and rendered figures under `figures/outputs/`. Open
`LOCAL_RESULTS.html` in a browser to browse the supplied manuscript figures and
search the saved tables and PDFs. These artifacts are ignored by Git; a source-only
checkout will not contain them. See `docs/LOCAL_ARTIFACTS.md` for scope and limits.

## Quick check

Python 3.10+ is required. Run commands from this directory.

```bash
python -m pip install -r requirements.txt
python pipeline.py doctor
python pipeline.py smoke --out outputs/smoke
python -m unittest discover -s tests -v
```

The smoke command creates synthetic records, trains a tiny CPU model, evaluates
validation predictions, fits validation-only dose calibration, evaluates the
synthetic test cohort, generates trajectories, and writes a summary and chart.
**These results check software operation, not scientific model performance.**
An existing output directory is rejected rather than overwritten.

## One complete run

```bash
python pipeline.py doctor --config configs/pipeline.example.json
python pipeline.py run --config configs/pipeline.example.json
```

Edit the example to point to authorized data. Paths in a run configuration are
relative to that configuration file. Missing inputs are reported before a run
starts. The example trains a new model; it does not reproduce the published
training schedule or checkpoint.

For the published checkpoint, use `configs/inference.example.json`. That example
skips training and enables the published FIRST/same-day correction. The checkpoint,
correction code and fitted parameters must match the hashes in
`docs/checkpoint_provenance.json`. `native` inference is also explicit and is the
only mode used for newly trained/synthetic checkpoints. Applying the published
correction to arbitrary new weights is rejected.

The validation input must match the validation-data hash in the checkpoint before
calibration is fitted. Test data cannot be used for calibration. The final test
run checks the calibration's mode, time quantiles, policy, correction and checkpoint
provenance. Medication calibration is never silently applied to generated
conditional medication logits.

Outputs are separate stage directories:

```text
outputs/run/
  pipeline.json       # RUNNING / FAILED / PASS, completed stages and configuration
  training/           # best/last checkpoints; omitted when checkpoint is supplied
  validation/         # raw predictions, reporting mask, metrics and provenance
  calibration.json
  evaluation/         # policy-filtered test metrics; raw native scores retained
  generation/         # intact histories/records, metrics, exclusions and cap flags
  report/             # summary.json, metrics.csv, overview.png and manifest
```

## Individual stages

Use `python pipeline.py COMMAND --help` for the exact options. The commands are
`train`, `evaluate`, `calibrate`, `generate`, `prepare-temporal`,
`generate-temporal`, and `report`. Each also has a direct Python entry point.

```bash
python train.py --config configs/train.json --train-data data/train.bin \
  --val-data data/val.bin --device cuda:0 --out outputs/training
python train.py --config configs/train.json --train-data data/train.bin \
  --val-data data/val.bin --device cuda:0 --out outputs/training \
  --resume outputs/training/last.pt
python evaluate.py --checkpoint outputs/training/best.pt --data data/val.bin \
  --labels data/labels.csv --mode landmark --out outputs/validation
python generate.py --checkpoint outputs/training/best.pt --data data/test.bin \
  --labels data/labels.csv --patients 128 --replicates 3 \
  --horizon-days 1096 --out outputs/generation
```

`train.py` supports single CPU/GPU training and exact resume of its own runs.
The historical four-GPU continuation program is retained as `train_model.py` for
checkpoint provenance; it is not the general training entry point. Multi-GPU
historical checkpoints can be loaded for inference, but are not interchangeable
with the new trainer's optimizer/RNG resume state.

Raw data are headerless little-endian uint32 rows `(ID, AGE, EVENT, DOSE, DUR)`.
Patients must be contiguous, ages chronological, and train/validation patient IDs
disjoint. The final model supports ten medication classes. `data/labels.csv` is
only the active token dictionary, not patient data.

## Evaluation definitions

- `prefix` / `landmark` evaluation predicts the original next clinical target,
  marginalizing predicted TIME. It is not fixed-horizon disease risk.
- Window-head metrics are reported separately with their own horizon and
  follow-up-proxy definition. They are not substituted for next-event metrics.
- P00–P96 exclusions are resolved from the supplied dictionary. Original excluded
  targets are removed from scoring; a later target is never substituted. Histories,
  training data, native probability mass, and generated records remain unchanged.
  Endpoint, target-row and window exclusions are reported with recomputed denominators.
- Generation counts and recovery summaries use policy-filtered scoring records.
  Raw generated and observed records remain available. Event caps, death and
  incomplete refill intervals are explicitly recorded.
- The evaluator defaults to at most 100,000 raw target observations. Its two EVENT
  buffers can require about 1 GB at that limit, excluding model/batch RAM. Larger
  prefix studies must use bounded patient shards; the guard fails before the
  over-limit prediction and never truncates silently. Choosing a larger explicit
  limit requires enough RAM. Shard-level AUCs must not be averaged as pooled AUCs.

Temporal preparation accepts history and a strictly later future file:

```bash
python prepare_temporal.py --history data/history.bin --future data/future.bin \
  --out outputs/temporal
python generate_temporal.py --checkpoint checkpoints/final.pt \
  --cohort outputs/temporal/cohort.bin --index outputs/temporal/index.npz \
  --cohort-manifest outputs/temporal/manifest.json --labels data/labels.csv \
  --correction final --out outputs/temporal_generation
```

The temporal history must end at a clinical token. A trailing laboratory row is
rejected even on the same day, rather than predicting from an unsupervised position
or silently deleting history. Temporal artifacts are checked against their hashes,
patient indices, chronology, boundaries and death semantics before generation.

## Research analysis and manuscript figures

`analysis/` and `figures/` contain the curated research workflows with functional
names. Install their optional dependencies with:

```bash
python -m pip install -r requirements-analysis.txt
```

`provenance/manuscript_map.csv` maps 41 manuscript items to source code and states
which matches are verified, historical, or unresolved. Numerical reproduction of
those analyses additionally needs their original cohorts, fitted arrays, trajectory
outputs and provenance files. Their inclusion does not mean all manuscript results
were rerun, or that older evaluators already applied the current exclusion policy.

Preserved workflows use `project_paths.py` for local code and historical assets.
The default asset location is `assets/`. Set `JMDC_ARTIFACT_ROOT` explicitly to an
archive with the expected historical layout, or add exact path mappings in
`configs/asset_paths.json`. No default read/write falls back to the old server's
absolute workspace. The public pipeline above uses its explicit run configuration
and does not need this historical asset archive.

Two different versions of a figure helper are not overwritten: current manuscript
helpers are in `figures/`, and the original portable runtime helpers are retained
in `legacy/runtime_figures/`. `legacy/root/` contains verbatim copies of every root
Python file from before consolidation. These are historical references, not active
alternatives to the final model modules.

## Validation and limits

The nine frozen model source hashes are preserved. Tests cover policy resolution,
raw-history/score preservation, ten medication classes, temporal rejection,
calibration leakage guards, bounded evaluation memory, source paths, exact resume,
and the full synthetic pipeline in an isolated copy of this directory.

See `docs/CODE_REVIEW.md` and `provenance/verification.json` for actual verification.
Real-cohort training/evaluation, GPU throughput, external APIs, and full manuscript
figure reproduction were not run during consolidation. The local results bundle
includes the hash-verified final checkpoint and copies of saved results, but no raw
cohort dataset, intermediate calculation caches or experimental checkpoints. The
download ZIP excludes runtime caches and synthetic smoke outputs. Git publication includes only the selected source files.
