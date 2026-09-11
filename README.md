# DM-CDHNet

DM-CDHNet is a standalone research implementation for longitudinal disease and drug prediction. It preserves the agreed model structure while leaving the final choice of the TIME distribution open. The code can run independently without relying on parent directories, previous experiment modules, or hard-coded data paths.

**This is a new implementation, not a direct port of an existing checkpoint. It must not be assumed to reproduce the performance of previous experiments.** Training on real-world data and matched comparisons with the existing model are required. Validation details are documented in `docs/VALIDATION.md`.

## Architecture

`observed history → TIME → time-conditioned EVENT → medication class/action → class/action/time FiLM → DOSE and DUR`

- The backbone uses RoPE, GQA, RMSNorm, SwiGLU MoE, and alternating local/global causal attention.
- EVENT combines a medication gate with a joint distribution over three actions and seven medication classes. Disease and Death tokens form a separately normalized non-medication distribution, and the complete EVENT distribution sums to one.
- Medication actions are `FIRST_RECORDED`, `CONTINUATION`, and `RESTART`. FIRST is determined from medication history before the current event on the same day. For an existing class, a soft penalty is applied when CONTINUATION or RESTART conflicts with whether the medication appeared within the recent 90-day window. These actions are distinct from increase, maintain, and decrease in DOSE.
- TIME conditions the EVENT and medication-action pathways. DOSE and DUR use FiLM conditioned on medication class, action, and time. Residual and FiLM paths are initialized to behave as identity mappings.
- DOSE uses a binary focal loss for increase versus non-increase. Platt calibration is fitted only on frozen validation predictions and is restricted by checkpoint hash.
- DUR uses **MDL-16**, a logistic mixture that assigns integer probability mass from 0 to 550 days. It is initialized only from the training-duration histogram. Labels outside the supported range raise an error instead of being silently clipped.
- The TIME target is the actual day gap between adjacent **supported clinical events**. Laboratory and demographic rows remain in the input but are excluded from TIME and EVENT target anchors. Zero-day gaps are preserved and are not altered through `mask_ties` or synthetic no-event targets.
- An auxiliary window head predicts whether selected disease codes will be **recorded at least once** within 30, 90, 365.25, 1,095.75, or 1,826.25 days. This objective is separate from the TIME head.

See `docs/DESIGN.md` for architectural decisions and implementation differences, and `docs/TIME.md` for the TIME and discrete-distribution definitions.

## Files

| File | Purpose |
|---|---|
| `model.py` | Backbone, EVENT, action, FiLM, DOSE, MDL-16, and window heads |
| `data.py`, `token_roles.py`, `audit_data.py` | Binary-data contract, clinical gaps, medication state, and input auditing |
| `train_model.py` | Single-GPU/CPU training, fixed validation, best/last checkpoints, and exact resume |
| `evaluate_auc.py` | Next-EVENT AUC/AUPRC, DOSE/DUR metrics, TIME MAE, and separate window evaluation |
| `calibrate_dose.py` | Validation-only Platt calibration |
| `generate.py`, `evaluate_generation.py` | Time-to-event sequential sampling, generated counts, refill intervals, and stopping reasons |
| `build_temporal_test.py`, `evaluate_temporal_auc.py`, `generate_temporal.py` | Evaluation and generation across a strict temporal boundary between corrected history and future-only test data |
| `time_distributions.py` | Pluggable TIME interface and reference adapters for supported distribution families |
| `config.py`, `configs/` | Explicit JSON configuration with validation of unsupported settings |
| `make_smoke_data.py`, `tests/` | Synthetic data generation and contract/integration tests |

## Installation and Input Format

Python 3.10 or later is required. Run commands from the repository root.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The input is a headerless, little-endian, five-column `uint32` binary with columns `(ID, AGE, EVENT, DOSE, DUR)`. AGE and DUR are measured in days. Rows for each patient must be contiguous and ordered chronologically, while preserving the original order of events recorded on the same day. Overlapping patient IDs between TRAIN and VAL are rejected. Patient IDs must be globally consistent across all splits.

| Field or code | Meaning |
|---|---|
| EVENT 0 | Padding only; not valid in stored observations |
| EVENT 1–21 | Non-clinical input; excluded from EVENT output |
| EVENT 22–1277 | Disease tokens |
| EVENT 1278–1284 | Seven medication classes |
| EVENT 1285–1287 | Excluded from output and not treated as medication |
| EVENT 1288 | Death token; terminates generation |
| DOSE 1 / 2 / 3 | Increase / maintain / decrease; valid only for medication rows |

```bash
python audit_data.py --data /absolute/path/train.bin --out outputs/train_audit.json
```

Training samples patients uniformly and predicts the next clinical event at valid clinical positions in a left-context prefix. The target at the final input position may be the next observed clinical event after the context. Future targets are never used to construct input hidden states or medication states. Window labels may use observations after the context boundary.

## Training, Evaluation, and Calibration

### Fixed evaluation roles

The project assigns fixed roles to the evaluation splits:

- **`kr_val`** is used only for single-point prediction evaluation. Metrics include **EVENT AUC, DOSE AUC, DOSE balanced accuracy, DOSE calibration, DUR R2, and DUR RMSE**.
- **`kr_test`** is used only for three-year patient-trajectory generation evaluation. Corrected historical records are attached as a prefix, and generated trajectories are compared with future observations from `kr_test`.

Do not use next-event metrics from `kr_test` for model selection, and do not use `kr_val` for three-year generation evaluation. The detailed data contract is documented in `docs/EVALUATION_PROTOCOL.md`.

The `time_family` field in `configs/base.json` is `null`. The `exponential` value below is an execution example, not the final distribution choice. Supported reference options are `exponential`, `weibull`, and `discrete`.

```bash
python train_model.py --config configs/base.json --time-family exponential --train-data /absolute/path/train.bin --val-data /absolute/path/val.bin --out outputs/exp_run --device cuda:0
python train_model.py --config configs/base.json --time-family exponential --train-data /absolute/path/train.bin --val-data /absolute/path/val.bin --out outputs/exp_run --device cuda:0 --resume outputs/exp_run/last.pt
```

Loss components are written to `metrics.jsonl`. `last.pt` is saved at validation points and at shutdown, while `best.pt` is updated when validation improves. Validation uses the same sampled patients on every pass. Resume requires matching configuration, data, and Python-source hashes. `--stop-after` can stop an intermediate run without changing the learning-rate schedule. The implementation is single-process and should not be assumed to support `torchrun` or DDP.

```bash
python evaluate_auc.py --checkpoint outputs/exp_run/best.pt --data /absolute/path/val.bin --mode prefix --quantiles 16 --out outputs/val_auc --device cuda:0
python calibrate_dose.py --evaluation outputs/val_auc --out outputs/dose_platt.json
python evaluate_auc.py --checkpoint outputs/exp_run/best.pt --data /absolute/path/test.bin --mode prefix --quantiles 16 --calibration outputs/dose_platt.json --out outputs/test_auc --device cuda:0
python evaluate_auc.py --checkpoint outputs/exp_run/best.pt --data /absolute/path/test.bin --mode landmark --out outputs/test_landmark --device cuda:0
```

`prefix` evaluates all valid transitions within prefixes sampled under the same definition used for training. `landmark` evaluates the next clinical event from the final record on the first clinical day at least 365 days after the initial clinical record. Results from these sampling definitions must not be compared as if they used the same population.

EVENT probabilities are approximately integrated over fixed quantiles of the predicted TIME distribution without using the actual future gap. DOSE and DUR are conditioned on the **observed next medication class**, while TIME and action are integrated over their predicted joint distribution. These are attribute-level evaluations and must be distinguished from joint generation accuracy.

Platt scaling calibrates probabilities under this attribute-evaluation definition. It is not automatically applied to a DOSE logit conditioned on a particular sampled time and action during generation. Generation calibration requires a separate validation protocol with the same conditioning definition. Calibration fitting fails if the evaluation was produced from a test-data hash.

## Simulation and Refill Evaluation

```bash
python generate.py --checkpoint outputs/exp_run/best.pt --data /absolute/path/test.bin --out outputs/simulation --patients 128 --replicates 3 --horizon-days 1826 --max-events 1024 --device cuda:0
python evaluate_generation.py --run outputs/simulation
```

At each generation step, the model first samples a gap and then conditions EVENT, medication action, DOSE, and DUR on the sampled time. Medication state is updated from generated history while retaining prescriptions outside the active context window. Zero-day repetitions are allowed. Death, horizon completion, and the event cap are recorded separately. KV-cache optimization is not implemented, so the complete active history is processed again at each step; throughput for large generation workloads must be evaluated separately.

The comparison interval is the shorter of the requested horizon and the last observed follow-up. This is a **proxy** for the actual observation period. Outputs include medication and disease counts, generated-to-observed ratios, count MAE, completed refill intervals, lower bounds for intervals without a refill before follow-up ends, and Death/cap rates. Same-class prescriptions recorded on the same day are merged only when calculating refill dates. Samples that reach the generation cap remain in the results and are explicitly marked; time after the cap is not treated as an event-free observation period.

Differences between completed refill intervals cannot be interpreted in isolation. Medication counts, class-level missing refills, follow-up truncation, generation caps, and prescription duration must be considered together. `records.csv` and patient-level JSON outputs provide the source data for further analysis. This implementation does not estimate medication adherence or causal treatment effects.

A future-only file such as `kr_test.bin`, which lacks sex and historical records, must not be used as a standalone cohort. First attach corrected history so that the sex-row AGE matches the AGE of the first raw token regardless of token type, and store the temporal boundary. The first clinical event (`EVENT >= 22`) is not the reference for this correction. Use verified `*_sexage_firsttoken.bin` history and the contract in `docs/EVALUATION_PROTOCOL.md`. Test targets and observed future events used for generation evaluation must be strictly later than that boundary.

```bash
python build_temporal_test.py --history /data/kr_train_corrected.bin /data/kr_val_corrected.bin --future /data/kr_test.bin --out outputs/kr_test_temporal
python evaluate_temporal_auc.py --checkpoint outputs/run/best.pt --cohort outputs/kr_test_temporal/cohort.bin --index outputs/kr_test_temporal/index.npz --cohort-manifest outputs/kr_test_temporal/manifest.json --out outputs/run/test_auc --device cuda:0
python generate_temporal.py --checkpoint outputs/run/best.pt --cohort outputs/kr_test_temporal/cohort.bin --index outputs/kr_test_temporal/index.npz --cohort-manifest outputs/kr_test_temporal/manifest.json --out outputs/run/test_generation --device cuda:0
```

## Synthetic Smoke Test

```bash
python make_smoke_data.py --out outputs/smoke_data
OMP_NUM_THREADS=1 python train_model.py --config configs/smoke.json --time-family exponential --train-data outputs/smoke_data/train.bin --val-data outputs/smoke_data/val.bin --out outputs/smoke_run --device cpu
OMP_NUM_THREADS=1 python evaluate_auc.py --checkpoint outputs/smoke_run/best.pt --data outputs/smoke_data/val.bin --mode prefix --quantiles 2 --max-patients 8 --out outputs/smoke_val
python calibrate_dose.py --evaluation outputs/smoke_val --out outputs/smoke_platt.json
OMP_NUM_THREADS=1 python generate.py --checkpoint outputs/smoke_run/best.pt --data outputs/smoke_data/test.bin --out outputs/smoke_generation --patients 2 --replicates 1 --horizon-days 90 --max-events 16
```

Synthetic smoke-test metrics are not evidence of research performance. On real-world data, compare against the existing checkpoint under fixed splits, seeds, and training budgets. When selecting a TIME distribution, evaluate not only next-event metrics but also medication counts, refill intervals, cap rates, and long-horizon distributions from multi-step generation.
