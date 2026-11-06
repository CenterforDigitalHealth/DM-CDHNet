# DM-CDHNet

DM-CDHNet is a research project for modeling longitudinal health records and
exploring how clinical events and treatment patterns evolve over time. It brings
together predictive modeling and trajectory generation within a shared framework.

The project focuses on learning from patient histories to represent clinical
context, estimate future events, and explore possible trajectories.

## Research overview

The framework combines sequence modeling with temporal information and treatment
attributes. Its main research areas include:

- **Clinical representation:** learning patterns across longitudinal records.
- **Prediction:** estimating clinical events and associated attributes from prior history.
- **Temporal modeling:** representing the timing and duration of events.
- **Trajectory generation:** producing possible continuations of clinical histories.
- **Evaluation:** examining predictive performance, calibration, and generated sequences.

## Modeling approach

A transformer-based architecture processes clinical histories as sequences of
structured events. The model combines shared representations with specialized
prediction components for clinical and temporal information.

Training, inference, evaluation, and generation are organized into separate modules.
Simulation utilities extend the generation workflow to explore alternative
trajectories and treatment conditions.

## Code organization

| Component | Files | Purpose |
| --- | --- | --- |
| Model | `model.py`, `config.py`, `token_roles.py` | Model architecture, configuration, and event definitions |
| Temporal modeling | `time_distributions.py` | Time and duration distributions |
| Training | `train.py`, `train_model.py` | Training workflows |
| Inference and generation | `inference.py`, `generate.py`, `generate_dynamic.py`, `generate_temporal.py` | Model loading and trajectory generation |
| Evaluation | `evaluate.py`, `evaluate_auc.py`, `evaluate_generation.py`, `evaluation_policy.py`, `calibrate.py` | Performance assessment, evaluation rules, and calibration |
| Simulation | `simulation/` | Trajectory sampling and supporting model adaptations |
| Workflow | `pipeline.py`, `utils.py` | Workflow coordination and shared utilities |

This repository shares selected source code for research review and methodological
reference. Datasets, trained model weights, generated outputs, and some supporting
execution resources are not included. Running the workflows requires the
corresponding resources and environment setup.
