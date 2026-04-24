# Urban Traffic Forecasting with Graph Neural Networks

This project builds a deep learning forecasting pipeline for **short-horizon traffic speed prediction** on a city-wide sensor network.  
Instead of treating each sensor independently, it models sensors as nodes in a graph and learns both:

- **Spatial structure** (how nearby roads influence one another)
- **Temporal dynamics** (how traffic evolves over time)

The repository implements and compares two architectures:

- `STGCN` (Spatio-Temporal Graph Convolutional Network)
- `ASTGCN` (Attention-based STGCN)

## Problem Statement

Traffic operations teams need accurate short-term predictions (next 30-60 minutes) to support:

- congestion mitigation,
- signal/control strategy planning,
- route guidance and ETA stabilization.

Classical time-series methods typically miss graph-level road dependencies.  
This project demonstrates how graph neural networks improve prediction quality by explicitly learning topology-aware interactions.

## Dataset and Forecasting Setup

- **Dataset:** PEMS04
- **Sensors (nodes):** 307
- **Sampling interval:** 5 minutes
- **Input channels per sensor:** flow, speed, occupancy
- **Lookback window:** 12 steps
- **Forecast horizon:** 12 steps
- **Split:** 70% train / 10% validation / 20% test

Dataset source: [Kaggle - PEMS dataset](https://www.kaggle.com/datasets/elmahy/pems-dataset)

Expected local placement:

```text
data/
  PEMS04.npz
  PEMS04.csv
```

## Model Design

### STGCN

- Graph convolution blocks for spatial message passing
- Temporal convolution blocks for sequence modeling
- Fully convolutional training flow (parallelizable)

### ASTGCN

Extends STGCN by adding attention over:

- spatial neighborhoods,
- temporal positions,
- feature channels.

This enables dynamic weighting under non-stationary traffic conditions (e.g., peak-hour transitions and incidents).

## Training and Evaluation

- **Framework:** PyTorch
- **Objective:** MAE-based training objective
- **Metrics:** MAE and RMSE on validation/test
- **Optimizer:** Adam
- **Epochs:** 10 (as configured in notebook run)

The notebook trains both models and records metrics and artifacts in `models/saved/`.

## Reproducibility Guide

### 1) Environment setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Run notebook

```bash
jupyter lab
```

Open `project.ipynb` and run top-to-bottom.

### 3) Outputs to expect

- saved checkpoint(s) under `models/saved/`
- printed validation/test metrics
- comparison output file(s)

## Results Summary

From current notebook outputs:

- **STGCN:** MAE `0.0097`, RMSE `0.0233`
- **ASTGCN:** MAE `0.0091`, RMSE `0.0227`

Interpretation:

- ASTGCN reduces both MAE and RMSE versus STGCN
- The attention mechanism improves robustness when sensor influence is context-dependent

## Repository Structure

- `project.ipynb` - end-to-end pipeline (load -> preprocess -> train -> evaluate)
- `requirements.txt` - dependency list
- `models/saved/` - saved checkpoints and comparison artifacts
- `.gitignore` - excludes local data/model artifacts

## Practical Extensions

- learned/dynamic adjacency construction,
- longer forecast windows,
- exogenous inputs (weather, holidays, incident signals),
- production API wrapping around trained checkpoints.
