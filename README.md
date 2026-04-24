# Urban Traffic Forecasting with Graph Neural Networks

This project predicts short-term traffic speed by modeling road sensors as a graph and learning spatial-temporal patterns with deep learning.

It implements and compares:
- `STGCN` (Spatio-Temporal Graph Convolutional Network)
- `ASTGCN` (Attention-based STGCN)

## Why it matters

City-scale traffic data is highly structured in both space (neighboring roads influence each other) and time (rush-hour and periodic effects). This project shows how graph neural networks can capture both dimensions and improve forecasting accuracy.

## Dataset

- **Dataset:** PEMS04
- **Sensors:** 307
- **Sampling rate:** 5-minute intervals
- **Features:** flow, speed, occupancy
- **Task:** use previous 12 steps to predict next 12 steps
- **Split:** 70% train / 10% validation / 20% test

Download: [Kaggle PEMS dataset](https://www.kaggle.com/datasets/elmahy/pems-dataset)

Expected local files:

```text
data/
  PEMS04.npz
  PEMS04.csv
```

## Tech stack

- Python 3.8+
- PyTorch
- NumPy, pandas, scikit-learn, matplotlib
- Jupyter Notebook

## How to run

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies.
4. Place dataset files in `data/`.
5. Launch Jupyter and run `project.ipynb`.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

## Results

From notebook test runs:

- **STGCN:** MAE `0.0097`, RMSE `0.0233`
- **ASTGCN:** MAE `0.0091`, RMSE `0.0227`

ASTGCN improves error metrics versus STGCN, showing the value of learned attention over static neighborhood weighting.

## Repository layout

- `project.ipynb` - training + evaluation pipeline
- `requirements.txt` - dependencies
- `models/saved/` - saved model artifacts and comparisons

## Next improvements

- Dynamic adjacency learning
- Longer forecast horizons
- External signals (weather/events/incidents)
