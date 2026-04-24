# Temporal Traffic Flow Prediction using STGCN and ASTGCN

## Project Overview
Traffic forecasting is a critical task in **Intelligent Transportation Systems (ITS)**, enabling better congestion management, route planning, and urban mobility decisions.  
This project addresses **short-term traffic speed prediction** by modeling traffic networks as **graphs** and learning both **spatial** and **temporal** dependencies using **Graph Neural Networks (GNNs)**.

We implement and compare two state-of-the-art models:
- **STGCN (Spatio-Temporal Graph Convolutional Network)**
- **ASTGCN (Attention-based Spatio-Temporal Graph Convolutional Network)**

Both models are evaluated on the real-world **PEMS04** traffic dataset.

---

## Theoretical Background

### Traffic as a Graph Problem
Traffic sensor networks naturally form a graph:
- **Nodes:** Traffic sensors (loop detectors)
- **Edges:** Road network connectivity or distance-based relationships
- **Node features:** Traffic flow, speed, and occupancy over time

Graph-based modeling allows the network to capture **spatial correlations**, such as congestion propagation across neighboring road segments.

---

### Temporal Modeling
Traffic data exhibits strong temporal dependencies due to:
- Rush-hour patterns
- Daily and weekly periodicity
- Sudden disruptions (accidents, weather conditions)

Instead of recurrent models, this project uses **temporal convolutions**, which:
- Enable parallel computation
- Are more stable during training
- Capture short- and mid-range temporal dependencies effectively

---

### STGCN (Spatio-Temporal Graph Convolutional Network)
STGCN integrates:
1. **Graph Convolutions (GCN)** to model spatial dependencies
2. **Temporal Convolutions (TCN)** to capture temporal dynamics

#### Key Characteristics:
- Aggregates information from neighboring sensors
- Uses a fully convolutional architecture
- Assumes **fixed spatial relationships** between sensors

---

### ASTGCN (Attention-based STGCN)
ASTGCN extends STGCN by incorporating **attention mechanisms**.

#### Attention Components:
- **Spatial Attention:** Learns dynamic importance of neighboring sensors
- **Temporal Attention:** Focuses on the most informative time steps
- **Feature (Channel) Attention:** Weighs the importance of input features

#### Advantages:
- Adapts to changing traffic conditions
- Handles non-stationary traffic patterns better
- Improves prediction accuracy, especially during peak congestion periods

---

## Dataset: PEMS04

- **Number of sensors:** 307  
- **Time interval:** 5 minutes  
- **Features per sensor:**
  - Traffic Flow
  - Traffic Speed
  - Occupancy  

### Prediction Task
- **Input:** Past **12** time steps  
- **Output:** Next **12** time steps  

### Data Split
- Train: 70%  
- Validation: 10%  
- Test: 20%  

### Dataset Source
The dataset is obtained from **Kaggle**:  
https://www.kaggle.com/datasets/elmahy/pems-dataset

Due to its large size, the dataset is **not included in this repository**.

### Setup Instructions
1. Download the dataset from the Kaggle link above.
2. Extract the files.
3. Place the dataset in the following location:

```
data/
└── PEMS04.npz
└── PEMS04.csv
```

---

## Repository Structure
```
.
├── project.ipynb
├── data/
│   └── PEMS04.npz
|   └── PEMS04.csv
├── models/
│   └── saved/
│       ├── STGCN_best.pth
│       └── comparison_results.txt
├── requirements.txt
├── README.md
├── .gitignore
```

---

## Cloning the Repository
```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_NAME>
```

---

## Environment Setup

### Requirements
- Python **3.8+**
- PyTorch
- CUDA-enabled GPU (recommended but optional)

### Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate         # Windows
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Project

### Jupyter Notebook
```bash
jupyter lab
```

Open `project.ipynb` and run all cells sequentially.

---

## Model Configuration
- **Optimizer:** Adam  
- **Loss Function:** Mean Absolute Error (MAE)  
- **Epochs:** 10  
- **Prediction Horizon:** 12 steps  
- **Device:** GPU / CPU  

---

## Results
- **ASTGCN consistently outperforms STGCN**
- ~6% reduction in MAE
- ~2.5% reduction in RMSE

---

## Limitations
- Fixed graph structure
- Short prediction horizon
- Limited hyperparameter tuning

---

## Future Work
- Dynamic or learned adjacency matrices
- Longer-term forecasting
- Integration with external factors
- Comparison with DCRNN and Graph WaveNet

---

## License
Academic and educational use only.

---

## Authors
Developed as part of an academic project on  
**Spatio-Temporal Traffic Flow Prediction using Graph Neural Networks**.
