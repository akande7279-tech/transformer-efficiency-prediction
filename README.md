# Transformer Efficiency Prediction

A deep learning model that predicts power transformer efficiency (%) from electrical, physical, and operational parameters, built with TensorFlow/Keras.

## Overview

Transformer efficiency depends on a mix of design parameters (rated power, voltage levels, losses) and operating conditions (ambient temperature, age, cooling mode). This project generates a realistic synthetic dataset based on standard transformer engineering relationships, then trains a fully connected neural network to predict efficiency from those parameters.

## Dataset

The dataset (5,000 samples) is synthetically generated using standard transformer parameter ranges and a physics-based efficiency equation:

```
Efficiency (%) = Output Power / (Output Power + Total Losses) × 100
```

where total losses combine no-load loss and copper loss, adjusted for an aging degradation factor. Small random noise is added to simulate real-world measurement variance.

**Features:**

| Feature | Description |
|---|---|
| `Rated_Power_kVA` | Transformer rated power |
| `HV_Voltage_kV` / `LV_Voltage_kV` | High/low voltage ratings |
| `HV_Rated_Current_A` / `LV_Rated_Current_A` | Rated currents, derived from power and voltage |
| `NoLoad_Loss_W` | No-load (core) loss |
| `FullLoad_Copper_Loss_W` | Full-load copper loss |
| `Copper_Loss_75C_W` | Copper loss at 75°C reference |
| `Impedance_Voltage_%` | Impedance voltage |
| `NoLoad_Current_%` | No-load current |
| `Cooling_Mode` | ONAN / ONAF / OFAF / ODAF |
| `Vector_Group` | Dyn11 / Yyn0 / Ynd11 |
| `Ambient_Temp_C` | Ambient temperature |
| `Oil_Type` | Mineral / Synthetic |
| `Age_Years` | Transformer age |

**Target:** `Efficiency_%` (clipped to a realistic 90–99.9% range)

## Model Architecture

A sequential dense network with batch normalization and dropout for regularization:

```
Input → Dense(128) → ReLU → BatchNorm → Dropout(0.20)
      → Dense(64)  → ReLU → BatchNorm → Dropout(0.30)
      → Dense(32)  → ReLU → BatchNorm → Dropout(0.10)
      → Dense(16)  → ReLU → BatchNorm → Dropout(0.20)
      → Dense(8)   → ReLU → BatchNorm → Dropout(0.20)
      → Dense(4)   → ReLU → BatchNorm → Dropout(0.10)
      → Dense(1, linear)
```

- **Loss:** MSE
- **Metric:** MAE
- **Optimizer:** Adam (lr=1e-3)
- Categorical features (`Cooling_Mode`, `Vector_Group`, `Oil_Type`) are one-hot encoded
- All features and the target are standardized with `StandardScaler` before training

## Results

| Metric | Value |
|---|---|
| RMSE | *fill in your value* |
| R² | *fill in your value* |

See `notebook.ipynb` for training curves and predicted-vs-actual plots.

## Getting Started

### Installation

```bash
git clone https://github.com/yourusername/transformer-efficiency-prediction.git
cd transformer-efficiency-prediction
pip install -r requirements.txt
```

### Usage

Open and run the notebook:

```bash
jupyter notebook notebook.ipynb
```

Run all cells top to bottom to regenerate the dataset, train the model, and view evaluation plots.

## Project Structure

```
transformer-efficiency-prediction/
├── notebook.ipynb        # Main notebook: data generation, training, evaluation
├── requirements.txt       # Python dependencies
├── README.md
└── .gitignore
```

## Future Improvements

- Train on real transformer nameplate/test data instead of synthetic data
- Hyperparameter tuning (layer sizes, dropout rates, learning rate schedule)
- Try gradient-boosted trees (XGBoost/LightGBM) as a baseline comparison
- Add feature importance / SHAP analysis for interpretability

## License

MIT
