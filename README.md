# Predict Robot Energy

Support Vector Regression (SVR) and Linear Regression for predicting a mobile robot’s energy consumption from mission parameters.

## Overview

Machine Learning for Robotics assignment (A#03, Spring 2026) by Mohammad Rohaan (i222327). Builds a small tabular dataset, scales features correctly (fit on train only), trains an RBF-kernel SVR, compares against Linear Regression, and discusses limits of tiny datasets.

### Target

`Energy_Consumption_Wh`

### Features

| Feature | Meaning |
|---------|---------|
| `Load_Weight_kg` | Payload mass |
| `Speed_m_per_s` | Travel speed |
| `Terrain_Roughness` | Roughness score |
| `Task_Duration_min` | Mission duration |

## Repository Contents

| File | Description |
|------|-------------|
| `i222327_Mohammad_Rohaan_A_03_ML.ipynb` | Full SVR + LR experiment |
| `Asg-3-ML for Robotics-Spring 2026-NU.pdf` | Assignment specification |

## Methodology

1. Create a compact DataFrame (10 labeled examples)
2. Train/test split (`test_size=0.2`, `random_state=42`)
3. `StandardScaler` — fit on training data only
4. Train `SVR(kernel='rbf')`; predict a new mission profile
5. Visualize actual vs predicted and a smooth SVR curve vs load weight
6. Evaluate with MSE and R²
7. Compare against `LinearRegression` on the same scaled features

### Key takeaway

With only ten samples, both models fail to generalize (negative R² on the tiny test split). SVR still beats Linear Regression on the reported metrics, but more diverse data is required for reliable energy planning.

## Tech Stack

Python 3 · Jupyter · NumPy · pandas · Matplotlib · scikit-learn (`SVR`, `LinearRegression`, `StandardScaler`, `train_test_split`, metrics)

## Getting Started

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
jupyter notebook i222327_Mohammad_Rohaan_A_03_ML.ipynb
```

No external dataset download — all rows are defined in the notebook.

## Project Structure

```
Predict_Robot_Energy/
├── i222327_Mohammad_Rohaan_A_03_ML.ipynb
└── Asg-3-ML for Robotics-Spring 2026-NU.pdf
```

## Author

Mohammad Rohaan — i222327 · [rohaan2802](https://github.com/rohaan2802)
