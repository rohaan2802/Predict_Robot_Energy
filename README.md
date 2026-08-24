# Predict Robot Energy — RBF SVR vs Linear Regression (ML for Robotics A#03)

Jupyter lab that predicts a mobile robot’s **energy consumption (Wh)** from four mission features using **RBF-kernel SVR**, then compares it with **ordinary least squares**. The table is **hard-coded (n = 10)**. On a 80/20 split that leaves **two test rows**, both models have **negative R²** (worse than predicting the mean). The notebook treats that failure as the result: **too little data**, not a production energy planner.

**Author:** Mohammad Rohaan · **22I-2327** · Spring 2026 · [rohaan2802](https://github.com/rohaan2802)  
**Notebook:** `i222327_Mohammad_Rohaan_A_03_ML.ipynb` (23 cells, executed).  
Assignment PDF: `Asg-3-ML for Robotics-Spring 2026-NU.pdf` (not text-extractable in this snapshot; the pipeline below is exactly what the notebook implements).

## Table of contents

- [Problem statement / academic context](#problem-statement--academic-context)
- [Features](#features)
- [Architecture / design](#architecture--design)
- [Algorithms and data structures](#algorithms-and-data-structures)
- [File-by-file reference](#file-by-file-reference)
- [Data formats / schemas](#data-formats--schemas)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Prerequisites and install](#prerequisites-and-install)
- [How to build and run](#how-to-build-and-run)
- [Usage walkthrough](#usage-walkthrough)
- [Configuration / constants](#configuration--constants)
- [Results / metrics](#results--metrics)
- [Controls / CLI](#controls--cli)
- [Known limitations / bugs](#known-limitations--bugs)
- [How to extend](#how-to-extend)
- [Author](#author)

## Problem statement / academic context

Energy used on a traverse depends on **payload, speed, terrain roughness, and duration**. The notebook asks whether a nonlinear SVR can map those four numbers to Wh better than a linear model, then discusses using such a predictor for **path planning** and **speed/load control**. With ten labeled missions and two held-out points, the fitted surfaces cannot generalize; the written interpretation states that explicitly.

## Features

| Step | Notebook heading | What runs |
|------|------------------|-----------|
| Create data | `### Creating Dataset` | Python dict → `DataFrame` (10×5) |
| Split | `### Train-Test Split` | `train_test_split`, print train **8** / test **2** |
| Scale | `### Feature Scaling` | `StandardScaler` **fit on train only**, `transform` test |
| SVR | `### Training SVR Model` | `SVR(kernel='rbf')` |
| Predict | `### Predictions` | New mission `[25 kg, 2.2 m/s, roughness 7, 35 min]` |
| Plots | `### Visualization` | Test scatter vs load; 200-point load sweep with other features at **mean** |
| Metrics | `### Model Evaluation` | `mean_squared_error`, `r2_score` on SVR test preds |
| Baseline | `### Training Linear Regression to compare with SVR` | `LinearRegression` on the same scaled split; overlay vs identity line |
| Write-up | `### Brief interpretation of results` | Negative R², terrain story, planning/control bullets |

## Architecture / design

```
hard-coded 10-row table
        │
   X = four inputs, y = Energy_Consumption_Wh
        │
   train_test_split(test_size=0.2, random_state=42)  →  8 / 2
        │
   StandardScaler.fit(X_train) → transform train and test
        │
   SVR(rbf) ──────────────────► y_pred, MSE, R², plots, one OOD query
   LinearRegression ──────────► y_pred_lr, MSE, R², identity scatter
```

Comment in the scaler cell: *“Fit only on training data (VERY IMPORTANT to avoid data leakage).”*

## Algorithms and data structures

**SVR.** sklearn `SVR(kernel='rbf')` with **library defaults** (not set in code): `C=1.0`, `epsilon=0.1`, `gamma='scale'`. RBF maps scaled inputs to a kernel ridge-style regressor. With eight training points the model is severely underdetermined relative to a flexible kernel.

**Linear regression.** `LinearRegression()` on the **same** `X_train_scaled` / `X_test_scaled`. No regularization, no polynomial terms.

**Smoothing curve.** `np.linspace` over observed load min–max (**5–22 kg**), 200 points; `Speed_m_per_s`, `Terrain_Roughness`, and `Task_Duration_min` held at the **full-X column means** (including test rows). Predictions use the **train-fitted** scaler.

**Metrics in code:** MSE and R² only. RMSE below is \(\sqrt{\text{MSE}}\) from the printed MSE. MAE is **not** printed; the two-row test table is.

## File-by-file reference

| File | Description |
|------|-------------|
| `i222327_Mohammad_Rohaan_A_03_ML.ipynb` | Entire assignment (23 cells, plots in outputs). |
| `Asg-3-ML for Robotics-Spring 2026-NU.pdf` | Course brief (repo root). |

No CSV: the dataset lives in cell 2.

## Data formats / schemas

Column names (exact):

```text
Load_Weight_kg, Speed_m_per_s, Terrain_Roughness, Task_Duration_min, Energy_Consumption_Wh
```

**All ten rows (as in the notebook):**

| Load (kg) | Speed (m/s) | Roughness | Duration (min) | Energy (Wh) |
|-----------|-------------|-----------|----------------|-------------|
| 10 | 1.2 | 3 | 15 | 120 |
| 15 | 1.5 | 5 | 20 | 180 |
| 8 | 1.0 | 2 | 10 | 90 |
| 20 | 1.8 | 6 | 25 | 250 |
| 12 | 1.3 | 4 | 18 | 150 |
| 18 | 1.6 | 7 | 22 | 220 |
| 5 | 0.8 | 3 | 12 | 80 |
| 22 | 2.0 | 8 | 30 | 300 |
| 14 | 1.4 | 5 | 19 | 170 |
| 16 | 1.7 | 6 | 21 | 210 |

Colab profiler on the frame: load min **5** max **22** (std ≈ 5); speed **0.8–2.0** (std ≈ 0.368); roughness **2–8**; duration **10–30**; energy **80–300**. Features rise together with energy in this hand-built table (heavier / faster / rougher / longer → more Wh), so a linear trend exists **in-sample**, but two test points cannot support a stable score.

Executed test predictions (`random_state=42`):

| Load_Weight | Actual_Energy | Predicted_Energy (SVR) |
|-------------|---------------|-------------------------|
| 14 | 170 | **179.713403** |
| 15 | 180 | **180.413269** |

SVR is almost flat on these two loads (~180 Wh), while actuals differ by 10 Wh.

## Tech stack

- Python 3 / Jupyter (executed on Colab; matplotlib **3.10.0** in plot metadata)
- pandas, numpy, matplotlib
- sklearn: `StandardScaler`, `train_test_split`, `SVR`, `LinearRegression`, `mean_squared_error`, `r2_score`

## Project structure

```
Predict_Robot_Energy/
├── i222327_Mohammad_Rohaan_A_03_ML.ipynb
└── Asg-3-ML for Robotics-Spring 2026-NU.pdf
```

## Prerequisites and install

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

## How to build and run

```bash
cd Predict_Robot_Energy
jupyter notebook i222327_Mohammad_Rohaan_A_03_ML.ipynb
```

Run all cells in order. No external files. Re-running with the same `random_state` reproduces the printed metrics.

## Usage walkthrough

1. Confirm `data.head()` shows the five columns and energy 120, 180, 90, 250, 150 for the first five rows.
2. Check **Training size: 8** and **Testing size: 2**.
3. After SVR fit, read the out-of-range query: **Predicted Energy Consumption (Wh): 180.893219405397** for `[25, 2.2, 7, 35]`. That point is **outside** the training hull (max train load 22 kg, max duration 30 min, max speed 2.0). The prediction sits near the sample mean energy (~177 Wh), which is typical of a weakly fitted RBF SVR (small `C`, few points).
4. Plot 1: actual (circles) vs predicted (crosses) vs load on the two test rows.
5. Plot 2: smooth SVR curve vs load; red markers are **test** actuals only.
6. Read MSE / R², then the linear comparison printout and the identity-line figure.
7. Read the interpretation markdown (negative R², terrain qualitative notes, planning/control).

## Configuration / constants

| Item | Value |
|------|-------|
| n | 10 |
| `test_size` | 0.2 → 2 test rows |
| `random_state` | 42 |
| SVR | `kernel='rbf'` (other SVR kwargs default) |
| Scaler | `StandardScaler`, train-only `fit` |
| OOD query | `[[25, 2.2, 7, 35]]` |
| Smooth curve | 200-point `linspace` on load |
| Figure sizes | (8, 5), (9, 6), (10, 5) |

## Results / metrics

**Printed evaluation (test n = 2):**

| Model | MSE (printed) | R² (printed) | RMSE \(=\sqrt{\mathrm{MSE}}\) |
|-------|---------------|--------------|-------------------------------|
| **SVR (RBF)** | **47.26049282170618** | **−0.8904197128682472** | **6.875** |
| **Linear regression** | **70.65189898206445** | **−1.8260759592825782** | **8.405** |

Markdown interpretation rounds these to SVR R² **−0.89**, MSE **47.26**; linear R² **−1.83**, MSE **70.65**.

Negative R² means each model is worse than predicting \(\bar{y}_{\text{test}}\). SVR’s test predictions (~179.7 and ~180.4) barely move; actuals are 170 and 180, so most of the MSE is the 170 vs ~180 miss. Linear regression is worse on the same two points.

**OOD SVR prediction:** **180.893 Wh**.

The write-up’s terrain claims (roughness 1–10, energy rises with roughness × load × speed) are **domain narrative**, not fitted coefficients (SVR has none to print; linear `coef_` is never printed).

## Controls / CLI

Notebook only.

## Known limitations / bugs

- **n = 10 / test = 2:** metrics are not a reliable ranking of SVR vs OLS.
- No MAE in code; no cross-validation, no residual diagnostics.
- SVR hyperparameters are **untuned**.
- Smooth curve uses **full-data** feature means, not train-only means.
- OOD query is far from the convex hull; the ~181 Wh answer is not a physics prediction.
- Planning/control section is speculative given the scores.
- Assignment PDF could not be OCR’d here; if the brief required extra models or a real log file, they are **not** in this notebook.

## How to extend

- Log dozens of real drive cycles (load × speed × terrain × time) before trusting R².
- Add `GridSearchCV` on `C`, `epsilon`, `gamma`; report MAE and RMSE in addition to MSE.
- Hold a terrain-stratified split; print linear coefficients.
- Keep the scaler and models with `joblib` if this becomes a planning module.

## Author

**Mohammad Rohaan** — 22I-2327 · [rohaan2802](https://github.com/rohaan2802)
