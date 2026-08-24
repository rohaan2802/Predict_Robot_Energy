# Predict Robot Energy (ML for Robotics A#03)

Predict a mobile robot’s **energy consumption (Wh)** from mission parameters using **RBF-kernel SVR**, compared with **ordinary linear regression**. The dataset has only **10 labeled rows**, so both models **fail to generalize** (negative R²) — the notebook treats that as the main lesson.

**Author:** Mohammad Rohaan · **i222327** · Spring 2026  
**Notebook:** `i222327_Mohammad_Rohaan_A_03_ML.ipynb`  
[rohaan2802](https://github.com/rohaan2802)

---

## Table of contents

1. [Features and target](#features-and-target)
2. [The 10-row dataset](#the-10-row-dataset)
3. [Pipeline](#pipeline)
4. [Reported metrics](#reported-metrics)
5. [Terrain discussion](#terrain-discussion)
6. [How to run](#how-to-run)

---

## Features and target

| Column | Meaning |
|--------|---------|
| `Load_Weight_kg` | Payload |
| `Speed_m_per_s` | Commanded speed |
| `Terrain_Roughness` | Roughness score (about 1–10 in the write-up) |
| `Task_Duration_min` | Mission time |
| **`Energy_Consumption_Wh`** | **Target** |

---

## The 10-row dataset

Hard-coded in the notebook (examples): load 5–22 kg, speed 0.8–2.0 m/s, roughness 2–8, duration 10–30 min, energy 80–300 Wh.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

With n=10, **test size 0.2 ⇒ 2 test points**. Metrics on two points are noisy by construction.

---

## Pipeline

1. Build `DataFrame`.  
2. Split; print train/test lengths.  
3. **`StandardScaler` fit on train only** (comment: avoid leakage), `transform` test.  
4. `SVR(kernel='rbf')` fit.  
5. Predict a **new mission** `[25 kg, 2.2 m/s, roughness 7, 35 min]` (outside the train hull).  
6. Scatter actual vs predicted vs load on the test set.  
7. Smooth curve: vary load on a 200-point linspace; **hold other features at mean**.  
8. `mean_squared_error`, `r2_score`.  
9. Fit `LinearRegression` on the same scaled features; overlay vs identity line.

---

## Reported metrics

From the interpretation markdown:

| Model | R² | MSE |
|-------|-----|-----|
| **SVR** | **−0.89** | **47.26** |
| Linear Regression | −1.83 | 70.65 |

Negative R² means worse than predicting the **mean energy**. SVR is **less bad** than OLS here, but neither is usable for planning. The notebook states the cause: **too little data**.

---

## Terrain discussion

Qualitative: higher roughness → more traction work → energy rises. That story is **not** identified as a stable fitted coefficient on 10 points; treat it as domain knowledge, not a discovered law.

---

## How to run

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
jupyter notebook i222327_Mohammad_Rohaan_A_03_ML.ipynb
```

Keep `Asg-3-ML for Robotics-Spring 2026-NU.pdf` with the assignment brief.

**Extension:** collect dozens of real drive cycles (load × speed × terrain × time) before trusting SVR for battery sizing.

---

## Author

**Mohammad Rohaan** — i222327 · [rohaan2802](https://github.com/rohaan2802)
