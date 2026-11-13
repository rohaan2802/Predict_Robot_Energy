<div align="center">

# Predict Robot Energy — RBF SVR vs Linear Regression

**Two models that predict a mobile robot’s energy use (Wh) from load, speed, terrain roughness, and task duration.**

[![Python](https://img.shields.io/badge/Python-3-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://github.com/rohaan2802/Predict_Robot_Energy)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-SVR%20%2B%20LinearRegression-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://github.com/rohaan2802/Predict_Robot_Energy)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://github.com/rohaan2802/Predict_Robot_Energy)
[![Repo Size](https://img.shields.io/github/repo-size/rohaan2802/Predict_Robot_Energy?style=for-the-badge)](https://github.com/rohaan2802/Predict_Robot_Energy)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**Mohammad Rohaan** · 22I-2327

</div>

---

## Live Links

| What | Link |
|------|------|
| Repository | [https://github.com/rohaan2802/Predict_Robot_Energy](https://github.com/rohaan2802/Predict_Robot_Energy) |
| Website | [https://rohaan2802.github.io/Predict_Robot_Energy/](https://rohaan2802.github.io/Predict_Robot_Energy/) |
| Download ZIP | [https://github.com/rohaan2802/Predict_Robot_Energy/archive/refs/heads/main.zip](https://github.com/rohaan2802/Predict_Robot_Energy/archive/refs/heads/main.zip) |

---

## Table of Contents

1. [Live Links](#live-links)
2. [Overview](#overview)
3. [What's Inside](#whats-inside)
4. [The Data](#the-data)
5. [How the Models Are Fitted](#how-the-models-are-fitted)
6. [Test Results](#test-results)
7. [The Linear Fit](#the-linear-fit)
8. [What the SVR Is Actually Doing](#what-the-svr-is-actually-doing)
9. [Training and Test Files](#training-and-test-files)
10. [Folder Structure](#folder-structure)
11. [How to Use This Repository](#how-to-use-this-repository)
12. [Running the Fit](#running-the-fit)
13. [Screenshots](#screenshots)
14. [License](#license)
15. [Author](#author)

---

## Overview

This is Assignment 03 from ML for Robotics (CS-4090), Spring 2026. I was asked to predict a mobile robot’s energy consumption, in watt-hours, from four mission numbers — load weight, speed, terrain roughness, and task duration — with a support-vector regressor, and then to compare that fit with ordinary least squares.

The table is the ten-row example from the assignment brief. There is no larger drive-cycle log in this repository. I hard-coded those ten missions in the notebook, and I have also written them to [data/missions.csv](data/missions.csv) so the command-line script can load the same rows.

I ran the two models on the same 8 / 2 split (`test_size=0.2`, `random_state=42`):

- An RBF-kernel support-vector regressor, through `sklearn.svm.SVR(kernel='rbf')`, with the library defaults (`C=1.0`, `epsilon=0.1`, `gamma='scale'`).
- Ordinary least squares, through `sklearn.linear_model.LinearRegression`, on the same scaled training features.

A `StandardScaler` is fitted on the training features only and then applied to the test rows, which is the leakage comment already in the notebook. The brief also asked for the target to be scaled; the notebook does not do that, and the script follows the notebook.

I re-ran the work from [src/predict_robot_energy.py](src/predict_robot_energy.py). The printed test scores match the executed notebook exactly: SVR MSE 47.26049282170618 and R² −0.8904197128682472; linear regression MSE 70.65189898206445 and R² −1.8260759592825782. Both are worse than predicting the mean of the two test energies. A train-mean baseline on those same two rows has MSE 31.25, which is ahead of both fitted models. With only two held-out points, I am not treating the SVR-vs-linear ranking as a stable result — only as what this split actually produced.

The raw numbers from that run are under [results/](results/). The assignment brief is under [docs/assignment/](docs/assignment/).

---

## What's Inside

| Section | Path | What's there |
|---------|------|----------------|
| Notebook | [notebooks/i222327_Mohammad_Rohaan_A_03_ML.ipynb](notebooks/i222327_Mohammad_Rohaan_A_03_ML.ipynb) | The original 23-cell walkthrough, with the executed outputs still in it |
| Script | [src/predict_robot_energy.py](src/predict_robot_energy.py) | The same split, scaler and two fits, runnable from the command line. It prints the notebook figures and writes the files under `results/` |
| Data | [data/missions.csv](data/missions.csv) | The ten-row table, same values as the notebook and the brief. Notes are in [data/README.md](data/README.md) |
| Results | [results/](results/) | `metrics.csv`, coefficients, the full prediction table with errors, the out-of-range query, and the figures |
| Assignment brief | [docs/assignment/Asg-3-ML-for-Robotics-Spring-2026-NU.pdf](docs/assignment/Asg-3-ML-for-Robotics-Spring-2026-NU.pdf) | The original brief (it was at the repo root as `Asg-3-ML for Robotics-Spring 2026-NU.pdf`) |
| Screenshots | [screenshots/](screenshots/) | 24 images, all 1600×1000, covering the table, the fits, the charts and the website |
| Website | [index.html](index.html) | The GitHub Pages site for this repo (pine, moss and mist, dark and light) |
| Dependencies | [requirements.txt](requirements.txt) | pandas, numpy, matplotlib, scikit-learn |

---

## The Data

`missions.csv` has 10 rows and 5 columns. Every cell is filled. The values are exactly the ones in the assignment brief and in the notebook’s “Creating Dataset” cell.

| Column | Role | What I actually saw |
|--------|------|---------------------|
| `Load_Weight_kg` | Predictor | 5 to 22 kg, mean 14, every value unique |
| `Speed_m_per_s` | Predictor | 0.8 to 2.0 m/s, mean 1.43 |
| `Terrain_Roughness` | Predictor | 2 to 8 on the brief’s 1–10 scale, mean 4.9 |
| `Task_Duration_min` | Predictor | 10 to 30 minutes, mean 19.2 |
| `Energy_Consumption_Wh` | Target | 80 to 300 Wh, mean 177, median 175 |

All ten rows:

| # | Load (kg) | Speed (m/s) | Roughness | Duration (min) | Energy (Wh) |
|--:|----------:|------------:|----------:|---------------:|------------:|
| 0 | 10 | 1.2 | 3 | 15 | 120 |
| 1 | 15 | 1.5 | 5 | 20 | 180 |
| 2 | 8 | 1.0 | 2 | 10 | 90 |
| 3 | 20 | 1.8 | 6 | 25 | 250 |
| 4 | 12 | 1.3 | 4 | 18 | 150 |
| 5 | 18 | 1.6 | 7 | 22 | 220 |
| 6 | 5 | 0.8 | 3 | 12 | 80 |
| 7 | 22 | 2.0 | 8 | 30 | 300 |
| 8 | 14 | 1.4 | 5 | 19 | 170 |
| 9 | 16 | 1.7 | 6 | 21 | 210 |

This is a hand-built example, not a logged fleet. Heavier, faster, rougher, longer missions are also the ones with more watt-hours, and the four inputs move together. Pearson correlations of each column with energy are all high — duration 0.9868, load 0.9867, speed 0.9830, roughness 0.9572 — and the inputs are strongly correlated with each other as well (0.92 to 0.99). I would not read a single coefficient as “the effect of roughness, holding the rest fixed.” The notebook’s write-up about terrain is a domain note, not a fitted isolation of that column.

The 80/20 split with `random_state=42` holds out rows 1 and 8 (loads 15 kg / 180 Wh and 14 kg / 170 Wh). The other eight rows are the training set.

---

## How the Models Are Fitted

```
ten-row table  (data/missions.csv)
        │
   X = four inputs,  y = Energy_Consumption_Wh
        │
   train_test_split(test_size=0.2, random_state=42)  →  8 / 2
        │
   StandardScaler.fit(X_train)  →  transform train and test
        │
   SVR(kernel='rbf')        ──►  test predictions, MSE, R², one out-of-range query
   LinearRegression()       ──►  test predictions, MSE, R², coefficients
```

The scaler comment in the notebook is the one I kept: fit only on the training features, so the test rows do not leak into the scale. The assignment brief also asked to standardise the target; that step is not in the notebook, and I did not add it. A side check that does scale `y` is worse on this split (test R² about −1.37), and it is not the reported result.

The out-of-range query from the brief is `[25 kg, 2.2 m/s, roughness 7, 35 min]`. Every one of those four numbers sits outside the training hull (training maxima are 22 kg, 2.0 m/s, and 30 minutes; roughness 7 is inside the training range). The SVR answers **180.893219405397 Wh**, which matches the notebook. That is almost the training-set mean energy (177.5 Wh). Linear regression on the same scaled features answers 332.50 Wh — it extrapolates the line past the heaviest training mission.

The high-resolution curve in the notebook sweeps load from 5 to 22 kg over 200 points, with speed, roughness and duration held at the **full-table** means (including the two test rows), then runs those points through the train-fitted scaler and the SVR. The script does the same.

---

## Test Results

**Test set, 2 rows. The SVR and linear-regression figures match the original notebook exactly.**

| Model | MSE | RMSE | MAE | R² |
|-------|-----|------|-----|-----|
| SVR (RBF) | 47.260492821706 | 6.874626740537 | 5.063336003029 | −0.890419712868 |
| Linear Regression | 70.651898982064 | 8.405468397541 | 8.327820078686 | −1.826075959283 |
| Train-mean baseline | 31.250000000000 | 5.590169943749 | 5.000000000000 | −0.250000000000 |

The baseline is not in the notebook; I added it because a negative R² already means “worse than the mean of the test energies,” and I wanted the same comparison against the mean of the *training* energies. That baseline is ahead of both fitted models on this split.

**Training set, 8 rows:**

| Model | MSE | RMSE | MAE | R² |
|-------|-----|------|-----|-----|
| SVR (RBF) | 5209.865034105771 | 72.179394802851 | 65.038876736685 | 0.060227276824 |
| Linear Regression | 20.275891092016 | 4.502875869044 | 3.983685333438 | 0.996342567560 |

Linear regression is almost perfect on the eight training rows and then falls apart on the two held-out ones. The SVR barely leaves a flat line on either split: training R² is only 0.06, and its two test predictions sit at 179.71 and 180.41 Wh against actuals of 170 and 180.

The two test rows, with signed errors:

| Row | Load | Actual | Pred. SVR | Pred. LR | Err. SVR | Err. LR |
|----:|-----:|-------:|----------:|---------:|---------:|--------:|
| 8 | 14 | 170 | 179.713403 | 177.187947 | +9.713403 | +7.187947 |
| 1 | 15 | 180 | 180.413269 | 189.467694 | +0.413269 | +9.467694 |

SVR is almost exact on the 180 Wh row and about ten watt-hours high on the 170 Wh row. Linear regression is a little closer on 170 and farther on 180. With two points I would not pick a winner for a real robot; I would collect more missions.

The notebook’s own write-up already says the same thing in shorter form: both models “are not better than guessing the average,” the table is too small, and SVR is only “a bit better” on the numbers it printed. I agree with that, and the training scores above make the reason clearer.

---

## The Linear Fit

The notebook fits linear regression on the *scaled* training features and never prints `coef_`. From the re-run, on those standardised inputs:

```
Energy  ≈  177.5
        +  24.6713  ×  z(Load_Weight_kg)
        +  13.9964  ×  z(Speed_m_per_s)
        +   9.4609  ×  z(Terrain_Roughness)
        +  27.2790  ×  z(Task_Duration_min)
```

The intercept is the training-set mean energy, 177.5 Wh. An equivalent OLS on the raw units (same predictions) is:

```
Energy  ≈  −39.7358
        +   4.3539  ×  Load_Weight_kg
        +  35.9148  ×  Speed_m_per_s
        +   4.6671  ×  Terrain_Roughness
        +   4.3344  ×  Task_Duration_min
```

Speed looks large in raw units because its range is only 0.8–2.0 m/s. I would not treat either set of coefficients as a physical breakdown: the four inputs are nearly collinear, and a two-row test set is enough to show that a near-perfect training line does not hold just outside those eight points.

---

## What the SVR Is Actually Doing

`SVR(kernel='rbf')` is used with the library defaults. `C=1.0` and `epsilon=0.1` are small relative to energies in the 80–300 Wh range. All eight training rows are support vectors. The fitted intercept is 179.66 Wh, and the training predictions stay in a narrow band around 177–183 Wh even when the actuals run from 80 to 300. That is why the training R² is only 0.06, and why the out-of-range query comes back at 180.89 Wh rather than something closer to a heavy, long mission.

The notebook’s path-planning and energy-aware control remarks are a discussion of how a *working* predictor could be used. On these scores I would not hook either model to a planner. The useful result of the assignment, for me, is the failure itself: ten hand-built rows and an untuned RBF SVR are not enough to estimate watt-hours for a new task.

---

## Training and Test Files

| File | What’s in it |
|------|----------------|
| [data/missions.csv](data/missions.csv) | All ten missions |
| [results/metrics.csv](results/metrics.csv) | Test and training MSE, RMSE, MAE and R² for both models, plus the train-mean baseline |
| [results/predictions_test.csv](results/predictions_test.csv) | The two test rows, both predictions, and signed errors |
| [results/predictions.csv](results/predictions.csv) | The same for all ten rows, with a `split` column |
| [results/coefficients_scaled.csv](results/coefficients_scaled.csv) | Linear coefficients on the standardised features |
| [results/coefficients_raw.csv](results/coefficients_raw.csv) | The equivalent raw-unit coefficients |
| [results/correlations.csv](results/correlations.csv) | The 5×5 Pearson matrix |
| [results/summary_stats.csv](results/summary_stats.csv) | `describe()`, with quartiles and the first mode |
| [results/ood_prediction.csv](results/ood_prediction.csv) | The `[25, 2.2, 7, 35]` query, SVR and linear |
| [results/train_test_split.csv](results/train_test_split.csv) | Which rows are train and which are test |
| [results/figures/](results/figures/) | Load-vs-energy, the smooth SVR curve, the identity scatter, and a correlation plot |

---

## Folder Structure

```
Predict_Robot_Energy/
├── notebooks/i222327_Mohammad_Rohaan_A_03_ML.ipynb
├── src/predict_robot_energy.py
├── data/missions.csv
├── data/README.md
├── results/                  metrics, coefficients, predictions, figures
├── docs/assignment/          the original brief
├── screenshots/              24 × 1600×1000
├── assets/css, js, favicon
├── index.html
├── requirements.txt
├── LICENSE                   MIT 2026
└── README.md
```

---

## How to Use This Repository

Python 3, with the packages in [requirements.txt](requirements.txt). The CSV is already in `data/`.

### Linux / macOS

```bash
git clone https://github.com/rohaan2802/Predict_Robot_Energy.git
cd Predict_Robot_Energy
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 src/predict_robot_energy.py
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src\predict_robot_energy.py
```

That prints the notebook’s test scores (they should match the figures above), the extra training metrics, and the out-of-range query, and writes files under `results/`. Optional flags: `--data-dir`, `--results-dir` and `--no-plots`. The same steps are in [notebooks/i222327_Mohammad_Rohaan_A_03_ML.ipynb](notebooks/i222327_Mohammad_Rohaan_A_03_ML.ipynb).

---

## Running the Fit

From the repository root, `python src/predict_robot_energy.py` is the full run. You should see:

```
Training size: 8
Testing size: 2
Predicted Energy Consumption (Wh): 180.893219405397
Mean Squared Error (MSE): 47.26049282170618
R2 Score: -0.8904197128682472
```

and the linear-regression pair MSE 70.65189898206445, R² −1.8260759592825782. If those printed values move, the environment is not reproducing the notebook.

---

## Screenshots

All 24 images are 1600×1000 (16:10).

### 1. Repository Folder Structure

![Repository folder structure](screenshots/01-repo-structure.png)

The script, notebook, data, results and website files.

### 2. The Problem

![The problem](screenshots/02-problem.png)

Four mission inputs, two models, a ten-row table.

### 3. The Data Table

![The data table](screenshots/03-data-schema.png)

All ten missions, all five columns.

### 4. Summary Statistics

![Summary statistics](screenshots/04-summary-statistics.png)

`describe()` with quartiles and the first mode.

### 5. Correlations

![Pearson correlation matrix](screenshots/05-correlations.png)

The 5×5 Pearson matrix, every cell labelled.

### 6. The Train / Test Split

![Train and test split](screenshots/06-train-test-split.png)

Which eight rows are training, which two are held out.

### 7. How Closely the Inputs Move Together

![Feature collinearity](screenshots/07-collinearity.png)

Pairwise correlations among the four predictors.

### 8. Load vs Energy

![Load vs energy](screenshots/08-scatter-load.png)

All ten missions, with the two test rows marked.

### 9. Duration vs Energy

![Duration vs energy](screenshots/09-scatter-duration.png)

Task duration against watt-hours, r = 0.9868.

### 10. Terrain Roughness vs Energy

![Terrain vs energy](screenshots/10-scatter-terrain.png)

Roughness against watt-hours. The trend is real in this table, but it is not isolated from load and speed.

### 11. Fitting the Two Models

![Fitting code](screenshots/11-fit-code.png)

The real split, scaler, SVR and linear-regression calls from the script.

### 12. The Out-of-Range Query

![Out-of-range prediction](screenshots/12-ood-prediction.png)

`[25, 2.2, 7, 35]`: SVR 180.89 Wh, linear regression 332.50 Wh.

### 13. Test and Training Metrics

![Evaluation metrics](screenshots/13-evaluation-metrics.png)

Test, training, and the train-mean baseline, with RMSE and MAE as well as MSE and R².

### 14. The Two Test Predictions

![Test predictions](screenshots/14-predictions.png)

Actuals, both models, and signed errors.

### 15. Linear-Regression Coefficients

![Coefficients](screenshots/15-coefficients.png)

Scaled-feature coefficients, and the equivalent raw-unit equation.

### 16. What the SVR Settled On

![SVR defaults and support vectors](screenshots/16-svr-defaults.png)

Library defaults, all eight training rows as support vectors, predictions in a narrow band.

### 17. Actual vs Predicted

![Actual vs predicted](screenshots/17-actual-vs-predicted.png)

Both models on the two test rows, against the identity line.

### 18. High-Resolution SVR Curve

![SVR smooth curve](screenshots/18-svr-smooth-curve.png)

Load swept from 5 to 22 kg, other features at the full-table means.

### 19. Training Predictions vs Actuals

![Training predictions](screenshots/19-train-predictions.png)

The eight training rows: a near-perfect line for OLS, a flat band for the SVR.

### 20. Website Home, Dark Theme

![Website home, dark](screenshots/20-website-home-dark.png)

The pine-and-moss Pages site, dark theme.

### 21. Website Home, Light Theme

![Website home, light](screenshots/21-website-home-light.png)

The same page in the mist light theme.

### 22. Website on a Phone

![Website on a phone](screenshots/22-website-phone.png)

The mobile layout, hamburger menu and stats.

### 23. Website — Results

![Website results](screenshots/23-website-results.png)

The metrics and the two-row prediction table on the site.

### 24. Website — How to Run

![Website how to run](screenshots/24-website-how-to-run.png)

Create a virtualenv, install, and run `src/predict_robot_energy.py`.

---

## License

The code in this repository is under the [MIT License](LICENSE), Copyright (c) 2026 Mohammad Rohaan.

---

## Author

**Mohammad Rohaan** — 22I-2327 · [rohaan2802](https://github.com/rohaan2802)
