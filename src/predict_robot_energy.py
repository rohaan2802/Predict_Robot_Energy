#!/usr/bin/env python3
"""Fit an RBF SVR and ordinary least squares on the ten-mission table.

The pipeline matches the original notebook: an 80/20 split with
random_state=42 (eight training rows, two test rows), a StandardScaler
fitted on the training features only, SVR(kernel='rbf') with library
defaults, and LinearRegression on the same scaled split.

Run from the repository root:

    python src/predict_robot_energy.py

Optional flags: --data-dir, --results-dir, --no-plots.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

FEATURES = [
    "Load_Weight_kg",
    "Speed_m_per_s",
    "Terrain_Roughness",
    "Task_Duration_min",
]
TARGET = "Energy_Consumption_Wh"
ALL_COLS = FEATURES + [TARGET]

# Out-of-sample query from the assignment brief (task 4).
OOD = {
    "Load_Weight_kg": 25,
    "Speed_m_per_s": 2.2,
    "Terrain_Roughness": 7,
    "Task_Duration_min": 35,
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_data(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "missions.csv"
    if not path.is_file():
        sys.exit(
            f"Could not find {path}. The ten-row table should sit in "
            "data/ as missions.csv. See data/README.md."
        )
    df = pd.read_csv(path)
    missing = [c for c in ALL_COLS if c not in df.columns]
    if missing:
        sys.exit(f"missions.csv is missing columns: {missing}")
    if len(df) != 10:
        sys.exit(f"Expected 10 mission rows, found {len(df)}.")
    return df[ALL_COLS].copy()


def mets(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "MSE": mse,
        "RMSE": float(math.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    desc = df[ALL_COLS].describe().T
    modes, mode_counts = [], []
    for col in ALL_COLS:
        mode = df[col].mode().iloc[0]
        modes.append(mode)
        mode_counts.append(int((df[col] == mode).sum()))
    desc["mode"] = modes
    desc["mode_count"] = mode_counts
    return desc.rename(columns={"50%": "median", "25%": "q1", "75%": "q3"})


def write_figures(
    out: Path,
    df: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    svr_te: np.ndarray,
    lr_te: np.ndarray,
    load_range: np.ndarray,
    y_smooth: np.ndarray,
) -> None:
    out.mkdir(parents=True, exist_ok=True)

    # 1. All ten missions: load vs energy, train/test marked later is in other figs.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["Load_Weight_kg"], df[TARGET], s=60, c="#2d6a38", zorder=3)
    ax.set_xlabel("Load weight (kg)")
    ax.set_ylabel("Energy consumption (Wh)")
    ax.set_title("All ten missions — load vs energy")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out / "load_vs_energy.png", dpi=150)
    plt.close(fig)

    # 2. High-resolution SVR curve (same construction as the notebook).
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(load_range, y_smooth, color="#2d6a38", lw=2.2, label="SVR, other features at full-table means")
    ax.scatter(
        X_test["Load_Weight_kg"],
        y_test,
        color="#b8472a",
        s=70,
        zorder=3,
        label="Test actuals (n = 2)",
    )
    ax.set_xlabel("Load weight (kg)")
    ax.set_ylabel("Energy consumption (Wh)")
    ax.set_title("High-resolution SVR curve (load vs energy)")
    ax.legend()
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out / "svr_smooth_curve.png", dpi=150)
    plt.close(fig)

    # 3. Identity: SVR vs linear on the two test rows.
    fig, ax = plt.subplots(figsize=(8, 5))
    lo = min(float(y_test.min()), float(svr_te.min()), float(lr_te.min())) - 8
    hi = max(float(y_test.max()), float(svr_te.max()), float(lr_te.max())) + 8
    ax.plot([lo, hi], [lo, hi], "--", color="#b8472a", label="identity (y = x)")
    ax.scatter(y_test, svr_te, marker="x", s=90, color="#2d6a38", label="SVR", zorder=3)
    ax.scatter(y_test, lr_te, marker="o", s=70, color="#1a4a6e", label="Linear regression", zorder=3)
    ax.set_xlabel("Actual energy (Wh)")
    ax.set_ylabel("Predicted energy (Wh)")
    ax.set_title("SVR vs linear regression — two test rows")
    ax.legend()
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)

    # 4. Correlation heatmap values are written as CSV; a simple imshow here.
    corr = df[ALL_COLS].corr()
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(corr, cmap="Greens", vmin=0.9, vmax=1.0)
    ax.set_xticks(range(5), ALL_COLS, rotation=30, ha="right")
    ax.set_yticks(range(5), ALL_COLS)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{corr.iloc[i, j]:.3f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Pearson correlations (all five columns)")
    fig.tight_layout()
    fig.savefig(out / "correlations.png", dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--results-dir", type=Path, default=root / "results")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    df = load_data(args.data_dir)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    # Fit only on training data (VERY IMPORTANT to avoid data leakage) — same comment as the notebook.
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svr = SVR(kernel="rbf")
    svr.fit(X_train_scaled, y_train)
    svr_te = svr.predict(X_test_scaled)
    svr_tr = svr.predict(X_train_scaled)

    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    lr_te = lr.predict(X_test_scaled)
    lr_tr = lr.predict(X_train_scaled)

    # Equivalent OLS on raw (unscaled) units — same test predictions, different coefficients.
    lr_raw = LinearRegression().fit(X_train, y_train)

    ood = pd.DataFrame([OOD], columns=FEATURES)
    ood_s = scaler.transform(ood)
    ood_svr = float(svr.predict(ood_s)[0])
    ood_lr = float(lr.predict(ood_s)[0])

    svr_test = mets(y_test, svr_te)
    svr_train = mets(y_train, svr_tr)
    lr_test = mets(y_test, lr_te)
    lr_train = mets(y_train, lr_tr)
    mean_base = mets(y_test, np.full(len(y_test), float(y_train.mean())))

    # Smooth curve, same as the notebook: other features at the FULL-table means.
    load_range = np.linspace(X["Load_Weight_kg"].min(), X["Load_Weight_kg"].max(), 200)
    X_smooth = pd.DataFrame({
        "Load_Weight_kg": load_range,
        "Speed_m_per_s": X["Speed_m_per_s"].mean(),
        "Terrain_Roughness": X["Terrain_Roughness"].mean(),
        "Task_Duration_min": X["Task_Duration_min"].mean(),
    })
    y_smooth = svr.predict(scaler.transform(X_smooth))

    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)

    # ---- tables ----
    summary_table(df).to_csv(results_dir / "summary_stats.csv")
    df.corr().to_csv(results_dir / "correlations.csv", float_format="%.8f")

    metrics = pd.DataFrame(
        [
            {"model": "SVR (RBF)", "split": "test", **svr_test},
            {"model": "SVR (RBF)", "split": "train", **svr_train},
            {"model": "Linear Regression", "split": "test", **lr_test},
            {"model": "Linear Regression", "split": "train", **lr_train},
            {"model": "Train-mean baseline", "split": "test", **mean_base},
        ]
    )
    metrics.to_csv(results_dir / "metrics.csv", index=False, float_format="%.12f")

    coef_scaled = pd.DataFrame({
        "term": ["intercept"] + FEATURES,
        "coefficient_on_scaled_features": [float(lr.intercept_)] + [float(c) for c in lr.coef_],
    })
    coef_raw = pd.DataFrame({
        "term": ["intercept"] + FEATURES,
        "coefficient_on_raw_units": [float(lr_raw.intercept_)] + [float(c) for c in lr_raw.coef_],
    })
    coef_scaled.to_csv(results_dir / "coefficients_scaled.csv", index=False, float_format="%.12f")
    coef_raw.to_csv(results_dir / "coefficients_raw.csv", index=False, float_format="%.12f")

    pred_test = pd.DataFrame({
        "row_index": X_test.index.values,
        "Load_Weight_kg": X_test["Load_Weight_kg"].values,
        "Speed_m_per_s": X_test["Speed_m_per_s"].values,
        "Terrain_Roughness": X_test["Terrain_Roughness"].values,
        "Task_Duration_min": X_test["Task_Duration_min"].values,
        "Actual": y_test.values,
        "Pred_SVR": svr_te,
        "Pred_LR": lr_te,
        "Err_SVR": svr_te - y_test.values,
        "Err_LR": lr_te - y_test.values,
        "AbsErr_SVR": np.abs(svr_te - y_test.values),
        "AbsErr_LR": np.abs(lr_te - y_test.values),
        "split": "test",
    })
    pred_train = pd.DataFrame({
        "row_index": X_train.index.values,
        "Load_Weight_kg": X_train["Load_Weight_kg"].values,
        "Speed_m_per_s": X_train["Speed_m_per_s"].values,
        "Terrain_Roughness": X_train["Terrain_Roughness"].values,
        "Task_Duration_min": X_train["Task_Duration_min"].values,
        "Actual": y_train.values,
        "Pred_SVR": svr_tr,
        "Pred_LR": lr_tr,
        "Err_SVR": svr_tr - y_train.values,
        "Err_LR": lr_tr - y_train.values,
        "AbsErr_SVR": np.abs(svr_tr - y_train.values),
        "AbsErr_LR": np.abs(lr_tr - y_train.values),
        "split": "train",
    })
    preds = pd.concat([pred_test, pred_train], ignore_index=True)
    preds.to_csv(results_dir / "predictions.csv", index=False, float_format="%.12f")
    pred_test.to_csv(results_dir / "predictions_test.csv", index=False, float_format="%.12f")

    split = pd.DataFrame({
        "row_index": list(range(len(df))),
        "split": ["test" if i in X_test.index else "train" for i in range(len(df))],
    })
    split = split.merge(df.reset_index().rename(columns={"index": "row_index"}), on="row_index")
    split.to_csv(results_dir / "train_test_split.csv", index=False)

    ood_df = pd.DataFrame([{
        **OOD,
        "Pred_SVR": ood_svr,
        "Pred_LR": ood_lr,
        "note": "outside the training hull (max train load 22 kg, speed 2.0, duration 30 min)",
    }])
    ood_df.to_csv(results_dir / "ood_prediction.csv", index=False, float_format="%.12f")

    scaler_df = pd.DataFrame({
        "feature": FEATURES,
        "train_mean": scaler.mean_,
        "train_scale": scaler.scale_,
        "train_var": scaler.var_,
    })
    scaler_df.to_csv(results_dir / "scaler.csv", index=False, float_format="%.12f")

    info = {
        "n": int(len(df)),
        "test_size": 0.2,
        "random_state": 42,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_indices": sorted(int(i) for i in X_train.index),
        "test_indices": sorted(int(i) for i in X_test.index),
        "svr": {
            "kernel": "rbf",
            "C": svr.C,
            "epsilon": svr.epsilon,
            "gamma": "scale",
            "gamma_value": float(svr._gamma),
            "n_support": int(svr.n_support_[0]),
            "intercept": float(svr.intercept_[0]),
        },
        "ood_svr": ood_svr,
        "ood_lr": ood_lr,
        "sklearn_note": "Numbers below should match the executed notebook outputs.",
    }
    (results_dir / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")

    smooth = pd.DataFrame({"Load_Weight_kg": load_range, "SVR_pred": y_smooth})
    smooth.to_csv(results_dir / "svr_smooth_curve.csv", index=False, float_format="%.8f")

    if not args.no_plots:
        write_figures(
            results_dir / "figures",
            df, X_test, y_test, svr_te, lr_te, load_range, y_smooth,
        )

    # ---- print, matching the notebook's wording, then the extra tables ----
    print("Training size:", len(X_train))
    print("Testing size:", len(X_test))
    print()
    print("SVR model trained successfully!")
    print()
    print("Predicted Energy Consumption (Wh):", ood_svr)
    print()
    results_nb = pd.DataFrame({
        "Load_Weight": X_test["Load_Weight_kg"].values,
        "Actual_Energy": y_test.values,
        "Predicted_Energy": svr_te,
    })
    print(results_nb.to_string(index=True))
    print()
    print("Mean Squared Error (MSE):", svr_test["MSE"])
    print("R2 Score:", svr_test["R2"])
    print()
    print("=== SVR Model ===")
    print("MSE:", svr_test["MSE"])
    print("R2:", svr_test["R2"])
    print()
    print("=== Linear Regression ===")
    print("MSE:", lr_test["MSE"])
    print("R2:", lr_test["R2"])
    print()
    print("--- extra report (not in the original notebook prints) ---")
    print(f"SVR    test   MSE={svr_test['MSE']:.12f}  RMSE={svr_test['RMSE']:.12f}  MAE={svr_test['MAE']:.12f}  R2={svr_test['R2']:.12f}")
    print(f"SVR    train  MSE={svr_train['MSE']:.12f}  RMSE={svr_train['RMSE']:.12f}  MAE={svr_train['MAE']:.12f}  R2={svr_train['R2']:.12f}")
    print(f"LR     test   MSE={lr_test['MSE']:.12f}  RMSE={lr_test['RMSE']:.12f}  MAE={lr_test['MAE']:.12f}  R2={lr_test['R2']:.12f}")
    print(f"LR     train  MSE={lr_train['MSE']:.12f}  RMSE={lr_train['RMSE']:.12f}  MAE={lr_train['MAE']:.12f}  R2={lr_train['R2']:.12f}")
    print(f"Mean   test   MSE={mean_base['MSE']:.12f}  RMSE={mean_base['RMSE']:.12f}  MAE={mean_base['MAE']:.12f}  R2={mean_base['R2']:.12f}")
    print(f"LR OOD prediction (Wh): {ood_lr:.12f}")
    print(f"Wrote tables under {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
