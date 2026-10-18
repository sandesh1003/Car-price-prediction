"""
Task 3: Car Price Prediction with Machine Learning
==================================================

Workflow
  1. Collect data      - generated dataset (brand goodwill, horsepower, mileage, ...) or your own CSV
  2. Preprocess        - clean rows, impute missing values, scale numbers, one-hot encode categories
  3. Feature engineering - car age, km per year, power per litre, goodwill x power, log(km)
  4. Train             - Linear Regression, Ridge, Random Forest, Gradient Boosting (+ tuned version)
  5. Evaluate          - 5-fold cross-validation, then MAE / RMSE / R2 on a held-out test set
  6. Explain           - plots and permutation feature importance

Run:
    python car_price_prediction.py
    python car_price_prediction.py --data my_cars.csv --target price

NOTE: the built-in dataset is SYNTHETIC (prices come from a made-up formula plus noise).
It lets the whole pipeline run end to end. For real conclusions, pass a real CSV with --data.
A custom CSV needs a numeric target column; numeric columns and text columns are detected
automatically, and feature engineering is applied only to columns that exist.
"""

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # save figures to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

RANDOM_STATE = 42
REFERENCE_YEAR = 2026  # used to turn "year" into "age"

# --------------------------------------------------------------------------- #
# 1. Data collection
# --------------------------------------------------------------------------- #
# brand: (goodwill score out of 10, mean horsepower, std horsepower, sampling weight)
BRANDS = {
    "Maruti Suzuki": (8.0, 85, 12, 22),
    "Hyundai": (7.6, 95, 15, 15),
    "Tata": (7.0, 100, 18, 10),
    "Mahindra": (7.3, 130, 30, 8),
    "Honda": (8.2, 115, 20, 8),
    "Toyota": (8.8, 130, 35, 9),
    "Kia": (7.5, 110, 20, 6),
    "Skoda": (7.0, 125, 20, 4),
    "Ford": (6.5, 110, 25, 4),
    "BMW": (9.2, 220, 50, 4),
    "Mercedes-Benz": (9.4, 230, 55, 4),
    "Audi": (9.0, 215, 50, 3),
}


def generate_data(n=3000, seed=RANDOM_STATE):
    """Synthetic used-car listings. Price is in INR lakh."""
    rng = np.random.default_rng(seed)
    names = list(BRANDS)
    weights = np.array([BRANDS[b][3] for b in names], dtype=float)
    brand = rng.choice(names, size=n, p=weights / weights.sum())

    goodwill = np.array([BRANDS[b][0] for b in brand])
    hp = np.clip(
        rng.normal([BRANDS[b][1] for b in brand], [BRANDS[b][2] for b in brand]), 55, 450
    ).round()
    engine_cc = (np.clip(hp * 13 + rng.normal(0, 150, n), 800, 5000) / 50).round() * 50
    year = rng.integers(2008, REFERENCE_YEAR, n)
    age = REFERENCE_YEAR - year
    km_driven = (age * rng.gamma(4, 3500, n)).round(-2)
    mileage = np.clip(30 - 0.05 * hp + rng.normal(0, 1.8, n), 7, 30).round(1)  # km per litre
    fuel = rng.choice(["Petrol", "Diesel", "CNG"], size=n, p=[0.55, 0.38, 0.07])
    p_auto = np.where(hp > 180, 0.9, np.where(hp > 120, 0.4, 0.15))
    automatic = rng.random(n) < p_auto
    owners = rng.choice([1, 2, 3, 4], size=n, p=[0.6, 0.25, 0.11, 0.04])
    seats = rng.choice([4, 5, 7], size=n, p=[0.05, 0.80, 0.15])

    log_price = (
        0.9
        + 0.0075 * hp
        + 0.12 * goodwill
        - 0.085 * age
        - 2.5e-6 * km_driven
        - 0.05 * (owners - 1)
        + 0.10 * automatic
        + 0.06 * (fuel == "Diesel")
        - 0.08 * (fuel == "CNG")
        + 0.010 * (mileage - 18)
        + 0.03 * (seats == 7)
        + rng.normal(0, 0.10, n)
    )

    df = pd.DataFrame(
        {
            "brand": brand,
            "brand_goodwill": goodwill,
            "year": year,
            "horsepower": hp,
            "engine_cc": engine_cc,
            "mileage_kmpl": mileage,
            "km_driven": km_driven,
            "fuel_type": fuel,
            "transmission": np.where(automatic, "Automatic", "Manual"),
            "owners": owners,
            "seats": seats,
            "price_lakh": np.exp(log_price).round(2),
        }
    )
    # real data is messy: knock out ~3% of a few columns so imputation has work to do
    for col in ("horsepower", "mileage_kmpl", "km_driven"):
        df.loc[rng.random(n) < 0.03, col] = np.nan
    return df


# --------------------------------------------------------------------------- #
# 2. Cleaning and 3. Feature engineering
# --------------------------------------------------------------------------- #
def clean(df, target):
    """Drop duplicates, rows without a valid target, and physically impossible values."""
    before = len(df)
    df = df.drop_duplicates()
    df = df.dropna(subset=[target])
    df = df[df[target] > 0]
    for col, lowest in {"km_driven": 0, "horsepower": 1, "engine_cc": 1}.items():
        if col in df:
            df = df[df[col].isna() | (df[col] >= lowest)]
    print(f"Cleaning: {before} rows -> {len(df)} rows")
    return df.reset_index(drop=True)


def add_features(df):
    """Stateless feature engineering (safe to run before the train/test split)."""
    df = df.copy()
    if "year" in df:
        df["age"] = REFERENCE_YEAR - df["year"]
        df = df.drop(columns="year")
    if {"km_driven", "age"} <= set(df.columns):
        df["km_per_year"] = df["km_driven"] / df["age"].clip(lower=1)
    if "km_driven" in df:
        df["log_km_driven"] = np.log1p(df["km_driven"])  # tames the long right tail
    if {"horsepower", "engine_cc"} <= set(df.columns):
        df["hp_per_litre"] = df["horsepower"] / (df["engine_cc"] / 1000)
    if {"brand_goodwill", "horsepower"} <= set(df.columns):
        df["goodwill_x_power"] = df["brand_goodwill"] * df["horsepower"] / 100
    return df.replace([np.inf, -np.inf], np.nan)


def build_model(regressor, X_sample):
    """raw DataFrame -> feature engineering -> impute/scale/encode -> regressor.

    The target is modelled on a log scale (prices are right-skewed) and converted back
    automatically, so predictions and metrics are in the original price unit.
    """
    engineered = add_features(X_sample)
    num_cols = engineered.select_dtypes(include="number").columns.tolist()
    cat_cols = engineered.select_dtypes(exclude="number").columns.tolist()

    preprocess = ColumnTransformer(
        [
            (
                "num",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )
    pipe = Pipeline(
        [
            ("features", FunctionTransformer(add_features)),
            ("prep", preprocess),
            ("model", regressor),
        ]
    )
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)


# --------------------------------------------------------------------------- #
# 5. Evaluation helpers and plots
# --------------------------------------------------------------------------- #
def scores(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


def plot_eda(df, target, label, out):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(df[target], bins=40, color="#4C72B0", edgecolor="white")
    ax[0].set(title="Price distribution", xlabel=label, ylabel="Cars")
    ax[1].hist(np.log1p(df[target]), bins=40, color="#55A868", edgecolor="white")
    ax[1].set(title="Log-transformed price", xlabel=f"log(1 + {label})", ylabel="Cars")
    fig.tight_layout()
    fig.savefig(out / "01_price_distribution.png", dpi=150)
    plt.close(fig)

    corr = df.select_dtypes("number").corr()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation of numeric columns")
    fig.tight_layout()
    fig.savefig(out / "02_correlation_heatmap.png", dpi=150)
    plt.close(fig)

    cols = [c for c in ("horsepower", "mileage_kmpl", "km_driven", "brand_goodwill") if c in df]
    if cols:
        rows = (len(cols) + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(11, 4 * rows), squeeze=False)
        for ax, col in zip(axes.ravel(), cols):
            ax.scatter(df[col], df[target], s=6, alpha=0.35)
            ax.set(xlabel=col, ylabel=label, title=f"Price vs {col}")
        for ax in axes.ravel()[len(cols):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out / "03_price_vs_features.png", dpi=150)
        plt.close(fig)


def plot_model_comparison(results, out):
    x = np.arange(len(results))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.2, results["CV_RMSE"], 0.4, label="Cross-validation RMSE (train)")
    ax.bar(x + 0.2, results["Test_RMSE"], 0.4, label="Test RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(results.index, rotation=15, ha="right")
    ax.set(ylabel="RMSE (lower is better)", title="Model comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "04_model_comparison.png", dpi=150)
    plt.close(fig)


def plot_predictions(y_test, pred, label, out):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_test, pred, s=10, alpha=0.5)
    lim = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    ax.plot(lim, lim, "r--", label="Perfect prediction")
    ax.set(xlabel=f"Actual {label}", ylabel=f"Predicted {label}", title="Actual vs predicted (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "05_actual_vs_predicted.png", dpi=150)
    plt.close(fig)

    resid = y_test - pred
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(resid, bins=40, color="#8172B2", edgecolor="white")
    ax[0].set(title="Residuals (actual - predicted)", xlabel=label, ylabel="Cars")
    ax[1].scatter(pred, resid, s=8, alpha=0.4)
    ax[1].axhline(0, color="r", linestyle="--")
    ax[1].set(title="Residuals vs predicted", xlabel=f"Predicted {label}", ylabel="Residual")
    fig.tight_layout()
    fig.savefig(out / "06_residuals.png", dpi=150)
    plt.close(fig)


def plot_importance(model, X_test, y_test, out):
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, scoring="r2"
    )
    imp = (
        pd.DataFrame({"mean": result.importances_mean, "std": result.importances_std}, index=X_test.columns)
        .sort_values("mean")
        .tail(12)
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp.index, imp["mean"], xerr=imp["std"], color="#4C72B0")
    ax.set(xlabel="Drop in R2 when the column is shuffled", title="Permutation feature importance (test set)")
    fig.tight_layout()
    fig.savefig(out / "07_feature_importance.png", dpi=150)
    plt.close(fig)
    return imp.sort_values("mean", ascending=False)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Car price prediction")
    parser.add_argument("--data", help="optional CSV with car data (default: generate a synthetic dataset)")
    parser.add_argument("--target", default="price_lakh", help="name of the price column")
    parser.add_argument("--out", default="outputs", help="folder for plots, tables and the saved model")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = args.target
    label = "Price (INR lakh)" if target == "price_lakh" else target

    # 1. data
    if args.data:
        df = pd.read_csv(args.data)
        print(f"Loaded {args.data}: {df.shape[0]} rows x {df.shape[1]} columns")
    else:
        df = generate_data()
        df.to_csv(out / "car_price_data.csv", index=False)
        print(f"Generated synthetic dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nMissing values per column:\n", df.isna().sum()[df.isna().sum() > 0].to_string(), sep="")

    # 2. clean + explore
    df = clean(df, target)
    print("\nPrice summary:\n", df[target].describe().round(2).to_string(), sep="")
    plot_eda(df, target, label, out)

    X, y = df.drop(columns=target), df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # 3. tune gradient boosting on the training set only
    print("\nTuning Gradient Boosting (grid search, 5-fold CV)...")
    grid = GridSearchCV(
        build_model(GradientBoostingRegressor(random_state=RANDOM_STATE), X_train),
        param_grid={
            "regressor__model__n_estimators": [200, 400],
            "regressor__model__learning_rate": [0.05, 0.1],
            "regressor__model__max_depth": [2, 3],
        },
        cv=KFold(5, shuffle=True, random_state=RANDOM_STATE),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    best_gb = {k.split("__")[-1]: v for k, v in grid.best_params_.items()}
    print("Best parameters:", best_gb)

    # 4. compare models
    candidates = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "Gradient Boosting (tuned)": GradientBoostingRegressor(random_state=RANDOM_STATE, **best_gb),
    }
    cv = KFold(5, shuffle=True, random_state=RANDOM_STATE)
    rows, fitted = {}, {}
    for name, regressor in candidates.items():
        model = build_model(regressor, X_train)
        cv_res = cross_validate(
            model, X_train, y_train, cv=cv, scoring=("r2", "neg_root_mean_squared_error")
        )
        model.fit(X_train, y_train)
        test = scores(y_test, model.predict(X_test))
        rows[name] = {
            "CV_R2": cv_res["test_r2"].mean(),
            "CV_RMSE": -cv_res["test_neg_root_mean_squared_error"].mean(),
            "Test_MAE": test["MAE"],
            "Test_RMSE": test["RMSE"],
            "Test_R2": test["R2"],
        }
        fitted[name] = model

    results = pd.DataFrame(rows).T
    results.round(4).to_csv(out / "model_comparison.csv")
    print(f"\nModel comparison (errors are in {label}):\n", results.round(3).to_string(), sep="")
    plot_model_comparison(results, out)

    # pick the final model using cross-validation only, never the test set
    best_name = results["CV_RMSE"].idxmin()
    best = fitted[best_name]
    print(f"\nSelected model (lowest CV RMSE): {best_name}")

    # 5. final evaluation + explanations
    pred = best.predict(X_test)
    final = scores(y_test, pred)
    print(
        f"Test set -> MAE {final['MAE']:.3f} | RMSE {final['RMSE']:.3f} | R2 {final['R2']:.3f}"
        f" | typical error {100 * np.median(np.abs(pred - y_test) / y_test):.1f}% of price (median)"
    )
    plot_predictions(y_test.to_numpy(), pred, label, out)
    importance = plot_importance(best, X_test, y_test, out)
    print("\nMost important raw columns:\n", importance["mean"].head(6).round(3).to_string(), sep="")

    sample = X_test.head(5)
    preview = sample.copy()
    preview["actual"] = y_test.head(5).values
    preview["predicted"] = best.predict(sample).round(2)
    print("\nSample predictions:\n", preview[["actual", "predicted"]].to_string(), sep="")

    # 6. save the whole pipeline (accepts raw car columns)
    joblib.dump(best, out / "car_price_model.joblib")
    print(f"\nSaved plots, tables and model to ./{out}/")


if __name__ == "__main__":
    main()
