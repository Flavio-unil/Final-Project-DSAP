"""
Main entry point for the project:
Can Economic Sentiment Predict Market Movements?

This script reproduces the full pipeline using the final processed dataset
and runs the out-of-sample ML evaluation.

Run from project root with:
    python main.py
"""

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import statsmodels.api as sm


# Configuration
DATA_PATH = "datasets/clean_data/merged_weekly.csv"
OUTPUT_DIR = "outputs"
SPLIT_DATE = "2019-01-01"
# -----------------------------
# Sentiment keyword lists
# -----------------------------

NEGATIVE_WORDS = [
    "recession",
    "economic_recession",
    "financial_crisis",
    "stock_market_crash",
    "bear_market",
    "inflation",
    "debt_crisis",
    "banking_crisis",
    "job_losses",
    "unemployement"
]

POSITIVE_WORDS = [
    "economic_growth",
    "economic_recovery",
    "consumer_confidence",
    "business_confidence",
    "stock_market_rally",
    "bull_market",
    "job_creation",
    "strong_economy",
    "low_unemployment",
    "stock_market_optimism"
]
RANDOM_STATE = 42

# Load data
print("Loading dataset...")

df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Target construction
# Next-week return prediction target
df["ret_next_week"] = df["SP500_return_w"].shift(-1)
df = df.dropna(subset=["ret_next_week"]).copy()


# Sentiment feature engineering
# Simple sentiment index: average(positive) - average(negative)
df["pos_index"] = df[POSITIVE_WORDS].mean(axis=1)
df["neg_index"] = df[NEGATIVE_WORDS].mean(axis=1)
df["sentiment_simple"] = df["pos_index"] - df["neg_index"]

# Train / test split (needed before weighted index)
split_date = pd.Timestamp(SPLIT_DATE)
train_df = df[df["Date"] < split_date].copy()
test_df = df[df["Date"] >= split_date].copy()

# Weighted sentiment: weights learned on train only (avoid look-ahead bias)
all_words = POSITIVE_WORDS + NEGATIVE_WORDS
weights = train_df[all_words].corrwith(train_df["ret_next_week"]).fillna(0.0)
df["sentiment_weighted"] = (df[all_words] * weights).sum(axis=1)

# PCA sentiment index
from sklearn.decomposition import PCA

X_words = df[all_words].values
X_words_scaled = StandardScaler().fit_transform(X_words)
pca = PCA(n_components=1, random_state=RANDOM_STATE)
df["sentiment_pca"] = pca.fit_transform(X_words_scaled)

# Final feature set
features = ["sentiment_simple", "sentiment_weighted", "sentiment_pca"]

# train/test with the final feature matrix
train_df = df[df["Date"] < split_date].copy()
test_df = df[df["Date"] >= split_date].copy()

X_train = train_df[features]
y_train = train_df["ret_next_week"]

X_test = test_df[features]
y_test = test_df["ret_next_week"]


print(f"Train observations: {len(train_df)}")
print(f"Test observations: {len(test_df)}")


# OLS (out-of-sample)

Xtr = sm.add_constant(X_train)
Xte = sm.add_constant(X_test)

ols = sm.OLS(y_train, Xtr).fit()
y_pred_ols = ols.predict(Xte)

results = []
results.append({
    "model": "OLS",
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_ols)),
    "MAE": mean_absolute_error(y_test, y_pred_ols),
    "R2": r2_score(y_test, y_pred_ols)
})


# Machine Learning models

models = {
    "Ridge": Pipeline([
        ("scaler", StandardScaler()),
        ("model", RidgeCV(alphas=np.logspace(-4, 4, 80)))
    ]),
    "Lasso": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LassoCV(
            alphas=np.logspace(-4, 1, 80),
            max_iter=30000,
            random_state=RANDOM_STATE
        ))
    ]),
    "RandomForest": RandomForestRegressor(
        n_estimators=600,
        min_samples_leaf=3,
        random_state=RANDOM_STATE
    )
}

predictions = {"OLS": y_pred_ols}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred

    results.append({
        "model": name,
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred)
    })

results_df = pd.DataFrame(results).sort_values("RMSE")


# Save outputs

os.makedirs(OUTPUT_DIR, exist_ok=True)
results_path = os.path.join(OUTPUT_DIR, "model_results.csv")
results_df.to_csv(results_path, index=False)

print("\nModel comparison (out-of-sample):")
print(results_df)
print(f"\nResults saved to {results_path}")

# Plot 1: Time series (Actual vs Predicted - Lasso)

plt.figure()
plt.plot(y_test.values, label="Actual")
plt.plot(predictions["Lasso"], label="Predicted (Lasso)")
plt.title("Out-of-sample: Actual vs Predicted Returns")
plt.xlabel("Test observations")
plt.ylabel("Next-week return")
plt.legend()
plt.tight_layout()
plt.show(block=True)


# Plot 2: Scatter (Predicted vs Actual - Lasso)


plt.figure()
plt.scatter(y_test.values, predictions["Lasso"], alpha=0.7)
plt.xlabel("Actual returns")
plt.ylabel("Predicted returns")
plt.title("Predicted vs Actual Returns (Lasso)")
plt.tight_layout()
plt.show(block=True)


print("\nPipeline completed successfully.")
