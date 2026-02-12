"""
Step 07 — Ticker Correlation Analysis.

Computes daily log returns, correlation matrix, PCA, hierarchical
clustering, and volatility — saving all plots as PNGs.
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram

from config import DATA_DIR, FORECASTS_DIR


def run() -> None:
    # --- Load cleaned stock data ---
    with open(os.path.join(DATA_DIR, "tickers_data_cleaned.pkl"), "rb") as f:
        stocks = pickle.load(f)

    # --- Compute daily log returns ---
    returns: dict[str, pd.Series] = {}
    for ticker, df in stocks.items():
        adj = df.set_index("index")["Adjusted"].astype(float)
        returns[ticker] = np.log(adj / adj.shift(1))

    merged = pd.DataFrame(returns).dropna(how="all")
    returns_matrix = merged.values

    # --- Correlation heatmap ---
    cor_matrix = merged.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cor_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                xticklabels=cor_matrix.columns, yticklabels=cor_matrix.columns)
    ax.set_title("Ticker Correlation Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(FORECASTS_DIR, "ticker_correlation_matrix.png"), dpi=150)
    plt.close(fig)
    print("[Step 07] Saved correlation matrix")

    # --- PCA ---
    clean = merged.dropna()
    clean = clean.loc[:, clean.std() > 0]
    pca = PCA()
    pca.fit(clean)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_,
           color="steelblue")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("PCA Scree Plot")
    fig.tight_layout()
    fig.savefig(os.path.join(FORECASTS_DIR, "ticker_pca_screeplot.png"), dpi=150)
    plt.close(fig)
    print("[Step 07] Saved PCA scree plot")

    # Variable contribution (loading plot for first 2 PCs)
    loadings = pd.DataFrame(pca.components_[:2].T, index=clean.columns, columns=["PC1", "PC2"])
    fig, ax = plt.subplots(figsize=(10, 8))
    contrib = np.sqrt((loadings ** 2).sum(axis=1))
    scatter = ax.scatter(loadings["PC1"], loadings["PC2"], c=contrib, cmap="YlOrRd", s=60)
    for ticker, row in loadings.iterrows():
        ax.annotate(ticker, (row["PC1"], row["PC2"]), fontsize=8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA Variable Contributions")
    plt.colorbar(scatter, label="Contribution")
    fig.tight_layout()
    fig.savefig(os.path.join(FORECASTS_DIR, "ticker_pca_varplot.png"), dpi=150)
    plt.close(fig)
    print("[Step 07] Saved PCA variable plot")

    # --- Hierarchical Clustering (by ticker using correlation distance) ---
    dist_matrix = 1 - cor_matrix.values  # correlation distance
    # condensed form (upper triangle)
    from scipy.spatial.distance import squareform
    dist_condensed = squareform(dist_matrix, checks=False)
    Z = linkage(dist_condensed, method="ward")
    fig, ax = plt.subplots(figsize=(12, 8))
    dendrogram(Z, labels=cor_matrix.columns.tolist(), ax=ax, leaf_rotation=90)
    ax.set_title("Hierarchical Clustering of Tickers (Correlation Distance)")
    fig.tight_layout()
    fig.savefig(os.path.join(FORECASTS_DIR, "ticker_hclust_dendrogram.png"), dpi=150)
    plt.close(fig)
    print("[Step 07] Saved clustering dendrogram")

    # --- Volatility ---
    volatility = merged.std().dropna().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(volatility.index, volatility.values, color="steelblue")
    ax.set_xlabel("Volatility (Std Dev of Log Returns)")
    ax.set_title("Ticker Volatility")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(FORECASTS_DIR, "ticker_volatility_barplot.png"), dpi=150)
    plt.close(fig)
    print("[Step 07] Saved volatility bar chart")


if __name__ == "__main__":
    run()
