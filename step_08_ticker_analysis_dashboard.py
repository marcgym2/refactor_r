"""
Step 08 — Ticker Analysis Dashboard (Streamlit).

Replaces the R Shiny dashboard. Run with:
    streamlit run step_08_ticker_analysis_dashboard.py
"""

from __future__ import annotations

import os
import streamlit as st

from config import FORECASTS_DIR


def main() -> None:
    st.set_page_config(page_title="Ticker Analytics Dashboard", layout="wide")
    st.title("📊 Ticker Analytics Dashboard")
    st.markdown("Explore **correlation**, **PCA**, **clustering**, and **volatility** analysis.")

    tabs = st.tabs(["Correlation", "PCA Scree", "PCA Variables", "Clustering", "Volatility"])

    plot_map = {
        0: "ticker_correlation_matrix.png",
        1: "ticker_pca_screeplot.png",
        2: "ticker_pca_varplot.png",
        3: "ticker_hclust_dendrogram.png",
        4: "ticker_volatility_barplot.png",
    }

    for i, tab in enumerate(tabs):
        with tab:
            img_path = os.path.join(FORECASTS_DIR, plot_map[i])
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
            else:
                st.warning(
                    f"`{plot_map[i]}` not found. Run **Step 07** first to generate plots."
                )


if __name__ == "__main__":
    main()
