"""
main.py — Pipeline Orchestrator.

Runs all steps of the stock ranking pipeline in order.
"""

from __future__ import annotations


def main() -> None:
    print("=" * 60)
    print("  Stock Ranking Pipeline — Python Refactor")
    print("=" * 60)

    # Step 1: Generate Ticker Metadata
    import step_01_generate_ticker_metadata
    step_01_generate_ticker_metadata.run()

    # Step 2: Download & Clean Raw Ticker Data
    import step_02_download_clean_ticker_data
    step_02_download_clean_ticker_data.run()

    # Step 4: Train Base + Meta Quantile Models
    #         (internally runs Step 3 feature engineering)
    import step_04_train_quantile_models
    step_04_train_quantile_models.run()

    # Step 6: Generate Final Ranked Forecast CSV
    import step_06_generate_ranked_forecast
    step_06_generate_ranked_forecast.run()

    # Step 9: Generate Portfolio Allocation
    import step_09_generate_portfolio
    step_09_generate_portfolio.run()

    # Step 7: (Optional) Ticker Correlation + Clustering Analysis
    import step_07_ticker_correlation_analysis
    step_07_ticker_correlation_analysis.run()

    # Step 8: (Optional) Launch Streamlit Dashboard
    #         Run separately via: streamlit run step_08_ticker_analysis_dashboard.py
    print("\n[main] ✅ Pipeline complete.")
    print("[main] To launch the dashboard:  streamlit run step_08_ticker_analysis_dashboard.py")


if __name__ == "__main__":
    main()
