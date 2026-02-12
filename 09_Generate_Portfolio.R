# === Load Required Libraries ===
library(data.table)

# === Load Forecast File ===
forecast_path <- "forecasts/ranked_forecast_2022-09-26 - 2025-04-06.csv"
forecast <- fread(forecast_path)

# === Drop Unused Columns (but keep quintile ranks) ===
forecast[, c("Decision", "Weight") := NULL]

# === Compute Expected Rank ===
forecast[, ExpectedRank := Rank1 * 1 + Rank2 * 2 + Rank3 * 3 + Rank4 * 4 + Rank5 * 5]

# === Compute Z-Score vs SPY ===
spy_rank <- forecast[ID == "SPY", ExpectedRank]
sigma <- sd(forecast$ExpectedRank)
forecast[, ZScore := (ExpectedRank - spy_rank) / sigma]

# === Flag Tickers Above SPY by Threshold ===
z_threshold <- 1.0
cutoff <- spy_rank + z_threshold * sigma
forecast[, Invest := ExpectedRank > cutoff]

# === Merge Sector Info ===
metadata <- readRDS("data/tickers_metadata.rds")
forecast <- merge(forecast, metadata[, .(ID = Symbol, Sector)], by = "ID", all.x = TRUE)

# === Merge Volatility Info ===
stocks <- readRDS("data/tickers_data_cleaned.rds")
volatility <- sapply(stocks, function(x) sd(diff(log(x$Adjusted)), na.rm = TRUE))
vol_dt <- data.table(ID = names(volatility), Volatility = as.numeric(volatility))
forecast <- merge(forecast, vol_dt, by = "ID", all.x = TRUE)

# === Save Full Enriched Portfolio Table ===
fwrite(forecast, "forecasts/portfolio_expected_rank_full.csv")

# === Optional: View Sorted Output ===
print(forecast[order(-ExpectedRank), .(ID, Sector, Volatility, Rank1, Rank2, Rank3, Rank4, Rank5, ExpectedRank, ZScore, Invest)])
