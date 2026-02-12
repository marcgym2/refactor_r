
# === Load Libraries ===
library(quantmod)
library(data.table)

# === Load Helper Scripts ===
source("scripts/config.R")
require(TTR)

# === Load Libraries ===
library(stringr)
library(imputeTS)
set.seed(1)


# === Load Helper Scripts ===
source("scripts/02a_DataCleaning_Helpers.R")

# Load the stock names for SOXX

# === Load Ticker Metadata ===
StockNames <- readRDS(file.path("data", "tickers_metadata.rds"))

# Initialize additional columns
StockNames[, MinDate := as.Date(NA)]
StockNames[, MaxDate := as.Date(NA)]
StockNames[, Activity := as.numeric(NA)]
StockNames[, missings := 0]

# Define the tickers

# === Define Ticker Symbols ===
tickers <- StockNames[, Symbol]

# Define date range for data download

# === Define Date Range ===
from = "2000-01-01"
to = Sys.Date()  # Update to today's date

# Downloading data

# === Download Stock Data ===
Stocks <- lapply(seq_along(tickers), function(i) {
  Sys.sleep(1)
  print(round(i/length(tickers), 3))
  ticker <- tickers[i]
  out <- try({
    temp <- as.data.table(getSymbols(ticker, from = from, to = to, auto.assign = FALSE))
    colnames(temp) <- c("index", "Open", "High", "Low", "Close", "Volume", "Adjusted")
    StockNames[i, MinDate := min(temp$index)]
    StockNames[i, MaxDate := max(temp$index)]
    StockNames[i, Activity := temp[max(1, .N - 100):.N, mean(Volume * Close)]]
    temp
  })
  if ("try-error" %in% class(out)) {
    out <- NULL
  }
  return(out)
})

# Check data integrity

# === Validate Downloaded Data ===
if (length(Stocks) == length(tickers)) {
  names(Stocks) = tickers
} else {
  stop("Data download incomplete")
}
print(sum(!sapply(Stocks, function(y) {!is.null(y)})))

if (sum(!(tickers %in% names(Stocks))) > 0) {
  warning(str_c(sum(!(tickers %in% names(Stocks))), " stock(s) missing"))
}

# Filter out stocks with missing data
Stocks <- Stocks[sapply(Stocks, function(y) {!is.null(y)})]
table(as.Date(sapply(Stocks, function(s) s[.N, index])))

# Clean the data

# === Clean Stock Data ===
StocksClean <- setNames(lapply(names(Stocks), function(ticker) {
  stock <- Stocks[[ticker]]
  naRows <- sum(apply(is.na(stock), 1, any))
  StockNames[Symbol == ticker, missings := naRows]
  if (naRows > 0) {
    print(str_c("Ticker: ", ticker, " Missing: ", naRows))
    return(cbind(stock[, .(index)], stock[, lapply(.SD, noisyInterpolation), .SDcols = names(stock)[-1]]))
  } else {
    return(stock)
  }
}), names(Stocks))

# Save the cleaned data

# === Save Cleaned Stock Data ===
saveRDS(StocksClean, file.path("data", "tickers_data_cleaned.rds"))

# Save the updated stock names with additional metadata

# === Save Updated Metadata ===
saveRDS(StockNames, file.path("data", "tickers_metadata.rds"))