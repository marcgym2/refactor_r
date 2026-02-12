
# === Load Libraries ===
library(quantmod)
library(data.table)

# === Load Helper Scripts ===
source("scripts/config.R")

# Crear carpeta Data si no existe
if (!dir.exists("data")) dir.create("data", recursive = TRUE)

# Símbolos y datos básicos de sectores del S&P500
sector_etfs <- c("SPY", "XLK", "XLF", "XLV", "XLI", "XLY", 
                 "XLP", "XLE", "XLB", "XLC", "XLU", "XLRE", "SH", "IAU")


# === Load Ticker Metadata ===
StockNames <- data.table(
  Symbol = sector_etfs,
  Name = c("S&P 500 ETF", "Technology Select Sector", "Financial Select Sector",
           "Health Care Select Sector", "Industrial Select Sector", "Consumer Discretionary",
           "Consumer Staples", "Energy Select Sector", "Materials Select Sector", "Gold",
           "Communication Services", "Utilities Select Sector", "Real Estate Select Sector", "Short S&P 500 ETF"),
  Sector = c("Broad Market", "Technology", "Financials", "Health Care", "Industrials",
             "Consumer Discretionary", "Consumer Staples", "Energy", "Materials",
             "Communication Services", "Utilities", "Real Estate", "Broad Market", "Gold"),
  ETF = TRUE
)

# Guardar a disco

# === Save Updated Metadata ===
saveRDS(StockNames, file.path("data", "tickers_metadata.rds"))
