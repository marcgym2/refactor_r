# === Load Libraries ===
library(data.table)
library(torch)
library(stringr)
library(ggplot2)
library(TTR)

# === Main Pipeline Script ===

# Step 1: Generate Ticker Metadata
source("scripts/01_Generate_Ticker_Metadata.R")

# Step 2: Download & Clean Raw Ticker Data
source("scripts/02_Download_Clean_TickerData.R")

# Step 3: Feature Engineering Helpers (imputation, standardization)
source("scripts/03a_FeatureEngineering_Helpers.R")

# Step 4: Feature Engineering (generate indicators)
source("scripts/03_FeatureEngineering.R")

# Step 5: MetaModel Structure Definition
source("scripts/04a_MetaModel_Module.R")

# Step 6: Train Base and Meta Quantile Models
source("scripts/04_Train_Quantile_Models.R")

# Step 7: Generate Final Ranked Forecast CSV
source("scripts/06_Generate_RankedForecast.R")

# Step 8: Generate Portfolio Allocation
source("scripts/09_Generate_Portfolio.R")

# Step 9: (Optional) Run Ticker Correlation + Clustering Analysis
source("scripts/07_Ticker_Correlation_Analysis.R")

# Step 10: (Optional) Launch Shiny Dashboard
source("scripts/08_Ticker_Analysis_Dashboard.R")
