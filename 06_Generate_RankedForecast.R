
# === Load Libraries ===
library(data.table)

# === Load Helper Scripts ===
source("scripts/config.R")

# === Load Libraries ===
library(stringr)
library(torch)

# === Load Libraries ===
library(ggplot2)
library(TTR)
rm(list=ls())

# Definir rutas

# === Set File Paths ===
data_dir <- "data"
precomputed_dir <- "features"
results_dir <- "forecasts"
if (!dir.exists(results_dir)) dir.create(results_dir, recursive = TRUE)
tempFilePath <- file.path("temp")


# === Load Helper Scripts ===
source(file.path("scripts", "05_RankingValidation_Helpers.R"))

# Crear ranked_forecast_ranked_forecast_template.csv si no existe
template_path <- "ranked_forecast_ranked_forecast_template.csv"
if (!file.exists(template_path)) {

# === Load Ticker Metadata ===
  StockNames <- readRDS(file.path(data_dir, "tickers_metadata.rds"))
  template <- data.table(
    ID = StockNames$Symbol,
    Rank1 = rep(0.2, nrow(StockNames)),
    Rank2 = rep(0.2, nrow(StockNames)),
    Rank3 = rep(0.2, nrow(StockNames)),
    Rank4 = rep(0.2, nrow(StockNames)),
    Rank5 = rep(0.2, nrow(StockNames)),
    Decision = rep(0.01, nrow(StockNames))
  )
  fwrite(template, template_path)
  cat("ranked_forecast_ranked_forecast_template.csv creado.\n")
} else {
  template <- fread(template_path)
}


# === Load Ticker Metadata ===
StockNames <- readRDS(file.path(data_dir, "tickers_metadata.rds"))

# === Generate Quantile Predictions ===
QuantilePredictions <- readRDS(file.path(precomputed_dir, "forecast_ranks_all.rds"))
QuantilePrediction <- QuantilePredictions$meta

# Filtrar datos de validación (asegúrate de que en QuantilePrediction exista la columna Split)
submission <- QuantilePrediction[Split == "Validation"]

# Calcular un único string para el periodo usando la fecha mínima y máxima
period <- paste0(min(submission$IntervalStart), " - ", max(submission$IntervalEnd))

# Ordenar submission de acuerdo al template y definir columnas finales
submission <- submission[match(template$ID, submission$Ticker), 
                         .(ID = Ticker, Rank1, Rank2, Rank3, Rank4, Rank5, Decision = 0)]
submission[, Decision := 0.01 * 0.25]

submission <- validateSubmission(submission, Round = TRUE)

# Guardar la submission en un único archivo CSV

# === Export Final Ranked Forecast ===
write.csv(submission, file.path(results_dir, paste0("ranked_forecast_", period, ".csv")), 
          row.names = FALSE, quote = FALSE)
