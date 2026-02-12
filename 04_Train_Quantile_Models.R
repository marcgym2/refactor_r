
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
rm(list=ls()); gc()

# Rutas de archivos y directorios

# === Set File Paths ===
data_dir <- "data"
precomputed_dir <- "features"
if (!dir.exists(precomputed_dir)) dir.create(precomputed_dir, recursive = TRUE)
tempFilePath <- file.path("temp")

# Cargar scripts auxiliares (ajusta las rutas según corresponda)

# === Load Helper Scripts ===
source(file.path("scripts", "03a_FeatureEngineering_Helpers.R"))
source(file.path("scripts", "03_FeatureEngineering.R"))

# === Load Helper Scripts ===
source(file.path("scripts", "04a_MetaModel_Module.R"))

# Definir lista de funciones de features

# === Define Feature Functions ===
featureList <- c(
  list(
    function(SD, BY) { Return(SD) },
    function(SD, BY) { LagVolatility(SD, lags = 1:7) },
    function(SD, BY) { LagReturn(SD, lags = 1:7) },
    function(SD, BY) { IsETF(SD, BY, StockNames = StockNames) }
  ),
  TTR
)

Shifts <- c(0, 7, 14, 21)
Submission <- 12

# === Generate Interval Information ===
IntervalInfos <- GenIntervalInfos(Submission = Submission, Shifts = Shifts)

# Generación o carga de StocksAggr
GenerateStockAggr <- TRUE 
precomputed_path <- file.path(precomputed_dir, "features_raw.rds")


# === Generate or Load Aggregated Features ===
if (GenerateStockAggr) {

# === Load Ticker Metadata ===
  StockNames <- readRDS(file.path(data_dir, "tickers_metadata.rds"))
  Stocks <- readRDS(file.path(data_dir, "tickers_data_cleaned.rds"))
  # Ordenar según Symbol
  temp <- StockNames[order(Symbol), .(Symbol)]
  Stocks <- Stocks[temp$Symbol]
  StocksAggr <- GenStocksAggr(Stocks, IntervalInfos, featureList, CheckLeakage = FALSE)
  saveRDS(StocksAggr, precomputed_path)
} else {
  if (file.exists(precomputed_path)) {
    StocksAggr <- readRDS(precomputed_path)
  } else {
    stop("El archivo 'Precomputed/features_raw.rds' no existe.")
  }
}

# Imputación y estandarización de features
featureNames <- setdiff(names(StocksAggr), c("Ticker", "Interval", "Return", 
                                             "Shift", "ReturnQuintile", 
                                             "IntervalStart", "IntervalEnd"))

# === Impute and Standardize Features ===
StocksAggr <- imputeFeatures(StocksAggr, featureNames = featureNames)
StocksAggr <- standartizeFeatures(StocksAggr, featureNames = setdiff(featureNames, "ETF"))
StocksAggr <- StocksAggr[order(IntervalStart, Ticker)]
saveRDS(StocksAggr, file.path(precomputed_dir, "features_standardized.rds"))
StocksAggr <- readRDS(file.path(precomputed_dir, "features_standardized.rds"))

# División en conjuntos de entrenamiento, test y validación

# === Split Data into Train/Test/Validation ===
TrainStart <- as.Date("2000-01-01")
interval_infos <- unique(StocksAggr[IntervalStart >= TrainStart, .(IntervalStart, IntervalEnd)])
num_intervals <- nrow(interval_infos)
train_end_index <- floor(0.8 * num_intervals)
test_end_index <- floor(0.9 * num_intervals)

TrainEnd <- interval_infos$IntervalEnd[train_end_index]
TestStart <- interval_infos$IntervalStart[train_end_index + 1]
TestEnd <- interval_infos$IntervalEnd[test_end_index]
ValidationStart <- interval_infos$IntervalStart[test_end_index + 1]
ValidationEnd <- interval_infos$IntervalEnd[num_intervals]

TrainRows <- which(StocksAggr$IntervalStart >= TrainStart & StocksAggr$IntervalEnd <= TrainEnd)
TestRows <- which(StocksAggr$IntervalStart >= TestStart & StocksAggr$IntervalEnd <= TestEnd)
ValidationRows <- which(StocksAggr$IntervalStart >= ValidationStart & StocksAggr$IntervalEnd <= ValidationEnd)

cat(sprintf("Validation Start: %s\nValidation End: %s\n", ValidationStart, ValidationEnd))
cat(sprintf("Rows - Train: %d | Test: %d | Validation: %d\n", 
            length(TrainRows), length(TestRows), length(ValidationRows)))

# Preparar tensores para entrenamiento

# === Prepare Training Tensors ===
y <- StocksAggr$ReturnQuintile
y_tensor <- torch_tensor(t(sapply(y, function(x) {
  if (is.na(x)) rep(NA, 5) else replace(numeric(5), x:5, 1)
})), dtype = torch_float())
x <- torch_tensor(as.matrix(StocksAggr[, ..featureNames]), dtype = torch_float())
xtype_factor <- as.factor(StocksAggr$Ticker)
i <- torch_tensor(t(cbind(seq_along(xtype_factor), as.integer(xtype_factor))),
                  dtype = torch_int64())
v <- torch_tensor(rep(1, length(xtype_factor)))
xtype <- torch_sparse_coo_tensor(i, v, c(length(xtype_factor), length(levels(xtype_factor))))$coalesce()

# División de tensores
y_train <- y_tensor[TrainRows, , drop = FALSE]
x_train <- x[TrainRows, , drop = FALSE]
xtype_train <- subsetTensor(xtype, rows = TrainRows)
y_test <- y_tensor[TestRows, , drop = FALSE]
x_test <- x[TestRows, , drop = FALSE]
xtype_test <- subsetTensor(xtype, rows = TestRows)
y_validation <- y_tensor[ValidationRows, , drop = FALSE]
x_validation <- x[ValidationRows, , drop = FALSE]
xtype_validation <- subsetTensor(xtype, rows = ValidationRows)
criterion <- function(y_pred, y) { ComputeRPSTensor(y_pred, y) }

# Entrenamiento del modelo base
horizon <- NULL
r <- 1
set.seed(r)
torch_manual_seed(r)
inputSize <- length(featureNames)
layerSizes <- c(32, 8, 5)
layerDropouts <- c(rep(0.2, length(layerSizes) - 1), 0)
layerTransforms <- c(lapply(seq_len(length(layerSizes) - 1), function(x) nnf_leaky_relu),
                     list(function(x) nnf_softmax(x, 2)))

# === Define and Train Base Model ===
baseModel <- constructFFNN(inputSize, layerSizes, layerTransforms, layerDropouts)
baseModel <- prepareBaseModel(baseModel, x = x_train)
minibatch <- 200
lr <- c(0.01)

start <- Sys.time()
fit <- trainModel2(model = baseModel, criterion, 
                   train = list(y_train, x_train),
                   test = list(y_test, x_test),
                   validation = list(y_validation, x_validation),
                   epochs = 100, minibatch = minibatch, tempFilePath = tempFilePath, 
                   patience = 5, printEvery = 1, lr = lr)
cat("Tiempo entrenamiento modelo base:", Sys.time() - start, "\n")
baseModel <- fit$model
baseModelProgress <- fit$progress
saveRDS(baseModelProgress, file.path(precomputed_dir, "training_log_base.rds"))
torch_save(baseModel, file.path(precomputed_dir, "model_base.t7"))

# Cargar y verificar el modelo base
baseModel <- torch_load(file.path(precomputed_dir, "model_base.t7"))
print(baseModel)
y_pred_base <- baseModel(x_validation)
loss_validation_base <- as.array(ComputeRPSTensor(y_pred_base, y_validation))
loss_validation_base_vector <- as.array(ComputeRPSTensorVector(y_pred_base, y_validation))

# Entrenamiento del meta-modelo
allowMetaStructure <- rapply(baseModel$stateStructure, 
                             function(x) torch_tensor(array(FALSE, dim = x)), 
                             how = "list")
includedLayers <- (length(baseModel$stateStructure) - 1):length(baseModel$stateStructure)
for (i in includedLayers) {
  allowMetaStructure[[i]] <- torch_tensor(array(TRUE, dim = dim(allowMetaStructure[[i]])))
}


# === Define and Train Meta Model ===
metaModel <- MetaModel(baseModel, xtype_train, mesaParameterSize = 1, allowBias = TRUE, 
                       pDropout = 0, initMesaRange = 0, initMetaRange = 1,
                       allowMetaStructure = allowMetaStructure)
minibatch_fn <- function() { minibatchSampler(100, xtype_train) }
lr_meta <- c(0.01, 0.001, 0.001, 0.0005, 0.0003, 0.0001, 0.00005)

set.seed(r)
torch_manual_seed(r)
start <- Sys.time()
fit_meta <- trainModel2(model = metaModel, criterion,
                        train = list(y_train, x_train, xtype_train),
                        test = list(y_test, x_test, xtype_test),
                        validation = list(y_validation, x_validation, xtype_validation),
                        epochs = 100, minibatch = minibatch_fn, tempFilePath = tempFilePath, 
                        patience = 5, printEvery = 1, lr = lr_meta)
cat("Tiempo entrenamiento meta-modelo:", Sys.time() - start, "\n")
metaModel <- fit_meta$model
metaModelProgress <- fit_meta$progress
saveRDS(metaModelProgress, file.path(precomputed_dir, "training_log_meta.rds"))
torch_save(metaModel, file.path(precomputed_dir, "model_meta.t7"))

metaModel <- torch_load(file.path(precomputed_dir, "model_meta.t7"))
print(metaModel)
y_pred_meta <- metaModel(x_validation, xtype_validation)
loss_validation_meta <- as.array(ComputeRPSTensor(y_pred_meta, y_validation))
loss_validation_meta_vector <- as.array(ComputeRPSTensorVector(y_pred_meta, y_validation))
cat("Loss meta-modelo:", loss_validation_meta, "\n")

# Visualización
temp_data <- rbind(
  melt(baseModelProgress, id.vars = "epoch")[, type := "base"],
  melt(metaModelProgress, id.vars = "epoch")[, epoch := epoch + nrow(baseModelProgress)][, type := "meta"]
)
temp_validation <- rbind(
  data.table(
    epoch = c(max(temp_data[type == "base", epoch]), 
              max(temp_data[type == "meta", epoch]), 
              max(temp_data[type == "meta", epoch]) + 1),
    variable = "loss_validation",
    value = c(loss_validation_base, loss_validation_meta, mean(loss_validation_meta_vector)),
    type = c("base", "meta", "mesa"),
    subset = "all"
  ),
  rbind(
    data.table(
      epoch = max(temp_data[type == "base", epoch]),
      variable = "loss_validation",
      value = loss_validation_base_vector,
      type = "base",
      subset = seq_along(loss_validation_base_vector)
    ),
    data.table(
      epoch = max(temp_data[type == "meta", epoch]),
      variable = "loss_validation",
      value = loss_validation_meta_vector,
      type = "meta",
      subset = seq_along(loss_validation_meta_vector)
    ),
    data.table(
      epoch = max(temp_data[type == "meta", epoch]) + 1,
      variable = "loss_validation",
      value = loss_validation_meta_vector,
      type = "mesa",
      subset = seq_along(loss_validation_meta_vector)
    )
  )
)

ggplot(temp_data, aes(x = epoch, y = value, colour = variable, shape = type)) +
  geom_line(aes(linetype = type)) +
  geom_point(data = temp_validation[subset == "all"]) +
  geom_text(data = temp_validation[subset == "all"], 
            aes(label = round(value, 4)), hjust = -0.5, vjust = 0.2, size = 1.9) +
  geom_text(data = temp_validation[subset != "all"], 
            aes(label = subset), hjust = 0.5, vjust = 0.3, size = 1.9) +
  coord_cartesian(ylim = c(0.145, 0.16)) +
  geom_hline(yintercept = 0.16, alpha = 0.5)


StocksAggrTrain <- StocksAggr[TrainRows, .SD, .SDcols = names(StocksAggr)[!(names(StocksAggr) %in% featureNames)]][, Split := "Train"]
StocksAggrTest <- StocksAggr[TestRows, .SD, .SDcols = names(StocksAggr)[!(names(StocksAggr) %in% featureNames)]][, Split := "Test"]
StocksAggrValidation <- StocksAggr[ValidationRows, .SD, .SDcols = names(StocksAggr)[!(names(StocksAggr) %in% featureNames)]][, Split := "Validation"]


# === Generate Quantile Predictions ===
QuantilePredictions <- list(
  base = rbind(
    cbind(StocksAggrTrain, setNames(as.data.table(as.array(baseModel(x_train))), paste0("Rank", 1:5))),
    cbind(StocksAggrTest, setNames(as.data.table(as.array(baseModel(x_test))), paste0("Rank", 1:5))),
    cbind(StocksAggrValidation, setNames(as.data.table(as.array(baseModel(x_validation))), paste0("Rank", 1:5)))
  ),
  meta = rbind(
    cbind(StocksAggrTrain, setNames(as.data.table(as.array(metaModel(x_train, xtype_train))), paste0("Rank", 1:5))),
    cbind(StocksAggrTest, setNames(as.data.table(as.array(metaModel(x_test, xtype_test))), paste0("Rank", 1:5))),
    cbind(StocksAggrValidation, setNames(as.data.table(as.array(metaModel(x_validation, xtype_validation))), paste0("Rank", 1:5)))
  )
)

saveRDS(QuantilePredictions, file.path(precomputed_dir, "forecast_ranks_all.rds"))
