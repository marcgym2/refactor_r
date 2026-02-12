
# === Load Libraries ===
library(data.table)

# === Load Helper Scripts ===
source("scripts/config.R")

# Completa los datos de acciones hasta TimeEnd, excluyendo fines de semana.
AugmentStock <- function(Stock, TimeEnd) {
  if (max(Stock$index) + 1 < TimeEnd) {
    temp <- seq(max(Stock$index) + 1, TimeEnd, by = 1)
    temp <- temp[!(weekdays(temp, abbreviate = T) %in% c("so", "ne"))]
    StockAug <- data.table(index = temp)
    rbind(Stock, StockAug, fill = TRUE)
  } else {
    Stock
  }
}

# Estandariza un vector numérico (media cero y desviación estándar uno).
standartize <- function(x) {
  (x - mean(x, na.rm = TRUE)) / (sd(x, na.rm = TRUE) + 1e-5)
}

# Calcula quintiles para un vector numérico.
computeQuintile <- function(x) { 
  nas <- is.na(x)
  out <- findInterval(rank(x[!nas]) / length(x[!nas]), c(0, 0.2, 0.4, 0.6, 0.8, 1), left.open = TRUE)
  ifelse(nas, NA, out[cumsum(!nas)])
}

# Calcula el “Ranked Probability Score” (RPS) para tensores y devuelve el promedio.
ComputeRPSTensor <- function(y_pred, y) {
  temp <- (y_pred$cumsum(2) - y)^2
  mean(temp$sum(2) / 5)
}

# Similar a ComputeRPSTensor pero devuelve un vector de puntajes.
ComputeRPSTensorVector <- function(y_pred, y) {
  temp <- (y_pred$cumsum(2) - y)^2
  temp$sum(2) / 5
}

# imputeNA(x)
imputeNA <- function(x) {
  ifelse(is.na(x) | is.infinite(x), median(x, na.rm = TRUE), x)
}

# Imputa valores NA en características específicas de una tabla de datos.

imputeFeatures <- function(StocksAggr, featureNames = NULL) {
  for (featureName in featureNames) {
    StocksAggr[[featureName]] <- imputeNA(StocksAggr[[featureName]])
  }
  StocksAggr
}

#	Estandariza características específicas de una tabla de datos.
standartizeFeatures <- function(StocksAggr, featureNames = NULL) {
  otherNames <- names(StocksAggr)[!(names(StocksAggr) %in% c("Interval", featureNames))]
  StocksAggr[, c(setNames(lapply(otherNames, function(x) get(x)), otherNames), lapply(.SD, function(x) standartize(x))), .(Interval), .SDcols = featureNames]
}

# Subconjunta un tensor basado en filas especificadas, considerando si es esparso o no.
subsetTensor <- function(x, rows, isSparse = NULL) {
  if (is.null(isSparse)) {
    isSparse = try({x$is_coalesced()}, silent = TRUE)
  }
  if (isSparse == TRUE) {
    rowIndices <- as.array(x$indices()[1,]) + 1L
    selection <- rowIndices[rowIndices %in% rows]
    i <- x$indices()[, selection] + 1L
    i[1, ] <- seq_along(selection)
    v <- x$values()[selection]
    out <- torch_sparse_coo_tensor(i, v, c(length(selection), ncol(x)))$coalesce()
  } else {
    out <- x[rows, ]
  }
  return(out)
}

# Genera agregaciones de acciones considerando información de intervalos y posibles fugas de datos.
GenStocksAggr <- function(Stocks, IntervalInfos, featureList, CheckLeakage = TRUE) {
  StocksAggr <- do.call(rbind, lapply(seq_along(Stocks), function(s) {
    if (s %% 1 == 0) {
      print(str_c("Stock:", s, " Time:", Sys.time()))
    }
    
    StockAggr <- lapply(IntervalInfos, function(IntervalInfo) {
      Stock <- Stocks[[s]]
      Ticker <- names(Stocks)[s]
      Stock <- AugmentStock(Stock[index >= IntervalInfo$Start & index <= IntervalInfo$End], IntervalInfo$End)
      Stock[, Interval := findInterval(index, IntervalInfo$TimeBreaks, left.open = TRUE)]
      Stock[, Interval := factor(Interval, levels = seq_along(IntervalInfo$IntervalNames), labels = IntervalInfo$IntervalNames)]
      Stock[, Ticker := Ticker]
      StockAggr <- Stock[, computeFeatures(.SD, .BY, featureList), .(Ticker)]
      
      if (CheckLeakage) {
        featureNames <- names(StockAggr)[!(names(StockAggr) %in% c("Ticker", "Interval", "Return"))]
        StockCensored <- copy(Stock)
        StockCensored <- StockCensored[index >= IntervalInfo$CheckLeakageStart, (c("Open", "High", "Low", "Close", "Volume", "Adjusted")) := lapply(1:6, function(x) NA)][]
        StockCensoredAggr <- StockCensored[, computeFeatures(.SD, .BY, featureList), .(Ticker)]
        for (featureName in featureNames) {
          same <- identical(StockAggr[, get(featureName)], StockCensoredAggr[, get(featureName)])
          if (!same) {
            print(str_c("Possible leakage in ", featureName, ", stock = ", Ticker, " (s = ", s, ")"))
          }
        }
      }
      StockAggr[, Shift := IntervalInfo$Shift]
      return(StockAggr)
    })
    return(do.call(rbind, StockAggr))
  }))
  StocksAggr[, ReturnQuintile := computeQuintile(Return), .(Interval)]
  StocksAggr[, IntervalStart := as.Date(str_sub(Interval, 1, 10))]
  StocksAggr[, IntervalEnd := as.Date(str_sub(Interval, 14, 23))]
  return(StocksAggr)
}

GenIntervalInfos <- function(Submission, Shifts = 0, TimeEnd = Sys.Date(), 
                             interval_days = 28, total_intervals = 1000) {
  lapply(Shifts, function(Shift) {
    # Calcular la fecha de inicio según total_intervals
    TimeStart <- (TimeEnd - Shift) - interval_days * total_intervals
    # Generar secuencia de puntos de ruptura
    TimeBreaks <- seq(TimeStart, TimeEnd, by = interval_days)
    # Si el último break es anterior a TimeEnd, se agrega TimeEnd para incluir datos parciales
    if (tail(TimeBreaks, 1) < TimeEnd) {
      TimeBreaks <- c(TimeBreaks, TimeEnd)
    }
    # Definir inicios y fines de cada intervalo
    IntervalStarts <- TimeBreaks[-length(TimeBreaks)] + 1
    IntervalEnds <- TimeBreaks[-1]
    IntervalNames <- str_c(IntervalStarts, " : ", IntervalEnds)
    list(
      Shift = Shift,
      TimeBreaks = TimeBreaks,
      IntervalStarts = IntervalStarts,
      IntervalEnds = IntervalEnds,
      IntervalNames = IntervalNames,
      Start = IntervalStarts[1],
      End = IntervalEnds[(length(IntervalEnds) - (12 - Submission))],
      CheckLeakageStart = IntervalStarts[(length(IntervalStarts) - (12 - Submission))]
    )
  })
}


# Entrena un modelo con parámetros específicos, gestión de mini-lotes, y optimización.
trainModel2 <- function(model, criterion, train, test = NULL, validation = NULL, epochs = 10, minibatch = Inf, tempFilePath = NULL, patience = 1, printEvery = Inf, lr = 0.001, weight_decay = 0, isSparse = NULL, optimizerType = "adam", ...) {
  modelOut <- model
  
  out <- lapply(seq_along(lr), function(rs) {
    model <- modelOut
    
    optimizer = switch(optimizerType,
                       "adam" = optim_adam(model$parameters, lr = lr[rs], weight_decay = weight_decay),
                       "sgd" = optim_sgd(model$parameters, lr = lr[rs], weight_decay = weight_decay),
                       "adadelta" = optim_adadelta(model$parameters, lr = lr[rs], weight_decay = weight_decay),
                       "asgd" = optim_asgd(model$parameters, lr = lr[rs], weight_decay = weight_decay),
                       "lbfgs" = optim_lbfgs(model$parameters, lr = lr[rs]),
                       "rmsprop" = optim_rmsprop(model$parameters, lr = lr[rs], weight_decay = weight_decay),
                       "rprop" = optim_rmsprop(model$parameters, lr = lr[rs])
    )
    
    progress <- data.table(
      epoch = seq_len(epochs),
      loss_train = rep(Inf, epochs),
      loss_test = rep(Inf, epochs),
      loss_validation = rep(Inf, epochs)
    )
    if (is.null(isSparse)) {
      isSparse <- c(rep(FALSE, 2), rep(TRUE, length(train) - 2))
    }
    if (length(minibatch) == 1) {
      minibatch <- lapply(1:epochs, function(e) {
        minibatch
      })
    }
    
    for (e in 1:epochs) {
      if (is.numeric(minibatch[[e]])) {
        temp <- sample(seq_len(nrow(train[[2]])))
                       mbs <- split(temp, ceiling(seq_along(temp) / minibatch[[e]]))
      } else {
        mbs <- minibatch[[e]]()
      }
      
      if (e > 1) {
        model$train()
        for (mb in seq_along(mbs)) {
          rows <- mbs[[mb]]
          train_mb <- lapply(seq_along(train), function(i) subsetTensor(train[[i]], rows = rows, isSparse = isSparse[i]))
          
          optimizer$zero_grad()
          y_pred_mb = do.call(model, c(train_mb[-1], list(...)))
          loss = criterion(y_pred_mb, train_mb[[1]])
          loss$backward()
          optimizer$step()
        }
        model$eval()
      }
      
      progress[e, loss_train := as.array(criterion(do.call(model, c(train[-1], list(...))), train[[1]]))]
      if (!is.null(test)) {
        progress[e, loss_test := as.array(criterion(do.call(model, c(test[-1], list(...))), test[[1]]))]
      }
      if (!is.null(validation)) {
        progress[e, loss_validation := as.array(criterion(do.call(model, c(validation[-1], list(...))), validation[[1]]))]
      }
      if (e %% printEvery == 0) {
        print(str_c("restart: ", rs, " epoch:", e, " train: ", round(progress[e, loss_train], 5), " test:", round(progress[e, loss_test], 5), " validation:", round(progress[e, loss_validation], 5), " Time:", Sys.time()))
      }
      
      if (!is.null(test)) {
        ebest <- progress[, which.min(loss_test)]
        if ((e == ebest) & !is.null(tempFilePath)) {
          torch_save(model, file.path(tempFilePath, str_c("temp", ".t7")))
        }
        if (e - ebest >= patience) {
          progress <- progress[1:e, ]
          break()
        }
      }
    }
    
    if (!is.null(tempFilePath) & !is.null(test)) {
      model <- torch_load(file.path(tempFilePath, str_c("temp", ".t7")))
      file.remove(file.path(tempFilePath, str_c("temp", ".t7")))
    }
    
    modelOut <<- model
    return(progress)
  })
        
        model <- modelOut
        progress <- do.call(rbind, out)
        
        return(list(
          model = model,
          progress = progress
        ))
}

# Genera muestras de mini-lotes para entrenamiento.
minibatchSampler <- function(batchSize, xtype_train) {
  rows <- as.array(xtype_train$indices()[1,]) + 1
  columns <- as.array(xtype_train$indices()[2,]) + 1
  uniqueColumns <- unique(columns)
  bs <- sample(seq_along(uniqueColumns), replace = FALSE)
  bs <- split(bs, ceiling(seq_along(bs) / batchSize))
  bs <- lapply(bs, function(x) uniqueColumns[x])
  bs <- lapply(bs, function(x) which(columns %in% x))
  return(bs)
}

constructFFNN = nn_module(
  initialize = function(inputSize, layerSizes, layerTransforms, layerDropouts = NULL) {
    self$layerSizes <- layerSizes
    self$layerTransforms <- layerTransforms
    self$layerSizesAll <- c(inputSize, layerSizes)
    self$Dropout <- !is.null(layerDropouts)
    for(i in seq_along(self$layerSizes)){
      self[[str_c("layer_",i)]] <- nn_linear(self$layerSizesAll[i], self$layerSizesAll[i+1])
    }
    if(self$Dropout){
      for(i in seq_along(self$layerSizes)){
        self[[str_c("layerDropout_",i)]] <- nn_dropout(p=layerDropouts[i])
      }
    }
  },
  forward = function(x) {
    for(i in seq_along(self$layerSizes)){
      x <- self$layerTransforms[[i]](self[[str_c("layer_",i)]](x))
      if(self$Dropout){
        x <- self[[str_c("layerDropout_",i)]](x)
      }
    }
    x
  },
  fforward = function(x,state){
    for(i in seq_along(self$layerSizes)){
      x <- self$layerTransforms[[i]](nnf_linear(x, weight = state[[str_c("layer_",i,".weight")]], bias = state[[str_c("layer_",i,".bias")]]))
    }
    x
  }
)

