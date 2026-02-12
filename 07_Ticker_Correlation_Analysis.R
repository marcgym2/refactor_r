

# === Load Libraries ===
library(data.table)
library(ggplot2)

# === Load Libraries ===
library(reshape2)
library(corrplot)

# === Load Libraries ===
library(cluster)
library(factoextra)

# Load cleaned stock data
stocks <- readRDS("data/tickers_data_cleaned.rds")

# Compute daily log returns
daily_returns <- lapply(stocks, function(df) {
  df[, Return := c(NA, diff(log(Adjusted)))]
  df[, .(index, Return)]
})

# Merge into a single data.table
merged_returns <- Reduce(function(x, y) merge(x, y, by = "index", all = TRUE),
                         Map(function(x, name) setnames(copy(x), "Return", name), daily_returns, names(daily_returns)))

# Filter out all-NA rows
merged_returns <- merged_returns[!apply(is.na(merged_returns[,-1, with=FALSE]), 1, all)]

# Get return matrix
returns_matrix <- as.matrix(merged_returns[,-1, with=FALSE])

# Correlation matrix
cor_matrix <- cor(returns_matrix, use = "pairwise.complete.obs")

# Plot correlation matrix
png("forecasts/ticker_correlation_matrix.png", width = 1000, height = 800)
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.7, tl.col = "black", order = "hclust", addrect = 2)
dev.off()

# ============================
# PCA Analysis
# ============================
# Remove rows with any NA or Inf before PCA
returns_clean <- returns_matrix[complete.cases(returns_matrix) & apply(returns_matrix, 1, function(x) all(is.finite(x))), ]

# Run PCA
pca_res <- prcomp(returns_clean, scale. = TRUE, center = TRUE)

# Scree plot
png("forecasts/ticker_pca_screeplot.png", width = 1000, height = 600)
fviz_eig(pca_res)
dev.off()

# Variable contribution
png("forecasts/ticker_pca_varplot.png", width = 1000, height = 800)
fviz_pca_var(pca_res, col.var = "contrib", gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"))
dev.off()

# ============================
# Clustering based on PCA scores
# ============================
pca_scores <- pca_res$x[, 1:5]
dist_mat <- dist(pca_scores)
hc <- hclust(dist_mat, method = "ward.D2")

png("forecasts/ticker_hclust_dendrogram.png", width = 1000, height = 800)
plot(hc, cex = 0.8, hang = -1, main = "Hierarchical Clustering of Tickers (PCA)")
rect.hclust(hc, k = 4, border = 2:5)
dev.off()


# ============================
# Volatility Analysis
# ============================
# Compute volatility per ticker (ignoring NAs)
volatility <- apply(returns_matrix, 2, function(x) sd(x, na.rm = TRUE))

# Convert to data.table and remove NAs/infs
vol_dt <- data.table(Ticker = names(volatility), Volatility = as.numeric(volatility))
vol_dt <- vol_dt[is.finite(Volatility)]

# Sort by volatility
vol_dt <- vol_dt[order(-Volatility)]

# Plot
png("forecasts/ticker_volatility_barplot.png", width = 1000, height = 600)
ggplot(vol_dt, aes(x = reorder(Ticker, -Volatility), y = Volatility)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Ticker Volatility (Standard Deviation of Returns)", x = "Ticker", y = "Volatility")
dev.off()
