# === Load Libraries ===
library(shiny)
library(ggplot2)

# === Load Libraries ===
library(data.table)
library(png)

# === Load Libraries ===
library(grid)
library(gridExtra)

ui <- fluidPage(
  titlePanel("Ticker Analytics Dashboard"),
  sidebarLayout(
    sidebarPanel(
      helpText("Explore correlation, PCA, clustering, and volatility.")
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Correlation", imageOutput("corrPlot")),
        tabPanel("PCA Scree", imageOutput("pcaScree")),
        tabPanel("PCA Variables", imageOutput("pcaVars")),
        tabPanel("Clustering", imageOutput("clusterPlot")),
        tabPanel("Volatility", imageOutput("volPlot"))
      )
    )
  )
)

server <- function(input, output) {
  renderLocalImage <- function(filename) {
    list(
      src = normalizePath(file.path("forecasts", filename)),
      contentType = 'image/png',
      width = "100%",
      alt = filename
    )
  }
  
  output$corrPlot <- renderImage(renderLocalImage("ticker_correlation_matrix.png"), deleteFile = FALSE)
  output$pcaScree <- renderImage(renderLocalImage("ticker_pca_screeplot.png"), deleteFile = FALSE)
  output$pcaVars <- renderImage(renderLocalImage("ticker_pca_varplot.png"), deleteFile = FALSE)
  output$clusterPlot <- renderImage(renderLocalImage("ticker_hclust_dendrogram.png"), deleteFile = FALSE)
  output$volPlot <- renderImage(renderLocalImage("ticker_volatility_barplot.png"), deleteFile = FALSE)
}

shinyApp(ui = ui, server = server)
