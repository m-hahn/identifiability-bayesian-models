library(ggplot2)
library(dplyr)

# Read the data
data <- read.csv("evaluateCrossValidationResults_Synthetic_Gardelle_VisualizeByNoiseCount_AndSize_ByP_ConfusMat.py.txt", sep="\t")

# Make sure images directory exists
if (!dir.exists("images")) {
  dir.create("images")
}

# Get unique combinations
unique_combinations <- unique(data %>% select(dataSize, noiseCount))

library(viridis)

# Loop over combinations
for (i in 1:nrow(unique_combinations)) {
  ds <- unique_combinations$dataSize[i]
  nc <- unique_combinations$noiseCount[i]
  
  filtered_data <- data %>%
    filter(dataSize == ds, noiseCount == nc) %>% group_by(PTrue, PFit) %>% summarise(NLL=median(NLL))
  
  p <- ggplot(filtered_data, aes(x = factor(PTrue), y = factor(PFit), fill = pmax(1,NLL))) +
    geom_tile() +
scale_fill_gradientn(
  colours = c("white", "skyblue", "blue", "purple", "red"),
  trans = "log",
  limits = c(1, 800),
  name = "NLL"
) +
    labs(x = "true", y = "fitted") +
    theme_minimal() +
    theme(
      legend.position = "none",   # ← turn off legend here
      axis.title.x    = element_text(size = 32),# face = "bold"),
      axis.title.y    = element_text(size = 32),#, face = "bold"),

      # ↑ axis tick labels (x/y) in 14pt
      axis.text.x     = element_text(size = 32),
      axis.text.y     = element_text(size = 32),

      # ↑ if you ever add a title, make it larger, too
      plot.title      = element_text(size = 18, face = "bold", hjust = 0.5)
    )
  
  ggsave(filename = paste0("images/heatmap_dataSize_", ds, "_noiseCount_", nc, ".pdf"),
         plot = p, width = 4.2, height = 4.2)
}


# Print Sample Size
for (i in 1:nrow(unique_combinations)) {
  ds <- unique_combinations$dataSize[i]
  nc <- unique_combinations$noiseCount[i]
  
  filtered_data <- data %>%
    filter(dataSize == ds, noiseCount == nc) %>% group_by(PTrue, PFit) %>% summarise(NLL=median(NLL))
  print(paste(ds, nc, "----", cat(((data %>%    filter(dataSize == ds, noiseCount == nc) %>% group_by(PTrue, PFit) %>% summarise(u = n())))$u), "\n"))

}

# Now build a “legend‐only” object and save it to a separate PDF.
# We’ll construct a dummy ggplot whose sole purpose is to display the scale_fill_gradientn legend.

library(ggplot2)
library(cowplot)
library(grid)

# Dummy data (only used to generate the color scale)
dummy_df <- data.frame(x = 1, y = 1, a = 1)

# Build a “legend‐only” plot, but force the breaks at (3,13,103) 
# and label them as (0,10,100).
legend_plot <- ggplot(dummy_df, aes(x, y, fill = a)) +
  geom_tile() +
  scale_fill_gradientn(
    colours = c("white", "skyblue", "blue", "purple", "red"),
    trans   = "log",
    limits  = c(1, 800),
    breaks  = c(1,  11, 101),          # these are (0+3, 10+3, 100+3)
    labels  = c("< 1", "10", "100"),     # what you want displayed
    name    = "NLL"
  ) +
  theme_minimal() +
  theme(
    axis.title       = element_blank(),
    axis.text        = element_blank(),
    axis.ticks       = element_blank(),
    panel.grid       = element_blank(),
    panel.background = element_blank(),
    legend.position  = "right", legend.margin = margin(0,0,0,0, "pt")
  )

# Extract and save just that legend as a PDF
legend_grob <- get_legend(legend_plot)
library(cowplot)
# save it, auto-cropped
save_plot(
  filename    = "images/legend_only.pdf",
  plot        = legend_grob,
  base_width  = 0.8,    # these are in inches, just enough to accommodate the key
  base_height = 2     # tweak if necessary to get exactly the right aspect‐ratio
)



library(dplyr)
library(ggplot2)


