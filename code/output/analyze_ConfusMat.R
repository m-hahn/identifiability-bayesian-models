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
#    scale_fill_viridis(
#  trans = "log",
#  limits = c(1, 800),
#  name = "NLL (log scale)"
#) +
scale_fill_gradientn(
  colours = c("white", "skyblue", "blue", "purple", "red"),
  trans = "log",
  limits = c(1, 800),
  name = "NLL"
) +
#    scale_fill_gradientn(
#      colours = c("white", "yellow", "red"),
#      trans = "log",
#      limits = c(-1, 800),
#      name = "NLL (log scale)"
#    ) +
    labs(x = "true", y = "fitted") +
#         title = paste0("Heatmap NLL by PTrue and PFit\n(dataSize = ", ds, ", noiseCount = ", nc, ")")) +
    theme_minimal() +
    theme(
      legend.position = "none",   # ← turn off legend here
 # ↑ axis titles (x/y) in 16pt bold
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
#
#library(dplyr)
#library(ggplot2)
#library(viridis)
#library(cowplot)   # ← get_legend() lives here
#library(grid)
#
# (5) Now build a “legend‐only” object and save it to a separate PDF.
#     We’ll construct a dummy ggplot whose sole purpose is to display the scale_fill_gradientn legend.

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

#pdf("images/legend_only.pdf", width = 0.8*3, height = 0.8*5)
#grid.newpage()
#grid.draw(legend_grob)
#dev.off()
##
#
#
###########
#crash()
#
#
##plot = ggplot(data %>% filter(PFit != PTrue) %>% group_by(dataSize, noiseCount) %>% summarise(NLL=median(NLL)), aes(x=noiseCount, y=NLL)) + geom_bar(stat="identity") + facet_grid(~dataSize)
##ggsave(filename = "images/confusion-summary.pdf", plot=plot, width=6, height=2)
##
##plot = ggplot(data %>% filter(PFit != PTrue) %>% group_by(dataSize, noiseCount) %>% summarise(NLL=median(NLL)), aes(x=noiseCount, y=NLL)) + geom_bar(stat="identity") + facet_grid(~dataSize)
##ggsave(filename = "images/confusion-summary.pdf", plot=plot, width=6, height=2)
#
#
#
#
#library(dplyr)
#library(ggplot2)
#
## 1. Build the summary data (median NLL per dataSize × noiseCount)
#summary_df <- data %>%
#  filter(PFit != PTrue) %>%
#  group_by(dataSize, noiseCount) %>%
#  summarise(NLL = mean(NLL), .groups = "drop")
#
## 2. Plot: first the bars (median NLL), then overlay the individual points
#plot <- ggplot() +
#  # (a) bars using the pre-computed summary
#  geom_bar(
#    data = summary_df,
#    aes(x = noiseCount, y = NLL),
#    stat = "identity",
#    fill = "skyblue",
#    width = 0.7
#  ) +
#  # (b) jittered points from the raw data
#  geom_jitter(
#    data = data %>% filter(PFit != PTrue),
#    aes(x = noiseCount, y = NLL),
#    width = 0.15,        # adjust horizontal “spread” so points don’t sit exactly under the bar
#    alpha = 0.5,         # semi‐transparent so you can see overlap
#    size = 1
#  ) +
#  facet_grid(~ dataSize) +
#  labs(
#    x = "Noise Count",
#    y = "Negative Log‐Likelihood (NLL)",
#    title = "Median NLL per Noise Level (bars) with Individual Observations (dots)"
#  ) +
#  theme_minimal()
#
## 3. Save to PDF as before
#ggsave(
#  filename = "images/confusion-summary-with-dots.pdf",
#  plot = plot,
#  width = 6,
#  height = 2
#)
#
#
#
#
#
#library(dplyr)
#library(ggplot2)
#
## 1. Compute median NLL per dataSize × noiseCount
#summary_df <- data %>%
#  filter(PFit != PTrue) %>%
#  group_by(dataSize, noiseCount) %>%
#  summarise(NLL = median(NLL), .groups = "drop")
#
## 2. Build the plot, adding scale_y_log10()
#plot <- ggplot() +
#  # (a) Bars for the median NLL
#  geom_bar(
#    data = summary_df,
#    aes(x = noiseCount, y = NLL),
#    stat = "identity",
#    fill = "skyblue",
#    width = 0.7
#  ) +
#  # (b) Jittered points for all individual NLL values
#  geom_jitter(
#    data = data %>% filter(PFit != PTrue),
#    aes(x = noiseCount, y = NLL),
#    width = 0.15,
#    alpha = 0.05,
#    size = 1
#  ) +
#  facet_grid(~ dataSize) +
#  # ← HERE: switch the y‐axis to log10 scale
#  scale_y_log10() +
#  labs(
#    x = "Noise Count",
#    y = "Negative Log‐Likelihood (NLL)\n(log scale)",
#    title = "Median NLL (bars) with Individual Observations (dots) on Log Scale"
#  ) +
#  theme_minimal()
#
## 3. Save to PDF
#ggsave(
#  filename = "images/confusion-summary-with-dots-logscale.pdf",
#  plot = plot,
#  width = 6,
#  height = 2
#)
#
#
#

library(dplyr)
library(ggplot2)

## 1. Compute median and SD on log10(NLL) per dataSize × noiseCount
#summary_df <- data %>%
#  filter(PFit != PTrue) %>%
#  mutate(logNLL = log10(pmax(NLL, 0.1))) %>% 
#  group_by(dataSize, noiseCount) %>%
#  summarise(
#    median_log = median(logNLL),
#    sd_log     = sd(logNLL),
#    .groups    = "drop"
#  ) %>%
#  # 2. Back‐transform to original scale for plotting
#  mutate(
#    NLL_med = 10^median_log,
#    ymin    = 10^(median_log - sd_log),
#    ymax    = 10^(median_log + sd_log)
#  )
#
## 3. Build the plot with bars, error bars, and jittered points:
#plot <- ggplot() +
#  # (a) Bars at 10^(median_log)
#  geom_bar(
#    data = summary_df,
#    aes(x = noiseCount, y = NLL_med),
#    stat = "identity",
#    fill = "skyblue",
#    width = 0.7
#  ) +
#  # (b) Error bars from ymin to ymax (log-scale ±1 SD)
#  geom_errorbar(
#    data = summary_df,
#    aes(
#      x    = noiseCount,
#      ymin = pmax(ymin, 1e-6),  # ensure strictly positive for log scale
#      ymax = ymax
#    ),
#    width = 0.2,
#    color = "darkblue"
#  ) +
#  # (c) Jittered points (raw NLL)
#  geom_jitter(
#    data = data %>% filter(PFit != PTrue),
#    aes(x = noiseCount, y = NLL),
#    width = 0.15,
#    alpha = 0.05,
#    size = 1
#  ) +
#  facet_grid(~ dataSize) +
#  scale_y_log10() +
#  labs(
#    x     = "Noise Count",
#    y     = "Negative Log‐Likelihood (NLL)\n(log scale)",
#    title = "Median NLL (bars) ±1 SD on Log Scale with Individual Observations (dots)"
#  ) +
#  theme_minimal()
#
## 4. Save
#ggsave(
#  filename = "images/confusion‐summary‐with‐dots‐log‐sd.pdf",
#  plot    = plot,
#  width   = 6,
#  height  = 2
#)

