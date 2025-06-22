library(ggplot2)
library(dplyr)
library(scales)  # for comma()
# Read the data
data <- read.csv("evaluateCrossValidationResults_Synthetic_Gardelle_VisualizeByNoiseCount_AndSize_ByP_ConfusMat.py.txt", sep="\t")

# Make sure images directory exists
if (!dir.exists("images")) {
  dir.create("images")
}

# Get unique combinations
unique_combinations <- unique(data %>% select(dataSize, noiseCount))

library(viridis)
library(dplyr)
library(ggplot2)
library(viridis)
library(cowplot)   # ← get_legend() lives here
library(grid)
#########
library(dplyr)
library(ggplot2)


data = data %>% mutate(NLL = pmax(NLL, 1))

# 1. Compute median and SD on log10(NLL) per dataSize × noiseCount
summary_df <- data %>%
  filter(PFit != PTrue) %>%
  filter(dataSize != 20000) %>% 
  mutate(logNLL = log10(NLL)) %>% 
  group_by(dataSize, noiseCount) %>%
  summarise(
    median_log = median(logNLL),
    sd_log     = sd(logNLL),
    .groups    = "drop"
  ) %>%
  # 2. Back‐transform to original scale for plotting
  mutate(
    NLL_med = 10^median_log,
    ymin    = 10^(median_log - sd_log),
    ymax    = 10^(median_log + sd_log)
  )

# 3. Build the plot with bars, error bars, and jittered points:
plot <- ggplot() +
  # (a) Bars at 10^(median_log)
  geom_bar(
    data = summary_df,
    aes(x = noiseCount, y = NLL_med),
    stat = "identity",
    fill = "skyblue",
    width = 0.7
  ) +
  # (b) Error bars from ymin to ymax (log-scale ±1 SD)
  geom_errorbar(
    data = summary_df,
    aes(
      x    = noiseCount,
      ymin = ymin, #pmax(ymin, 1e-6),  # ensure strictly positive for log scale
      ymax = ymax
    ),
    width = 0.2,
    color = "darkblue"
  ) +
  # (c) Jittered points (raw NLL)
  geom_jitter(
    data = data %>% filter(PFit != PTrue, dataSize != 20000),
    aes(x = noiseCount, y = NLL, alpha = 0.08 / choose(4, noiseCount)),
    width = 0.15,
    #,
    size = 1
  ) + scale_alpha_identity() +
#  facet_grid(rows=NULL, cols= vars(dataSize), labeller=labeller(dataSize = function(x) comma(as.numeric(x)))) +
  facet_grid(rows=NULL, cols= vars(dataSize), labeller=labeller(dataSize = function(x) as.numeric(x))) +
  scale_y_log10(
    breaks = c(1, 10, 100, 1000),
    labels = c("<1", "10", "100", "")) +
  labs(
    x     = "Number of Noise Levels",
    y     = "NLL",
#    title = "Median NLL (bars) ±1 SD on Log Scale with Individual Observations (dots)"
  ) +
  theme_minimal(base_size = 14) +   # ↑ bump up the base font size from default (~11)
  theme(
    # ───── Axis titles ──────────────────────────────────────────────────
    axis.title.x   = element_text(size = 16), #, face = "bold"),
    axis.title.y   = element_text(size = 16), #, face = "bold"),

    # ───── Axis tick labels ─────────────────────────────────────────────
    axis.text.x    = element_text(size = 16),
    axis.text.y    = element_text(size = 16),

    # ───── Facet strip labels (dataSize headers) ───────────────────────
    strip.text     = element_text(size = 16), #, face = "bold"),

    # ───── Legend (if added later) ─────────────────────────────────────
    legend.title   = element_text(size = 14), #, face = "bold"),
    legend.text    = element_text(size = 12),

    # ───── Plot title (if you uncomment `title = ...` above) ───────────
#    plot.title     = element_text(size = 18, face = "bold", hjust = 0.5)
  )


# 4. Save
ggsave(
  filename = "images/confusion‐summary‐with‐dots‐log‐sd.pdf",
  plot    = plot,
  width   = 1.1*6,
  height  = 1.1*2
)

