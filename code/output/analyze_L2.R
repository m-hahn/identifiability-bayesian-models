
library(ggplot2)
library(dplyr) 

# Read the data 
data <- read.csv("evaluateCrossValidationResults_Synthetic_Gardelle_VisualizeByNoiseCount_AndSize_ByP_ConfusMat.py.txt", sep="\t")

data = data %>% filter(dataSize==40000, PTrue!=PFit)

# install.packages("tidyverse")  # if not already installed
library(dplyr)
library(ggplot2)



library(dplyr)
library(ggplot2)

library(dplyr)
library(ggplot2)

# 1. Add NLL_log = log(pmax(NLL, 0.01)) and split sensoryLevels into digits
data2 <- data %>% filter(noiseCount==2) %>%
  mutate(
    NLL_log      = log(pmax(NLL, 0.01)),
    first_digit  = (5-((floor(sensoryLevels / 10))-1)),
    second_digit = (5-((sensoryLevels %% 10)-1))
  )

data3 = data %>% filter(noiseCount==1) %>%
  mutate(
    NLL_log      = log(pmax(NLL, 0.01)),
    first_digit  = (5-((sensoryLevels-1))),
    second_digit = (5-((sensoryLevels)-1))
  )

data2 = rbind(data2, data3)
data2$diff = data2$first_digit - data2$second_digit

# 2. Create the faceted jitter plot with custom y‐axis ticks
p <- ggplot(data2, aes(x = 0, y = NLL_log)) +
  geom_jitter(width = 0.3, height = 0, alpha = 0.6, size = 1.5) +
  facet_grid(second_digit ~ first_digit) +
  scale_x_continuous(NULL, breaks = NULL) +
  scale_y_continuous(
    breaks = c(log(0.01), 0, log(100)),
    labels = c("< 0.01", "1", "100")
  ) +
  labs(
    x     = NULL,
    y     = "NLL (log scale)"
#    title = "Jittered dot‐plots of log(pmax(NLL, 0.01)) by sensoryLevels digits"
  ) +
  theme_minimal() +
  theme(
    strip.text         = element_text(size = 10),
    axis.text.x        = element_blank(),
    axis.ticks.x       = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank()
  ) +
  theme(
    # draw a thin black border around each facet
    panel.border      = element_rect(color = "black", fill = NA, size = 0.3),
    # add a bit of space between facet panels
    panel.spacing     = unit(0.5, "lines"),
    # remove any background fill inside each panel (so the border stands out)
    panel.background  = element_blank(),
    # hide x-axis text/ticks since we’re only using x for jitter
    axis.text.x       = element_blank(),
    axis.ticks.x      = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    strip.background  = element_rect(fill = "grey90", color = "black", size = 0.3),
    strip.text        = element_text(size = 10)
  )

# 3. Save to a taller-than‐wide PDF
ggsave("images/sensory_LVL_dotplots.pdf", plot = p, width = 6, height = 10)


# Linear Regression
summary(lm(NLL_log ~ first_digit + diff, data=data2))

