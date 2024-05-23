library(arrow)
library(dplyr)
library(arrow)
library(ggplot2)
library(tidyr)
library(brms)
library(tidybayes)

options(
  mc.cores = 8,
  brms.file_refit = "on_change",
  brms.backend = "cmdstanr"
)

hdr <- read_csv_arrow(file.path("out", "hdr.csv"),
  as_data_frame = TRUE
) |>
  relocate(file) |>
  pivot_longer(-c(time, epoch, file, condition),
    names_to = "channel",
    values_to = "amp"
  ) |>
  mutate(
    chroma = if_else(grepl("hbo", channel), "HbO", "HbR"),
    channel = gsub(" hbo| hbr", "", channel),
    file = basename(file)
  ) |>
  arrange(file, channel, chroma, time) |>
  summarise(
    amp = mean(amp),
    .by = c(channel, condition, chroma, time)
  )

fit <- brm(
  amp ~ condition + s(time, by = interaction(chroma, condition)) +
    (1 + chroma * condition | channel),
  data = hdr, iter = 1000, chains = 4, cores = 4,
  seed = 4,
  file = file.path("out", "fit.rds"),
  file_refit = "on_change"
)
