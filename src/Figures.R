library(arrow)
library(dplyr)
library(ggplot2)
library(tidyr)
library(purrr)


roi_lst <- list(
  occipital = c(
    "S8_D6", "S8_D8", "S7_D6", "S8_D7", "S10_D8",
    "S7_D17", "S7_D7", "S10_D7", "S10_D18", "S9_D17"
  ),
  right_stg = c(
    "S12_D12", "S12_D20", "S11_D12",
    "S11_D19", "S11_D10", "S11_D9"
  ),
  left_stg = c(
    "S4_D2", "S5_D15", "S5_D2", "S6_D2", "S5_D3", "S6_D16",
    "S6_D3", "S6_D5"
  ),
  left_ifg = c(
    "S1_D13", "S1_D1", "S3_D14", "S3_D1", "S2_D1", "S4_D1"
  )
)

roi_df <- tibble(
  roi = names(roi_lst),
  channel = roi_lst
) |>
  unnest(channel)

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
    file = basename(file),
    condition = factor(condition, levels = c("Silence", "Noise", "Speech"))
  ) |>
  inner_join(roi_df) |>
  mutate(
    roi = factor(roi,
      levels = c("left_ifg", "left_stg", "right_stg", "occipital"),
      labels = c(
        "Left inferior frontal gyrus",
        "Left superior temporal gyrus",
        "Right superior temporal gyrus",
        "Occipital lobe"
      )
    )
  ) |>
  arrange(file, roi, channel, chroma, time) |>
  summarise(
    amp = mean(amp),
    .by = c(channel, roi, condition, chroma, time)
  )

hdr |>
  mutate(amp = amp * 100) |>
  summarise(
    amp_sd = sd(amp),
    amp = mean(amp),
    n = n(),
    .by = c(roi, condition, chroma, time)
  ) |>
  mutate(amp_se = amp_sd / sqrt(n)) |>
  # filter(file %in% unique(hdr$file[c(1)])) |>
  ggplot(aes(time, amp,
    ymin = amp - amp_se,
    ymax = amp + amp_se,
    colour = chroma,
    fill = chroma,
    linetype = chroma
  )) +
  facet_grid(condition ~ roi, scale = "free_y") +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey") +
  geom_ribbon(colour = NA, alpha = 1 / 3) +
  geom_line(aes(group = interaction(chroma, condition))) +
  scale_fill_brewer(palette = "Set2") +
  scale_colour_brewer(palette = "Set2") +
  theme_minimal() +
  theme(
    panel.grid = element_blank(),
    panel.background = element_rect(fill = "white", colour = NA),
    plot.background = element_rect(fill = "white", colour = NA)
  )

ggsave(file.path("img", "hdr.png"), dpi = 1000, width = 10, height = 7)
