#!/usr/bin/env Rscript
# Generate one spatial image using ggplot2 with optional Palo palette optimization.
# Usage:
# Rscript plot_spatial_with_palo.R <coords_csv> <assignments_csv> <id_col> <output_png> <title> <width> <height> <dpi> <use_palo>

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 9) {
  stop("Usage: Rscript plot_spatial_with_palo.R <coords_csv> <assignments_csv> <id_col> <output_png> <title> <width> <height> <dpi> <use_palo>")
}

coords_csv <- args[1]
assignments_csv <- args[2]
id_col <- args[3]
output_png <- args[4]
plot_title <- args[5]
plot_width <- suppressWarnings(as.numeric(args[6]))
plot_height <- suppressWarnings(as.numeric(args[7]))
plot_dpi <- suppressWarnings(as.numeric(args[8]))
use_palo <- tolower(trimws(args[9])) %in% c("true", "1", "yes", "y")

if (is.na(plot_width) || plot_width <= 0) {
  plot_width <- 10
}
if (is.na(plot_height) || plot_height <= 0) {
  plot_height <- 10
}
if (is.na(plot_dpi) || plot_dpi <= 0) {
  plot_dpi <- 300
}

if (!requireNamespace("ggplot2", quietly = TRUE)) {
  stop("R package 'ggplot2' is required for spatial plotting.")
}

coords_df <- read.csv(coords_csv, stringsAsFactors = FALSE)
assignments_df <- read.csv(assignments_csv, stringsAsFactors = FALSE)

if (!id_col %in% names(coords_df)) {
  stop(paste("ID column '", id_col, "' not found in coordinates CSV", sep = ""))
}
if (!id_col %in% names(assignments_df)) {
  stop(paste("ID column '", id_col, "' not found in assignments CSV", sep = ""))
}
if (!"x" %in% names(coords_df) || !"y" %in% names(coords_df)) {
  stop("Coordinates CSV must have 'x' and 'y' columns")
}
if (!"cluster" %in% names(assignments_df)) {
  stop("Assignments CSV must have 'cluster' column")
}

df <- merge(
  coords_df[, c(id_col, "x", "y")],
  assignments_df[, c(id_col, "cluster")],
  by = id_col,
  all = FALSE,
  sort = FALSE
)

df <- df[!is.na(df$x) & !is.na(df$y) & !is.na(df$cluster), ]
df <- df[is.finite(df$x) & is.finite(df$y), ]
if (nrow(df) == 0) {
  stop("No valid rows remain after merging and filtering.")
}

df$cluster <- as.character(df$cluster)
cluster_levels <- sort(unique(df$cluster))

gg_color_hue <- function(n) {
  if (n <= 0) {
    return(character())
  }
  hues <- seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}

initial_palette <- gg_color_hue(length(cluster_levels))
names(initial_palette) <- cluster_levels

cluster_colors <- initial_palette
if (use_palo) {
  if (!requireNamespace("Palo", quietly = TRUE)) {
    stop("R package 'Palo' is required when use_palo=TRUE.")
  }
  coords_matrix <- as.matrix(df[, c("x", "y")])
  cluster_vec <- as.character(df$cluster)
  palo_palette <- Palo::Palo(position = coords_matrix, cluster = cluster_vec, palette = initial_palette)
  cluster_colors <- palo_palette
}

p <- ggplot2::ggplot(df, ggplot2::aes(x = x, y = y, color = cluster)) +
  ggplot2::geom_point(size = 0.5, alpha = 0.7) +
  ggplot2::scale_color_manual(values = cluster_colors, breaks = cluster_levels, guide = "none") +
  ggplot2::coord_equal() +
  ggplot2::labs(
    x = "X coordinate",
    y = "Y coordinate",
    title = if (nchar(trimws(plot_title)) > 0) plot_title else NULL
  ) +
  ggplot2::theme_minimal(base_size = 10) +
  ggplot2::theme(
    legend.position = "none",
    panel.grid.major = ggplot2::element_line(color = "#EAEAEA", linewidth = 0.2),
    panel.grid.minor = ggplot2::element_blank(),
    plot.title = ggplot2::element_text(hjust = 0.5, face = "bold")
  )

ggplot2::ggsave(
  filename = output_png,
  plot = p,
  width = plot_width,
  height = plot_height,
  dpi = plot_dpi,
  bg = "white"
)

cat(sprintf("Spatial image saved to: %s\n", output_png))
