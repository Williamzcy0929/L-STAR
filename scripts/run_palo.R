#!/usr/bin/env Rscript
# R script to call Palo::Palo() for color palette optimization
# Usage: Rscript run_palo.R <coords_csv> <assignments_csv> <id_col> <output_csv> [rgb_weight] [color_blind_fun]

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 4) {
  stop("Usage: Rscript run_palo.R <coords_csv> <assignments_csv> <id_col> <output_csv> [rgb_weight] [color_blind_fun]")
}

coords_csv <- args[1]
assignments_csv <- args[2]
id_col <- args[3]
output_csv <- args[4]
rgb_weight <- if (length(args) >= 5) as.numeric(args[5]) else NULL
color_blind_fun <- if (length(args) >= 6) args[6] else NULL

# Check if Palo is installed
if (!require("Palo", quietly = TRUE)) {
  stop("Palo package is not installed. Please install it before running this script.")
}

# Read data
coords_df <- read.csv(coords_csv, stringsAsFactors = FALSE)
assignments_df <- read.csv(assignments_csv, stringsAsFactors = FALSE)

# Validate columns
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

# Merge data by ID
merged <- merge(coords_df[, c(id_col, "x", "y")], 
                assignments_df[, c(id_col, "cluster")],
                by = id_col, 
                all = FALSE, 
                sort = FALSE)

if (nrow(merged) == 0) {
  stop("No matching rows found after merging coordinates and assignments")
}

# Extract coordinates matrix (2 columns: x, y)
u <- as.matrix(merged[, c("x", "y")])

# Extract cluster labels as character vector
cl <- as.character(merged$cluster)

# Get unique clusters and their count
unique_clusters <- sort(unique(cl))
n_clusters <- length(unique_clusters)

# Generate initial palette
# Use gg_color_hue or RColorBrewer for initial colors
if (n_clusters <= 12) {
  library(RColorBrewer)
  init_pal <- brewer.pal(max(3, n_clusters), "Set3")
  if (n_clusters < 3) {
    init_pal <- init_pal[1:n_clusters]
  }
} else {
  # For more clusters, use colorRampPalette
  library(RColorBrewer)
  init_pal <- colorRampPalette(brewer.pal(12, "Set3"))(n_clusters)
}

# Ensure palette length matches number of clusters
if (length(init_pal) != n_clusters) {
  init_pal <- init_pal[1:n_clusters]
}

# Call Palo::Palo()
# Palo API: Palo(u, cl, pal, rgb_weight, color_blind_fun)
# u: 2-column coordinate matrix
# cl: cluster labels (character vector)
# pal: initial palette (character vector of hex colors)
# rgb_weight: optional numeric vector of length 3 (default: c(1, 1, 1))
# color_blind_fun: optional function name (e.g., "deutan", "protan", "tritan")

palo_args <- list(
  u = u,
  cl = cl,
  pal = init_pal
)

# Add optional parameters if provided
if (!is.null(rgb_weight)) {
  palo_args$rgb_weight <- rgb_weight
}
if (!is.null(color_blind_fun)) {
  palo_args$color_blind_fun <- color_blind_fun
}

# Call Palo
opt_pal <- do.call(Palo::Palo, palo_args)

# opt_pal is a named vector: names are cluster labels, values are optimized hex colors
# Convert to data frame with cluster -> r, g, b (0-1 floats)
rgb_matrix <- col2rgb(opt_pal) / 255.0

# Create output data frame
output_df <- data.frame(
  cluster = names(opt_pal),
  r = rgb_matrix[1, ],
  g = rgb_matrix[2, ],
  b = rgb_matrix[3, ],
  stringsAsFactors = FALSE
)

# Ensure clusters are sorted for consistency
# Try numeric sort first, fallback to character sort if needed
tryCatch({
  output_df <- output_df[order(as.numeric(output_df$cluster)), ]
}, error = function(e) {
  output_df <<- output_df[order(output_df$cluster), ]
})

# Write output CSV
write.csv(output_df, file = output_csv, row.names = FALSE, quote = FALSE)

# Print success message
cat(sprintf("Palo optimization complete. Generated %d colors for %d clusters.\n", 
            nrow(output_df), n_clusters))
