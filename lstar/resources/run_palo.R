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

# Helper function to parse rgb_weight from comma/semicolon/space-separated string
parse_rgb_weight <- function(x, default = c(1, 1, 1)) {
  # Handle missing, NULL, empty string, or "NULL" (case-insensitive)
  if (missing(x) || is.null(x) || length(x) == 0 ||
      (is.character(x) && (nchar(trimws(x)) == 0 || tolower(trimws(x)) == "null"))) {
    return(default)
  }

  # Convert to character and trim
  x_str <- trimws(as.character(x))

  # Split by comma, semicolon, or whitespace
  # Try comma first, then semicolon, then whitespace
  if (grepl(",", x_str, fixed = TRUE)) {
    parts <- strsplit(x_str, ",", fixed = TRUE)[[1]]
  } else if (grepl(";", x_str, fixed = TRUE)) {
    parts <- strsplit(x_str, ";", fixed = TRUE)[[1]]
  } else {
    # Split by whitespace (one or more spaces/tabs)
    parts <- strsplit(x_str, "\\s+")[[1]]
  }

  # Trim each part and convert to numeric
  parts <- trimws(parts)
  parts <- parts[parts != ""]  # Remove empty strings

  if (length(parts) == 0) {
    return(default)
  }

  # Convert to numeric
  rgb_vals <- suppressWarnings(as.numeric(parts))

  # Validate: length must be exactly 3
  if (length(rgb_vals) != 3) {
    stop(sprintf(
      "rgb_weight must have exactly 3 values, but received %d value(s) from string '%s'. Expected format: 'r,g,b' (e.g., '1,1,1' or '3,4,2') or space/semicolon-separated.",
      length(rgb_vals), x
    ))
  }

  # Validate: no NA values
  if (any(is.na(rgb_vals))) {
    stop(sprintf(
      "rgb_weight contains invalid numeric values. Received string '%s', which could not be parsed to 3 numbers. Expected format: 'r,g,b' (e.g., '1,1,1' or '3,4,2').",
      x
    ))
  }

  # Validate: all values must be finite and >= 0
  if (any(!is.finite(rgb_vals)) || any(rgb_vals < 0)) {
    stop(sprintf(
      "rgb_weight values must be finite and >= 0. Received: c(%s) from string '%s'.",
      paste(rgb_vals, collapse = ", "), x
    ))
  }

  return(rgb_vals)
}

# Helper function to parse color_blind_fun
parse_color_blind_fun <- function(x) {
  # Handle missing, NULL, empty string, or "NULL" (case-insensitive)
  if (missing(x) || is.null(x) || length(x) == 0 ||
      (is.character(x) && (nchar(trimws(x)) == 0 || tolower(trimws(x)) == "null"))) {
    return(NULL)
  }

  # Convert to character and trim
  fun_name <- trimws(as.character(x))

  # Try to resolve the function name
  # Common color blindness functions: deutan, protan, tritan
  # We'll pass it as a string to Palo, but validate it exists if it's a known function
  known_funs <- c("deutan", "protan", "tritan")

  if (fun_name %in% known_funs) {
    # It's a known function name, return as-is (Palo will handle it)
    return(fun_name)
  } else {
    # Unknown function name - warn but allow it (Palo may accept custom functions)
    warning(sprintf(
      "Unknown color_blind_fun '%s'. Known functions: %s. Proceeding anyway.",
      fun_name, paste(known_funs, collapse = ", ")
    ))
    return(fun_name)
  }
}

# Parse optional arguments
# If rgb_weight is not provided (args length < 5), use default c(1,1,1)
# If provided (even if empty string), parse_rgb_weight will handle it and return default if needed
rgb_weight <- if (length(args) >= 5) {
  parse_rgb_weight(args[5])
} else {
  parse_rgb_weight()  # Use default c(1,1,1) when not provided
}
color_blind_fun <- if (length(args) >= 6) parse_color_blind_fun(args[6]) else NULL

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
position = as.matrix(merged[, c("x", "y")])

# Extract cluster labels as character vector
cluster = as.character(merged$cluster)

# Get unique clusters and their count
unique_clusters = sort(unique(cluster))
n_clusters = length(unique_clusters)

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
# Palo API: Palo(position, cluster, palette, rgb_weight, color_blind_fun)
# position: 2-column coordinate matrix
# cluster: cluster labels (character vector)
# palette: initial palette (character vector of hex colors)
# rgb_weight: optional numeric vector of length 3 (default: c(1, 1, 1))
# color_blind_fun: optional function name (e.g., "deutan", "protan", "tritan")

palo_args <- list(
  position = position,
  cluster = cluster,
  palette = init_pal
)

# Add optional parameters if provided
# rgb_weight is always set (defaults to c(1,1,1) if not provided)
palo_args$rgb_weight <- rgb_weight
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
