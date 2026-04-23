#!/usr/bin/env Rscript
# Smoke test script for Palo installation
# Usage: Rscript scripts/test_palo.R

cat("Testing Palo installation...\n")

# Check if Palo is installed
if (!require("Palo", quietly = TRUE)) {
  stop("Palo package is not installed. Run: Rscript scripts/install_palo.R")
}

# Print version
version <- as.character(packageVersion("Palo"))
cat(sprintf("Palo version: %s\n", version))

# Run minimal test
cat("Running minimal Palo::Palo() test...\n")

# Create test data: 6 spots with 2 clusters
position = matrix(c(
  1.0, 1.0,
  2.0, 1.0,
  1.0, 2.0,
  3.0, 3.0,
  4.0, 3.0,
  3.0, 4.0
), ncol = 2, byrow = TRUE)

cluster = c("1", "1", "1", "2", "2", "2")
palette = c("#FF0000", "#00FF00")  # Red, Green

# Call Palo
result = Palo::Palo(position = position, cluster = cluster, palette = palette)

# Check result
if (is.null(result) || length(result) == 0) {
  stop("Palo::Palo() returned empty result")
}

cat(sprintf("Palo::Palo() test passed. Result: %d optimized colors\n", length(result)))
cat("Optimized colors:\n")
for (i in seq_along(result)) {
  cat(sprintf("  Cluster %s: %s\n", names(result)[i], result[i]))
}

cat("\nAll tests passed!\n")
