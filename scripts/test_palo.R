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

# Create test data: 3 spots with 2 clusters
u <- matrix(c(
  1.0, 1.0,
  2.0, 1.0,
  3.0, 2.0
), ncol = 2, byrow = TRUE)

cl <- c("1", "1", "2")
pal <- c("#FF0000", "#00FF00")  # Red, Green

# Call Palo
result <- Palo::Palo(u = u, cl = cl, pal = pal)

# Check result
if (is.null(result) || length(result) == 0) {
  stop("Palo::Palo() returned empty result")
}

cat(sprintf("✓ Palo::Palo() test passed. Result: %d optimized colors\n", length(result)))
cat("Optimized colors:\n")
for (i in seq_along(result)) {
  cat(sprintf("  Cluster %s: %s\n", names(result)[i], result[i]))
}

cat("\n✓ All tests passed!\n")
