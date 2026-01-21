#!/usr/bin/env Rscript
# Script to install Palo R package
# Usage: Rscript scripts/install_palo.R [commit_sha_or_tag]
# If no argument provided, installs from main branch

args <- commandArgs(trailingOnly = TRUE)

# Check if remotes is installed
if (!require("remotes", quietly = TRUE)) {
  cat("Installing remotes package...\n")
  install.packages("remotes", repos = "https://cloud.r-project.org")
}

# Determine installation source
if (length(args) >= 1 && nchar(args[1]) > 0) {
  # Install from specific commit/tag
  commit_or_tag <- args[1]
  cat(sprintf("Installing Palo from commit/tag: %s\n", commit_or_tag))
  remotes::install_github("Winnie09/Palo", ref = commit_or_tag, repos = "https://cloud.r-project.org")
} else {
  # Install from main branch (default)
  cat("Installing Palo from main branch...\n")
  remotes::install_github("Winnie09/Palo", repos = "https://cloud.r-project.org")
}

# Verify installation
if (require("Palo", quietly = TRUE)) {
  version <- as.character(packageVersion("Palo"))
  cat(sprintf("✓ Palo successfully installed (version: %s)\n", version))
  
  # Run a quick smoke test
  cat("Running smoke test...\n")
  tryCatch({
    # Create minimal test data
    u <- matrix(c(1, 2, 3, 1, 2, 3), ncol = 2)
    cl <- c("1", "2", "1")
    pal <- c("#FF0000", "#00FF00")
    
    result <- Palo::Palo(u = u, cl = cl, pal = pal)
    cat("✓ Palo::Palo() smoke test passed\n")
  }, error = function(e) {
    cat(sprintf("⚠ Palo smoke test failed: %s\n", e$message))
  })
} else {
  stop("Failed to install Palo package")
}
