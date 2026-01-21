# L-STAR R Dependencies

This directory contains scripts for installing and testing the Palo R package, which is used for spatially-aware color palette optimization.

## Palo Installation

### Option 1: Using the installation script (recommended)

```bash
# Install Palo from main branch
Rscript scripts/install_palo.R

# Or install from a specific commit/tag
Rscript scripts/install_palo.R <commit_sha_or_tag>
```

### Option 2: Using R directly

```r
# Install remotes if not already installed
install.packages("remotes", repos = "https://cloud.r-project.org")

# Install Palo
remotes::install_github("Winnie09/Palo", repos = "https://cloud.r-project.org")
```

### Option 3: Using conda

```bash
# Create conda environment (includes R and dependencies)
conda env create -f environment.yml
conda activate lstar

# Install Palo
Rscript scripts/install_palo.R
```

### Option 4: Using Docker

```bash
# Build Docker image (Palo is preinstalled)
docker build -t lstar .

# Run container
docker run -it lstar
```

## Testing Palo Installation

Run the smoke test to verify Palo is working:

```bash
Rscript scripts/test_palo.R
```

Expected output:
```
Testing Palo installation...
Palo version: <version>
Running minimal Palo::Palo() test...
✓ Palo::Palo() test passed. Result: 2 optimized colors
Optimized colors:
  Cluster 1: #<hex_color>
  Cluster 2: #<hex_color>

✓ All tests passed!
```

## Palo API

The Palo package provides the `Palo::Palo()` function with the following signature:

```r
Palo::Palo(
  u,              # 2-column coordinate matrix (x, y positions)
  cl,             # Cluster labels (character vector)
  pal,            # Initial palette (character vector of hex colors)
  rgb_weight,     # Optional: numeric vector of length 3 (default: c(1, 1, 1))
  color_blind_fun # Optional: color blindness function ("deutan", "protan", "tritan")
)
```

Returns: Named character vector mapping cluster labels to optimized hex colors.

## Troubleshooting

### Rscript not found

Ensure R is installed and in your PATH:
```bash
which Rscript
Rscript --version
```

### Palo installation fails

1. Check internet connection (for GitHub access)
2. Ensure remotes/devtools are installed:
   ```r
   install.packages(c("remotes", "devtools"), repos = "https://cloud.r-project.org")
   ```
3. Try installing from a specific commit:
   ```bash
   Rscript scripts/install_palo.R <commit_sha>
   ```

### Runtime errors

If Palo fails at runtime, L-STAR will automatically fall back to default color palettes. Check logs for error messages.
