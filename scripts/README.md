# L-STAR R Dependencies

This directory contains helper scripts for installing and testing Palo in source checkouts.
In packaged installs, L-STAR uses bundled R scripts from `lstar/resources/`.

## Runtime R Components (when `use_palo=True`)

L-STAR uses two R scripts at runtime:

- `run_palo.R`: computes Palo-optimized palettes
- `plot_spatial_with_palo.R`: renders per-method spatial images with `ggplot2` (no legend)

Required R packages:

- `Palo`
- `ggplot2`
- `RColorBrewer`

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

# Install plotting dependencies
install.packages(c("ggplot2", "RColorBrewer"), repos = "https://cloud.r-project.org")

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

```text
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
  position,       # 2-column coordinate matrix (x, y positions)
  cluster,        # Cluster labels (character vector)
  palette,        # Initial palette (character vector of hex colors)
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

If Palo or R plotting fails at runtime, L-STAR automatically falls back to matplotlib/default color rendering. Check logs for details.

### Path dependency / script resolution

L-STAR resolves runtime scripts in this order:

1. Environment variable override
2. Bundled package resources (`lstar/resources/*.R`)
3. Local fallback paths for editable/development installs

Optional environment variables:

- `LSTAR_RUN_PALO_SCRIPT` (override path for `run_palo.R`)
- `LSTAR_PLOT_SPATIAL_SCRIPT` (override path for `plot_spatial_with_palo.R`)
