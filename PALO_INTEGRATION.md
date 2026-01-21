# Palo R Package Integration

This document describes the integration of the Palo R package for spatially-aware color palette optimization in L-STAR.

## Overview

L-STAR now uses the [Palo](https://github.com/Winnie09/Palo) R package to optimize color palettes when generating spatial domain visualization images. Palo ensures that spatially neighboring clusters are assigned visually distinct colors, improving the interpretability of spatial transcriptomics visualizations.

## Implementation Details

### Core Function: `get_palo_optimized_colors()`

**Location:** `lstar/image_generation.py`

**Changes:**
- Removed custom neighbor-graph and color-swapping heuristics
- Now calls `Palo::Palo()` via external R script (`scripts/run_palo.R`)
- Properly handles Palo's API: `Palo::Palo(u, cl, pal, rgb_weight, color_blind_fun)`
- Maintains cluster ID consistency when round-tripping between Python and R
- Graceful fallback to default colors if Palo is unavailable

**API:**
```python
get_palo_optimized_colors(
    assignments: pd.Series,           # Cluster assignments indexed by spot_id
    spatial_coords: pd.DataFrame,      # DataFrame with id_col, x, y columns
    id_col: str = "spot_id",          # ID column name
    rgb_weight: Optional[Tuple[float, float, float]] = None,  # Optional RGB weights
    color_blind_fun: Optional[str] = None  # Optional color blindness function
) -> Dict[int, Tuple[float, float, float]]  # Returns cluster_id -> (r, g, b) mapping
```

### R Script: `scripts/run_palo.R`

**Purpose:** Standalone R script that calls Palo::Palo() with proper data formatting.

**Input:**
- Coordinates CSV: `id_col, x, y`
- Assignments CSV: `id_col, cluster`
- Optional: `rgb_weight`, `color_blind_fun`

**Output:**
- CSV with columns: `cluster, r, g, b` (RGB values 0-1)

**Key Features:**
- Validates Palo is installed (fails fast if not available)
- Properly formats data for Palo API:
  - `u`: 2-column coordinate matrix
  - `cl`: Character vector of cluster labels
  - `pal`: Initial palette (RColorBrewer Set3 or colorRampPalette)
- Handles cluster ID types (numeric/string) consistently
- Clear error messages if Palo is missing

## Preinstallation Mechanisms

### 1. Docker (Recommended for CI/HPC)

**File:** `Dockerfile`

**Features:**
- Based on `rocker/r-ver:4.3.0` (official R Docker image)
- Preinstalls R system dependencies
- Installs Palo via `remotes::install_github('Winnie09/Palo')`
- Verifies installation with version check
- Includes Python dependencies

**Usage:**
```bash
docker build -t lstar .
docker run -it lstar
```

**Benefits:**
- No runtime network access needed
- Reproducible builds
- Works in HPC/CI environments with network restrictions

### 2. Conda Environment

**File:** `environment.yml`

**Features:**
- Includes R 4.3.0, remotes, devtools, RColorBrewer
- Python dependencies via pip
- Post-install step: `Rscript scripts/install_palo.R`

**Usage:**
```bash
conda env create -f environment.yml
conda activate lstar
Rscript scripts/install_palo.R
```

**Benefits:**
- Cross-platform (Linux, macOS, Windows)
- Version-pinned dependencies
- Easy to share across team members

### 3. Installation Script

**File:** `scripts/install_palo.R`

**Features:**
- Installs remotes if needed
- Supports installing from specific commit/tag (for reproducibility)
- Runs smoke test after installation
- Clear error messages

**Usage:**
```bash
# Install from main branch
Rscript scripts/install_palo.R

# Install from specific commit/tag
Rscript scripts/install_palo.R <commit_sha_or_tag>
```

### 4. Smoke Test

**File:** `scripts/test_palo.R`

**Purpose:** Verify Palo installation and basic functionality.

**Tests:**
- Checks Palo is installed
- Prints version
- Runs minimal `Palo::Palo()` call with test data
- Validates output format

**Usage:**
```bash
Rscript scripts/test_palo.R
```

## How Palo is Actually Used

### Data Flow

1. **Python:** `get_palo_optimized_colors()` receives assignments and coordinates
2. **Python:** Writes temporary CSV files (coordinates, assignments)
3. **Python:** Calls `Rscript scripts/run_palo.R <args>`
4. **R:** Reads CSVs, merges data
5. **R:** Formats data for Palo:
   - `u <- as.matrix(merged[, c("x", "y")])`  # 2-column coordinate matrix
   - `cl <- as.character(merged$cluster)`      # Character vector of cluster labels
   - `pal <- brewer.pal(...)` or `colorRampPalette(...)`  # Initial palette
6. **R:** Calls `Palo::Palo(u, cl, pal)` (with optional parameters)
7. **R:** Converts optimized hex colors to RGB (0-1 floats)
8. **R:** Writes CSV: `cluster, r, g, b`
9. **Python:** Reads CSV, builds `Dict[cluster_id -> (r, g, b)]`
10. **Python:** Returns color mapping for image generation

### Why This Approach Works

1. **No Runtime Installation:** Palo is preinstalled via Docker/conda/scripts
2. **Proper API Usage:** Directly calls `Palo::Palo()`, not custom heuristics
3. **Robust Fallback:** If R/Palo unavailable, uses default colors (pipeline continues)
4. **Cluster ID Consistency:** Handles numeric/string cluster IDs correctly
5. **Clear Errors:** Fails fast with helpful messages if Palo missing

## Verification

### Ensure Palo is Used

1. **Check R script calls Palo::Palo():**
   ```bash
   grep -n "Palo::Palo" scripts/run_palo.R
   ```
   Should show: `opt_pal <- do.call(Palo::Palo, palo_args)`

2. **Run smoke test:**
   ```bash
   Rscript scripts/test_palo.R
   ```
   Should pass with "✓ All tests passed!"

3. **Check Python function:**
   ```python
   # In get_palo_optimized_colors(), verify it calls run_palo.R
   # Not implementing custom neighbor-graph logic
   ```

### Preinstallation Verification

1. **Docker:**
   ```bash
   docker build -t lstar .
   docker run lstar Rscript -e "library(Palo); print(packageVersion('Palo'))"
   ```

2. **Conda:**
   ```bash
   conda activate lstar
   Rscript -e "library(Palo); print(packageVersion('Palo'))"
   ```

3. **Manual:**
   ```bash
   Rscript scripts/test_palo.R
   ```

## Rationale

### Why Preinstallation?

- **HPC/CI Environments:** Often block outbound network at runtime
- **Reproducibility:** Pinned versions ensure consistent results
- **Performance:** No delay from runtime installation
- **Reliability:** Installation failures caught at build time, not runtime

### Why External R Script?

- **Separation of Concerns:** R logic isolated from Python
- **Easier Debugging:** Can test R script independently
- **Version Control:** R script changes tracked separately
- **Reusability:** Script can be used outside Python context

### Why Not rpy2?

- **Dependency Complexity:** rpy2 requires R headers, compilation
- **Platform Issues:** rpy2 can be difficult on some systems
- **Subprocess Simplicity:** `Rscript` is universally available
- **Isolation:** R process failures don't crash Python

## Troubleshooting

### Palo Not Found at Runtime

**Symptom:** Logs show "Palo package is not installed" or "Using default colors"

**Solution:**
1. Run `Rscript scripts/test_palo.R` to verify installation
2. If missing, run `Rscript scripts/install_palo.R`
3. Check R is in PATH: `which Rscript`

### Cluster ID Mismatch

**Symptom:** Colors not matching expected clusters

**Solution:**
- Ensure cluster IDs are consistent (numeric vs string)
- Check CSV output from R script matches Python expectations
- Verify `unique_clusters` sorting matches R output

### Runtime Installation Attempts

**Symptom:** R script tries to install Palo at runtime (shouldn't happen)

**Solution:**
- Ensure `scripts/run_palo.R` checks for Palo before calling it
- Preinstall Palo using one of the mechanisms above
- Remove any `install.packages()` or `devtools::install_github()` calls from runtime code
