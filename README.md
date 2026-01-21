# L-STAR: LLM-Guided Spatial Domain Detection

L-STAR is a Python package for performing LLM-based pairwise model comparisons and consensus clustering for spatial transcriptomics data. The pipeline uses Large Language Models (LLMs) to evaluate clustering models through pairwise image comparisons, then aggregates the top-performing models using Evidence Accumulation Clustering (EAC) to produce a robust consensus clustering result.

## Overview

The L-STAR pipeline consists of three main stages:

### **Step 0. Spatial Domain Visualization Reconstruction (Palo-based)**

Before any model comparison, spatial domain visualizations are **reconstructed from raw clustering assignments** to ensure fair, color-consistent, and spatially interpretable inputs for downstream evaluation.

Given spatial coordinates and per-spot domain labels from each method, L-STAR regenerates domain images using the **Palo** color optimization strategy, which assigns visually distinct colors to spatially adjacent domains. This avoids palette-induced bias and ensures that visual differences reflect structural discrepancies rather than arbitrary color choices.

This step is fully automated, supports multiple coordinate formats, and produces standardized PNG images for all methods (and optional H&E references) that are later consumed by the LLM comparison stage.
(Related code for this reconstruction step is provided in the repository. )

### **Step 1. Pairwise Visual Comparisons via LLMs**

L-STAR uses a large language model (e.g., GPT-5) to perform **pairwise visual comparisons** between spatial domain detection results. For each dataset, the LLM is shown reconstructed domain images (from Step 0), optionally alongside histology references, and asked to judge which method produces a more biologically plausible and spatially coherent partition.

These pairwise decisions are aggregated across repeated runs and model pairs to compute **winning rates**, which quantify the relative visual performance of each method. The results are summarized in a ranking CSV that serves as the empirical basis for downstream model selection.

### **Step 2. Top-Performing Model Selection**

Based on the aggregated pairwise comparison outcomes, L-STAR selects a subset of high-performing methods for consensus construction.

This subset can be:

* **Manually specified**, or
* **Automatically determined**, for example by choosing the top-k methods according to LLM-derived winning rates.

This step filters out systematically underperforming methods while retaining complementary high-quality solutions, balancing robustness and diversity for the consensus stage.

### **Step 3. Consensus Clustering via Evidence Accumulation**

The selected top-performing methods are integrated using **Evidence Accumulation Clustering (EAC)**. Pairwise co-assignment frequencies across methods are accumulated into a consensus similarity matrix, which is then clustered to produce a final spatial domain assignment.

The resulting consensus labels are reported as **L-STAR**, representing an ensemble spatial domain detection that leverages both human-interpretable visual judgment (via LLMs) and classical clustering theory. This consensus is subsequently evaluated against ground truth using standard metrics such as ARI and AMI.

## Installation

Install from source (this Repo):

```bash
git clone https://github.com/Williamzcy0929/lstar.git
cd lstar
pip install -e .
```
OR
```
pip install "git+https://github.com/Williamzcy0929/L-STAR.git"
```

### R Dependencies (for Palo color optimization)

By default, L-STAR uses the [Palo](https://github.com/Winnie09/Palo) R package for spatially-aware color palette optimization when generating spatial visualization images from CSV files. Palo optimizes colors so that spatially neighboring clusters have visually distinct colors, improving visualization interpretability.

**Install Palo (recommended):**
```bash
Rscript scripts/install_palo.R
```

**Verify installation:**
```bash
Rscript scripts/test_palo.R
```

**Note:** If Palo is not installed, L-STAR will automatically fall back to default color palettes. The pipeline will still function correctly, but color optimization for spatially neighboring clusters will not be performed.

See `scripts/README.md` for detailed installation instructions and troubleshooting.

## Quick Start

L-STAR supports two modes for spatial visualization:

### Mode 1: Using Pre-generated Images (Original Mode)

```python
import lstar

# Run the full L-STAR pipeline with pre-generated images
df = lstar.l_star(
    image_dir="path/to/images",           # Directory with model output images and the optional H&E image
    dataset_name="DLPFC (from 10X Visium Human Brain)",
    assignments_csv="path/to/assignments.csv",  # Combined assignments CSV
    id_col="spot_id",                     # ID column name
    fixed_k=7,                             # Fixed number of clusters
    api_key="your-openai-api-key"          # Or set OPENAI_API_KEY env var
)

print(df.head())
# Output includes 'L-STAR' column with consensus cluster labels
```

### Mode 2: Generate Images from CSV Files (Default with Palo)

By default, L-STAR can generate spatial visualization images internally from spatial locations and domain assignments, using Palo for color palette optimization:

```python
import lstar

# Generate images internally using Palo for color optimization
df = lstar.l_star(
    dataset_name="DLPFC (from 10X Visium Human Brain)",
    spatial_locations_csv="path/to/spatial_locations.csv",  # CSV with spot_id, x, y columns
    assignments_csv="path/to/assignments.csv",              # CSV with spot_id and method columns
    id_col="spot_id",                                       # ID column name
    use_palo=True,                                          # Use Palo for color optimization (default: True)
    fixed_k=7,
    api_key="your-openai-api-key"
)

print(df.head())
# Output includes 'L-STAR' column with consensus cluster labels
# Generated images are saved to output_dir/generated_images/
```

**Key Points:**
- When `spatial_locations_csv` and `assignments_csv` are provided, images are generated internally
- Palo is used by default (`use_palo=True`) to optimize colors for spatially neighboring clusters
- To use pre-generated images instead, provide `image_dir` and omit `spatial_locations_csv`
- Generated images are saved to `output_dir/generated_images/` with filenames matching method names

## Input Format

L-STAR supports two input modes:

### Mode 1: Pre-generated Images

The `image_dir` should contain:
- `he.png` (or custom name with extensions .png, .jpg, .jpeg, or .pdf): H&E reference image (optional)
- `Model1.png`, `Model2.jpg`, etc.: Clustering visualization images for each model
  - Supported formats: `.png`, `.jpg`, `.jpeg`, `.pdf`
  - If multiple formats exist for the same model name, PNG is preferred over JPG/JPEG, which is preferred over PDF

### Mode 2: CSV Files for Image Generation (Default with Palo)

When generating images internally, provide two CSV files:

**1. Spatial Locations CSV** (`spatial_locations_csv`):
- Required columns: `spot_id` (or custom `id_col`), `x`, `y`
- Contains spatial coordinates for each spot/cell
- Example:
```csv
spot_id,x,y
spot_1,10.5,20.3
spot_2,11.2,21.1
spot_3,12.0,19.8
...
```

**2. Assignments CSV** (`assignments_csv`):
- Required columns: `spot_id` (or custom `id_col`), plus one column per method
- Column names (except `spot_id`) are treated as method names
- Each method column contains cluster assignments for that method
- Example:
```csv
spot_id,GraphST,SpaGCN,BayesSpace,STAGATE
spot_1,1,2,1,3
spot_2,2,2,2,3
spot_3,1,1,1,2
...
```

**Image Generation Process:**
- L-STAR internally generates spatial visualization images for each method
- By default, uses Palo R package to optimize color palettes so that spatially neighboring clusters have visually distinct colors
- Generated images are saved to `output_dir/generated_images/` with filenames matching method names (e.g., `GraphST.png`, `SpaGCN.png`)
- If `he_image_path` is provided, the H&E image is copied to the generated images directory

**Palo Color Optimization:**
- Palo optimizes colors based on spatial adjacency, ensuring neighboring clusters are visually distinct
- Set `use_palo=False` to disable Palo and use default color palettes
- If Palo is not installed, L-STAR automatically falls back to default colors

### Assignment CSVs (Legacy Mode)

For the legacy mode with separate CSV files per model, each model should have a CSV file with clustering assignments. The CSV should contain:
- An ID column (first column, e.g., `spot_id`, `cell_id`)
- A clustering column (e.g., `cluster`, `label`, or model name)
- Optionally, a ground truth column (e.g., `Ground`, `ground_truth`)

Example:
```csv
spot_id,cluster
spot_1,1
spot_2,2
spot_3,1
...
```

## API Reference

### High-Level Pipeline

#### `l_star()`

Main entry point for the full L-STAR pipeline.

```python
lstar.l_star(
    dataset_name: str,
    *,
    image_dir: str | Path | None = None,
    spatial_locations_csv: str | Path | None = None,
    assignments_csv: str | Path | None = None,
    id_col: str | None = None,
    use_separate_csvs: bool = False,
    assignments_dir: str | Path | None = None,
    assignment_csv_list: Sequence[str | Path] | None = None,
    output_dir: str | Path = "lstar_output",
    simple_mode: bool = True,
    reps: int = 5,
    top_k: int = 5,
    top_k_mode: Literal["fixed", "elbow"] = "fixed",
    selection_mode: Literal["manual", "top_k"] = "top_k",
    model_names: Sequence[str] | None = None,
    k_mode: Literal["fixed", "auto"] = "auto",
    fixed_k: Optional[int] = None,
    use_palo: bool = True,
    he_image_path: str | Path | None = None,
    api_key: Optional[str] = None,
    **kwargs
) -> pd.DataFrame
```

**Key Parameters:**
- `dataset_name`: Background information about the dataset (required)
- `image_dir`: Directory with pre-generated images (use this OR provide CSV files for image generation)
- `spatial_locations_csv`: CSV with spatial coordinates (required for image generation mode)
- `assignments_csv`: CSV with assignments where column names are method names (required for image generation mode)
- `id_col`: Name of the ID column (default: "spot_id")
- `use_palo`: Whether to use Palo for color optimization when generating images (default: True)
- `he_image_path`: Optional path to H&E image to copy when generating images

### Pairwise Comparisons

#### `run_pairwise_comparisons()`

Run LLM-based pairwise comparisons and generate ranking.

```python
ranking_df, pairwise_dir, ranking_csv = lstar.run_pairwise_comparisons(
    image_dir="path/to/images",
    reps=5,
    top_k=5,
    simple_mode=True,
    output_dir="lstar_output",
    api_key="your-api-key"
)
```

**Key Parameters:**
- `reps`: Number of pairwise comparison repetitions (default: 5)
- `simple_mode`: Use simple prompts (True) or complex prompts with bias warnings (False)
- `top_k_mode`: "fixed" or "elbow" for top-k selection
- `force_rerun`: Ignore cache and recompute all comparisons
- `skip_pairwise`: Skip LLM calls and reuse existing results

**Caching:** Pairwise comparisons are automatically cached to avoid redundant LLM calls. Each comparison is stored as a JSON file in `output_dir/pairwise/cache_*.json`.

### Consensus Clustering

#### `run_consensus_clustering()`

Perform consensus clustering on selected models.

```python
consensus_df = lstar.run_consensus_clustering(
    ranking_csv="lstar_output/ranking.csv",
    assignments_dir="path/to/assignments",
    model_names=["Model1", "Model2", "Model3"],
    k_mode="auto",
    output_csv="lstar_output/L_STAR_consensus.csv"
)
```

**Key Parameters:**
- `selection_mode`: "manual" (use `model_names`) or "top_k" (select by ranking)
- `k_mode`: "fixed" (use `fixed_k`) or "auto" (determine from models)
- `k_method`: "median_from_models" or "mode_from_models" for auto k selection
- `ground_truth_col`: Optional column name for ARI evaluation

## Output Files

The pipeline generates the following outputs in `output_dir`:

- `pairwise/`: Directory containing:
  - `pairwise_results_rep*.jsonl`: Pairwise comparison results (one per repetition)
  - `cache_*.json`: Cached individual pairwise comparisons
- `ranking.csv`: Model ranking with winning rates, games, wins, losses, ties, points
- `L_STAR_consensus.csv`: Final consensus clustering with 'L-STAR' column

## Advanced Usage

### Custom Model Selection

```python
# Manually specify models for consensus (using pre-generated images)
df = lstar.l_star(
    image_dir="images",
    dataset_name="DLPFC (from 10X Visium Human Brain)",
    assignments_csv="assignments.csv",
    id_col="spot_id",
    model_names=["GraphST", "STAGATE", "SpaGCN", "BayesSpace"],
    fixed_k=7
)
```

### Generate Images with Custom Palo Settings

```python
# Generate images from CSV with custom Palo parameters
df = lstar.l_star(
    dataset_name="DLPFC (from 10X Visium Human Brain)",
    spatial_locations_csv="spatial_locations.csv",
    assignments_csv="assignments.csv",
    id_col="spot_id",
    use_palo=True,  # Enable Palo color optimization
    he_image_path="path/to/he_image.png",  # Optional H&E image
    fixed_k=7,
    api_key="your-api-key"
)
```

### Disable Palo Color Optimization

```python
# Generate images without Palo (use default colors)
df = lstar.l_star(
    dataset_name="DLPFC (from 10X Visium Human Brain)",
    spatial_locations_csv="spatial_locations.csv",
    assignments_csv="assignments.csv",
    id_col="spot_id",
    use_palo=False,  # Disable Palo, use default color palettes
    fixed_k=7,
    api_key="your-api-key"
)
```

### Top-K Selection

```python
# Automatically select top 5 models by ranking
df = lstar.l_star(
    image_dir="images",
    assignments_dir="assignments",
    selection_mode="top_k",
    top_k=5,
    k_mode="auto"
)
```

### Custom LLM Settings

```python
df = lstar.l_star(
    image_dir="images",
    assignments_dir="assignments",
    model_names=["Model1", "Model2", "Model3"],
    model_name="gpt-5.1-2025-11-13",
    pairwise_temperature=1.0,
    pairwise_reasoning_effort="medium",
    second_round_reasoning_effort="high",
    api_key="your-api-key"
)
```

### Step-by-Step Execution

```python
# Step 1: Run pairwise comparisons
ranking_df, pairwise_dir, ranking_csv = lstar.run_pairwise_comparisons(
    image_dir="images",
    output_dir="output",
    api_key="your-api-key"
)

# Step 2: Run consensus clustering
consensus_df = lstar.run_consensus_clustering(
    ranking_csv=ranking_csv,
    assignments_dir="assignments",
    model_names=["Model1", "Model2", "Model3"],
    output_dir="output"
)
```

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: OpenAI API key (can be set instead of passing `api_key` parameter)

### Default Values

- Output directory: `lstar_output`
- Repetitions: 5
- Top-K: 5
- K range: 2-15
- Model: `gpt-5-2025-08-07`
- Temperature: 1.0
- Reasoning effort: "medium" (pairwise), "high" (second-round if applicable)

## Error Handling

The package provides informative error messages for common issues:

- Missing assignment CSVs
- Mismatched row counts between CSVs
- Missing models in ranking
- Invalid k values
- API connection errors

## Logging

L-STAR uses Python's `logging` module. To enable verbose output:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Citation

If you use L-STAR in your research, please cite:

```bibtex
@software{lstar,
  title={L-STAR: LLM-Guided Spatial Domain Detection},
  author={Changyue Zhao, Zhicheng Ji},
  year={2025},
  url={https://github.com/Williamzcy0929/L-STAR}
}
```

## License

MIT License

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Contact

Authors: Changyue (William) Zhao ([changyue.zhao@duke.edu](mailto:changyue.zhao@duke.edu)) and Dr. Zhicheng Ji ([zhicheng.ji@duke.edu](mailto:zhicheng.ji@duke.edu))

For questions and issues, please open an issue on GitHub or [send an email to the maintainer](mailto:changyue.zhao@duke.edu).
