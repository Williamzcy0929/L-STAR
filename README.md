# L-STAR: LLM-Guided Spatial Domain Detection

L-STAR is a Python package for performing LLM-based pairwise model comparisons and consensus clustering for spatial transcriptomics data. The pipeline uses Large Language Models (LLMs) to evaluate clustering models through pairwise image comparisons, then aggregates the top-performing models using Evidence Accumulation Clustering (EAC) to produce a robust consensus clustering result.

## Overview

The L-STAR pipeline consists of two main stages:

1. **Pairwise Comparisons**: Uses an LLM (e.g., GPT-5) to compare clustering results from different models based on images. The LLM evaluates which model performs better for each pair, and these comparisons are aggregated into a ranking CSV with winning rates.

2. **Consensus Clustering**: Selects a subset of top-performing models (either manually specified or top-k by ranking) and performs Evidence Accumulation Clustering (EAC) to produce a final consensus clustering assignment labeled "L-STAR".

## Installation

```bash
pip install lstar
```

Or install from source:

```bash
git clone https://github.com/yourusername/lstar.git
cd lstar
pip install -e .
```

## Quick Start

```python
import lstar

# Run the full L-STAR pipeline
df = lstar.l_star(
    image_dir="path/to/images",           # Directory with H&E and model output images
    assignments_dir="path/to/assignments", # Directory with per-model assignment CSVs
    model_names=["Model1", "Model2", "Model3", "Model4", "Model5"],
    fixed_k=7,                             # Fixed number of clusters
    api_key="your-openai-api-key"          # Or set OPENAI_API_KEY env var
)

print(df.head())
# Output includes 'L-STAR' column with consensus cluster labels
```

## Input Format

### Image Directory

The `image_dir` should contain:
- `he.png` (or custom name with extensions .png, .jpg, .jpeg, or .pdf): H&E reference image (optional)
- `Model1.png`, `Model2.jpg`, etc.: Clustering visualization images for each model
  - Supported formats: `.png`, `.jpg`, `.jpeg`, `.pdf`
  - If multiple formats exist for the same model name, PNG is preferred over JPG/JPEG, which is preferred over PDF

### Assignment CSVs

Each model should have a CSV file with clustering assignments. The CSV should contain:
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
    image_dir: str | Path,
    *,
    assignments_dir: str | Path | None = None,
    assignment_csvs: Sequence[str | Path] | None = None,
    output_dir: str | Path = "lstar_output",
    simple_mode: bool = True,
    reps: int = 5,
    top_k: int = 5,
    top_k_mode: Literal["fixed", "elbow"] = "fixed",
    selection_mode: Literal["manual", "top_k"] = "manual",
    model_names: Sequence[str] | None = None,
    k_mode: Literal["fixed", "auto"] = "auto",
    fixed_k: Optional[int] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> pd.DataFrame
```

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
# Manually specify models for consensus
df = lstar.l_star(
    image_dir="images",
    assignments_dir="assignments",
    model_names=["GraphST", "STAGATE", "SpaGCN", "BayesSpace"],
    fixed_k=7
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
- Model: `gpt-5.1-2025-11-13`
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

Authors: Changyue (William) Zhao ([changyue (dot) zhao (at) duke (dot) edu](mailto:changyue.zhao@duke.edu)) and Dr. Zhicheng Ji ([zhicheng (dot) ji (at) duke (dot) edu](mailto:zhicheng.ji@duke.edu]))

For questions and issues, please open an issue on GitHub or [send an email to the maintainer](mailto:changyue.zhao@duke.edu).
