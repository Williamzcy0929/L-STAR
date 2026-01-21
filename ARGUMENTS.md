# L-STAR Arguments Reference

This document provides detailed documentation for all arguments in the L-STAR package, including default values for different modes and usage examples.

## Table of Contents

1. [Main Function: `l_star()`](#main-function-l_star)
2. [Pairwise Comparisons: `run_pairwise_comparisons()`](#pairwise-comparisons-run_pairwise_comparisons)
3. [Second-Round Reasoning: `run_second_round_reasoning()`](#second-round-reasoning-run_second_round_reasoning)
4. [Consensus Clustering: `run_consensus_clustering()`](#consensus-clustering-run_consensus_clustering)
5. [Default Settings by Mode](#default-settings-by-mode)
6. [Examples](#examples)

---

## Main Function: `l_star()`

The high-level entry point that runs the complete L-STAR pipeline: pairwise comparisons → (optional second-round reasoning) → consensus clustering.

### Function Signature

```python
def l_star(
    image_dir: Union[str, Path],
    dataset_name: str,
    *,
    assignments_csv: Union[str, Path, None] = None,
    id_col: Optional[str] = None,
    use_separate_csvs: bool = False,
    assignments_dir: Union[str, Path, None] = None,
    assignment_csv_list: Optional[Sequence[Union[str, Path]]] = None,
    output_dir: Union[str, Path] = "lstar_output",
    simple_mode: bool = True,
    reps: int = 5,
    top_k: int = 5,
    top_k_mode: Literal["fixed", "elbow"] = "fixed",
    selection_mode: Literal["manual", "top_k"] = "top_k",
    model_names: Sequence[str] | None = None,
    k_mode: Literal["fixed", "auto"] = "auto",
    fixed_k: Optional[int] = None,
    use_second_round: bool = False,
    **kwargs,
) -> pd.DataFrame
```

### Required Arguments

#### `image_dir` (required)
- **Type**: `str` or `Path`
- **Description**: Directory containing H&E reference image and model output images
- **Supported Image Formats**: `.png`, `.jpg`, `.jpeg` (case-insensitive)
- **Structure**:
  - H&E image: `he.png` (or `he.jpg`, `he.jpeg`) - optional
  - Model images: `ModelName.png`, `ModelName.jpg`, etc. - at least 2 required
- **Example**: `"data/DLPFC/images"`

#### `dataset_name` (required)
- **Type**: `str`
- **Description**: Background information about the dataset name provided in LLM prompts. This helps the LLM understand the context of the spatial transcriptomics data being analyzed.
- **Usage**: Included in prompts to give the LLM context about the tissue type, technology, and biological system
- **Examples**:
  - `"DLPFC (from 10X Visium Human Brain)"`
  - `"STARmap Mouse Ventricular Cardiomyocytes"`
  - `"Stereo-seq Axolotl Brain"`
  - `"SlideV2 Mouse Embryo E8.5"`
  - `"SpatialMouseAtlas2020 (from SeqFISH dataset)"`

### Assignment Data Arguments

#### `assignments_csv` (optional, default: `None`)
- **Type**: `str` or `Path` or `None`
- **Description**: Path to a single CSV file containing all model assignments (one column per model)
- **Default Mode**: This is the default mode when `use_separate_csvs=False`
- **Format**: CSV with ID column + one column per model
  ```csv
  spot_id,GraphST,SpaGCN,STMGCN,SpaceFlow,BayesSpace
  spot_1,1,2,1,3,2
  spot_2,2,2,2,3,2
  ...
  ```
- **Fuzzy Matching**: When using this mode, model names are automatically normalized and matched between:
  - Ranking CSV model names
  - Assignment CSV column names
  - Image filenames
- **Required When**: `use_separate_csvs=False` (default mode)
- **Example**: `"data/DLPFC/combined_assignments.csv"`

#### `id_col` (optional, default: `None`)
- **Type**: `str` or `None`
- **Description**: Name of the ID column in `assignments_csv`
- **Required When**: `assignments_csv` is provided
- **Default**: `None` (must be specified when using combined CSV mode)
- **Example**: `"spot_id"`, `"cell_id"`, `"barcode"`

#### `use_separate_csvs` (optional, default: `False`)
- **Type**: `bool`
- **Description**: If `True`, use legacy mode with separate CSV files (one CSV per model)
- **Default**: `False` (uses combined CSV mode)
- **When `True`**: Requires either `assignments_dir` or `assignment_csv_list`
- **When `False`**: Requires `assignments_csv` and `id_col`
- **Example**: `False` (default), `True` (legacy mode)

#### `assignments_dir` (optional, default: `None`)
- **Type**: `str` or `Path` or `None`
- **Description**: Directory containing per-model assignment CSV files
- **Only Used When**: `use_separate_csvs=True`
- **File Naming**: Searches for `{model_name}.csv`, `{model_name}_pred_label.csv`, or `{model_name}_labels.csv`
- **Example**: `"data/DLPFC/assignments"`

#### `assignment_csv_list` (optional, default: `None`)
- **Type**: `Sequence[str | Path]` or `None`
- **Description**: Explicit list of per-model assignment CSV file paths
- **Only Used When**: `use_separate_csvs=True`
- **Order**: Should match the order of models in `model_names` (if provided)
- **Example**: `["model1.csv", "model2.csv", "model3.csv"]`

### Output Arguments

#### `output_dir` (optional, default: `"lstar_output"`)
- **Type**: `str` or `Path`
- **Description**: Base directory for all output files
- **Output Structure**:
  - `output_dir/pairwise/`: Pairwise comparison results and cache
  - `output_dir/ranking.csv`: Model ranking with winning rates
  - `output_dir/L_STAR_consensus.csv`: Final consensus clustering
  - `output_dir/pairwise/*_second_round_reasoning.json`: Second-round reasoning results (if enabled)
- **Example**: `"results/DLPFC"`

### Pairwise Comparison Arguments

#### `simple_mode` (optional, default: `True`)
- **Type**: `bool`
- **Description**: Prompt style for pairwise comparisons
- **When `True`**: Simple, concise prompts
- **When `False`**: Complex prompts with bias warnings about not favoring "smooth" clusters
- **Example**: `True` (default), `False` (for more detailed analysis)

#### `reps` (optional, default: `5`)
- **Type**: `int`
- **Description**: Number of repetitions for pairwise comparisons
- **Purpose**: Multiple repetitions increase robustness of rankings
- **Range**: Typically 3-10
- **Example**: `5` (default), `10` (more robust)

#### `top_k` (optional, default: `5`)
- **Type**: `int`
- **Description**: Number of top models to consider
- **Usage**:
  - Used in pairwise ranking for top-k selection
  - Used in consensus clustering when `selection_mode="top_k"`
  - Used as input to second-round reasoning when `use_second_round=True`
- **Example**: `5` (default), `10` (more models)

#### `top_k_mode` (optional, default: `"fixed"`)
- **Type**: `Literal["fixed", "elbow"]`
- **Description**: Method for selecting top-k models
- **Options**:
  - `"fixed"`: Select exactly `top_k` models
  - `"elbow"`: Use elbow detection to find optimal k (at most `top_k`)
- **Example**: `"fixed"` (default), `"elbow"` (adaptive)

### Model Selection Arguments

#### `selection_mode` (optional, default: `"top_k"`)
- **Type**: `Literal["manual", "top_k"]`
- **Description**: How to select models for consensus clustering
- **Options**:
  - `"manual"`: Use `model_names` parameter (explicit list)
  - `"top_k"`: Select top k models from ranking by win_rate (default)
- **Priority**: Manual override > Second-round reasoning > Top-k
- **Example**: `"top_k"` (default), `"manual"` (explicit selection)

#### `model_names` (optional, default: `None`)
- **Type**: `Sequence[str]` or `None`
- **Description**: Manually specified list of model names for consensus clustering
- **Required When**: `selection_mode="manual"`
- **Usage**: Overrides all other selection methods (highest priority)
- **Example**: `["GraphST", "STAGATE", "SpaGCN", "BayesSpace"]`

#### `use_second_round` (optional, default: `False`)
- **Type**: `bool`
- **Description**: Enable second-round reasoning to filter out "poison pill" models
- **When `True`**:
  - Reads second-round JSON file (if present) from `output_dir/pairwise/`
  - Extracts `final_model_ids` from the JSON
  - Uses those models for consensus (overrides top-k, but not manual)
- **When `False`**: Uses default top-k selection (or manual if specified)
- **Priority**: Manual > Second-round > Top-k
- **Note**: Second-round reasoning is performed by the separate `second_round.py` module. The JSON file must be generated separately or by running `run_second_round_reasoning()` before calling `l_star()`.
- **Example**: `False` (default), `True` (enable filtering)

### Cluster Number Arguments

#### `k_mode` (optional, default: `"auto"`)
- **Type**: `Literal["fixed", "auto"]`
- **Description**: How to determine the number of clusters
- **Options**:
  - `"fixed"`: Use `fixed_k` parameter
  - `"auto"`: Automatically determine from selected models (default)
- **Example**: `"auto"` (default), `"fixed"` (specify k)

#### `fixed_k` (optional, default: `None`)
- **Type**: `int` or `None`
- **Description**: Fixed number of clusters for consensus
- **Required When**: `k_mode="fixed"`
- **Example**: `7`, `10`, `15`

### Additional Arguments (`**kwargs`)

These arguments are passed through to underlying functions:

#### `api_key` (optional)
- **Type**: `str` or `None`
- **Description**: OpenAI API key
- **Alternative**: Set `OPENAI_API_KEY` environment variable
- **Example**: `"sk-..."`

#### `api_base` (optional)
- **Type**: `str` or `None`
- **Description**: Custom API base URL (for non-OpenAI endpoints)
- **Default**: OpenAI's default base URL
- **Example**: `"https://api.openai.com/v1"`

#### `model_name` (optional, default: `"gpt-5.1-2025-11-13"`)
- **Type**: `str`
- **Description**: LLM model name to use for pairwise comparisons
- **Example**: `"gpt-5.1-2025-11-13"` (default)

#### `pairwise_temperature` (optional, default: `1.0`)
- **Type**: `float`
- **Description**: Temperature for pairwise comparison LLM calls
- **Range**: Typically 0.0-2.0
- **Example**: `1.0` (default), `0.7` (more deterministic)

#### `pairwise_reasoning_effort` (optional, default: `"medium"`)
- **Type**: `Literal["minimal", "medium", "high"]`
- **Description**: Reasoning effort level for pairwise comparisons
- **Example**: `"medium"` (default), `"high"` (more thorough)

#### `second_round_temperature` (optional, default: `1.0`)
- **Type**: `float`
- **Description**: Temperature for second-round reasoning LLM calls (used by `second_round.py` module)
- **Note**: This parameter is used by the separate `run_second_round_reasoning()` function, not by `run_pairwise_comparisons()`
- **Example**: `1.0` (default)

#### `second_round_reasoning_effort` (optional, default: `"high"`)
- **Type**: `Literal["minimal", "medium", "high"]`
- **Description**: Reasoning effort level for second-round reasoning (used by `second_round.py` module)
- **Note**: This parameter is used by the separate `run_second_round_reasoning()` function, not by `run_pairwise_comparisons()`
- **Example**: `"high"` (default, more careful analysis)

#### `k_method` (optional, default: `"median_from_models"`)
- **Type**: `Literal["median_from_models", "mode_from_models", "silhouette", "gap_statistic"]`
- **Description**: Method to determine optimal number of clusters (when `k_mode="auto"`)
- **Options**:
  - `"median_from_models"`: Use median of cluster counts from selected models (default)
  - `"mode_from_models"`: Use most common cluster count from selected models
  - `"silhouette"`: Maximize silhouette score across k values
  - `"gap_statistic"`: Use gap statistic to find optimal k
- **Example**: `"median_from_models"` (default)

#### `k_range` (optional, default: `range(2, 16)`)
- **Type**: `range` or `list`
- **Description**: Valid range of k values for auto-determination
- **Example**: `range(2, 16)` (default), `range(3, 20)`

#### `ground_truth_col` (optional, default: `None`)
- **Type**: `str` or `None`
- **Description**: Name of ground truth column for ARI evaluation
- **Note**: Only used for evaluation, not for clustering
- **Example**: `"Ground"`, `"ground_truth"`

#### `he_basename` (optional, default: `"he.png"`)
- **Type**: `str`
- **Description**: Basename of H&E image (with or without extension)
- **Supported**: If no extension, searches for `.png`, `.jpg`, `.jpeg`
- **Example**: `"he.png"` (default), `"he"` (searches multiple formats), `"he.jpg"`

#### `skip_pairwise` (optional, default: `False`)
- **Type**: `bool`
- **Description**: Skip pairwise comparisons and reuse existing ranking CSV
- **Example**: `False` (default), `True` (skip if ranking exists)

#### `force_rerun` (optional, default: `False`)
- **Type**: `bool`
- **Description**: Ignore cache and recompute all pairwise comparisons
- **Example**: `False` (default), `True` (force recomputation)

---

## Pairwise Comparisons: `run_pairwise_comparisons()`

Runs LLM-based pairwise comparisons and generates model ranking.

### Function Signature

```python
def run_pairwise_comparisons(
    image_dir: Union[str, Path],
    dataset_name: str,
    *,
    reps: int = 5,
    top_k: int = 5,
    top_k_mode: Literal["fixed", "elbow"] = "fixed",
    he_basename: str = "he.png",
    skip_pairwise: bool = False,
    simple_mode: bool = True,
    output_dir: Union[str, Path] = "lstar_output",
    force_rerun: bool = False,
    model_name: str = "gpt-5.1-2025-11-13",
    pairwise_temperature: float = 1.0,
    pairwise_reasoning_effort: Literal["minimal", "medium", "high"] = "medium",
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> Tuple[pd.DataFrame, Path, Path]
```

### Key Arguments

Most arguments are the same as `l_star()`. See above for details.

**Note**: This function only handles first-round pairwise comparisons. Second-round reasoning is handled by the separate `second_round.py` module.

**Returns:**
- `ranking_df`: DataFrame with model rankings
- `pairwise_dir`: Path to pairwise results directory
- `ranking_csv_path`: Path to ranking CSV file

---

## Second-Round Reasoning: `run_second_round_reasoning()`

Performs second-round "poison pill" screening to filter out problematic models. This function is in the `second_round.py` module.

### Function Signature

```python
def run_second_round_reasoning(
    image_dir: Path,
    pairwise_dir: Path,
    ranking_csv_path: Path,
    he_basename: str,
    top_k: int,
    model_name: str,
    second_round_temperature: float,
    second_round_reasoning_effort: str,
    api_key: Optional[str],
    api_base: Optional[str],
    dataset_name: str,
    output_dir: Path,
) -> List[str]
```

### Key Arguments

#### `image_dir` (required)
- **Type**: `Path`
- **Description**: Directory containing H&E and model output images

#### `pairwise_dir` (required)
- **Type**: `Path`
- **Description**: Directory containing pairwise comparison results (from `run_pairwise_comparisons()`)

#### `ranking_csv_path` (required)
- **Type**: `Path`
- **Description**: Path to ranking CSV file (from `run_pairwise_comparisons()`)

#### `he_basename` (required)
- **Type**: `str`
- **Description**: Basename of H&E image

#### `top_k` (required)
- **Type**: `int`
- **Description**: Number of top models to consider for second-round screening

#### `model_name` (required)
- **Type**: `str`
- **Description**: LLM model name to use

#### `second_round_temperature` (required)
- **Type**: `float`
- **Description**: Temperature for second-round LLM calls

#### `second_round_reasoning_effort` (required)
- **Type**: `str`
- **Description**: Reasoning effort level ("minimal", "medium", or "high")

#### `api_key` (optional)
- **Type**: `str` or `None`
- **Description**: OpenAI API key

#### `api_base` (optional)
- **Type**: `str` or `None`
- **Description**: Custom API base URL

#### `dataset_name` (required)
- **Type**: `str`
- **Description**: Dataset name for context in prompts

#### `output_dir` (required)
- **Type**: `Path`
- **Description**: Output directory where JSON file will be written

**Returns:**
- `List[str]`: List of selected model IDs (after filtering)

**Output File:**
- Writes `{dataset_name}_second_round_reasoning.json` (or `second_round_reasoning.json` if dataset_name is empty) to `pairwise_dir`
- JSON contains `final_model_ids` which can be read by `read_second_round_results()`

---

## Consensus Clustering: `run_consensus_clustering()`

Performs consensus clustering on selected models.

### Function Signature

```python
def run_consensus_clustering(
    ranking_csv: Union[str, Path],
    *,
    assignments_csv: Union[str, Path, None] = None,
    id_col: Optional[str] = None,
    use_separate_csvs: bool = False,
    assignments_dir: Union[str, Path, None] = None,
    assignment_csv_list: Optional[Sequence[Union[str, Path]]] = None,
    output_csv: Union[str, Path, None] = None,
    output_dir: Union[str, Path] = "lstar_output",
    model_names: Sequence[str] | None = None,
    top_k: Optional[int] = 5,
    selection_mode: Literal["manual", "top_k"] = "manual",
    k_method: Literal["median_from_models", "mode_from_models", "silhouette", "gap_statistic"] = "median_from_models",
    k_range: range = range(2, 16),
    k_mode: Literal["fixed", "auto"] = "auto",
    fixed_k: Optional[int] = None,
    ground_truth_col: Optional[str] = None,
    random_state: Optional[int] = 0,
) -> pd.DataFrame
```

### Key Arguments

Most arguments are the same as `l_star()`. Additional arguments:

#### `ranking_csv` (required)
- **Type**: `str` or `Path`
- **Description**: Path to ranking CSV file (output from pairwise comparisons)
- **Example**: `"lstar_output/ranking.csv"`

#### `output_csv` (optional, default: `None`)
- **Type**: `str` or `Path` or `None`
- **Description**: Explicit output CSV path for consensus results
- **Default**: `output_dir / "L_STAR_consensus.csv"`
- **Example**: `"results/consensus.csv"`

---

## Default Settings by Mode

### Default Mode (Combined CSV with Fuzzy Matching)

**When**: `use_separate_csvs=False` (default)

| Argument | Default Value |
|----------|---------------|
| `assignments_csv` | `None` (must be provided) |
| `id_col` | `None` (must be provided) |
| `use_separate_csvs` | `False` |
| `selection_mode` | `"top_k"` |
| `use_second_round` | `False` |
| `k_mode` | `"auto"` |
| `k_method` | `"median_from_models"` |
| `top_k` | `5` |
| `reps` | `5` |
| `simple_mode` | `True` |

**Model Selection Priority**:
1. Manual (`selection_mode="manual"` + `model_names`)
2. Second-round reasoning (if `use_second_round=True` and JSON exists)
3. Top-k (default)

### Legacy Mode (Separate CSV Files)

**When**: `use_separate_csvs=True`

| Argument | Default Value |
|----------|---------------|
| `assignments_dir` | `None` (must be provided) |
| `assignment_csv_list` | `None` (alternative to `assignments_dir`) |
| `use_separate_csvs` | `True` |
| `selection_mode` | `"top_k"` |
| `use_second_round` | `False` |
| `k_mode` | `"auto"` |
| `k_method` | `"median_from_models"` |

**Note**: Fuzzy matching is NOT used in legacy mode (exact name matching).

### Second-Round Reasoning Mode

**When**: `use_second_round=True`

| Argument | Default Value |
|----------|---------------|
| `use_second_round` | `True` |
| `second_round_temperature` | `1.0` |
| `second_round_reasoning_effort` | `"high"` |
| Model Selection | Uses `final_model_ids` from second-round JSON |

**Output File**: `{dataset_name}_second_round_reasoning.json` in `output_dir/pairwise/`

**Note**: Second-round reasoning must be run separately using `run_second_round_reasoning()` before calling `l_star()` with `use_second_round=True`, or the JSON file must already exist.

---

## Examples

### Example 1: Minimal Runnable Example (Combined CSV)

The simplest way to run L-STAR with combined assignments CSV:

```python
import lstar

df = lstar.l_star(
    image_dir="data/DLPFC/images",
    dataset_name="DLPFC (from 10X Visium Human Brain)",
    assignments_csv="data/DLPFC/combined_assignments.csv",
    id_col="spot_id",
    api_key="your-openai-api-key"
)

print(df.head())
# Output:
#    spot_id  L-STAR
# 0  spot_1       0
# 1  spot_2       1
# 2  spot_3       0
# ...
```

**What happens**:
- Uses default top-k selection (`selection_mode="top_k"`, `top_k=5`)
- Automatically determines k using median method
- No second-round reasoning
- Fuzzy name matching enabled

### Example 2: With Assignment CSV List (Legacy Mode)

Using separate CSV files per model:

```python
import lstar

df = lstar.l_star(
    image_dir="data/DLPFC/images",
    dataset_name="DLPFC (from 10X Visium Human Brain)",
    use_separate_csvs=True,
    assignments_dir="data/DLPFC/assignments",
    selection_mode="top_k",
    top_k=5,
    k_mode="auto",
    api_key="your-openai-api-key"
)

print(df.head())
```

**File structure**:
```
data/DLPFC/assignments/
  ├── GraphST.csv
  ├── STAGATE.csv
  ├── SpaGCN.csv
  ├── BayesSpace.csv
  └── ...
```

**What happens**:
- Uses legacy mode with separate CSV files
- Top-k selection from ranking
- Exact name matching (no fuzzy matching)

### Example 3: With Second-Round Reasoning

Enabling second-round reasoning to filter out problematic models:

**Step 1: Run pairwise comparisons and second-round reasoning**

```python
import lstar
from lstar.second_round import run_second_round_reasoning
from pathlib import Path

# Step 1a: Run pairwise comparisons
ranking_df, pairwise_dir, ranking_csv = lstar.run_pairwise_comparisons(
    image_dir="data/DLPFC/images",
    dataset_name="DLPFC (from 10X Visium Human Brain)",
    output_dir="results/DLPFC",
    top_k=5,
    api_key="your-api-key"
)

# Step 1b: Run second-round reasoning
final_model_ids = run_second_round_reasoning(
    image_dir=Path("data/DLPFC/images"),
    pairwise_dir=pairwise_dir,
    ranking_csv_path=ranking_csv,
    he_basename="he.png",
    top_k=5,
    model_name="gpt-5.1-2025-11-13",
    second_round_temperature=1.0,
    second_round_reasoning_effort="high",
    api_key="your-api-key",
    api_base=None,
    dataset_name="DLPFC (from 10X Visium Human Brain)",
    output_dir=Path("results/DLPFC"),
)
```

**Step 2: Run consensus with second-round results**

```python
# Step 2: Run consensus clustering (reads JSON automatically)
df = lstar.l_star(
    image_dir="data/DLPFC/images",
    dataset_name="DLPFC (from 10X Visium Human Brain)",
    assignments_csv="data/DLPFC/combined_assignments.csv",
    id_col="spot_id",
    use_second_round=True,  # Enable second-round reasoning
    api_key="your-api-key"
)

print(df.head())
```

**What happens**:
1. Runs pairwise comparisons (5 reps)
2. Selects top 5 models from ranking
3. Runs second-round reasoning on those 5 models
4. Reads `*_second_round_reasoning.json` file
5. Uses `final_model_ids` from JSON for consensus (may be fewer than 5)
6. Performs consensus clustering on filtered models

**Output file**: `results/DLPFC/pairwise/DLPFC_second_round_reasoning.json` (or similar)

### Example 4: Full Argument Example

Complete example with all major arguments:

```python
import lstar

df = lstar.l_star(
    # Required
    image_dir="data/DLPFC/images",
    dataset_name="DLPFC (from 10X Visium Human Brain)",
    
    # Assignment data (combined CSV mode)
    assignments_csv="data/DLPFC/combined_assignments.csv",
    id_col="spot_id",
    use_separate_csvs=False,  # Default
    
    # Output
    output_dir="results/DLPFC",
    
    # Pairwise comparison settings
    simple_mode=True,
    reps=5,
    top_k=5,
    top_k_mode="fixed",
    
    # Model selection
    selection_mode="top_k",  # Default
    use_second_round=False,  # Default
    # model_names=None,  # Not needed for top-k
    
    # Cluster number determination
    k_mode="auto",
    k_method="median_from_models",  # Options: median_from_models, mode_from_models, silhouette, gap_statistic
    k_range=range(2, 16),
    # fixed_k=None,  # Not needed for auto mode
    
    # LLM settings
    model_name="gpt-5.1-2025-11-13",
    pairwise_temperature=1.0,
    pairwise_reasoning_effort="medium",
    
    # Other
    ground_truth_col="Ground",  # For ARI evaluation only
    he_basename="he.png",
    
    # API
    api_key="your-openai-api-key"
)

print(f"Consensus clustering complete. Output: {df.shape[0]} spots, {df['L-STAR'].nunique()} clusters")
```

### Example 5: Manual Model Selection

Explicitly specify which models to use:

```python
import lstar

df = lstar.l_star(
    image_dir="data/DLPFC/images",
    dataset_name="DLPFC (from 10X Visium Human Brain)",
    assignments_csv="data/DLPFC/combined_assignments.csv",
    id_col="spot_id",
    selection_mode="manual",
    model_names=["GraphST", "STAGATE", "SpaGCN", "BayesSpace", "SEDR"],
    k_mode="fixed",
    fixed_k=7,
    api_key="your-openai-api-key"
)
```

**What happens**:
- Manual selection takes highest priority
- Uses exactly the 5 specified models
- Fixed k=7 clusters
- Ignores ranking and second-round results

### Example 6: Second-Round with Manual Override

Even with second-round enabled, manual selection takes priority:

```python
import lstar

df = lstar.l_star(
    image_dir="data/DLPFC/images",
    dataset_name="DLPFC (from 10X Visium Human Brain)",
    assignments_csv="data/DLPFC/combined_assignments.csv",
    id_col="spot_id",
    selection_mode="manual",
    model_names=["GraphST", "STAGATE", "SpaGCN"],  # Explicit list
    use_second_round=True,  # Will be ignored due to manual mode
    k_mode="auto",
    api_key="your-openai-api-key"
)
```

**What happens**:
- Manual selection (Priority 1) takes precedence
- Second-round reasoning JSON (if present) is ignored
- Uses the 3 manually specified models

### Example 7: Different k Determination Methods

Using different methods to determine the number of clusters:

```python
import lstar

# Method 1: Median (default)
df1 = lstar.l_star(
    image_dir="data/DLPFC/images",
    dataset_name="DLPFC (from 10X Visium Human Brain)",
    assignments_csv="data/DLPFC/combined_assignments.csv",
    id_col="spot_id",
    k_method="median_from_models",
    api_key="your-api-key"
)

# Method 2: Mode (most common)
df2 = lstar.l_star(
    image_dir="data/DLPFC/images",
    dataset_name="DLPFC (from 10X Visium Human Brain)",
    assignments_csv="data/DLPFC/combined_assignments.csv",
    id_col="spot_id",
    k_method="mode_from_models",
    api_key="your-api-key"
)

# Method 3: Silhouette score
df3 = lstar.l_star(
    image_dir="data/DLPFC/images",
    dataset_name="DLPFC (from 10X Visium Human Brain)",
    assignments_csv="data/DLPFC/combined_assignments.csv",
    id_col="spot_id",
    k_method="silhouette",
    k_range=range(3, 12),
    api_key="your-api-key"
)

# Method 4: Gap statistic
df4 = lstar.l_star(
    image_dir="data/DLPFC/images",
    dataset_name="DLPFC (from 10X Visium Human Brain)",
    assignments_csv="data/DLPFC/combined_assignments.csv",
    id_col="spot_id",
    k_method="gap_statistic",
    k_range=range(3, 12),
    api_key="your-api-key"
)
```

### Example 8: Step-by-Step Execution

Running pairwise and consensus separately:

```python
import lstar

# Step 1: Pairwise comparisons
ranking_df, pairwise_dir, ranking_csv = lstar.run_pairwise_comparisons(
    image_dir="data/DLPFC/images",
    dataset_name="DLPFC (from 10X Visium Human Brain)",
    output_dir="results/DLPFC",
    reps=5,
    api_key="your-api-key"
)

print(ranking_df.head())
# Output: Model ranking with win rates

# Step 2: Consensus clustering
consensus_df = lstar.run_consensus_clustering(
    ranking_csv=ranking_csv,
    assignments_csv="data/DLPFC/combined_assignments.csv",
    id_col="spot_id",
    selection_mode="top_k",
    top_k=5,
    k_mode="auto",
    output_dir="results/DLPFC"
)

print(consensus_df.head())
```

### Example 9: Using Environment Variable for API Key

```python
import os
import lstar

# Set API key as environment variable
os.environ["OPENAI_API_KEY"] = "your-api-key"

# No need to pass api_key parameter
df = lstar.l_star(
    image_dir="data/DLPFC/images",
    dataset_name="DLPFC (from 10X Visium Human Brain)",
    assignments_csv="data/DLPFC/combined_assignments.csv",
    id_col="spot_id"
)
```

### Example 10: Custom H&E Image Name

```python
import lstar

df = lstar.l_star(
    image_dir="data/DLPFC/images",
    dataset_name="DLPFC (from 10X Visium Human Brain)",
    assignments_csv="data/DLPFC/combined_assignments.csv",
    id_col="spot_id",
    he_basename="he_reference.jpg",  # Custom name with extension
    api_key="your-api-key"
)

# Or without extension (searches multiple formats)
df = lstar.l_star(
    image_dir="data/DLPFC/images",
    dataset_name="DLPFC (from 10X Visium Human Brain)",
    assignments_csv="data/DLPFC/combined_assignments.csv",
    id_col="spot_id",
    he_basename="he",  # Will search for he.png, he.jpg, he.jpeg
    api_key="your-api-key"
)
```

---

## Model Selection Priority Summary

The model selection follows this priority order:

1. **Manual Override** (Highest Priority)
   - If `selection_mode="manual"` AND `model_names` is provided and non-empty
   - Uses the user-provided model list
   - Example: `selection_mode="manual"`, `model_names=["Model1", "Model2"]`

2. **Second-Round Reasoning** (Medium Priority)
   - If `use_second_round=True` AND second-round JSON file exists with valid `final_model_ids`
   - Uses models selected by second-round reasoning
   - Example: `use_second_round=True`

3. **Top-K Selection** (Default, Lowest Priority)
   - If no manual override and no usable second-round results
   - Uses `selection_mode="top_k"` with `top_k` value
   - Example: Default behavior, or `selection_mode="top_k"`, `top_k=5`

---

## Common Patterns

### Pattern 1: Quick Analysis (Minimal Arguments)
```python
df = lstar.l_star(
    image_dir="images",
    dataset_name="Your Dataset Name",
    assignments_csv="assignments.csv",
    id_col="spot_id",
    api_key="key"
)
```

### Pattern 2: Robust Analysis (More Repetitions)
```python
df = lstar.l_star(
    image_dir="images",
    dataset_name="Your Dataset Name",
    assignments_csv="assignments.csv",
    id_col="spot_id",
    reps=10,  # More repetitions
    top_k=7,  # More models
    api_key="key"
)
```

### Pattern 3: Careful Filtering (Second-Round)
```python
# First run second-round reasoning separately
from lstar.second_round import run_second_round_reasoning
from pathlib import Path

# ... run second-round reasoning ...

# Then use it in l_star
df = lstar.l_star(
    image_dir="images",
    dataset_name="Your Dataset Name",
    assignments_csv="assignments.csv",
    id_col="spot_id",
    use_second_round=True,  # Filter problematic models
    top_k=8,  # Start with more, filter down
    api_key="key"
)
```

### Pattern 4: Fixed Configuration
```python
df = lstar.l_star(
    image_dir="images",
    dataset_name="Your Dataset Name",
    assignments_csv="assignments.csv",
    id_col="spot_id",
    selection_mode="manual",
    model_names=["Model1", "Model2", "Model3"],
    k_mode="fixed",
    fixed_k=7,  # Known number of clusters
    api_key="key"
)
```

---

## Troubleshooting

### Common Issues

1. **"assignments_csv must be provided"**
   - Solution: Provide `assignments_csv` and `id_col`, or set `use_separate_csvs=True`

2. **"id_col must be provided"**
   - Solution: Specify the ID column name when using combined CSV mode

3. **"Could not match ranking model"**
   - Solution: Check that model names in ranking CSV match column names in assignments CSV (fuzzy matching handles variations)

4. **"No second-round reasoning JSON found"**
   - Solution: Ensure `run_second_round_reasoning()` has been run and generated the JSON file, or that the JSON file exists in `output_dir/pairwise/`

5. **"Need at least 2 model images"**
   - Solution: Ensure image directory contains at least 2 model output images

---

## See Also

- [README.md](README.md) - General package overview
- [PACKAGE_STRUCTURE.md](PACKAGE_STRUCTURE.md) - Package structure and organization
