# L-STAR: LLM-Guided Spatial Domain Detection

L-STAR is a Python package for LLM-based model comparison, consensus clustering, and post-consensus biological domain annotation for spatial transcriptomics data. It supports performance-first pairwise comparisons and cost-first simultaneous all-wise comparisons, aggregates the top-performing methods using Evidence Accumulation Clustering (EAC), and annotates the resulting consensus domains from molecular, spatial, and dataset context.

## Overview

The L-STAR pipeline includes comparison, consensus construction, and an optional post-consensus annotation stage:

### **Step 0. Spatial Domain Visualization Reconstruction**

Before any model comparison, spatial domain visualizations are **reconstructed from raw clustering assignments** to ensure fair, color-consistent, and spatially interpretable inputs for downstream evaluation.

Given spatial coordinates and per-spot domain labels from each method, L-STAR regenerates domain images using the **Palo** color optimization strategy, which assigns visually distinct colors to spatially adjacent domains. This avoids palette-induced bias and ensures that visual differences reflect structural discrepancies rather than arbitrary color choices.

This step is fully automated, supports multiple coordinate formats, and produces standardized PNG images for all methods (and optional H&E references) that are later consumed by the LLM comparison stage.
(Related code for this reconstruction step is provided in the repository. )

### **Step 1. Visual Comparisons via LLMs**

L-STAR provides two comparison modes. `mode="performance"` is the default and performs exhaustive pairwise comparisons. `mode="cost"` includes all candidate method images in one request per repetition and returns a strict ranking. Both modes optionally include the histology reference.

Repeated comparison results are aggregated into a ranking CSV that serves as the basis for downstream model selection.

### **Step 2. Top-Performing Model Selection**

Based on the aggregated comparison outcomes, L-STAR selects a subset of high-performing methods for consensus construction.

This subset can be:

* **Manually specified**, or
* **Automatically determined**, by choosing the top-k methods according to the comparison score and including all methods tied at the cutoff.

This step filters out systematically underperforming methods while retaining complementary high-quality solutions, balancing robustness and diversity for the consensus stage.

### **Step 3. Consensus Clustering via Evidence Accumulation**

The selected top-performing methods are integrated using **Evidence Accumulation Clustering (EAC)**. Pairwise co-assignment frequencies across methods are accumulated into a consensus similarity matrix, which is then clustered to produce a final spatial domain assignment.

The resulting consensus labels are reported as **L-STAR**, representing an ensemble spatial domain detection that leverages both human-interpretable visual judgment (via LLMs) and classical clustering theory. This consensus is subsequently evaluated against ground truth using standard metrics such as ARI and AMI.

### **Step 4. Post-Consensus Domain Annotation**

`annotate_domain()` names each L-STAR consensus domain from its ranked marker genes in a single dataset-level LLM call, then maps that name back to every aligned spot or cell. It reads expression from H5AD or 10x H5/HDF5, automatically uses the methods recorded in `lstar_run_manifest.json`, resolves feature identities, and names every evaluable domain together in one request — there is no separate candidate-generation or reconciliation stage, and no annotation-confidence output. This follows the single-call, marker-list-to-name pattern validated for single-cell cluster annotation by [GPTCelltype](https://doi.org/10.1038/s41592-024-02235-4) (Hou & Ji, *Nature Methods*, 2024). A domain with no usable positive-marker evidence is named `"Unknown"` without spending a call on it, and if the model's response never validates after repair attempts, affected domains are named `"Unknown"` deterministically so a run always completes.

Expression preprocessing and marker computation are delegated to [scanpy](https://scanpy.readthedocs.io/). Wherever scanpy exposes a default we adopt it rather than tuning per dataset; where scanpy has no default, or its default cannot apply, the value and its source are stated. Normalization is `scanpy.pp.normalize_total` followed by `scanpy.pp.log1p`, both at scanpy's defaults — `target_sum` is not passed at all, so each observation is scaled to the dataset's median total count rather than a fixed constant. Domain markers come from `scanpy.tl.rank_genes_groups` with **every** argument left unset, including `method`, which scanpy resolves to a t-test; markers are the top of the ranking scanpy itself returns, truncated to the top 25 per domain (configurable via `annotate_domain`'s `max_positive_markers`). Selection keeps positive (enriched) genes that pass the two prevalence criteria of `scanpy.tl.filter_rank_genes_groups` at that function's own default values and strict comparisons (`pct_in > 0.25`, `pct_out < 0.5`). Its third criterion, `min_fold_change`, is deliberately not adopted: a greater-than-twofold floor suits discrete cell-type clusters but starves spatial domains, whose neighbours differ by gradients — on cortical data it reduced a genuine layer-6 domain to three incoherent genes by rejecting TBR1 at 0.87. No adjusted-p-value criterion is applied, matching that function. `annotate_domain`'s `marker_test_method` selects a different test (`"wilcoxon"`, for instance) when comparing against scanpy's default is the point.

Two things are deliberately **not** removed, because removing them would impose a judgement scanpy does not make. There is no hand-curated "technical gene" list — deciding what counts as a housekeeping gene has no principled cutoff — and mitochondrial genes are not dropped from the matrix, since scanpy provides no such step (it computes a per-cell mitochondrial *fraction* for QC and leaves the genes in place). A gene expressed almost everywhere fails `max_pct_out` on its own statistics, so it is excluded by measurement rather than by symbol; for the residue that still slips through, the judgement sits where a human annotator applies it: the naming prompt instructs the model to read `MT-`/`mt-` and ribosomal genes as cell-state or capture-quality signals rather than evidence of regional identity, and to answer `Unknown` if too little remains once they are set aside. Quality control is the one place with no scanpy default at all — `sc.pp.filter_cells` and `sc.pp.filter_genes` leave every bound unset — so thresholds are stated explicitly: transcriptome-wide assays (>2000 features) require ≥200 detected genes and <20% mitochondrial counts, following the scanpy PBMC3k tutorial, while targeted panels (≤2000 features), on which those cutoffs would remove nearly every observation, require ≥50 counts and ≥20 detected genes with no mitochondrial gate. On both platforms a gene must additionally be detected in at least 3 QC-passed observations to enter marker computation, the same tutorial's `sc.pp.filter_genes` bound. QC restricts marker computation only — every aligned observation still receives a domain label.

The user declares only whether the platform is `"spot"` or `"cell"` level. Dataset, species, and tissue context guide interpretation, and an optional `notes` field carries structural context the other fields don't capture (for example, that a tissue is known to have a layered organization) — used only to choose naming conventions and break ties among names the markers already support, never to override marker evidence. The naming prompt's granularity and terminology rules follow standard single-cell/spatial annotation practice (see the [sc-best-practices annotation chapter](https://www.sc-best-practices.org/cellular_structure/annotation.html) and Clarke et al., *Nature Protocols*, 2021) and, for well-atlased tissue such as brain or embryo, atlas-style regional nomenclature (for example, the [Allen Brain Atlas / Common Coordinate Framework](https://atlas.brain-map.org/)) and standard anatomical ontology usage (for example, [UBERON](https://obophenotype.github.io/uberon/)). When available, a minimal L-STAR domain map and H&E image are included as visual background. Raw expression values are never sent to the LLM.

## Installation

Install from source (this Repo):

```bash
git clone https://github.com/Williamzcy0929/L-STAR.git
cd L-STAR
pip install -e .
```
OR
```
pip install "git+https://github.com/Williamzcy0929/L-STAR.git"
```

### R Dependencies

When generating images from CSVs with `use_palo=True`, L-STAR uses **R scripts** bundled inside the Python package:

- `run_palo.R`: computes Palo-optimized palettes
- `plot_spatial_with_palo.R`: renders per-method spatial PNGs with **ggplot2** (no legend)

Required R packages:

- [Palo](https://github.com/Winnie09/Palo)
- `ggplot2`
- `RColorBrewer`

Install example:
```r
install.packages(c("ggplot2", "RColorBrewer"), repos = "https://cloud.r-project.org")
remotes::install_github("Winnie09/Palo", repos = "https://cloud.r-project.org")
```

If R/Palo/ggplot dependencies are unavailable at runtime, L-STAR automatically falls back to matplotlib/default color rendering so the pipeline can still run.

For source checkouts, `scripts/install_palo.R` and `scripts/test_palo.R` are still available for setup/testing convenience.

## Quick Start

L-STAR supports two modes for spatial visualization:

### Default Mode: Generate Images from CSV Files

By default, L-STAR can generate spatial visualization images internally from spatial locations and domain assignments. With `use_palo=True`, images are rendered by R/ggplot2 using Palo-optimized palettes (one PNG per method, legend disabled):

```python
import lstar

# Generate images internally using Palo for color optimization
df = lstar.l_star(
    dataset_name="DLPFC (from 10X Visium Human Brain)",
    spatial_locations_csv="path/to/spatial_locations.csv",  # CSV with spot_id, x, y columns
    assignments_csv="path/to/assignments.csv",              # CSV with spot_id and method columns
    id_col="spot_id",                                       # ID column name
    mode="performance",                                     # Default; use "cost" for all-wise comparison
    use_palo=True,                                          # Use Palo for color optimization (default: True)
    k_mode="fixed",                                         # Required for fixed_k to take effect
    fixed_k=7,
    api_key="your-openai-api-key"
)

print(df.head())
# Output includes 'L-STAR' column with consensus cluster labels
# Generated images are saved to output_dir/generated_images/
```

### Image Mode: Using Pre-generated Images

```python
import lstar

# Run the full L-STAR pipeline with pre-generated images
df = lstar.l_star(
    image_dir="path/to/images",           # Directory with model output images and the optional H&E image
    dataset_name="DLPFC (from 10X Visium Human Brain)",
    assignments_csv="path/to/assignments.csv",  # Combined assignments CSV
    id_col="spot_id",                     # ID column name
    k_mode="fixed",                       # Required for fixed_k to take effect
    fixed_k=7,                            # Fixed number of clusters
    api_key="your-openai-api-key"         # Or set OPENAI_API_KEY env var
)

print(df.head())
# Output includes 'L-STAR' column with consensus cluster labels
```

### Annotate the L-STAR Consensus Domains

Run annotation against the same `output_dir` used by `l_star()`. The run manifest supplies the consensus-selected method columns, assignment path, ID column, and consensus path automatically.

```python
import lstar

annotations = lstar.annotate_domain(
    expression_h5="path/to/filtered_feature_bc_matrix.h5",
    output_dir="lstar_output",
    dataset_context="10x Visium human dorsolateral prefrontal cortex",
    species="Homo sapiens",
    tissue="dorsolateral prefrontal cortex",
    notes="DLPFC has a layered cortical structure (L1-L6 + white matter)",  # Optional
    sampling_level="spot",
    he_image_path="path/to/he_image.png",  # Optional
    api_key="your-openai-api-key",         # Or set OPENAI_API_KEY
)

print(annotations.head())
# Columns: spot_id, L-STAR, domain_name
```

For H5AD input, replace `expression_h5=...` with `expression_h5ad=...`. Expression CSV files are not accepted. By default, observation IDs must match the consensus exactly; use `allow_partial_observations=True` only when an explicit intersection is intended.

**Key Points:**
- When `spatial_locations_csv` and `assignments_csv` are provided, images are generated internally
- With `use_palo=True`, L-STAR uses bundled R scripts (`run_palo.R` + `plot_spatial_with_palo.R`) for color optimization and rendering
- Output remains one image per method, with no legend, consistent dimensions, and `coord_equal`-style geometry
- To use pre-generated images instead, provide `image_dir` and omit `spatial_locations_csv`
- Generated images are saved to `output_dir/generated_images/` with filenames matching method names

## Input Format

L-STAR supports two input modes:

### Default Mode: CSV Files for Image Generation

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
- L-STAR internally generates one spatial visualization image per method column
- With `use_palo=True`, palette optimization and plotting run through bundled R scripts (Palo + ggplot2)
- Generated images are saved to `output_dir/generated_images/` with filenames matching method names (e.g., `GraphST.png`, `SpaGCN.png`)
- If `he_image_path` is provided, the H&E image is copied to the generated images directory

**Palo Color Optimization:**
- Palo optimizes colors based on spatial adjacency, ensuring neighboring clusters are visually distinct
- Set `use_palo=False` to disable Palo and use matplotlib/default color palettes
- If Palo/R/ggplot dependencies are unavailable, L-STAR automatically falls back to matplotlib rendering

### Image Mode: Pre-generated Images

The `image_dir` should contain:
- `he.png` (or custom name with extensions .png, .jpg, .jpeg, or .pdf): H&E reference image (optional)
- `Model1.png`, `Model2.jpg`, etc.: Clustering visualization images for each model
  - Supported formats: `.png`, `.jpg`, `.jpeg`, `.pdf`
  - If multiple formats exist for the same model name, PNG is preferred over JPG/JPEG, which is preferred over PDF

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
    image_dir: str | Path | None = None,
    dataset_name: str | None = None,
    *,
    spatial_locations_csv: str | Path | None = None,
    assignments_csv: str | Path | None = None,
    id_col: str | None = None,
    use_separate_csvs: bool = False,
    assignments_dir: str | Path | None = None,
    assignment_csv_list: Sequence[str | Path] | None = None,
    output_dir: str | Path = "lstar_output",
    mode: Literal["performance", "cost"] = "performance",
    simple_mode: bool = True,
    reps: int = 5,
    top_k: int = 5,
    top_k_mode: Literal["fixed", "elbow"] = "fixed",
    selection_mode: Literal["manual", "top_k"] = "top_k",
    model_names: Sequence[str] | None = None,
    k_mode: Literal["fixed", "auto"] = "auto",
    fixed_k: Optional[int] = None,
    use_second_round: bool = False,
    use_palo: bool = True,
    he_image_path: str | Path | None = None,
    **kwargs
) -> pd.DataFrame
```

**Key Parameters:**
- `dataset_name`: Background information about the dataset. Required in practice — despite the `None` default in the signature, `l_star()` raises `ValueError` when it is missing
- `image_dir`: Directory with pre-generated images (use this OR provide CSV files for image generation). It is the first positional parameter, so pass `dataset_name` by keyword
- `spatial_locations_csv`: CSV with spatial coordinates (required for image generation mode)
- `assignments_csv`: CSV with assignments where column names are method names (required in the default, combined-CSV mode)
- `id_col`: Name of the ID column. Required in combined-CSV mode; defaults to `"spot_id"` only when images are generated from `spatial_locations_csv`
- `mode`: `"performance"` for pairwise comparisons or `"cost"` for all-wise comparisons (default: `"performance"`)
- `selection_mode`: `"top_k"` (default) selects models from the ranking; `model_names` is used **only** when `selection_mode="manual"` is also passed
- `k_mode`: `"auto"` (default) derives k from the selected models; `fixed_k` is used **only** when `k_mode="fixed"` is also passed
- `use_second_round`: Read a second-round reasoning JSON already written by `lstar.second_round.run_second_round_reasoning()` and use its models for consensus. `l_star()` never runs the second round itself
- `use_palo`: Whether to use Palo for color optimization when generating images (default: True)
- `he_image_path`: Optional path to H&E image to copy when generating images
- `**kwargs`: Forwarded to the comparison and consensus stages — `api_key`, `api_base`, `model_name`, `pairwise_temperature`, `pairwise_reasoning_effort`, `he_basename`, `force_rerun`, `skip_comparisons` (alias `skip_pairwise`), `disable_cache`, `k_method`, `k_range`, `ground_truth_col`, `random_state`. Unrecognized keywords are ignored with a warning rather than raising

### Pairwise Comparisons

#### `run_pairwise_comparisons()`

Run LLM-based pairwise comparisons and generate ranking.

```python
ranking_df, pairwise_dir, ranking_csv = lstar.run_pairwise_comparisons(
    image_dir="path/to/images",
    dataset_name="DLPFC (from 10X Visium Human Brain)",  # Required
    reps=5,
    top_k=5,
    simple_mode=True,
    output_dir="lstar_output",
    api_key="your-api-key"
)
```

**Key Parameters:**
- `image_dir`, `dataset_name`: Both required — neither has a default
- `reps`: Number of pairwise comparison repetitions (default: 5)
- `simple_mode`: Use simple prompts (True) or complex prompts with bias warnings (False)
- `top_k_mode`: "fixed" or "elbow" for top-k selection
- `force_rerun`: Ignore cache and recompute all comparisons
- `skip_pairwise`: Skip LLM calls and reuse existing results
- `disable_cache`: Do not read or write `cache_*.json` files at all

**Caching:** Pairwise comparisons are automatically cached to avoid redundant LLM calls. Each comparison is stored as a JSON file in `output_dir/pairwise/cache_*.json`.

### All-Wise Comparisons

#### `run_allwise_comparisons()`

Run cost-first all-wise comparisons and generate a position-score ranking.

```python
comparison_result = lstar.run_allwise_comparisons(
    image_dir="path/to/images",
    dataset_name="DLPFC (from 10X Visium Human Brain)",
    reps=5,
    output_dir="lstar_output",
    api_key="your-api-key"
)
```

### Consensus Clustering

#### `run_consensus_clustering()`

Perform consensus clustering on selected models.

```python
consensus_df = lstar.run_consensus_clustering(
    ranking_csv="lstar_output/ranking.csv",
    assignments_csv="path/to/assignments.csv",
    id_col="spot_id",
    selection_mode="manual",
    model_names=["Model1", "Model2", "Model3"],
    k_mode="auto",
    output_csv="lstar_output/L_STAR_consensus.csv"
)
```

**Key Parameters:**
- `assignments_csv` + `id_col`: The default, combined-CSV input. To pass `assignments_dir` or `assignment_csv_list` instead, `use_separate_csvs=True` is also required
- `selection_mode`: "manual" (use `model_names`, the default for this function) or "top_k" (select by ranking). Note `l_star()` defaults to "top_k" instead
- `k_mode`: "fixed" (use `fixed_k`) or "auto" (determine from models)
- `k_method`: "median_from_models" or "mode_from_models" for auto k selection (default is unconstrained by `k_range`), or "silhouette"/"gap_statistic" (uses `k_range`)
- `k_range`: Only consulted by "silhouette"/"gap_statistic". Defaults to `range(2, 30)` when this function is called directly, while `l_star()` passes `range(2, 16)`
- `ground_truth_col`: Optional column name for ARI evaluation

### Domain Annotation

#### `annotate_domain()`

Annotate the domains produced by an L-STAR consensus run.

```python
lstar.annotate_domain(
    expression_h5ad=None,
    *,
    expression_h5=None,
    expression_source=None,
    dataset_context=None,
    species=None,
    tissue=None,
    notes=None,
    sampling_level=None,
    expression_scale=None,
    max_positive_markers=25,
    marker_test_method=None,
    resolution_resource_paths=(),
    auto_discover_resolution_resources=True,
    output_dir="lstar_output",
    assignments_csv=None,
    lstar_consensus_csv=None,
    run_manifest=None,
    id_col=None,
    allow_partial_observations=False,
    observation_id_transform=None,
    allowed_context_fields=(),
    evaluation_only_fields=(),
    forbidden_annotation_fields=(),
    model_name=None,
    reasoning_effort="high",
    api_key=None,
    api_base=None,
    he_image_path=None,
    include_visual_background=True,
    show_progress=True,
) -> pandas.DataFrame
```

Required inputs are exactly one of `expression_h5ad` or `expression_h5`, plus non-empty `dataset_context` and `sampling_level` (`"spot"` or `"cell"`). When `output_dir` contains the standard L-STAR run manifest, `assignments_csv`, `lstar_consensus_csv`, `id_col`, and consensus method columns are resolved automatically; `run_manifest` points at a manifest stored elsewhere or under a non-default name. `species` and `tissue` are optional biological priors, and `notes` is a separate optional field for structural context they don't capture (see [Step 4](#step-4-post-consensus-domain-annotation) above). The remaining arguments are advanced controls for expression-source selection, feature-resolution resources, explicit ID transformation, and H5AD context-field governance.

Every evaluable domain is named in one dataset-level call; there is no per-domain candidate generation, no dataset-level reconciliation call, and no annotation-confidence output. A domain with no usable positive-marker evidence, or that fails a deterministic upstream evidence gate (see the run manifest), is named `"Unknown"` without spending a call on it. If the model's response never passes validation — including up to two repair attempts — the affected domains are named `"Unknown"` by a deterministic fallback so the run always completes; this is recorded in `annotation_response.json` under `annotation_artifacts/`.

## Output Files

The pipeline generates the following outputs in `output_dir`:

- `pairwise/` in performance mode, containing:
  - `pairwise_results_rep*.jsonl`: Pairwise comparison results (one per repetition)
  - `cache_*.json`: Cached individual pairwise comparisons
- `allwise/` in cost mode, containing `allwise_results_rep*.jsonl`
- `ranking.csv`: Model ranking produced by the selected comparison mode
- `L_STAR_consensus.csv`: Final consensus clustering with 'L-STAR' column
- `lstar_run_manifest.json`: Exact consensus-selected models, their matched
  assignment columns, and the input/output paths needed by downstream domain
  annotation
- `L_STAR_domain_assignment.csv`: Spot- or cell-level consensus assignment and
  `domain_name` produced by `annotate_domain()` — exactly three columns (the
  observation ID, `L-STAR`, and `domain_name`); versions before 0.4.0 also
  included a fourth `annotation_confidence` column, which v3's single-call
  naming flow no longer produces
- `annotation_artifacts/`: Feature-resolution, quality-control (scanpy QC
  thresholds and per-cell outcomes), marker, evidence, and visual-context
  audit files, plus the single naming call's payload and response
  (`annotation_prompt_payload.json`, `annotation_response.json`)

## Advanced Usage

### Custom Model Selection

```python
# Manually specify models for consensus (using pre-generated images)
df = lstar.l_star(
    image_dir="images",
    dataset_name="DLPFC (from 10X Visium Human Brain)",
    assignments_csv="assignments.csv",
    id_col="spot_id",
    selection_mode="manual",  # Without this, model_names is ignored and top-k applies
    model_names=["GraphST", "STAGATE", "SpaGCN", "BayesSpace"],
    k_mode="fixed",
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
    k_mode="fixed",
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
    k_mode="fixed",
    fixed_k=7,
    api_key="your-api-key"
)
```

### Top-K Selection

```python
# Automatically select top 5 models by ranking
df = lstar.l_star(
    image_dir="images",
    dataset_name="DLPFC (from 10X Visium Human Brain)",
    assignments_csv="assignments.csv",
    id_col="spot_id",
    selection_mode="top_k",
    top_k=5,
    k_mode="auto"
)
```

To read per-model CSVs from a directory instead, set `use_separate_csvs=True` alongside `assignments_dir`; otherwise `assignments_dir` is ignored and the missing `assignments_csv` raises a `ValueError`.

### Custom LLM Settings

```python
df = lstar.l_star(
    image_dir="images",
    dataset_name="DLPFC (from 10X Visium Human Brain)",
    assignments_csv="assignments.csv",
    id_col="spot_id",
    selection_mode="manual",
    model_names=["Model1", "Model2", "Model3"],
    model_name="gpt-5.1-2025-11-13",
    pairwise_temperature=1.0,
    pairwise_reasoning_effort="medium",
    api_key="your-api-key"
)
```

Second-round settings (`second_round_temperature`, `second_round_reasoning_effort`) are not accepted here — `l_star()` would log them as unused. They belong to `lstar.second_round.run_second_round_reasoning()`, which is run separately; `l_star(use_second_round=True)` then reads the JSON it wrote.

### Step-by-Step Execution

```python
# Step 1: Run pairwise comparisons
ranking_df, pairwise_dir, ranking_csv = lstar.run_pairwise_comparisons(
    image_dir="images",
    dataset_name="DLPFC (from 10X Visium Human Brain)",
    output_dir="output",
    api_key="your-api-key"
)

# Step 2: Run consensus clustering
consensus_df = lstar.run_consensus_clustering(
    ranking_csv=ranking_csv,
    assignments_csv="assignments.csv",
    id_col="spot_id",
    model_names=["Model1", "Model2", "Model3"],
    output_dir="output"
)
```

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: OpenAI API key (can be set instead of passing `api_key` parameter)
- `LSTAR_RUN_PALO_SCRIPT`: Optional absolute path override for `run_palo.R`
- `LSTAR_PLOT_SPATIAL_SCRIPT`: Optional absolute path override for `plot_spatial_with_palo.R`

### Default Values

- Output directory: `lstar_output`
- Comparison mode: `"performance"`
- Repetitions: 5
- Top-K: 5
- Model selection: `"top_k"` in `l_star()`, `"manual"` in `run_consensus_clustering()`
- K selection: `k_mode="auto"` with `k_method="median_from_models"`
- K range (used only by `k_method="silhouette"`/`"gap_statistic"`): 2-15 via `l_star()`, 2-29 when calling `run_consensus_clustering()` directly
- Model: `gpt-5-2025-08-07` everywhere — `l_star()`, `run_pairwise_comparisons()`, `run_allwise_comparisons()`, and `annotate_domain()`
- Temperature: 1.0
- Reasoning effort: "medium" (pairwise), "high" (annotation, and second-round if applicable)

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
