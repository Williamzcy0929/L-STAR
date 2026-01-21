# L-STAR Package Structure

## Directory Tree

```{}
lstar/
├── lstar/                    # Main package directory
│   ├── __init__.py           # Package exports
│   ├── types.py              # Type definitions (TypedDict, etc.)
│   ├── config.py             # Default configuration and constants
│   ├── io_utils.py           # CSV/JSON I/O helpers
│   ├── pairwise.py           # Pairwise LLM comparisons and ranking
│   ├── consensus.py          # Consensus clustering (EAC)
│   └── pipeline.py           # High-level l_star() function
├── tests/                    # Unit tests
│   ├── __init__.py
│   ├── test_pairwise.py      # Tests for pairwise module
│   ├── test_consensus.py     # Tests for consensus module
│   ├── test_io_utils.py      # Tests for I/O utilities
│   └── test_pipeline.py      # Tests for high-level pipeline
├── pyproject.toml            # Package metadata and dependencies
├── README.md                 # User documentation
└── PACKAGE_STRUCTURE.md      # This file
```

## Module Responsibilities

### `lstar/__init__.py`
- Exports main public APIs: `run_pairwise_comparisons`, `run_consensus_clustering`, `l_star`
- Package version

### `lstar/types.py`
- Type definitions using TypedDict for structured data
- `PairwiseResult`, `ConsensusResult`

### `lstar/config.py`
- Default values for all configurable parameters
- Constants for file/directory names
- Default hyperparameters

### `lstar/io_utils.py`
- `read_ranking_csv()`: Read and validate ranking CSV
- `read_assignment_csvs()`: Read multiple assignment CSVs with alignment checks
- `find_assignment_csvs()`: Find CSVs in directory by model name
- `append_jsonl()`, `read_jsonl()`: JSONL file operations
- `write_consensus_csv()`: Write final consensus output

### `lstar/pairwise.py`
- `discover_models()`: Find H&E and model images in directory
- `build_pairwise_messages()`: Construct LLM prompts (simple/complex modes)
- `ask_llm_with_retries()`: LLM API calls with retry logic
- `get_pairwise_cache_key()`, `get_cached_pairwise_result()`, `cache_pairwise_result()`: Caching
- `compute_winning_rates()`: Aggregate pairwise results into ranking
- `select_top_models()`: Select top-k models (fixed or elbow mode)
- `run_pairwise_comparisons()`: Main pairwise comparison function

### `lstar/consensus.py`
- `determine_optimal_k()`: Auto-determine k from model cluster counts (median/mode)
- `perform_eac_consensus()`: Evidence Accumulation Clustering implementation
- `run_consensus_clustering()`: Main consensus clustering function

### `lstar/pipeline.py`
- `l_star()`: High-level pipeline function that orchestrates pairwise + consensus

## Key Features

1. **Caching**: Pairwise comparisons are cached to avoid redundant LLM calls
2. **Error Handling**: Clear error messages for missing files, mismatched data, etc.
3. **Logging**: Uses Python logging module (INFO/WARNING/ERROR levels)
4. **Type Hints**: Full type annotations for all public functions
5. **Flexible I/O**: Supports both directory-based and explicit file list inputs
6. **Reproducibility**: Random state support for deterministic results

## Usage Patterns

### Full Pipeline
```python
import lstar
df = lstar.l_star(image_dir="...", assignments_dir="...", model_names=[...])
```

### Step-by-Step
```python
# Step 1: Pairwise comparisons
ranking_df, pairwise_dir, ranking_csv = lstar.run_pairwise_comparisons(...)

# Step 2: Consensus clustering
consensus_df = lstar.run_consensus_clustering(ranking_csv=ranking_csv, ...)
```

## Output Structure

```
output_dir/
├── pairwise/
│   ├── pairwise_results_rep01.jsonl
│   ├── pairwise_results_rep02.jsonl
│   ├── ...
│   └── cache_*.json          # Cached individual comparisons
├── ranking.csv                # Model ranking with win rates
└── L_STAR_consensus.csv        # Final consensus clustering
```
