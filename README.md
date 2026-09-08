# L-STAR: LLM-Guided Spatial Domain Detection

L-STAR is a Python package for comparing spatial domain detection methods with LLMs and combining selected methods into a consensus clustering. An optional annotation step assigns biological names to the consensus domains using marker genes and dataset context.

## Workflow

1. **Visualize:** Generate spatial domain images from coordinates and clustering assignments, with Palo color optimization, or supply pre-generated images.
2. **Compare:** Rank methods using repeated LLM comparisons. The default `mode="performance"` compares methods pairwise; `mode="cost"` ranks all methods in one request per repetition. An H&E reference is optional.
3. **Combine:** Select the top-ranked methods, including ties at the cutoff, or specify methods manually. Evidence Accumulation Clustering (EAC) combines their co-assignment frequencies into a consensus domain assignment.
4. **Annotate (Optional):** Use expression data to identify domain markers and assign biological names with an LLM.

## Installation

Requires Python 3.10 or later.

```bash
pip install "git+https://github.com/Williamzcy0929/L-STAR.git"
```

For an editable source installation:

```bash
git clone https://github.com/Williamzcy0929/L-STAR.git
cd L-STAR
pip install -e .
```

Set your API key before running LLM comparisons or annotation:

```bash
export OPENAI_API_KEY="your-api-key"
```

### Optional: Palo color optimization

Palo rendering requires R and the following packages:

```r
install.packages(c("remotes", "ggplot2", "RColorBrewer"),
                 repos = "https://cloud.r-project.org")
remotes::install_github("Winnie09/Palo")
```

L-STAR uses bundled R scripts to optimize colors for adjacent domains and render spatial images. If these dependencies are unavailable, it falls back to matplotlib. Set `use_palo=False` to use matplotlib directly.

## Input Data

Provide a combined assignments CSV with an observation ID column and one column per method:

```csv
spot_id,GraphST,SpaGCN,BayesSpace
spot_1,1,2,1
spot_2,2,2,2
spot_3,1,1,1
```

To generate images, also provide a spatial coordinates CSV with matching IDs:

```csv
spot_id,x,y
spot_1,10.5,20.3
spot_2,11.2,21.1
spot_3,12.0,19.8
```

Use `id_col` to specify a different ID column name.

Alternatively, supply pre-generated images whose filenames match the method columns, such as `GraphST.png` and `SpaGCN.png`. Supported formats are PNG, JPG/JPEG, and PDF. An optional H&E reference can be included as `he.png`; use `he_basename` for a different reference name.

## Quick Start

### Generate images and run consensus clustering

```python
import lstar

consensus = lstar.l_star(
    dataset_name="10x Visium human dorsolateral prefrontal cortex",
    spatial_locations_csv="path/to/spatial_locations.csv",
    assignments_csv="path/to/assignments.csv",
    id_col="spot_id",
    output_dir="lstar_output",
)

print(consensus.head())
```

The result contains an `L-STAR` column with consensus domain labels. By default, L-STAR uses pairwise comparisons, five repetitions, and the top five methods plus cutoff ties. The number of consensus domains is derived from the selected methods. Add `he_image_path="path/to/he.png"` to include an H&E reference.

### Use pre-generated images

Provide `image_dir` and omit `spatial_locations_csv`:

```python
consensus = lstar.l_star(
    dataset_name="10x Visium human dorsolateral prefrontal cortex",
    image_dir="path/to/images",
    assignments_csv="path/to/assignments.csv",
    id_col="spot_id",
    output_dir="lstar_output",
)
```

### Annotate consensus domains

Run `annotate_domain()` with the same `output_dir` as the consensus run. The saved run manifest supplies the assignment and consensus paths, observation ID column, and selected methods.

```python
annotations = lstar.annotate_domain(
    expression_h5="path/to/filtered_feature_bc_matrix.h5",
    output_dir="lstar_output",
    dataset_context="10x Visium human dorsolateral prefrontal cortex",
    species="Homo sapiens",
    tissue="dorsolateral prefrontal cortex",
    sampling_level="spot",
)

print(annotations.head())
# Columns: spot_id, L-STAR, domain_name
```

Annotation requires:

- **Expression data:** Exactly one of `expression_h5` for 10x H5/HDF5 or `expression_h5ad` for H5AD. Expression CSV is not supported.
- **Dataset context:** A non-empty `dataset_context` and `sampling_level="spot"` or `"cell"`. Optional `species`, `tissue`, and `notes` guide interpretation without overriding marker evidence.
- **Aligned IDs:** Expression observation IDs must match the consensus exactly by default. Set `allow_partial_observations=True` only when you intend to annotate their intersection.

L-STAR applies expression QC, normalization, and marker ranking with Scanpy, using a t-test by default and up to 25 positive markers per domain. QC restricts marker computation; every aligned observation still receives its domain's name. Evaluable domains are named together in one dataset-level LLM request, with optional visual context. Raw expression values are not sent to the LLM.

Domains without usable marker evidence, or whose responses remain invalid after repair attempts, receive `Unknown`. Marker and response records are saved in `annotation_artifacts/`.

## Common Options

Pass these options to `l_star()`:

| Option | Purpose |
| --- | --- |
| `mode="cost"` | Rank all methods together instead of comparing pairs. |
| `reps=5` | Set the number of comparison repetitions. |
| `top_k=5` | Set how many top-ranked methods enter consensus; cutoff ties are included. |
| `selection_mode="manual", model_names=["GraphST", "SpaGCN"]` | Select consensus methods explicitly. Both arguments are needed. |
| `k_mode="fixed", fixed_k=7` | Set the number of consensus domains. Both arguments are needed. |
| `model_name="your-model-name"` | Select the LLM (OpenAI models only). |
| `api_key="your-api-key"` | Supply a key instead of using `OPENAI_API_KEY`. |
| `force_rerun=True` | Recompute comparisons instead of using cached results. |

For annotation, `max_positive_markers` controls the marker list length and `marker_test_method` selects an alternative test, such as `"wilcoxon"`. Use `he_image_path` to supply an H&E reference and `include_visual_background=False` to omit visual context.

## Output Files

Results are saved under `output_dir` (default: `lstar_output`):

| File or directory | Contents |
| --- | --- |
| `generated_images/` | Per-method spatial images, when generated from CSVs. |
| `pairwise/` or `allwise/` | Comparison results and, for pairwise mode, cached calls. |
| `ranking.csv` | Aggregated method ranking. |
| `L_STAR_consensus.csv` | Consensus assignments in the `L-STAR` column. |
| `lstar_run_manifest.json` | Selected methods and paths needed for annotation. |
| `L_STAR_domain_assignment.csv` | Annotation output: observation ID, `L-STAR`, and `domain_name`. |
| `annotation_artifacts/` | Annotation QC, markers, evidence, and LLM request/response records. |

The final two outputs are created by `annotate_domain()`.

## Advanced Usage

Individual stages are available through `run_pairwise_comparisons()`, `run_allwise_comparisons()`, and `run_consensus_clustering()`. For example:

```python
ranking, pairwise_dir, ranking_csv = lstar.run_pairwise_comparisons(
    image_dir="path/to/images",
    dataset_name="10x Visium human dorsolateral prefrontal cortex",
    output_dir="lstar_output",
)

consensus = lstar.run_consensus_clustering(
    ranking_csv=ranking_csv,
    assignments_csv="path/to/assignments.csv",
    id_col="spot_id",
    selection_mode="top_k",
    top_k=5,
    k_mode="auto",
    output_csv="lstar_output/L_STAR_consensus.csv",
)
```

Unlike `l_star()`, `run_consensus_clustering()` defaults to manual model selection, so specify `selection_mode="top_k"` when selecting from a ranking.

For separate assignment CSVs, set `use_separate_csvs=True` and provide `assignments_dir` or `assignment_csv_list`. Each file should contain an observation ID column and a clustering column.

Use `help(lstar.l_star)` or `help(lstar.annotate_domain)` for the complete parameter reference.

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

## License and Contact

MIT License.

For questions, bug reports, or contributions, [open an issue](https://github.com/Williamzcy0929/L-STAR/issues) or submit a pull request.

Authors: Changyue (William) Zhao ([changyue.zhao@duke.edu](mailto:changyue.zhao@duke.edu)) and Zhicheng Ji ([zhicheng.ji@duke.edu](mailto:zhicheng.ji@duke.edu)).
