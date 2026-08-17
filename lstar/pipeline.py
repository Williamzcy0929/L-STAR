"""High-level L-STAR comparison and consensus-clustering pipeline."""

import logging
from pathlib import Path
from typing import Literal, Optional, Sequence, Union

import pandas as pd

from lstar.allwise import run_allwise_comparisons
from lstar.consensus import run_consensus_clustering
from lstar.config import (
    DEFAULT_HE_BASENAME,
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PAIRWISE_REASONING_EFFORT,
    DEFAULT_PAIRWISE_TEMPERATURE,
)
from lstar.image_generation import generate_images_from_csvs
from lstar.io_utils import read_second_round_results
from lstar.pairwise import run_pairwise_comparisons
from lstar.types import ComparisonResult

logger = logging.getLogger(__name__)


def l_star(
    image_dir: Union[str, Path, None] = None,
    dataset_name: Optional[str] = None,
    *,
    spatial_locations_csv: Union[str, Path, None] = None,
    assignments_csv: Union[str, Path, None] = None,
    id_col: Optional[str] = None,
    use_separate_csvs: bool = False,
    assignments_dir: Union[str, Path, None] = None,
    assignment_csv_list: Optional[Sequence[Union[str, Path]]] = None,
    output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
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
    he_image_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    High-level L-STAR pipeline: LLM comparison + ranking + consensus.
    
    This function:
      1) Optionally generates images from spatial locations and assignments (if spatial_locations_csv provided)
      2) Runs performance-first pairwise or cost-first all-wise comparisons
      3) Uses the ranking to select models for consensus clustering
      4) Performs consensus clustering and outputs L-STAR assignments
    
    Parameters
    ----------
    dataset_name : str
        Background information about the dataset name provided in LLM prompts.
        This helps the LLM understand the context of the spatial transcriptomics data.
        Example: "DLPFC (from 10X Visium Human Brain)" or "STARmap Mouse Visual Cortex"
    
    image_dir : str or Path, optional
        Directory containing H&E and model output images.
        If not provided, images will be generated from spatial_locations_csv and assignments_csv.
        Either image_dir or (spatial_locations_csv + assignments_csv) must be provided.
    
    spatial_locations_csv : str or Path, optional
        Path to CSV file with spatial coordinates (id_col, x, y).
        Required when image_dir is not provided. Used to generate images internally.
        Example columns: spot_id, x, y
    
    assignments_csv : str or Path, optional
        Path to a single CSV file containing all method assignments.
        Format: The CSV should have one column named `id_col` (e.g., "spot_id") and 
        all other columns should be named after the spatial domain detection methods
        (e.g., "GraphST", "SpaGCN", "BayesSpace", etc.). Each method column contains
        cluster assignments for that method.
        
        Example format:
            spot_id,GraphST,SpaGCN,BayesSpace,STAGATE
            spot_1,1,2,1,3
            spot_2,2,2,2,3
            ...
        
        This is the default mode. If not provided and use_separate_csvs=False, will raise an error.
        When using this mode, fuzzy name matching is automatically enabled to match
        method names between ranking CSV, assignment columns, and image filenames.
        Required when generating images from CSV files.
    
    id_col : str, optional
        Name of the ID column in assignments_csv and spatial_locations_csv.
        Required when assignments_csv is provided.
        Default: "spot_id"
    
    use_separate_csvs : bool, default False
        If True, use the legacy mode with separate CSV files per model (one CSV per model).
        Requires either assignments_dir or assignment_csv_list to be provided.
        If False (default), uses assignments_csv mode.
    
    assignments_dir : str or Path, optional
        Directory containing per-model clustering assignment CSVs.
        Only used if use_separate_csvs=True.
    
    assignment_csv_list : sequence of paths, optional
        Explicit list of per-model clustering assignment CSVs.
        Only used if use_separate_csvs=True.
    
    output_dir : str or Path
        Base directory for all outputs:
          - output_dir / "pairwise" / ...  (performance-mode artifacts)
          - output_dir / "allwise" / ...   (cost-mode artifacts)
          - output_dir / "ranking.csv"     (model ranking)
          - output_dir / "L_STAR_consensus.csv"  (final consensus)
          - output_dir / "lstar_run_manifest.json"  (selected-method provenance)
    
    mode : {"performance", "cost"}, default "performance"
        Comparison implementation. Performance mode uses exhaustive pairwise
        comparisons. Cost mode ranks all candidate methods in one request per
        repetition.

    simple_mode : bool
        If True, use simple prompts for comparisons.
        If False, use complex prompts with bias warnings.
    
    reps : int
        Number of comparison repetitions
    
    top_k : int
        Minimum number of top models to consider. All models tied at the
        cutoff score are included.
    
    top_k_mode : {"fixed", "elbow"}
        Mode for top-k selection in ranking
    
    selection_mode : {"manual", "top_k"}
        How to select models for consensus:
        - "manual": Use model_names parameter
        - "top_k": Select at least top k models from the ranking score,
          including all models tied at the cutoff (default)
    
    use_second_round : bool, default False
        If True, read second-round reasoning JSON file (if present) and use its selected models for consensus.
        When enabled, second-round results override the default top-k selection
        (unless selection_mode="manual" with explicit model_names).
        Note: Second-round reasoning is performed by the separate second_round.py module, not within l_star.
    
    use_palo : bool, default True
        Whether to use Palo R package for color palette optimization when generating images.
        Only used when spatial_locations_csv is provided.
    
    he_image_path : str or Path, optional
        Optional path to H&E image file. If provided and images are generated from CSV,
        the H&E image will be copied to the generated images directory.
    
    model_names : sequence of str, optional
        Manually specified list of model names for consensus clustering.
        Required if selection_mode="manual".
    
    k_mode : {"fixed", "auto"}
        Whether to use fixed_k or auto-determine k from models
    
    fixed_k : int, optional
        Fixed number of clusters (used when k_mode="fixed")
    
    **kwargs
        Additional arguments passed to comparison and consensus functions:
        - api_key, api_base, model_name
        - pairwise_temperature, pairwise_reasoning_effort
        - k_method, k_range, ground_truth_col, etc.
        
        Note: Second-round reasoning parameters (second_round_temperature, second_round_reasoning_effort)
        are handled by the separate second_round.py module.
    
    Returns
    -------
    consensus_df : pd.DataFrame
        DataFrame with 'L-STAR' column containing consensus cluster labels.
        Also includes ID column(s) from input assignment CSVs.
    
    Examples
    --------
    >>> import lstar
    >>> # Example 1: Using pre-generated images (original mode)
    >>> df = lstar.l_star(
    ...     image_dir="path/to/images",
    ...     dataset_name="DLPFC (from 10X Visium Human Brain)",
    ...     assignments_csv="path/to/combined_assignments.csv",
    ...     id_col="spot_id",
    ...     selection_mode="top_k",
    ...     top_k=5,
    ...     k_mode="auto",
    ...     api_key="your-api-key"
    ... )
    >>> 
    >>> # Example 2: Generate images from CSV files (new mode)
    >>> df = lstar.l_star(
    ...     dataset_name="DLPFC (from 10X Visium Human Brain)",
    ...     spatial_locations_csv="path/to/spatial_locations.csv",
    ...     assignments_csv="path/to/combined_assignments.csv",
    ...     id_col="spot_id",
    ...     use_palo=True,
    ...     selection_mode="top_k",
    ...     top_k=5,
    ...     k_mode="auto",
    ...     api_key="your-api-key"
    ... )
    >>> 
    >>> # Example 3: Legacy mode (separate CSV files per model)
    >>> df = lstar.l_star(
    ...     image_dir="path/to/images",
    ...     dataset_name="DLPFC (from 10X Visium Human Brain)",
    ...     use_separate_csvs=True,
    ...     assignments_dir="path/to/assignments",
    ...     model_names=["Model1", "Model2", "Model3"],
    ...     fixed_k=7,
    ...     api_key="your-api-key"
    ... )
    >>> print(df.head())
    """
    logger.info("=" * 60)
    logger.info("STARTING L-STAR PIPELINE")
    logger.info("=" * 60)
    
    # Validate required arguments
    if dataset_name is None:
        raise ValueError("dataset_name is required")
    if mode not in {"performance", "cost"}:
        raise ValueError(
            f"Unknown comparison mode: {mode}. Expected 'performance' or 'cost'."
        )
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine input mode: pre-generated images vs. CSV-based generation
    if image_dir is None:
        # Mode 1: Generate images from CSV files
        if spatial_locations_csv is None or assignments_csv is None:
            raise ValueError(
                "Either image_dir must be provided, or both spatial_locations_csv and "
                "assignments_csv must be provided for image generation."
            )
        if id_col is None:
            id_col = "spot_id"  # Default ID column name
            logger.info(f"Using default id_col='{id_col}'")
        
        logger.info("=" * 60)
        logger.info("GENERATING IMAGES FROM CSV FILES")
        logger.info("=" * 60)
        logger.info(f"Spatial locations CSV: {spatial_locations_csv}")
        logger.info(f"Assignments CSV: {assignments_csv}")
        logger.info(f"ID column: {id_col}")
        logger.info(f"Using Palo for color optimization: {use_palo}")
        
        # Generate images
        generated_image_dir, model_images = generate_images_from_csvs(
            spatial_locations_csv=spatial_locations_csv,
            assignments_csv=assignments_csv,
            id_col=id_col,
            output_dir=output_dir / "generated_images",
            use_palo=use_palo,
            he_image_path=he_image_path,
        )
        
        image_dir = generated_image_dir
        logger.info(f"Generated images saved to: {image_dir}")
    else:
        # Mode 2: Use pre-generated images (original mode)
        image_dir = Path(image_dir)
        if not image_dir.is_dir():
            raise NotADirectoryError(f"Image directory not found: {image_dir}")
        logger.info(f"Using pre-generated images from: {image_dir}")
    
    # Extract kwargs for comparison and consensus stages.
    skip_comparisons = kwargs.pop(
        "skip_comparisons",
        kwargs.pop("skip_pairwise", False),
    )
    comparison_kwargs = {
        "image_dir": image_dir,
        "dataset_name": dataset_name,
        "reps": reps,
        "he_basename": kwargs.pop("he_basename", DEFAULT_HE_BASENAME),
        "simple_mode": simple_mode,
        "output_dir": output_dir,
        "force_rerun": kwargs.pop("force_rerun", False),
        "model_name": kwargs.pop("model_name", DEFAULT_MODEL_NAME),
        "pairwise_temperature": kwargs.pop(
            "pairwise_temperature", DEFAULT_PAIRWISE_TEMPERATURE
        ),
        "pairwise_reasoning_effort": kwargs.pop(
            "pairwise_reasoning_effort", DEFAULT_PAIRWISE_REASONING_EFFORT
        ),
        "api_key": kwargs.pop("api_key", None),
        "api_base": kwargs.pop("api_base", None),
    }
    disable_cache = kwargs.pop("disable_cache", False)
    
    # Build initial consensus_kwargs (updated after the comparison phase).
    consensus_kwargs = {
        "output_dir": output_dir,
        "selection_mode": selection_mode,  # Will be updated based on priority
        "model_names": model_names,  # Will be updated based on priority
        "top_k": top_k,  # Will be updated based on priority
        "k_mode": k_mode,
        "fixed_k": fixed_k,
        "k_method": kwargs.pop("k_method", "median_from_models"),
        "k_range": kwargs.pop("k_range", range(2, 16)),
        "ground_truth_col": kwargs.pop("ground_truth_col", None),
        "random_state": kwargs.pop("random_state", 0),
        "assignments_csv": assignments_csv,
        "id_col": id_col,
        "use_separate_csvs": use_separate_csvs,
    }
    
    # Set up assignment source based on mode
    if use_separate_csvs:
        # Legacy mode: separate CSV files
        if assignments_dir is not None:
            consensus_kwargs["assignments_dir"] = assignments_dir
            logger.info(f"Using separate CSV files from directory: {assignments_dir}")
        elif assignment_csv_list is not None:
            consensus_kwargs["assignment_csv_list"] = assignment_csv_list
            logger.info(f"Using separate CSV files: {len(assignment_csv_list)} files")
        else:
            raise ValueError(
                "When use_separate_csvs=True, either assignments_dir or assignment_csv_list must be provided."
            )
    else:
        # Default mode: combined CSV
        if assignments_csv is None:
            raise ValueError(
                "assignments_csv must be provided when use_separate_csvs=False (default mode). "
                "Either provide assignments_csv, or set use_separate_csvs=True to use separate CSV files."
            )
        if id_col is None:
            raise ValueError(
                "id_col must be provided when using assignments_csv mode. "
                "Specify the name of the ID column in the combined assignments CSV."
            )
        logger.info(f"Using combined assignments CSV: {assignments_csv}")
    
    # Warn about unused kwargs
    if kwargs:
        logger.warning(f"Unused keyword arguments: {list(kwargs.keys())}")
    
    # Step 1: Run the selected comparison implementation.
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Model Comparisons (%s mode)", mode)
    logger.info("=" * 60)

    if mode == "performance":
        ranking_df, artifact_dir, ranking_csv_path = run_pairwise_comparisons(
            **comparison_kwargs,
            top_k=top_k,
            top_k_mode=top_k_mode,
            skip_pairwise=skip_comparisons,
            disable_cache=disable_cache,
        )
        comparison_result = ComparisonResult(
            ranking_df=ranking_df,
            ranking_csv_path=ranking_csv_path,
            artifact_dir=artifact_dir,
            score_column="win_rate",
        )
    else:
        comparison_result = run_allwise_comparisons(
            **comparison_kwargs,
            skip_comparisons=skip_comparisons,
        )

    ranking_csv_path = comparison_result.ranking_csv_path
    logger.info(
        "Comparisons complete. Ranking CSV: %s",
        comparison_result.ranking_csv_path,
    )
    
    # Determine model selection based on priority: manual > second-round JSON > top-k
    # Priority 1: Manual override (explicit by user)
    if selection_mode == "manual" and model_names is not None and len(model_names) > 0:
        logger.info(f"Using manual model selection: {model_names}")
        final_selection_mode = "manual"
        final_model_names = model_names
        final_top_k = None
    
    # Priority 2: Second-round reasoning JSON (when requested and present)
    elif use_second_round:
        # Do not run the second round inside l_star (second-round is a separate script/module)
        # Instead, assume the separate second-round script has already written a JSON file
        second_round_models = read_second_round_results(output_dir)
        if second_round_models is not None and len(second_round_models) > 0:
            logger.info(f"Using second-round reasoning selected models from JSON: {second_round_models}")
            final_selection_mode = "manual"
            final_model_names = second_round_models
            final_top_k = None
        else:
            logger.warning("Second-round reasoning enabled but no valid JSON found, falling back to default top-k selection")
            # Fall through to Priority 3 (top-k)
            final_selection_mode = "top_k"
            final_model_names = None
            final_top_k = top_k
    
    # Priority 3: Default top-k selection from the comparison ranking.
    else:
        final_selection_mode = "top_k"
        final_model_names = None
        final_top_k = top_k
        logger.info(f"Using default top-k selection: top_k={final_top_k}")
    
    # Update consensus_kwargs with final selection
    consensus_kwargs["selection_mode"] = final_selection_mode
    consensus_kwargs["model_names"] = final_model_names
    consensus_kwargs["top_k"] = final_top_k if final_selection_mode == "top_k" else None
    
    # Step 2: Run consensus clustering
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Consensus Clustering")
    logger.info("=" * 60)
    
    consensus_kwargs["ranking_csv"] = ranking_csv_path
    consensus_df = run_consensus_clustering(**consensus_kwargs)
    
    logger.info("\n" + "=" * 60)
    logger.info("L-STAR PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"  - Comparison results: {comparison_result.artifact_dir}")
    logger.info(f"  - Ranking CSV: {ranking_csv_path}")
    logger.info(f"  - Consensus CSV: {output_dir / 'L_STAR_consensus.csv'}")
    logger.info(f"  - Run manifest: {output_dir / 'lstar_run_manifest.json'}")
    logger.info("=" * 60)
    
    return consensus_df
