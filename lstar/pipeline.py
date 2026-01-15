"""High-level L-STAR pipeline combining pairwise comparisons and consensus clustering."""

import logging
from pathlib import Path
from typing import Optional, Sequence, Literal, Union

import pandas as pd

from lstar.pairwise import run_pairwise_comparisons
from lstar.consensus import run_consensus_clustering
from lstar.config import DEFAULT_OUTPUT_DIR
from lstar.io_utils import read_second_round_results

logger = logging.getLogger(__name__)


def l_star(
    image_dir: Union[str, Path],
    dataset_name: str,
    *,
    assignments_csv: Union[str, Path, None] = None,
    id_col: Optional[str] = None,
    use_separate_csvs: bool = False,
    assignments_dir: Union[str, Path, None] = None,
    assignment_csv_list: Optional[Sequence[Union[str, Path]]] = None,
    output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
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
) -> pd.DataFrame:
    """
    High-level L-STAR pipeline: pairwise LLM comparisons + ranking + consensus.
    
    This function:
      1) Runs pairwise LLM comparisons to rank models
      2) Uses the ranking to select models for consensus clustering
      3) Performs consensus clustering and outputs L-STAR assignments
    
    Parameters
    ----------
    image_dir : str or Path
        Directory containing H&E and model output images
    
    dataset_name : str
        Background information about the dataset name provided in LLM prompts.
        This helps the LLM understand the context of the spatial transcriptomics data.
        Example: "DLPFC (from 10X Visium Human Brain)" or "STARmap Mouse Ventricular Cardiomyocytes"
    
    assignments_csv : str or Path, optional
        Path to a single CSV file containing all model assignments (one column per model).
        This is the default mode. If not provided and use_separate_csvs=False, will raise an error.
        When using this mode, fuzzy name matching is automatically enabled to match
        model names between ranking CSV, assignment columns, and image filenames.
    
    id_col : str, optional
        Name of the ID column in assignments_csv (only used in combined CSV mode).
        Required when assignments_csv is provided.
    
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
          - output_dir / "pairwise" / ...  (pairwise comparison cache)
          - output_dir / "ranking.csv"     (model ranking)
          - output_dir / "L_STAR_consensus.csv"  (final consensus)
    
    simple_mode : bool
        If True, use simple prompts for pairwise comparisons.
        If False, use complex prompts with bias warnings.
    
    reps : int
        Number of pairwise comparison repetitions
    
    top_k : int
        Number of top models to consider (used in pairwise ranking and/or consensus selection)
    
    top_k_mode : {"fixed", "elbow"}
        Mode for top-k selection in pairwise ranking
    
    selection_mode : {"manual", "top_k"}
        How to select models for consensus:
        - "manual": Use model_names parameter
        - "top_k": Select top k models from ranking by win_rate (default)
    
    use_second_round : bool, default False
        If True, read second-round reasoning JSON file (if present) and use its selected models for consensus.
        When enabled, second-round results override the default top-k selection
        (unless selection_mode="manual" with explicit model_names).
        Note: Second-round reasoning is performed by the separate second_round.py module, not within l_star.
    
    model_names : sequence of str, optional
        Manually specified list of model names for consensus clustering.
        Required if selection_mode="manual".
    
    k_mode : {"fixed", "auto"}
        Whether to use fixed_k or auto-determine k from models
    
    fixed_k : int, optional
        Fixed number of clusters (used when k_mode="fixed")
    
    **kwargs
        Additional arguments passed to run_pairwise_comparisons and run_consensus_clustering:
        - api_key, api_base, model_name
        - pairwise_temperature, pairwise_reasoning_effort
        - k_method, k_range, ground_truth_col, etc.
        
        Note: Second-round reasoning parameters (second_round_temperature, second_round_reasoning_effort)
        are handled by the separate second_round.py module, not by run_pairwise_comparisons.
    
    Returns
    -------
    consensus_df : pd.DataFrame
        DataFrame with 'L-STAR' column containing consensus cluster labels.
        Also includes ID column(s) from input assignment CSVs.
    
    Examples
    --------
    >>> import lstar
    >>> # Example 1: Default mode (combined CSV with fuzzy matching)
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
    >>> # Example 2: Legacy mode (separate CSV files per model)
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
    
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract kwargs for pairwise and consensus
    pairwise_kwargs = {
        "image_dir": image_dir,
        "dataset_name": dataset_name,
        "reps": reps,
        "top_k": top_k,
        "top_k_mode": top_k_mode,
        "simple_mode": simple_mode,
        "output_dir": output_dir,
        "force_rerun": kwargs.pop("force_rerun", False),
        "skip_pairwise": kwargs.pop("skip_pairwise", False),
        "model_name": kwargs.pop("model_name", "gpt-5.1-thinking"),
        "pairwise_temperature": kwargs.pop("pairwise_temperature", 1.0),
        "pairwise_reasoning_effort": kwargs.pop("pairwise_reasoning_effort", "medium"),
        "api_key": kwargs.pop("api_key", None),
        "api_base": kwargs.pop("api_base", None),
        "disable_cache": kwargs.pop("disable_cache", False),
    }
    
    # Build initial consensus_kwargs (will be updated after pairwise phase)
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
    
    # Step 1: Run pairwise comparisons
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Pairwise Comparisons")
    logger.info("=" * 60)
    
    ranking_df, pairwise_dir, ranking_csv_path = run_pairwise_comparisons(**pairwise_kwargs)
    
    logger.info(f"Pairwise comparisons complete. Ranking CSV: {ranking_csv_path}")
    
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
    
    # Priority 3: Default: top-k winning-rate models (no second round)
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
    logger.info(f"  - Pairwise results: {pairwise_dir}")
    logger.info(f"  - Ranking CSV: {ranking_csv_path}")
    logger.info(f"  - Consensus CSV: {output_dir / 'L_STAR_consensus.csv'}")
    logger.info("=" * 60)
    
    return consensus_df

