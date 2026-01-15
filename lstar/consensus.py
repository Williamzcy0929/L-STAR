"""Consensus clustering using Evidence Accumulation Clustering (EAC)."""

import logging
from pathlib import Path
from typing import Literal, Optional, Sequence, Dict, Union
from collections import Counter

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score, silhouette_score

from lstar.config import (
    DEFAULT_OUTPUT_DIR,
    CONSENSUS_CSV_NAME,
    DEFAULT_K_RANGE,
)
from lstar.io_utils import (
    read_ranking_csv,
    read_assignment_csvs,
    find_assignment_csvs,
    write_consensus_csv,
    read_combined_assignments_csv,
    build_name_mappings,
    match_models_with_fuzzy_matching,
    normalize_name,
)

logger = logging.getLogger(__name__)


def determine_optimal_k(
    pred_data_for_mode: Optional[pd.DataFrame] = None,
    distance_matrix: Optional[np.ndarray] = None,
    method: Literal["median_from_models", "mode_from_models", "silhouette", "gap_statistic"] = "median_from_models",
    k_range: range = DEFAULT_K_RANGE,
) -> Dict[str, any]:
    """
    Determine optimal k from model predictions or using statistical methods.
    
    Parameters:
    -----------
    pred_data_for_mode : pd.DataFrame, optional
        Matrix of model labels (each column is a model). Required for median/mode methods.
    distance_matrix : np.ndarray, optional
        Distance/dissimilarity matrix. Required for silhouette and gap_statistic methods.
    method : str
        One of: "median_from_models", "mode_from_models", "silhouette", "gap_statistic"
    k_range : range or list
        Valid range of k values
    
    Returns:
    --------
    dict with keys: k_optimal, method, k_counts (or scores for silhouette/gap_statistic)
    """
    valid_methods = ["median_from_models", "mode_from_models", "silhouette", "gap_statistic"]
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}")
    
    k_range = list(k_range) if isinstance(k_range, range) else k_range
    
    if method in ["median_from_models", "mode_from_models"]:
        if pred_data_for_mode is None or pred_data_for_mode.empty:
            raise ValueError(f"pred_data_for_mode (matrix of model labels) is required for {method}")
        
        # Number of clusters implied by each model
        k_counts = [pred_data_for_mode[col].nunique() for col in pred_data_for_mode.columns]
        
        # Clip to the user-specified k_range to avoid wild k's
        k_counts = [k for k in k_counts if k in k_range]
        if len(k_counts) == 0:
            raise ValueError(
                f"No model-implied k falls inside k_range {list(k_range)}. "
                f"Model k values: {[pred_data_for_mode[col].nunique() for col in pred_data_for_mode.columns]}"
            )
        
        if method == "mode_from_models":
            # Most common k
            count_freq = Counter(k_counts)
            k_optimal = count_freq.most_common(1)[0][0]
        else:  # "median_from_models"
            k_optimal = int(round(np.median(k_counts)))
        
        logger.info(f"Determined optimal k={k_optimal} using {method} (from k_counts: {k_counts})")
        
        return {
            'k_optimal': k_optimal,
            'method': method,
            'k_counts': k_counts
        }
    
    elif method == "silhouette":
        if distance_matrix is None:
            raise ValueError("distance_matrix is required for silhouette method")
        
        # For large datasets, subsample for k determination
        n_total = len(distance_matrix)
        max_sample = 5000
        
        if n_total > max_sample:
            np.random.seed(2025)  # For reproducibility
            sample_idx = np.random.choice(n_total, max_sample, replace=False)
            D_sample = distance_matrix[np.ix_(sample_idx, sample_idx)]
        else:
            D_sample = distance_matrix
            sample_idx = None
        
        # Clean and symmetrize the distance matrix
        D_clean = D_sample.copy()
        D_clean = np.nan_to_num(D_clean, nan=1.0, posinf=1.0, neginf=0.0)
        D_clean = (D_clean + D_clean.T) / 2
        np.fill_diagonal(D_clean, 0)
        D_clean = np.clip(D_clean, 0, 1)
        
        # Convert to condensed form for linkage
        D_condensed = squareform(D_clean, checks=False)
        hc = linkage(D_condensed, method='average')
        
        silhouette_scores = []
        valid_k_values = []
        
        for k in k_range:
            if k < 2 or k >= len(D_clean):
                continue
            
            try:
                clusters = cut_tree(hc, n_clusters=k).flatten()
                n_clusters = len(np.unique(clusters))
                
                if n_clusters < 2 or n_clusters >= len(D_clean):
                    continue
                
                # Calculate silhouette score
                sil_score = silhouette_score(D_clean, clusters, metric='precomputed')
                silhouette_scores.append(sil_score)
                valid_k_values.append(k)
            except Exception:
                continue
        
        if len(silhouette_scores) == 0:
            raise ValueError("Could not compute silhouette scores for any k in range")
        
        best_idx = np.argmax(silhouette_scores)
        k_optimal = valid_k_values[best_idx]
        
        logger.info(f"Determined optimal k={k_optimal} using silhouette (max score: {silhouette_scores[best_idx]:.4f})")
        
        return {
            'k_optimal': k_optimal,
            'method': 'silhouette',
            'scores': dict(zip(valid_k_values, silhouette_scores)),
            'max_score': silhouette_scores[best_idx],
            'subsampled': sample_idx is not None,
            'sample_size': max_sample if sample_idx is not None else n_total
        }
    
    elif method == "gap_statistic":
        if distance_matrix is None:
            raise ValueError("distance_matrix is required for gap_statistic method")
        
        # Simplified gap statistic implementation
        n_total = len(distance_matrix)
        max_sample = 5000
        
        if n_total > max_sample:
            np.random.seed(2025)
            sample_idx = np.random.choice(n_total, max_sample, replace=False)
            D_sample = distance_matrix[np.ix_(sample_idx, sample_idx)]
        else:
            D_sample = distance_matrix
            sample_idx = None
        
        # Clean distance matrix
        D_clean = D_sample.copy()
        D_clean = np.nan_to_num(D_clean, nan=1.0, posinf=1.0, neginf=0.0)
        D_clean = (D_clean + D_clean.T) / 2
        np.fill_diagonal(D_clean, 0)
        D_clean = np.clip(D_clean, 0, 1)
        
        D_condensed = squareform(D_clean, checks=False)
        hc = linkage(D_condensed, method='average')
        
        gap_scores = []
        valid_k_values = []
        n_refs = 10  # Number of reference distributions
        
        for k in k_range:
            if k < 2 or k >= len(D_clean):
                continue
            
            try:
                clusters = cut_tree(hc, n_clusters=k).flatten()
                n_clusters = len(np.unique(clusters))
                
                if n_clusters < 2 or n_clusters >= len(D_clean):
                    continue
                
                # Calculate within-cluster dispersion
                W_k = 0.0
                for cluster_id in np.unique(clusters):
                    cluster_mask = clusters == cluster_id
                    cluster_distances = D_clean[np.ix_(cluster_mask, cluster_mask)]
                    if len(cluster_distances) > 1:
                        W_k += np.sum(cluster_distances) / (2.0 * np.sum(cluster_mask))
                
                # Calculate expected dispersion under null reference
                W_kb_list = []
                for ref_idx in range(n_refs):
                    # Generate random reference data (uniform distribution)
                    np.random.seed(2025 + ref_idx)
                    random_indices = np.random.permutation(len(D_clean))
                    D_ref = D_clean[np.ix_(random_indices, random_indices)]
                    D_ref_condensed = squareform(D_ref, checks=False)
                    hc_ref = linkage(D_ref_condensed, method='average')
                    clusters_ref = cut_tree(hc_ref, n_clusters=k).flatten()
                    
                    W_kb = 0.0
                    for cluster_id in np.unique(clusters_ref):
                        cluster_mask = clusters_ref == cluster_id
                        cluster_distances = D_ref[np.ix_(cluster_mask, cluster_mask)]
                        if len(cluster_distances) > 1:
                            W_kb += np.sum(cluster_distances) / (2.0 * np.sum(cluster_mask))
                    W_kb_list.append(np.log(W_kb + 1e-10))
                
                gap_k = np.mean(W_kb_list) - np.log(W_k + 1e-10)
                gap_scores.append(gap_k)
                valid_k_values.append(k)
            except Exception:
                continue
        
        if len(gap_scores) == 0:
            raise ValueError("Could not compute gap statistic for any k in range")
        
        # Find k that maximizes gap
        best_idx = np.argmax(gap_scores)
        k_optimal = valid_k_values[best_idx]
        
        logger.info(f"Determined optimal k={k_optimal} using gap_statistic (max score: {gap_scores[best_idx]:.4f})")
        
        return {
            'k_optimal': k_optimal,
            'method': 'gap_statistic',
            'scores': dict(zip(valid_k_values, gap_scores)),
            'max_score': gap_scores[best_idx],
            'subsampled': sample_idx is not None,
            'sample_size': max_sample if sample_idx is not None else n_total
        }


def perform_eac_consensus(
    label_mat: pd.DataFrame,
    k_optimal: int,
    ground_truth: Optional[pd.Series] = None,
) -> Dict[str, any]:
    """
    Perform Evidence Accumulation Clustering (EAC) consensus.
    
    Parameters:
    -----------
    label_mat : pd.DataFrame
        Matrix of model labels (each column is a model, rows are spots/cells)
    k_optimal : int
        Number of clusters for final consensus
    ground_truth : pd.Series, optional
        Ground truth labels for ARI evaluation
    
    Returns:
    --------
    dict with consensus results
    """
    n = len(label_mat)
    if n == 0:
        raise ValueError("label_mat is empty")
    
    if len(label_mat.columns) < 2:
        raise ValueError(f"Need at least 2 models for consensus, got {len(label_mat.columns)}")
    
    # Build co-association matrix C
    C = np.zeros((n, n))
    
    for col in label_mat.columns:
        lab = label_mat[col].values
        for g in np.unique(lab):
            idx = np.where(lab == g)[0]
            # Increment co-association for all pairs in same cluster
            C[np.ix_(idx, idx)] += 1
    
    C = C / len(label_mat.columns)
    
    # Dissimilarity matrix
    D = 1 - C
    np.fill_diagonal(D, 0)
    
    # Hierarchical clustering and cut at k_optimal
    D_condensed = squareform(D, checks=False)
    hc = linkage(D_condensed, method='average')
    consensus_labels = cut_tree(hc, n_clusters=k_optimal).flatten()
    
    # Consensus ARI (evaluation-only)
    consensus_ari = None
    if ground_truth is not None:
        consensus_ari = adjusted_rand_score(ground_truth.values, consensus_labels)
        logger.info(f"Consensus ARI: {consensus_ari:.4f}")
    
    return {
        'consensus_labels': consensus_labels,
        'consensus_ari': consensus_ari,
        'k_optimal': k_optimal,
    }


def run_consensus_clustering(
    ranking_csv: Union[str, Path],
    *,
    assignments_csv: Union[str, Path, None] = None,
    id_col: Optional[str] = None,
    use_separate_csvs: bool = False,
    assignments_dir: Union[str, Path, None] = None,
    assignment_csv_list: Optional[Sequence[Union[str, Path]]] = None,
    output_csv: Union[str, Path, None] = None,
    output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
    model_names: Sequence[str] | None = None,
    top_k: Optional[int] = 5,
    selection_mode: Literal["manual", "top_k"] = "manual",
    k_method: Literal["median_from_models", "mode_from_models", "silhouette", "gap_statistic"] = "median_from_models",
    k_range: range = DEFAULT_K_RANGE,
    k_mode: Literal["fixed", "auto"] = "auto",
    fixed_k: Optional[int] = None,
    ground_truth_col: Optional[str] = None,
    random_state: Optional[int] = 0,
) -> pd.DataFrame:
    """
    Run consensus clustering over selected models and produce the L-STAR assignment.
    
    Parameters
    ----------
    ranking_csv : str or Path
        Path to ranking CSV with model names and win rates
    
    assignments_csv : str or Path, optional
        Path to a single CSV file containing all model assignments (one column per model).
        This is the default mode. If not provided and use_separate_csvs=False, will raise an error.
        When using this mode, fuzzy name matching is automatically enabled.
    
    id_col : str, optional
        Name of the ID column in assignments_csv (only used in combined CSV mode).
        Required when assignments_csv is provided.
    
    use_separate_csvs : bool, default False
        If True, use the legacy mode with separate CSV files per model (one CSV per model).
        Requires either assignments_dir or assignment_csv_list to be provided.
        If False (default), uses assignments_csv mode.
    
    assignments_dir : str or Path, optional
        Directory containing per-model assignment CSVs (one CSV per model).
        Only used if use_separate_csvs=True.
    
    assignment_csv_list : sequence of paths, optional
        Explicit list of per-model assignment CSV paths.
        Only used if use_separate_csvs=True.
    
    output_csv : str or Path, optional
        Output CSV path. If None, uses output_dir / "L_STAR_consensus.csv"
    
    output_dir : str or Path
        Output directory for consensus CSV
    
    model_names : sequence of str, optional
        Manually specified model names (required if selection_mode="manual")
    
    top_k : int, optional
        Number of top models to select (used if selection_mode="top_k")
    
    selection_mode : {"manual", "top_k"}
        How to select models: "manual" uses model_names, "top_k" selects from ranking
    
    k_method : {"median_from_models", "mode_from_models", "silhouette", "gap_statistic"}
        Method to determine optimal number of clusters
    
    k_range : range
        Valid range of k values for auto-determination
    
    k_mode : {"fixed", "auto"}
        Whether to use fixed_k or auto-determine k
    
    fixed_k : int, optional
        Fixed number of clusters (used if k_mode="fixed")
    
    ground_truth_col : str, optional
        Name of ground truth column for ARI evaluation
    
    Returns
    -------
    consensus_df : pd.DataFrame
        DataFrame with 'L-STAR' column containing consensus cluster labels
    """
    ranking_csv = Path(ranking_csv)
    output_dir = Path(output_dir)
    
    # Read ranking CSV
    ranking_df = read_ranking_csv(ranking_csv)
    logger.info(f"Loaded ranking CSV: {ranking_csv}")
    
    # Select models
    if selection_mode == "manual":
        if model_names is None:
            raise ValueError(
                "selection_mode='manual' requires model_names parameter. "
                "Provide a list of model names to use for consensus clustering."
            )
        selected_models = list(model_names)
        logger.info(f"Using manually specified models: {selected_models}")
        
        # Validate that all models exist in ranking
        model_col = "model" if "model" in ranking_df.columns else ranking_df.columns[0]
        available_models = set(ranking_df[model_col].values)
        missing = [m for m in selected_models if m not in available_models]
        if missing:
            raise ValueError(
                f"Models not found in ranking CSV: {', '.join(missing)}. "
                f"Available models: {', '.join(sorted(available_models))}"
            )
    
    elif selection_mode == "top_k":
        if top_k is None or top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        
        # Get win_rate column
        if "win_rate" in ranking_df.columns:
            win_col = "win_rate"
        elif "winning_rate" in ranking_df.columns:
            win_col = "winning_rate"
        elif "WinningRate" in ranking_df.columns:
            win_col = "WinningRate"
        else:
            win_col = ranking_df.columns[-1]
        
        # Get model column
        if "model" in ranking_df.columns:
            model_col = "model"
        elif "Model" in ranking_df.columns:
            model_col = "Model"
        else:
            model_col = ranking_df.columns[0]
        
        sorted_models = ranking_df.sort_values(win_col, ascending=False)
        available_count = len(sorted_models)
        
        if top_k > available_count:
            logger.warning(
                f"top_k={top_k} is larger than available models ({available_count}), "
                f"using all {available_count} models"
            )
            top_k = available_count
        
        selected_models = sorted_models[model_col].head(top_k).tolist()
        logger.info(f"Selected top {len(selected_models)} models by win_rate: {selected_models}")
    
    else:
        raise ValueError(f"Unknown selection_mode: {selection_mode}")
    
    if len(selected_models) < 2:
        raise ValueError(f"Need at least 2 models for consensus, got {len(selected_models)}")
    
    # Determine which mode to use
    # Default: combined CSV mode (unless explicitly using separate CSVs)
    if use_separate_csvs:
        # Legacy mode: separate CSV files per model
        use_combined_csv = False
        use_fuzzy_matching = False
    else:
        # Default mode: combined CSV
        use_combined_csv = True
        use_fuzzy_matching = True
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
    
    # Load assignment data
    if use_combined_csv:
        # NEW MODE: Combined CSV with fuzzy matching
        combined_csv_path = Path(assignments_csv)
        combined_df, assignment_normalized_map = read_combined_assignments_csv(
            combined_csv_path, id_col=id_col
        )
        
        # Build name mappings for fuzzy matching
        ranking_map, assignment_map, _ = build_name_mappings(
            ranking_df=ranking_df,
            combined_df=combined_df,
            id_col=id_col,
            use_fuzzy_matching=use_fuzzy_matching,
        )
        
        # Match selected models to assignment columns
        model_to_assignment_col, _ = match_models_with_fuzzy_matching(
            selected_models=selected_models,
            ranking_map=ranking_map,
            assignment_map=assignment_map,
            image_map={},  # Images not needed for consensus clustering
            ranking_csv_path=ranking_csv,
            combined_csv_path=combined_csv_path,
            require_images=False,  # Images are not required for consensus clustering
        )
        
        # Get ID values from combined CSV
        id_values = combined_df[id_col].values
        
        # Build label matrix from combined CSV
        label_mat = pd.DataFrame(index=range(len(id_values)))
        for model_name in selected_models:
            assignment_col = model_to_assignment_col[model_name]
            if assignment_col not in combined_df.columns:
                raise ValueError(
                    f"Assignment column '{assignment_col}' not found in combined CSV {combined_csv_path}. "
                    f"Available columns: {list(combined_df.columns)}"
                )
            label_mat[model_name] = combined_df[assignment_col].values
        
        # Store combined_df for ground truth access later
        assignment_dfs = None
        first_df = combined_df
        
    else:
        # EXISTING MODE: Separate CSV files per model (unchanged behavior)
        if assignment_csv_list is not None:
            assignment_paths = [Path(p) for p in assignment_csv_list]
            assignment_dfs = read_assignment_csvs(assignment_paths, model_names=selected_models)
        elif assignments_dir is not None:
            assignment_paths_dict = find_assignment_csvs(Path(assignments_dir), selected_models)
            assignment_paths = list(assignment_paths_dict.values())
            assignment_dfs = read_assignment_csvs(assignment_paths, model_names=selected_models)
        else:
            raise ValueError(
                "Either assignments_dir, assignment_csv_list, or assignments_csv must be provided. "
                "Specify the directory containing assignment CSVs, a list of CSV paths, or a combined CSV file."
            )
        
        # Align all DataFrames by ID column (assume first column is ID)
        first_df = list(assignment_dfs.values())[0]
        id_col = first_df.columns[0]
        id_values = first_df[id_col].values
        
        # Build label matrix
        label_mat = pd.DataFrame(index=range(len(id_values)))
        for model_name in selected_models:
            if model_name not in assignment_dfs:
                raise ValueError(
                    f"Model '{model_name}' not found in assignment CSVs. "
                    f"Available models: {', '.join(assignment_dfs.keys())}"
                )
            
            df = assignment_dfs[model_name]
            
            # Find clustering column (exclude ID and ground truth columns)
            clustering_cols = [c for c in df.columns 
                              if c != id_col and c != ground_truth_col]
            if not clustering_cols:
                raise ValueError(
                    f"No clustering column found in assignment CSV for model '{model_name}'. "
                    f"Columns: {list(df.columns)}"
                )
            
            # Use first non-ID, non-GT column as clustering labels
            clustering_col = clustering_cols[0]
            if len(clustering_cols) > 1:
                logger.warning(
                    f"Multiple clustering columns found for model '{model_name}', "
                    f"using first: {clustering_col}"
                )
            
            # Align by ID
            df_aligned = df.set_index(id_col).reindex(id_values)
            label_mat[model_name] = df_aligned[clustering_col].values
    
    # Convert labels to integer codes
    for col in label_mat.columns:
        label_mat[col] = pd.Categorical(label_mat[col]).codes
    
    # Build co-association matrix to get distance matrix for silhouette/gap_statistic
    n = len(label_mat)
    C = np.zeros((n, n))
    for col in label_mat.columns:
        lab = label_mat[col].values
        for g in np.unique(lab):
            idx = np.where(lab == g)[0]
            C[np.ix_(idx, idx)] += 1
    C = C / len(label_mat.columns)
    D = 1 - C
    np.fill_diagonal(D, 0)
    
    # Determine k
    if k_mode == "fixed":
        if fixed_k is None:
            raise ValueError("k_mode='fixed' requires fixed_k parameter")
        k_optimal = fixed_k
        logger.info(f"Using fixed k={k_optimal}")
    else:  # k_mode == "auto"
        if k_method in ["silhouette", "gap_statistic"]:
            k_result = determine_optimal_k(
                distance_matrix=D,
                method=k_method,
                k_range=k_range
            )
        else:  # median_from_models or mode_from_models
            k_result = determine_optimal_k(
                pred_data_for_mode=label_mat,
                method=k_method,
                k_range=k_range
            )
        k_optimal = k_result['k_optimal']
        logger.info(f"Auto-determined k={k_optimal} using {k_method}")
    
    # Get ground truth if available
    ground_truth = None
    if ground_truth_col:
        if use_combined_csv:
            # For combined CSV, ground truth is in the same DataFrame
            if ground_truth_col in combined_df.columns:
                ground_truth = combined_df[ground_truth_col]
            else:
                logger.warning(f"Ground truth column '{ground_truth_col}' not found, skipping ARI evaluation")
        else:
            # For separate CSVs, use first DataFrame
            if ground_truth_col in first_df.columns:
                ground_truth = first_df.set_index(id_col).reindex(id_values)[ground_truth_col]
            else:
                logger.warning(f"Ground truth column '{ground_truth_col}' not found, skipping ARI evaluation")
    
    # Perform EAC consensus (reuse the distance matrix we already built)
    # Note: perform_eac_consensus will rebuild it, but that's fine for now
    consensus_result = perform_eac_consensus(label_mat, k_optimal, ground_truth)
    consensus_labels = consensus_result['consensus_labels']
    
    # Build output DataFrame with proper column order: [id_col, "L-STAR"]
    consensus_df = pd.DataFrame({
        id_col: id_values,
        "L-STAR": consensus_labels,
    })
    
    # Ensure column order: id_col first, then L-STAR
    consensus_df = consensus_df[[id_col, "L-STAR"]]
    
    # Write output CSV
    if output_csv is None:
        output_csv = output_dir / CONSENSUS_CSV_NAME
    else:
        output_csv = Path(output_csv)
    
    write_consensus_csv(
        output_csv,
        consensus_labels,
        id_column=pd.Series(id_values),
        id_column_name=id_col,
        keep_original_columns=False,
    )
    
    logger.info(f"Consensus clustering complete. Output: {output_csv}")
    logger.info(f"  - Models used: {selected_models}")
    logger.info(f"  - k={k_optimal}")
    if consensus_result['consensus_ari'] is not None:
        logger.info(f"  - ARI={consensus_result['consensus_ari']:.4f}")
    
    return consensus_df

