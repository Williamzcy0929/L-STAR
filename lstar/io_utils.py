"""I/O utilities for reading/writing CSVs, JSON, and handling paths."""

import json
import csv
import logging
import re
from pathlib import Path
from typing import Optional, Sequence, Dict, Any, Tuple, Union, List
import pandas as pd

logger = logging.getLogger(__name__)


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, creating if necessary."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_name(name: str) -> str:
    """
    Normalize a model name by:
    - Removing all non-alphanumeric characters,
    - Converting to lowercase.
    
    Examples:
        "GraphST" -> "graphst"
        "GraphST_v1" -> "graphstv1"
        "Graph-ST (v1)" -> "graphstv1"
        "spa_gcn" -> "spagcn"
        "SpaGCN.png" -> "spagcn" (after stripping extension first)
    """
    return re.sub(r"[^0-9A-Za-z]+", "", name).lower()


def read_ranking_csv(csv_path: Path) -> pd.DataFrame:
    """
    Read ranking CSV and validate required columns.
    
    Raises:
        FileNotFoundError: If CSV doesn't exist
        ValueError: If required columns are missing
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Ranking CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Check for model column
    if "model" not in df.columns and "Model" not in df.columns:
        if len(df.columns) == 0:
            raise ValueError(f"Ranking CSV {csv_path} has no columns")
        # Use first column as model column
        logger.warning(f"Ranking CSV {csv_path} missing 'model' column, using first column: {df.columns[0]}")
    
    # Check for win_rate column
    if "win_rate" not in df.columns and "winning_rate" not in df.columns and "WinningRate" not in df.columns:
        if len(df.columns) < 2:
            raise ValueError(f"Ranking CSV {csv_path} missing win_rate column")
        # Use last column as win_rate
        logger.warning(f"Ranking CSV {csv_path} missing 'win_rate' column, using last column: {df.columns[-1]}")
    
    return df


def read_assignment_csvs(
    assignment_csvs: Sequence[Path],
    model_names: Optional[Sequence[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Read multiple assignment CSVs and validate alignment.
    
    Args:
        assignment_csvs: List of CSV file paths
        model_names: Optional list of model names (if None, inferred from filenames)
    
    Returns:
        Dict mapping model_name -> DataFrame
    
    Raises:
        ValueError: If CSVs have mismatched row counts or missing models
    """
    if not assignment_csvs:
        raise ValueError("No assignment CSVs provided")
    
    dfs = {}
    first_df = None
    first_id_cols = None
    
    for idx, csv_path in enumerate(assignment_csvs):
        if not csv_path.exists():
            raise FileNotFoundError(f"Assignment CSV not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Infer model name from filename if not provided
        if model_names and idx < len(model_names):
            model_name = model_names[idx]
        else:
            model_name = csv_path.stem
        
        # Check row count alignment
        if first_df is None:
            first_df = df
            first_id_cols = list(df.columns)
            first_n_rows = len(df)
        else:
            if len(df) != first_n_rows:
                raise ValueError(
                    f"Assignment CSV {csv_path} has {len(df)} rows, "
                    f"but expected {first_n_rows} rows (from {assignment_csvs[0]})"
                )
            
            # Check ID column alignment (assume first column is ID)
            if list(df.columns)[0] != first_id_cols[0]:
                logger.warning(
                    f"Assignment CSV {csv_path} has different first column name "
                    f"({df.columns[0]}) than first CSV ({first_id_cols[0]})"
                )
        
        dfs[model_name] = df
        logger.info(f"Loaded assignment CSV for model '{model_name}': {csv_path} ({len(df)} rows)")
    
    return dfs


def find_assignment_csvs(
    assignments_dir: Path,
    model_names: Sequence[str],
) -> Dict[str, Path]:
    """
    Find assignment CSV files in a directory.
    
    Args:
        assignments_dir: Directory to search
        model_names: List of model names to find
    
    Returns:
        Dict mapping model_name -> CSV path
    
    Raises:
        FileNotFoundError: If directory doesn't exist
        ValueError: If any model CSV is missing
    """
    assignments_dir = Path(assignments_dir)
    if not assignments_dir.is_dir():
        raise FileNotFoundError(f"Assignments directory not found: {assignments_dir}")
    
    found = {}
    missing = []
    
    for model_name in model_names:
        # Try multiple naming conventions
        candidates = [
            assignments_dir / f"{model_name}.csv",
            assignments_dir / f"{model_name}_pred_label.csv",
            assignments_dir / f"{model_name}_labels.csv",
        ]
        
        found_path = None
        for candidate in candidates:
            if candidate.exists():
                found_path = candidate
                break
        
        if found_path:
            found[model_name] = found_path
        else:
            missing.append(model_name)
    
    if missing:
        raise ValueError(
            f"Could not find assignment CSVs for models: {', '.join(missing)}. "
            f"Searched in: {assignments_dir}"
        )
    
    return found


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    """Append a JSON object to a JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[Dict[str, Any]]:
    """Read all JSON objects from a JSONL file."""
    results = []
    if not path.exists():
        return results
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON line in {path}")
                continue
    
    return results


def read_combined_assignments_csv(
    csv_path: Union[str, Path],
    id_col: str = "spot_id",
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Read a combined assignments CSV where all models are in one file.
    
    Args:
        csv_path: Path to combined assignments CSV
        id_col: Name of the ID column
    
    Returns:
        Tuple of (DataFrame, normalized_to_original) where:
        - DataFrame: The full CSV with ID column and model columns
        - normalized_to_original: Dict mapping normalized column name -> original column name
    
    Raises:
        FileNotFoundError: If CSV doesn't exist
        ValueError: If id_col is missing or no model columns found
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Combined assignments CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Check for ID column
    if id_col not in df.columns:
        raise ValueError(
            f"ID column '{id_col}' not found in combined assignments CSV {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )
    
    # Get model columns (all columns except id_col)
    model_columns = [col for col in df.columns if col != id_col]
    
    if len(model_columns) == 0:
        raise ValueError(
            f"No model columns found in combined assignments CSV {csv_path}. "
            f"Only found ID column: {id_col}"
        )
    
    # Build normalized name mapping
    normalized_to_original = {}
    for col in model_columns:
        norm_name = normalize_name(col)
        if norm_name in normalized_to_original:
            # Ambiguous: multiple columns normalize to same name
            existing = normalized_to_original[norm_name]
            raise ValueError(
                f"Ambiguous normalized name '{norm_name}' in combined assignments CSV {csv_path}: "
                f"columns ['{existing}', '{col}'] both normalize to '{norm_name}'"
            )
        normalized_to_original[norm_name] = col
    
    logger.info(
        f"Loaded combined assignments CSV: {csv_path} "
        f"({len(df)} rows, {len(model_columns)} model columns)"
    )
    
    return df, normalized_to_original


def build_name_mappings(
    ranking_df: pd.DataFrame,
    combined_df: Optional[pd.DataFrame] = None,
    id_col: Optional[str] = None,
    image_dir: Optional[Path] = None,
    use_fuzzy_matching: bool = False,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Build normalized name mappings for ranking, assignments, and images.
    
    Args:
        ranking_df: Ranking DataFrame with model names
        combined_df: Optional combined assignments DataFrame
        id_col: Optional ID column name (for combined_df)
        image_dir: Optional image directory
        use_fuzzy_matching: If True, use normalized matching; if False, exact matching
    
    Returns:
        Tuple of (ranking_map, assignment_map, image_map) where each maps:
        normalized_name -> original_name
    """
    ranking_map = {}
    assignment_map = {}
    image_map = {}
    
    # Build ranking map
    if "model" in ranking_df.columns:
        model_col = "model"
    elif "Model" in ranking_df.columns:
        model_col = "Model"
    else:
        model_col = ranking_df.columns[0]
    
    for model_name in ranking_df[model_col].unique():
        if use_fuzzy_matching:
            norm_name = normalize_name(model_name)
        else:
            norm_name = model_name
        if norm_name in ranking_map and ranking_map[norm_name] != model_name:
            # Shouldn't happen, but handle it
            logger.warning(
                f"Duplicate normalized name '{norm_name}' in ranking: "
                f"'{ranking_map[norm_name]}' and '{model_name}'"
            )
        ranking_map[norm_name] = model_name
    
    # Build assignment map (only if combined_df provided)
    if combined_df is not None and id_col is not None:
        for col in combined_df.columns:
            if col == id_col:
                continue
            if use_fuzzy_matching:
                norm_name = normalize_name(col)
            else:
                norm_name = col
            if norm_name in assignment_map:
                raise ValueError(
                    f"Ambiguous normalized name '{norm_name}' in assignments: "
                    f"columns ['{assignment_map[norm_name]}', '{col}'] both normalize to '{norm_name}'"
                )
            assignment_map[norm_name] = col
    
    # Build image map (only if image_dir provided)
    if image_dir is not None:
        image_dir = Path(image_dir)
        if image_dir.is_dir():
            # Support multiple image formats: png, jpg, jpeg
            image_extensions = ["*.png", "*.jpg", "*.jpeg"]
            for ext in image_extensions:
                for img_file in image_dir.glob(ext):
                    basename = img_file.stem
                    if use_fuzzy_matching:
                        norm_name = normalize_name(basename)
                    else:
                        norm_name = basename
                    if norm_name in image_map:
                        existing_file = image_map[norm_name]
                        raise ValueError(
                            f"Ambiguous normalized name '{norm_name}' in images: "
                            f"files ['{existing_file}.{Path(existing_file).suffix}', '{basename}.{img_file.suffix}'] "
                            f"both normalize to '{norm_name}'"
                        )
                    image_map[norm_name] = basename
    
    return ranking_map, assignment_map, image_map


def match_models_with_fuzzy_matching(
    selected_models: Sequence[str],
    ranking_map: Dict[str, str],
    assignment_map: Dict[str, str],
    image_map: Dict[str, str],
    ranking_csv_path: Optional[Path] = None,
    combined_csv_path: Optional[Path] = None,
    image_dir: Optional[Path] = None,
    require_images: bool = False,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Match selected models from ranking to assignment columns and images using fuzzy matching.
    
    Args:
        selected_models: List of model names from ranking CSV
        ranking_map: Mapping normalized -> original for ranking
        assignment_map: Mapping normalized -> original for assignments
        image_map: Mapping normalized -> original for images
        ranking_csv_path: Optional path for error messages
        combined_csv_path: Optional path for error messages
        image_dir: Optional path for error messages
        require_images: If True, log warnings when images cannot be matched. If False (default),
            suppress warnings since images are optional in this context.
    
    Returns:
        Tuple of (model_to_assignment_col, model_to_image) where:
        - model_to_assignment_col: Maps ranking model name -> assignment column name
        - model_to_image: Maps ranking model name -> image basename
    
    Raises:
        ValueError: If any model cannot be matched or has ambiguous matches
    """
    model_to_assignment_col = {}
    model_to_image = {}
    
    ranking_csv_str = f" in ranking CSV {ranking_csv_path}" if ranking_csv_path else ""
    combined_csv_str = f" in combined assignments CSV {combined_csv_path}" if combined_csv_path else ""
    image_dir_str = f" in image directory {image_dir}" if image_dir else ""
    
    for model_name in selected_models:
        # Normalize the ranking model name
        norm_name = normalize_name(model_name)
        
        # Find original ranking name (should always exist)
        if norm_name not in ranking_map:
            raise ValueError(
                f"Internal error: normalized name '{norm_name}' not found in ranking map "
                f"for model '{model_name}'{ranking_csv_str}"
            )
        
        # Match to assignment column
        if norm_name not in assignment_map:
            available = list(assignment_map.keys())
            raise ValueError(
                f"Could not match ranking model '{model_name}' (normalized '{norm_name}') "
                f"to any column{combined_csv_str}. "
                f"Available normalized column names: {available}"
            )
        model_to_assignment_col[model_name] = assignment_map[norm_name]
        
        # Match to image (optional - images might not exist for all models)
        if norm_name in image_map:
            model_to_image[model_name] = image_map[norm_name]
        else:
            # Only log warning if images are actually required
            if require_images:
                logger.warning(
                    f"Could not match ranking model '{model_name}' (normalized '{norm_name}') "
                    f"to any image file{image_dir_str}. This is OK if images are not required."
                )
            else:
                # Images are not required, so suppress the warning (use debug level if needed)
                logger.debug(
                    f"Could not match ranking model '{model_name}' (normalized '{norm_name}') "
                    f"to any image file{image_dir_str}. Images are not required in this context."
                )
    
    return model_to_assignment_col, model_to_image


def read_second_round_results(output_dir: Path) -> Optional[List[str]]:
    """
    Look for a second-round reasoning JSON file under output_dir (and/or
    output_dir / 'pairwise'), and return the list of REAL model IDs stored
    under the 'final_model_ids' key.
    
    Returns:
        A list of model IDs if found and valid, otherwise None.
    """
    output_dir = Path(output_dir)
    second_round_files = []
    
    # 1. Search for *_second_round_reasoning.json under:
    #    - output_dir / "pairwise", then
    #    - output_dir
    candidate_dirs = [output_dir / "pairwise", output_dir]
    
    for candidate_dir in candidate_dirs:
        if candidate_dir.exists():
            found = list(candidate_dir.glob("*second_round*.json"))
            second_round_files.extend(found)
    
    # 2. Choose the most recent matching file (by modification time) if multiple exist
    if not second_round_files:
        logger.debug(f"No second-round reasoning JSON found in {output_dir} or {output_dir / 'pairwise'}")
        return None
    
    second_round_file = sorted(second_round_files, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    
    # 3. Open it, parse as JSON, and return data.get("final_model_ids") if it is a non-empty list of strings
    try:
        with open(second_round_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        final_model_ids = data.get("final_model_ids", [])
        
        # 4. If no file found, or parsing fails, or final_model_ids is missing/empty, return None (do not raise)
        if isinstance(final_model_ids, list) and len(final_model_ids) > 0 and all(isinstance(x, str) for x in final_model_ids):
            logger.info(f"Read second-round reasoning results from {second_round_file}: {len(final_model_ids)} models selected")
            return final_model_ids
        else:
            logger.warning(f"Second-round JSON {second_round_file} has invalid or empty 'final_model_ids'")
            return None
    
    except (FileNotFoundError, json.JSONDecodeError, KeyError, Exception) as e:
        logger.warning(f"Could not read second-round reasoning file {second_round_file}: {e}")
        return None


def write_consensus_csv(
    output_path: Path,
    consensus_labels: Sequence[int],
    id_column: Optional[pd.Series] = None,
    id_column_name: str = "spot_id",
    keep_original_columns: bool = False,
    original_df: Optional[pd.DataFrame] = None,
) -> None:
    """
    Write consensus clustering results to CSV.
    
    Ensures column order: [id_column, "L-STAR", <other columns>]
    
    Args:
        output_path: Path to write CSV
        consensus_labels: Consensus cluster labels
        id_column: Optional ID column (e.g., spot_id, cell_id)
        id_column_name: Name for ID column if id_column is provided
        keep_original_columns: If True, include original columns from original_df
        original_df: Original DataFrame to copy columns from
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {"L-STAR": consensus_labels}
    
    if id_column is not None:
        data[id_column_name] = id_column.values
    
    df = pd.DataFrame(data)
    
    if keep_original_columns and original_df is not None:
        # Merge with original, keeping only non-clustering columns
        for col in original_df.columns:
            if col not in data and col != "L-STAR":
                df[col] = original_df[col].values
    
    # Reorder columns: [id_column, "L-STAR", <other columns>]
    columns = list(df.columns)
    if id_column_name in columns and "L-STAR" in columns:
        other_cols = [c for c in columns if c not in (id_column_name, "L-STAR")]
        new_cols = [id_column_name, "L-STAR"] + other_cols
        df = df[new_cols]
    elif "L-STAR" in columns:
        # If no ID column, just ensure L-STAR is first
        other_cols = [c for c in columns if c != "L-STAR"]
        new_cols = ["L-STAR"] + other_cols
        df = df[new_cols]
    
    df.to_csv(output_path, index=False)
    logger.info(f"Wrote consensus CSV: {output_path} ({len(df)} rows)")

