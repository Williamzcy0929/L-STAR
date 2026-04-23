"""Image generation from spatial locations and domain assignments using Palo for color optimization."""

import os
from importlib import resources
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def parse_spatial_locations(
    spatial_locations_csv: Union[str, Path],
    id_col: str = "spot_id",
) -> pd.DataFrame:
    """
    Read spatial locations CSV file.
    
    Supports two formats:
    1. CSV with explicit x, y columns: spot_id, x, y
    2. CSV with coordinates embedded in spot_id (e.g., "4.92x30.97"): spot_id, ...
       In this case, coordinates are extracted from the spot_id format "x_coordxy_coord"
    
    Args:
        spatial_locations_csv: Path to CSV file with spatial coordinates
        id_col: Name of the ID column (default: "spot_id")
    
    Returns:
        DataFrame with id_col, x, y columns
    
    Raises:
        FileNotFoundError: If CSV doesn't exist
        ValueError: If required columns are missing
    """
    spatial_locations_csv = Path(spatial_locations_csv)
    if not spatial_locations_csv.exists():
        raise FileNotFoundError(f"Spatial locations CSV not found: {spatial_locations_csv}")
    
    df = pd.read_csv(spatial_locations_csv)
    
    if id_col not in df.columns:
        raise ValueError(
            f"ID column '{id_col}' not found in spatial locations CSV {spatial_locations_csv}. "
            f"Available columns: {list(df.columns)}"
        )
    
    # Try to find x and y columns (case-insensitive)
    x_col = None
    y_col = None
    
    # Common column name patterns for x and y coordinates
    x_patterns = ['x', 'x_coord', 'x_coordinate', 'coord_x', 'xcoord', 'xpos', 'x_position']
    y_patterns = ['y', 'y_coord', 'y_coordinate', 'coord_y', 'ycoord', 'ypos', 'y_position']
    
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in x_patterns and x_col is None:
            x_col = col
        elif col_lower in y_patterns and y_col is None:
            y_col = col
    
    # If still not found, try to infer from column order (assume id_col is first, then x, then y)
    if x_col is None or y_col is None:
        remaining_cols = [c for c in df.columns if c != id_col]
        if len(remaining_cols) >= 2:
            if x_col is None:
                x_col = remaining_cols[0]
                logger.info(f"Inferred x column: {x_col}")
            if y_col is None:
                y_col = remaining_cols[1] if len(remaining_cols) > 1 else remaining_cols[0]
                logger.info(f"Inferred y column: {y_col}")
    
    # If x and y columns are found, use them
    if x_col is not None and y_col is not None:
        result_df = df[[id_col, x_col, y_col]].copy()
        result_df.columns = [id_col, 'x', 'y']
        logger.info(
            f"Loaded spatial locations CSV: {spatial_locations_csv} "
            f"({len(result_df)} spots, x_col={x_col}, y_col={y_col})"
        )
        return result_df
    
    # If x and y columns are not found, try to extract from spot_id format "x_coordxy_coord"
    logger.info("No explicit x, y columns found. Attempting to extract coordinates from spot_id...")
    
    # Try to parse coordinates from spot_id (format: "x_coordxy_coord" or similar)
    import re
    coords_extracted = []
    for spot_id in df[id_col]:
        # Try pattern like "4.92x30.97" or "4.92,30.97" or "4.92 30.97"
        match = re.search(r'([\d.]+)[x, ]([\d.]+)', str(spot_id))
        if match:
            x_val = float(match.group(1))
            y_val = float(match.group(2))
            coords_extracted.append((x_val, y_val))
        else:
            coords_extracted.append((None, None))
    
    if all(c[0] is not None and c[1] is not None for c in coords_extracted):
        result_df = df[[id_col]].copy()
        result_df['x'] = [c[0] for c in coords_extracted]
        result_df['y'] = [c[1] for c in coords_extracted]
        logger.info(
            f"Extracted coordinates from spot_id format: {spatial_locations_csv} "
            f"({len(result_df)} spots)"
        )
        return result_df
    
    # If all else fails, raise an error
    raise ValueError(
        f"Could not find x and y coordinate columns in spatial locations CSV {spatial_locations_csv}. "
        f"Available columns: {list(df.columns)}. "
        f"Expected either: (1) explicit x, y columns, or (2) coordinates embedded in {id_col} "
        f"in format like '4.92x30.97'."
    )


def get_palo_optimized_colors(
    assignments: pd.Series,
    spatial_coords: pd.DataFrame,
    id_col: str = "spot_id",
    rgb_weight: Optional[Tuple[float, float, float]] = None,
    color_blind_fun: Optional[str] = None,
) -> Dict[int, Tuple[float, float, float]]:
    """
    Use Palo R package to optimize color palette for spatial domain assignments.
    
    This function calls Palo::Palo() via Rscript, which optimizes colors so that
    spatially neighboring clusters have visually distinct colors.
    
    Args:
        assignments: Series of cluster assignments (indexed by spot_id)
        spatial_coords: DataFrame with id_col, x, y columns
        id_col: Name of the ID column
        rgb_weight: Optional tuple of (r_weight, g_weight, b_weight) for Palo
        color_blind_fun: Optional color blindness function name ("deutan", "protan", "tritan")
    
    Returns:
        Dictionary mapping cluster_id -> (r, g, b) tuple (values 0-1)
    
    Raises:
        RuntimeError: If R/Palo execution fails (falls back to default colors)
    """
    # Get unique clusters
    unique_clusters = sorted(assignments.unique())
    n_clusters = len(unique_clusters)
    
    if n_clusters == 0:
        logger.warning("No clusters found, using default colors")
        return {}
    
    # Merge assignments with spatial coordinates
    merged = pd.merge(
        pd.DataFrame({id_col: assignments.index, 'cluster': assignments.values}),
        spatial_coords[[id_col, 'x', 'y']],
        on=id_col,
        how='inner'
    )
    
    if len(merged) == 0:
        raise ValueError("No matching spots found between assignments and spatial coordinates")
    
    # Create temporary files for R script
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Write data files
        assignments_file = tmpdir_path / "assignments.csv"
        coords_file = tmpdir_path / "coords.csv"
        output_file = tmpdir_path / "colors.csv"
        
        # Write assignments (spot_id, cluster) - ensure cluster is string for R
        merged_assignments = merged[[id_col, 'cluster']].copy()
        merged_assignments['cluster'] = merged_assignments['cluster'].astype(str)
        merged_assignments.to_csv(assignments_file, index=False)
        
        # Write coordinates (spot_id, x, y)
        merged[[id_col, 'x', 'y']].to_csv(coords_file, index=False)
        
        success = run_palo_script(
            coords_file=coords_file,
            assignments_file=assignments_file,
            id_col=id_col,
            output_file=output_file,
            rgb_weight=rgb_weight,
            color_blind_fun=color_blind_fun,
        )
        if not success:
            return _get_default_colors(unique_clusters)

        colors_df = pd.read_csv(output_file)
        color_dict = {}
        for _, row in colors_df.iterrows():
            # Handle both numeric and string cluster IDs
            cluster_id_str = str(row['cluster'])
            try:
                # Try to convert to int if possible, otherwise keep as string key
                cluster_id = int(float(cluster_id_str))
            except (ValueError, TypeError):
                cluster_id = cluster_id_str

            r = float(row['r'])
            g = float(row['g'])
            b = float(row['b'])

            # Map back to original cluster ID type from unique_clusters
            # Find matching cluster in unique_clusters
            for orig_cluster in unique_clusters:
                if str(orig_cluster) == cluster_id_str:
                    color_dict[orig_cluster] = (r, g, b)
                    break
            else:
                # If not found, use the converted ID
                color_dict[cluster_id] = (r, g, b)

        logger.info(f"Generated {len(color_dict)} optimized colors using Palo::Palo()")
        return color_dict


def _resolve_r_script_path(
    script_filename: str,
    env_var_name: str,
) -> Optional[Path]:
    """Resolve an R script path from env var, packaged resources, or local fallbacks."""
    script_locations = []

    env_script = os.getenv(env_var_name)
    if env_script:
        script_locations.append(Path(env_script))

    try:
        bundled_script = resources.files("lstar").joinpath(f"resources/{script_filename}")
        script_locations.append(Path(str(bundled_script)))
    except Exception:
        # Ignore resource resolution errors; fallback paths are tried below.
        pass

    # Fallbacks for editable installs and local development.
    script_locations.extend([
        Path(__file__).parent.parent / "scripts" / script_filename,
        Path(__file__).parent.parent.parent / "scripts" / script_filename,
        Path("scripts") / script_filename,
    ])

    return next((loc for loc in script_locations if loc.exists()), None)


def run_palo_script(
    coords_file: Path,
    assignments_file: Path,
    id_col: str,
    output_file: Path,
    rgb_weight: Optional[Tuple[float, float, float]] = None,
    color_blind_fun: Optional[str] = None,
) -> bool:
    """Run the bundled Palo R script and return whether it succeeded."""
    r_script_path = _resolve_r_script_path(
        script_filename="run_palo.R",
        env_var_name="LSTAR_RUN_PALO_SCRIPT",
    )
    if r_script_path is None:
        logger.warning(
            "run_palo.R script not found. Using default colors. "
            "Please ensure run_palo.R is packaged or set LSTAR_RUN_PALO_SCRIPT."
        )
        return False

    cmd = [
        "Rscript",
        str(r_script_path),
        str(coords_file),
        str(assignments_file),
        id_col,
        str(output_file),
    ]
    if rgb_weight is not None:
        cmd.extend([f"{rgb_weight[0]},{rgb_weight[1]},{rgb_weight[2]}"])
        if color_blind_fun is not None:
            cmd.append(color_blind_fun)
    elif color_blind_fun is not None:
        cmd.extend(["", color_blind_fun])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        if result.returncode != 0:
            logger.warning(
                f"Palo R script failed with exit code {result.returncode}. "
                f"stderr: {result.stderr[:500] if result.stderr else 'No error message'}. "
                f"Using default colors."
            )
            return False
        if not output_file.exists():
            logger.warning("Palo output file not found, using default colors")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error("Palo R script timed out, using default colors")
        return False
    except FileNotFoundError:
        logger.error(
            "Rscript not found. Please install R and ensure it's in PATH. "
            "Using default colors."
        )
        return False
    except Exception as e:
        logger.error(f"Unexpected error in Palo color optimization: {e}. Using default colors.")
        return False


def run_spatial_plot_script(
    coords_file: Path,
    assignments_file: Path,
    id_col: str,
    output_path: Path,
    title: Optional[str],
    figsize: Tuple[float, float],
    dpi: int,
    use_palo: bool,
) -> bool:
    """Generate a spatial plot using the bundled ggplot2 R script."""
    r_script_path = _resolve_r_script_path(
        script_filename="plot_spatial_with_palo.R",
        env_var_name="LSTAR_PLOT_SPATIAL_SCRIPT",
    )
    if r_script_path is None:
        logger.warning(
            "plot_spatial_with_palo.R script not found. Falling back to matplotlib."
        )
        return False

    plot_title = title if title is not None else ""
    cmd = [
        "Rscript",
        str(r_script_path),
        str(coords_file),
        str(assignments_file),
        id_col,
        str(output_path),
        plot_title,
        str(figsize[0]),
        str(figsize[1]),
        str(dpi),
        "TRUE" if use_palo else "FALSE",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        if result.returncode != 0:
            logger.warning(
                f"Spatial plotting R script failed with exit code {result.returncode}. "
                f"stderr: {result.stderr[:500] if result.stderr else 'No error message'}. "
                f"Falling back to matplotlib."
            )
            return False
        if not output_path.exists():
            logger.warning("Spatial plotting script did not produce output file.")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error("Spatial plotting R script timed out; falling back to matplotlib.")
        return False
    except FileNotFoundError:
        logger.error("Rscript not found. Falling back to matplotlib.")
        return False
    except Exception as e:
        logger.error(f"Unexpected error in R plotting: {e}. Falling back to matplotlib.")
        return False


def _get_default_colors(cluster_ids: list) -> Dict[int, Tuple[float, float, float]]:
    """Generate default colors when Palo is unavailable."""
    n = len(cluster_ids)
    if n == 0:
        return {}
    
    # Use matplotlib's tab20 colormap for up to 20 clusters, then cycle
    if n <= 20:
        cmap = plt.cm.get_cmap('tab20')
        colors = [cmap(i / max(1, n - 1))[:3] for i in range(n)]
    else:
        cmap = plt.cm.get_cmap('tab20')
        colors = [cmap(i % 20 / 19)[:3] for i in range(n)]
    
    return {cluster_id: color for cluster_id, color in zip(cluster_ids, colors)}


def _generate_spatial_image_matplotlib(
    merged: pd.DataFrame,
    color_dict: Dict[Union[int, str], Tuple[float, float, float]],
    output_path: Path,
    title: Optional[str],
    figsize: Tuple[float, float],
    dpi: int,
    show_legend: bool,
) -> None:
    """Render a spatial image using matplotlib."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    for cluster_id in sorted(merged['cluster'].unique()):
        cluster_data = merged[merged['cluster'] == cluster_id]
        color = color_dict.get(cluster_id, (0.5, 0.5, 0.5))
        ax.scatter(
            cluster_data['x'],
            cluster_data['y'],
            c=[color],
            label=f'Cluster {cluster_id}',
            s=20,
            alpha=0.7,
            edgecolors='none',
        )

    ax.set_xlabel('X coordinate', fontsize=12)
    ax.set_ylabel('Y coordinate', fontsize=12)
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    if show_legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def generate_spatial_image(
    assignments: pd.Series,
    spatial_coords: pd.DataFrame,
    id_col: str = "spot_id",
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    use_palo: bool = True,
    figsize: Tuple[float, float] = (10, 10),
    dpi: int = 300,
) -> Path:
    """
    Generate a spatial domain visualization image.
    
    Args:
        assignments: Series of cluster assignments (indexed by spot_id)
        spatial_coords: DataFrame with id_col, x, y columns
        id_col: Name of the ID column
        output_path: Path to save the image (if None, returns a temporary file)
        title: Title for the plot
        use_palo: Whether to use Palo for color optimization
        figsize: Figure size (width, height) in inches
        dpi: Resolution in dots per inch
    
    Returns:
        Path to the generated image file
    """
    # Merge assignments with spatial coordinates
    merged = pd.merge(
        pd.DataFrame({id_col: assignments.index, 'cluster': assignments.values}),
        spatial_coords[[id_col, 'x', 'y']],
        on=id_col,
        how='inner'
    )
    
    if len(merged) == 0:
        raise ValueError("No matching spots found between assignments and spatial coordinates")
    
    if output_path is None:
        output_path = Path(tempfile.mktemp(suffix='.png'))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if use_palo:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            assignments_file = tmpdir_path / "assignments.csv"
            coords_file = tmpdir_path / "coords.csv"

            merged_assignments = merged[[id_col, 'cluster']].copy()
            merged_assignments['cluster'] = merged_assignments['cluster'].astype(str)
            merged_assignments.to_csv(assignments_file, index=False)
            merged[[id_col, 'x', 'y']].to_csv(coords_file, index=False)

            plotted = run_spatial_plot_script(
                coords_file=coords_file,
                assignments_file=assignments_file,
                id_col=id_col,
                output_path=output_path,
                title=title,
                figsize=figsize,
                dpi=dpi,
                use_palo=True,
            )
            if plotted:
                logger.info(f"Generated spatial image via R/ggplot: {output_path}")
                return output_path

            logger.warning("Falling back to matplotlib renderer for spatial image.")

    unique_clusters = sorted(merged['cluster'].unique())
    if use_palo:
        try:
            color_dict = get_palo_optimized_colors(assignments, spatial_coords, id_col)
        except Exception as e:
            logger.warning(f"Palo color optimization failed: {e}. Using default colors.")
            color_dict = _get_default_colors(unique_clusters)
    else:
        color_dict = _get_default_colors(unique_clusters)

    _generate_spatial_image_matplotlib(
        merged=merged,
        color_dict=color_dict,
        output_path=output_path,
        title=title,
        figsize=figsize,
        dpi=dpi,
        show_legend=False,
    )

    logger.info(f"Generated spatial image: {output_path}")

    return output_path


def generate_images_from_csvs(
    spatial_locations_csv: Union[str, Path],
    assignments_csv: Union[str, Path],
    id_col: str = "spot_id",
    output_dir: Union[str, Path, None] = None,
    use_palo: bool = True,
    he_image_path: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Dict[str, Path]]:
    """
    Generate spatial domain visualization images for all methods in assignments CSV.
    
    The assignments CSV should have the following structure:
    - One column named `id_col` (e.g., "spot_id") containing spot/cell identifiers
    - All other columns should be named after the spatial domain detection methods
      (e.g., "GraphST", "SpaGCN", "BayesSpace", etc.)
    - Each method column contains cluster assignments for that method
    
    Example assignments CSV:
        spot_id,GraphST,SpaGCN,BayesSpace,STAGATE
        spot_1,1,2,1,3
        spot_2,2,2,2,3
        ...
    
    Args:
        spatial_locations_csv: Path to CSV with spatial coordinates (id_col, x, y)
        assignments_csv: Path to CSV with assignments where column names (except id_col) 
                         are method names (id_col, Method1, Method2, ...)
        id_col: Name of the ID column (default: "spot_id")
        output_dir: Directory to save images (if None, creates a temporary directory)
        use_palo: Whether to use Palo for color optimization
        he_image_path: Optional path to H&E image to copy to output directory
    
    Returns:
        Tuple of (output_dir, model_images_dict) where:
        - output_dir: Path to directory containing generated images
        - model_images_dict: Dict mapping method_name -> image_path
            The method names are taken directly from the column names in assignments_csv
    """
    # Read spatial locations
    spatial_coords = parse_spatial_locations(spatial_locations_csv, id_col)
    
    # Read assignments CSV
    # Column names (except id_col) are assumed to be method names
    assignments_df = pd.read_csv(assignments_csv)
    if id_col not in assignments_df.columns:
        raise ValueError(
            f"ID column '{id_col}' not found in assignments CSV {assignments_csv}. "
            f"Available columns: {list(assignments_df.columns)}"
        )
    
    # Get method columns: all columns except id_col are assumed to be method names
    method_columns = [col for col in assignments_df.columns if col != id_col]
    
    if len(method_columns) == 0:
        raise ValueError(
            f"No method columns found in assignments CSV {assignments_csv}. "
            f"Only found ID column: {id_col}. "
            f"Expected format: {id_col},Method1,Method2,... where Method1, Method2, etc. "
            f"are the names of spatial domain detection methods."
        )
    
    logger.info(
        f"Found {len(method_columns)} methods in assignments CSV: {method_columns}. "
        f"These column names will be used as method names for image generation."
    )
    
    # Set up output directory
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix='lstar_images_'))
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating images for {len(method_columns)} methods in {output_dir}")
    
    # Generate image for each method
    # Method names are taken directly from the column names in assignments_csv
    method_images = {}
    
    for method_name in method_columns:
        # Get assignments for this method (using column name as method name)
        method_assignments = assignments_df.set_index(id_col)[method_name]
        
        # Generate image with filename based on method name
        output_path = output_dir / f"{method_name}.png"
        
        try:
            generate_spatial_image(
                assignments=method_assignments,
                spatial_coords=spatial_coords,
                id_col=id_col,
                output_path=output_path,
                title=f"{method_name} Spatial Domains",
                use_palo=use_palo,
            )
            method_images[method_name] = output_path
            logger.info(f"Generated image for method '{method_name}': {output_path}")
        except Exception as e:
            logger.error(f"Failed to generate image for method '{method_name}': {e}")
            continue
    
    # Copy H&E image if provided
    if he_image_path is not None:
        he_image_path = Path(he_image_path)
        if he_image_path.exists():
            he_dest = output_dir / "he.png"
            import shutil
            shutil.copy2(he_image_path, he_dest)
            logger.info(f"Copied H&E image to {he_dest}")
        else:
            logger.warning(f"H&E image not found: {he_image_path}")
    
    logger.info(f"Generated {len(method_images)} images in {output_dir}")
    
    if len(method_images) == 0:
        raise RuntimeError(
            "Failed to generate any images. Check that assignments CSV has valid data "
            "and that spatial coordinates match the spot IDs."
        )
    
    return output_dir, method_images
