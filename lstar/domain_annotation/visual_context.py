"""Build auditable histology and L-STAR domain visual context."""

from __future__ import annotations

import base64
import colorsys
import hashlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps


PathLike = Union[str, Path]
SUPPORTED_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff"})
DEFAULT_MAX_IMAGE_DIMENSION = 1600


@dataclass(frozen=True)
class VisualContext:
    """Images and coordinate evidence approved for multimodal prompting."""

    prompt_images: Tuple[Mapping[str, str], ...]
    audit: Mapping[str, Any]
    spatial_coordinates: Optional[np.ndarray]
    spatial_key: Optional[str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_to_data_url(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return "data:{};base64,{}".format(mime, encoded)


def _as_uint8_image(values: Any) -> Image.Image:
    array = np.asarray(values)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] not in {3, 4}:
        raise ValueError("Histology image must be a 2D grayscale, RGB, or RGBA array")
    if np.issubdtype(array.dtype, np.floating):
        finite = array[np.isfinite(array)]
        maximum = float(finite.max()) if finite.size else 0.0
        array = array * 255.0 if maximum <= 1.0 else array
    array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def _publish_histology_image(
    image: Image.Image,
    output_path: Path,
    *,
    max_dimension: int,
) -> Path:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="JPEG", quality=90, optimize=True)
    return output_path


def _external_histology_candidates(dataset_dir: Path) -> Sequence[Path]:
    candidates = []
    for directory in (dataset_dir, dataset_dir / "spatial"):
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                continue
            lower = path.stem.lower().replace("&", "e")
            score = 0
            if lower == "tissue_hires_image":
                score = 100
            elif "histology" in lower or lower in {"he", "h_e", "h-e"}:
                score = 95
            elif "tissue_lowres_image" in lower:
                score = 90
            elif "full_image" in lower:
                score = 80
            if score:
                candidates.append((score, path))
    return [path for _, path in sorted(candidates, key=lambda item: (-item[0], str(item[1])))]


def _embedded_h5ad_histology(path: Path) -> Optional[Tuple[np.ndarray, str]]:
    if path.suffix.lower() != ".h5ad":
        return None
    try:
        import h5py
    except ImportError:
        return None
    candidates = []
    with h5py.File(path, "r") as handle:
        def visit(name: str, item: Any) -> None:
            if not isinstance(item, h5py.Dataset):
                return
            normalized = "/" + name.strip("/")
            if not normalized.startswith("/uns/spatial/") or "/images/" not in normalized:
                return
            if item.ndim not in {2, 3}:
                return
            leaf = normalized.rsplit("/", 1)[-1].lower()
            priority = 2 if leaf == "hires" else 1 if leaf == "lowres" else 0
            candidates.append((priority, normalized))

        handle.visititems(visit)
        if not candidates:
            return None
        _, selected = max(candidates, key=lambda item: (item[0], item[1]))
        return np.asarray(handle[selected]), selected


def _prepare_histology(
    expression_path: Path,
    output_dir: Path,
    *,
    explicit_path: Optional[PathLike],
    max_dimension: int,
) -> Tuple[Optional[Path], Mapping[str, Any]]:
    if explicit_path is not None:
        source = Path(explicit_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError("he_image_path not found: {}".format(source))
        if source.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            raise ValueError("he_image_path must be PNG, JPEG, or TIFF")
        with Image.open(source) as image:
            published = _publish_histology_image(
                image.copy(), output_dir / "histology_background.jpg",
                max_dimension=max_dimension,
            )
        return published, {"source": str(source), "source_type": "explicit_image"}

    external = _external_histology_candidates(expression_path.parent)
    if external:
        source = external[0]
        with Image.open(source) as image:
            published = _publish_histology_image(
                image.copy(), output_dir / "histology_background.jpg",
                max_dimension=max_dimension,
            )
        return published, {"source": str(source), "source_type": "discovered_image"}

    embedded = _embedded_h5ad_histology(expression_path)
    if embedded is not None:
        values, source_key = embedded
        published = _publish_histology_image(
            _as_uint8_image(values), output_dir / "histology_background.jpg",
            max_dimension=max_dimension,
        )
        return published, {
            "source": str(expression_path),
            "source_key": source_key,
            "source_type": "h5ad_embedded_image",
        }
    return None, {"source": None, "source_type": "unavailable"}


def _read_tissue_positions(path: Path, observation_ids: Sequence[str]) -> Optional[np.ndarray]:
    first_line = path.open("r", encoding="utf-8", errors="replace").readline().lower()
    if "barcode" in first_line:
        frame = pd.read_csv(path, dtype=str)
        required = {"barcode", "pxl_row_in_fullres", "pxl_col_in_fullres"}
        if not required.issubset(frame.columns):
            return None
        selected = frame.loc[:, ["barcode", "pxl_col_in_fullres", "pxl_row_in_fullres"]]
    else:
        frame = pd.read_csv(path, header=None, dtype=str)
        if frame.shape[1] < 6:
            return None
        selected = frame.iloc[:, [0, 5, 4]].copy()
        selected.columns = ["barcode", "pxl_col_in_fullres", "pxl_row_in_fullres"]
    if selected["barcode"].duplicated().any():
        return None
    selected = selected.set_index("barcode").reindex(list(map(str, observation_ids)))
    coordinates = selected.iloc[:, :2].apply(pd.to_numeric, errors="coerce").to_numpy()
    return coordinates if np.all(np.isfinite(coordinates)) else None


def _read_metadata_coordinates(path: Path, observation_ids: Sequence[str]) -> Optional[np.ndarray]:
    header = pd.read_csv(path, sep="\t", nrows=0).columns
    if not {"barcode", "imagecol", "imagerow"}.issubset(header):
        return None
    frame = pd.read_csv(
        path,
        sep="\t",
        usecols=["barcode", "imagecol", "imagerow"],
        dtype={"barcode": str},
    )
    if frame["barcode"].duplicated().any():
        return None
    selected = frame.set_index("barcode").reindex(list(map(str, observation_ids)))
    coordinates = selected.loc[:, ["imagecol", "imagerow"]].to_numpy(dtype=float)
    return coordinates if np.all(np.isfinite(coordinates)) else None


def resolve_spatial_coordinates(
    expression_path: Path,
    observation_ids: Sequence[str],
    existing_coordinates: Optional[np.ndarray],
    existing_key: Optional[str],
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Use aligned H5AD coordinates or safe local Visium sidecars."""
    if existing_coordinates is not None:
        coordinates = np.asarray(existing_coordinates, dtype=float)
        if coordinates.shape == (len(observation_ids), 2) and np.all(np.isfinite(coordinates)):
            return coordinates, existing_key
    spatial_dir = expression_path.parent / "spatial"
    for name in ("tissue_positions.csv", "tissue_positions_list.csv"):
        path = spatial_dir / name
        if path.is_file():
            coordinates = _read_tissue_positions(path, observation_ids)
            if coordinates is not None:
                return coordinates, "sidecar:{}".format(name)
    metadata = expression_path.parent / "metadata.tsv"
    if metadata.is_file():
        coordinates = _read_metadata_coordinates(metadata, observation_ids)
        if coordinates is not None:
            return coordinates, "sidecar:metadata.tsv[imagecol,imagerow]"
    return None, None


def _default_palette(labels: Sequence[str]) -> Mapping[str, Tuple[int, int, int]]:
    base = (
        "1f77b4", "ff7f0e", "2ca02c", "d62728", "9467bd", "8c564b",
        "e377c2", "7f7f7f", "bcbd22", "17becf", "aec7e8", "ffbb78",
        "98df8a", "ff9896", "c5b0d5", "c49c94", "f7b6d2", "c7c7c7",
        "dbdb8d", "9edae5",
    )
    colors = []
    for index, _ in enumerate(labels):
        if index < len(base):
            value = base[index]
            colors.append(tuple(int(value[offset:offset + 2], 16) for offset in (0, 2, 4)))
        else:
            rgb = colorsys.hsv_to_rgb((index * 0.61803398875) % 1.0, 0.65, 0.85)
            colors.append(tuple(int(channel * 255) for channel in rgb))
    return dict(zip(labels, colors))


def _palo_palette(
    ids: Sequence[str],
    coordinates: np.ndarray,
    labels: Sequence[str],
) -> Optional[Mapping[str, Tuple[int, int, int]]]:
    rscript = shutil.which("Rscript")
    script = Path(__file__).resolve().parents[1] / "resources" / "run_palo.R"
    if rscript is None or not script.is_file():
        return None
    with tempfile.TemporaryDirectory(prefix="lstar_annotation_palo_") as directory:
        root = Path(directory)
        coords_path = root / "coordinates.csv"
        assignments_path = root / "assignments.csv"
        colors_path = root / "colors.csv"
        pd.DataFrame({"observation_id": ids, "x": coordinates[:, 0], "y": coordinates[:, 1]}).to_csv(
            coords_path, index=False
        )
        pd.DataFrame({"observation_id": ids, "cluster": labels}).to_csv(
            assignments_path, index=False
        )
        result = subprocess.run(
            [rscript, str(script), str(coords_path), str(assignments_path),
             "observation_id", str(colors_path)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if result.returncode != 0 or not colors_path.is_file():
            return None
        frame = pd.read_csv(colors_path, dtype={"cluster": str})
        if not {"cluster", "r", "g", "b"}.issubset(frame.columns):
            return None
        palette = {
            str(row.cluster): tuple(
                int(round(255 * float(value))) for value in (row.r, row.g, row.b)
            )
            for row in frame.itertuples(index=False)
        }
        return palette if set(map(str, labels)).issubset(palette) else None


def _render_domain_assignment(
    observation_ids: Sequence[str],
    coordinates: np.ndarray,
    domain_labels: Sequence[str],
    output_path: Path,
    *,
    use_palo: bool,
) -> Tuple[Path, str, Mapping[str, str]]:
    labels = list(map(str, domain_labels))
    unique_labels = sorted(
        set(labels),
        key=lambda value: (not value.lstrip("-.").isdigit(), value),
    )
    palette = (
        _palo_palette(observation_ids, coordinates, labels)
        if use_palo
        else None
    )
    renderer = (
        "palo_palette_with_pillow"
        if palette is not None
        else "deterministic_palette_with_pillow"
    )
    if palette is None:
        palette = _default_palette(unique_labels)

    width, height = 1200, 1200
    padding = 15
    plot_size = width - 2 * padding
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    x = coordinates[:, 0].astype(float)
    y = coordinates[:, 1].astype(float)
    x_range = max(float(np.ptp(x)), 1.0)
    y_range = max(float(np.ptp(y)), 1.0)
    px = padding + ((x - float(x.min())) / x_range * plot_size)
    py = padding + ((y - float(y.min())) / y_range * plot_size)
    radius = max(
        2,
        min(7, int(round(160.0 / np.sqrt(max(1, len(labels)))))),
    )
    for x_value, y_value, label in zip(px, py, labels):
        color = palette[str(label)]
        draw.ellipse(
            (x_value - radius, y_value - radius, x_value + radius, y_value + radius),
            fill=color,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)
    color_mapping = {
        str(label): "#{:02X}{:02X}{:02X}".format(*palette[str(label)])
        for label in unique_labels
    }
    return output_path, renderer, color_mapping


def build_visual_context(
    *,
    expression_path: Path,
    observation_ids: Sequence[str],
    domain_labels: Sequence[str],
    existing_coordinates: Optional[np.ndarray],
    existing_spatial_key: Optional[str],
    output_dir: Path,
    he_image_path: Optional[PathLike] = None,
    use_palo: bool = True,
    max_image_dimension: int = DEFAULT_MAX_IMAGE_DIMENSION,
) -> VisualContext:
    """Create visual background only when a domain visualization is available."""
    if not isinstance(use_palo, (bool, np.bool_)):
        raise ValueError("use_palo must be Boolean")
    if max_image_dimension < 256:
        raise ValueError("max_image_dimension must be at least 256")
    coordinates, coordinate_key = resolve_spatial_coordinates(
        expression_path,
        observation_ids,
        existing_coordinates,
        existing_spatial_key,
    )
    base_audit: Dict[str, Any] = {
        "policy": (
            "Send H&E together with the reconstructed L-STAR domain map when both "
            "are available; send only the domain map when H&E is unavailable; send "
            "no images when the domain map is unavailable."
        ),
        "spatial_coordinate_source": coordinate_key,
        "domain_visualization_available": False,
        "histology_available": False,
        "images_prepared": [],
        "images_transmitted": [],
    }
    if coordinates is None or len(coordinates) != len(domain_labels):
        base_audit["unavailable_reason"] = (
            "Aligned spatial coordinates are unavailable."
        )
        return VisualContext(tuple(), base_audit, coordinates, coordinate_key)

    output_dir.mkdir(parents=True, exist_ok=True)
    domain_path = output_dir / "lstar_domain_assignment.png"
    try:
        domain_path, renderer, color_mapping = _render_domain_assignment(
            observation_ids,
            coordinates,
            domain_labels,
            domain_path,
            use_palo=use_palo,
        )
    except Exception as error:
        base_audit["unavailable_reason"] = "Domain visualization failed: {}".format(error)
        return VisualContext(tuple(), base_audit, coordinates, coordinate_key)

    base_audit.update(
        {
            "domain_visualization_available": True,
            "domain_visualization_path": str(domain_path.resolve()),
            "domain_visualization_sha256": _sha256(domain_path),
            "domain_visualization_renderer": renderer,
            "domain_color_mapping": color_mapping,
            "domain_visualization_elements": {
                "spatial_points": True,
                "domain_assignment_colors": True,
                "legend": False,
                "title": False,
                "x_axis": False,
                "y_axis": False,
                "grid": False,
                "border": False,
            },
        }
    )
    prompt_images = []
    histology_path = None
    histology_record: Mapping[str, Any] = {
        "source": None,
        "source_type": "unavailable",
    }
    try:
        histology_path, histology_record = _prepare_histology(
            expression_path,
            output_dir,
            explicit_path=he_image_path,
            max_dimension=max_image_dimension,
        )
    except Exception as error:
        histology_record = {
            "source": None,
            "source_type": "unavailable",
            "error": str(error),
        }
    if histology_path is not None:
        base_audit.update(
            {
                "histology_available": True,
                "histology_path": str(histology_path.resolve()),
                "histology_sha256": _sha256(histology_path),
                "histology_source": dict(histology_record),
            }
        )
        prompt_images.append(
            {
                "label": "H&E histology reference",
                "image_url": _image_to_data_url(histology_path),
            }
        )
    else:
        base_audit["histology_source"] = dict(histology_record)
    prompt_images.append(
        {
            "label": (
                "L-STAR consensus domain assignment visualization; domain-to-color "
                "mapping: {}"
            ).format(
                ", ".join(
                    "{}={}".format(domain_id, color)
                    for domain_id, color in color_mapping.items()
                )
            ),
            "image_url": _image_to_data_url(domain_path),
        }
    )
    base_audit["images_prepared"] = [item["label"] for item in prompt_images]
    return VisualContext(tuple(prompt_images), base_audit, coordinates, coordinate_key)


__all__ = ["VisualContext", "build_visual_context", "resolve_spatial_coordinates"]
