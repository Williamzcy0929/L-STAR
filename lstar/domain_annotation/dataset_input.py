"""Unified, sparse-safe expression and consensus input loading."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse

if __package__:
    from .input_validation import (
        CANONICAL_CONSENSUS_FILENAME,
        CONSENSUS_COLUMN,
        MANIFEST_FILENAME,
        load_run_manifest,
        read_csv_header,
        resolve_method_columns,
        resolve_run_manifest_path,
    )
    from .preprocessing import profile_expression_matrix, select_expression_scale
else:
    from input_validation import (
        CANONICAL_CONSENSUS_FILENAME,
        CONSENSUS_COLUMN,
        MANIFEST_FILENAME,
        load_run_manifest,
        read_csv_header,
        resolve_method_columns,
        resolve_run_manifest_path,
    )
    from preprocessing import profile_expression_matrix, select_expression_scale


PathLike = Union[str, Path]
LABEL_LEAKAGE_PATTERN = re.compile(
    r"(?:ground|truth|manual|annotat|cell.?type|reference.?label|"
    r"predicted.?label|cluster|domain|class)",
    flags=re.IGNORECASE,
)
# Non-LLM QC channel: recognized obs columns are extracted regardless of
# allowed_context_fields and are never merged into observation_metadata, the
# evidence card, or any LLM prompt. They exist only to feed
# qc_filtering.apply_quality_control's precomputed-obs mode.
RECOGNIZED_QC_OBS_COLUMNS = (
    "n_genes_by_counts",
    "total_counts",
    "pct_counts_mt",
    "pct_counts_mito",
    "n_counts",
    "n_genes",
)


@dataclass(frozen=True)
class AnnotationInputs:
    """Validated H5AD or 10x HDF5 input aligned to L-STAR tabular outputs."""

    expression_matrix: Any
    observation_ids: Tuple[str, ...]
    feature_ids: Tuple[str, ...]
    feature_metadata: pd.DataFrame
    observation_metadata: pd.DataFrame
    assignments: pd.DataFrame
    consensus: pd.DataFrame
    id_col: str
    consensus_id_col: str
    consensus_domain_col: str
    method_columns: Tuple[str, ...]
    expression_data_path: Path
    input_format: str
    assignments_csv: Path
    lstar_consensus_csv: Path
    manifest: Mapping[str, Any]
    inspection_report: Mapping[str, Any]
    alignment_table: pd.DataFrame
    alignment_summary: Mapping[str, Any]
    expression_source_manifest: Mapping[str, Any]
    spatial_coordinates: Optional[np.ndarray]
    spatial_key: Optional[str]
    qc_metadata: Mapping[str, Any]

    @property
    def gene_columns(self) -> Tuple[str, ...]:
        return self.feature_ids


def _path(value: PathLike, name: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError("{} must be a string or pathlib.Path".format(name))
    if isinstance(value, str) and not value.strip():
        raise ValueError("{} must not be empty".format(name))
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError("{} not found: {}".format(name, path))
    return path.resolve()


def _decode(values: Any) -> Tuple[str, ...]:
    return tuple(
        value.decode("utf-8", errors="replace")
        if isinstance(value, bytes)
        else str(value)
        for value in values
    )


def _manifest_path(value: Any, output_dir: Path, key: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Manifest key {} must be a non-empty path".format(key))
    path = Path(value).expanduser()
    return path if path.is_absolute() else output_dir / path


def _resolve_tabular_inputs(
    *,
    output_dir: Path,
    assignments_csv: Optional[PathLike],
    lstar_consensus_csv: Optional[PathLike],
    id_col: Optional[str],
    consensus_id_col: Optional[str],
    consensus_domain_col: Optional[str],
    method_columns: Optional[Sequence[str]],
    run_manifest: Optional[PathLike] = None,
) -> Tuple[Path, Path, str, str, Tuple[str, ...], Mapping[str, Any]]:
    manifest = load_run_manifest(output_dir, run_manifest)
    # An explicitly supplied manifest may sit outside output_dir, so relative
    # paths inside it are resolved against the directory holding it.
    manifest_file = resolve_run_manifest_path(output_dir, run_manifest)
    manifest_root = manifest_file.parent if manifest_file is not None else output_dir
    if assignments_csv is not None:
        assignment_path = _path(assignments_csv, "assignments_csv")
    elif manifest.get("assignments_csv"):
        assignment_path = _path(
            _manifest_path(
                manifest["assignments_csv"], manifest_root, "assignments_csv"
            ),
            "assignments_csv",
        )
    else:
        raise ValueError(
            "assignments_csv is required unless recorded in a {} at or above "
            "{}".format(MANIFEST_FILENAME, output_dir)
        )
    if lstar_consensus_csv is not None:
        consensus_path = _path(lstar_consensus_csv, "lstar_consensus_csv")
    elif manifest.get("lstar_consensus_csv"):
        consensus_path = _path(
            _manifest_path(
                manifest["lstar_consensus_csv"], manifest_root, "lstar_consensus_csv"
            ),
            "lstar_consensus_csv",
        )
    else:
        consensus_path = _path(
            output_dir / CANONICAL_CONSENSUS_FILENAME,
            "lstar_consensus_csv",
        )

    consensus_header = read_csv_header(consensus_path)
    selected_consensus_id = consensus_id_col or id_col or manifest.get("id_col")
    if selected_consensus_id is None:
        selected_consensus_id = consensus_header[0]
    if selected_consensus_id not in consensus_header:
        raise ValueError("Consensus observation-ID column is absent")
    selected_domain = consensus_domain_col
    if selected_domain is None:
        if CONSENSUS_COLUMN in consensus_header:
            selected_domain = CONSENSUS_COLUMN
        else:
            remaining = [x for x in consensus_header if x != selected_consensus_id]
            if len(remaining) != 1:
                raise ValueError(
                    "consensus_domain_col is required when it cannot be inferred uniquely"
                )
            selected_domain = remaining[0]
    if selected_domain not in consensus_header or selected_domain == selected_consensus_id:
        raise ValueError("Consensus domain column is absent or conflicts with ID column")

    assignment_header = read_csv_header(assignment_path)
    assignment_id = id_col or manifest.get("id_col") or selected_consensus_id
    if assignment_id not in assignment_header:
        raise ValueError("Assignment observation-ID column is absent")
    methods = resolve_method_columns(
        assignment_header,
        assignment_id,
        output_dir,
        method_columns=method_columns,
        manifest=manifest,
    )
    return (
        assignment_path,
        consensus_path,
        str(assignment_id),
        str(selected_consensus_id),
        tuple(methods),
        {**dict(manifest), "consensus_domain_col": str(selected_domain)},
    )


def _candidate_record(
    name: str,
    matrix: Any,
    feature_ids: Sequence[str],
) -> Dict[str, Any]:
    profile = profile_expression_matrix(matrix)
    return {
        "source": name,
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "feature_count": int(len(feature_ids)),
        "sparsity": float(profile["zero_fraction"]),
        "minimum": float(profile["minimum"]),
        "maximum": float(profile["maximum"]),
        "nonzero_quantile_proxy": {
            "nonzero_value_percentiles": list(
                profile["nonzero_value_percentiles"]
            ),
            "row_total_percentiles": list(profile["row_total_percentiles"]),
        },
        "fraction_integer_like_nonzero": float(
            1.0 - profile["nonzero_noninteger_fraction"]
        ),
        "fraction_zeros": float(profile["zero_fraction"]),
        "has_negative_values": bool(profile["has_negative_values"]),
        "candidate_expression_scale": profile["detected_scale"],
        "scale_inference_confidence": profile["detection_confidence"],
        "scale_evidence": list(profile["diagnostics"]),
        "storage_type": profile["storage_type"],
        "profile": profile,
    }


def _select_candidate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    expression_source: Optional[str],
    expression_scale: Optional[str],
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    if expression_source is not None:
        matching = [row for row in candidates if row["source"] == expression_source]
        if not matching:
            raise ValueError(
                "expression_source {!r} is unavailable; candidates={}".format(
                    expression_source, [row["source"] for row in candidates]
                )
            )
        selected = matching[0]
    else:
        scale_rank = {
            "raw_counts": 5,
            "log1p_normalized": 4,
            "normalized_linear": 3,
            "binary_detection": 2,
            "scaled": 1,
            "corrected": 1,
            "unknown": 0,
        }
        confidence_rank = {"High": 3, "Medium": 2, "Low": 1}
        source_rank = lambda name: 2 if name == "X" else 1
        compatible = []
        for row in candidates:
            try:
                selection = select_expression_scale(row["profile"], expression_scale)
            except ValueError:
                continue
            compatible.append((row, selection))
        if not compatible:
            raise ValueError("No expression source is compatible with expression_scale")
        selected, _ = max(
            compatible,
            key=lambda item: (
                scale_rank.get(str(item[1]["selected_scale"]), 0),
                confidence_rank.get(str(item[1]["selection_confidence"]), 0),
                source_rank(str(item[0]["source"])),
            ),
        )
    selection = select_expression_scale(selected["profile"], expression_scale)
    if selection["selection_confidence"] == "Low" and expression_scale is None:
        raise ValueError(
            "Expression scale is uncertain for {}; declare expression_scale explicitly".format(
                selected["source"]
            )
        )
    if selection["selected_scale"] in {
        "scaled", "corrected", "imputed", "binary_detection", "unknown"
    }:
        raise ValueError(
            "Selected expression source has incompatible scale {!r}".format(
                selection["selected_scale"]
            )
        )
    return selected, selection


def _load_h5ad(
    path: Path,
    *,
    expression_source: Optional[str],
    expression_scale: Optional[str],
    allowed_context_fields: Sequence[str],
    evaluation_only_fields: Sequence[str],
    forbidden_annotation_fields: Sequence[str],
) -> Dict[str, Any]:
    try:
        import anndata as ad
    except ImportError as error:
        raise ImportError("anndata is required for H5AD annotation input") from error
    adata = ad.read_h5ad(path)
    candidates = []
    payloads: Dict[str, Tuple[Any, Tuple[str, ...], pd.DataFrame]] = {}

    def add(name: str, matrix: Any, ids: Sequence[str], metadata: pd.DataFrame) -> None:
        ids_tuple = tuple(map(str, ids))
        candidates.append(_candidate_record(name, matrix, ids_tuple))
        payloads[name] = (matrix, ids_tuple, metadata.copy())

    add("X", adata.X, adata.var_names, adata.var)
    for key in adata.layers.keys():
        if isinstance(key, str) and key:
            add("layers/{}".format(key), adata.layers[key], adata.var_names, adata.var)
    if adata.raw is not None:
        add("raw.X", adata.raw.X, adata.raw.var_names, adata.raw.var)

    selected_record, scale_selection = _select_candidate(
        candidates,
        expression_source=expression_source,
        expression_scale=expression_scale,
    )
    matrix, feature_ids, feature_metadata = payloads[str(selected_record["source"])]
    obs_columns = list(map(str, adata.obs.columns))
    evaluation = set(map(str, evaluation_only_fields))
    forbidden = set(map(str, forbidden_annotation_fields))
    leakage = sorted(
        set(column for column in obs_columns if LABEL_LEAKAGE_PATTERN.search(column))
        | evaluation
        | forbidden
    )
    allowed = [
        field
        for field in map(str, allowed_context_fields)
        if field in adata.obs.columns and field not in evaluation and field not in forbidden
    ]
    missing_allowed = [field for field in allowed_context_fields if field not in adata.obs]
    if missing_allowed:
        raise ValueError(
            "allowed_context_fields are absent from H5AD obs: {}".format(missing_allowed)
        )
    observation_metadata = adata.obs.loc[:, allowed].copy()
    observation_metadata.index = observation_metadata.index.astype(str)

    recognized_qc_columns = [
        column for column in RECOGNIZED_QC_OBS_COLUMNS if column in adata.obs.columns
    ]
    precomputed_obs_qc = (
        adata.obs.loc[:, recognized_qc_columns].copy()
        if recognized_qc_columns
        else None
    )
    if precomputed_obs_qc is not None:
        precomputed_obs_qc.index = precomputed_obs_qc.index.astype(str)

    spatial_candidates = []
    for key in adata.obsm.keys():
        value = np.asarray(adata.obsm[key])
        if value.ndim == 2 and value.shape[0] == adata.n_obs and value.shape[1] >= 2:
            spatial_candidates.append(str(key))
    selected_spatial = (
        "spatial" if "spatial" in spatial_candidates
        else next((key for key in spatial_candidates if "spatial" in key.lower()), None)
    )
    coordinates = (
        np.asarray(adata.obsm[selected_spatial])[:, :2].astype(float)
        if selected_spatial is not None
        else None
    )
    inspection = {
        "input_format": "h5ad",
        "path": str(path),
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "obs_names_uniqueness": bool(adata.obs_names.is_unique),
        "var_names_uniqueness": bool(adata.var_names.is_unique),
        "X_shape": [int(value) for value in adata.shape],
        "X_storage_type": candidates[0]["storage_type"],
        "X_numeric_summary": candidates[0]["profile"],
        "available_layers": [
            str(key) for key in adata.layers.keys() if isinstance(key, str) and key
        ],
        "raw_availability": adata.raw is not None,
        "raw_shape": (
            [int(adata.raw.n_obs), int(adata.raw.n_vars)]
            if adata.raw is not None else None
        ),
        "obs_columns": obs_columns,
        "var_columns": list(map(str, adata.var.columns)),
        "raw_var_columns": (
            list(map(str, adata.raw.var.columns)) if adata.raw is not None else []
        ),
        "obsm_keys": list(map(str, adata.obsm.keys())),
        "obsp_keys": list(map(str, adata.obsp.keys())),
        "uns_keys": list(map(str, adata.uns.keys())),
        "candidate_expression_sources": [
            {key: value for key, value in row.items() if key != "profile"}
            for row in candidates
        ],
        "candidate_feature_identity_columns": sorted(
            set(map(str, adata.var.columns))
            | (
                set(map(str, adata.raw.var.columns))
                if adata.raw is not None else set()
            )
        ),
        "candidate_spatial_coordinates": spatial_candidates,
        "selected_spatial_coordinates": selected_spatial,
        "candidate_label_leakage_columns": leakage,
        "allowed_context_fields": allowed,
        "evaluation_only_fields": sorted(evaluation),
        "forbidden_annotation_fields": sorted(forbidden),
    }
    return {
        "matrix": matrix,
        "observation_ids": tuple(map(str, adata.obs_names)),
        "feature_ids": feature_ids,
        "feature_metadata": feature_metadata.reset_index(drop=False),
        "observation_metadata": observation_metadata,
        "inspection": inspection,
        "candidates": candidates,
        "selected": selected_record,
        "scale_selection": scale_selection,
        "spatial_coordinates": coordinates,
        "spatial_key": selected_spatial,
        "qc_metadata": {
            "precomputed_obs_qc": precomputed_obs_qc,
            "recognized_obs_qc_columns": recognized_qc_columns,
            "counts_available_from_selected_source": (
                str(scale_selection["selected_scale"]) == "raw_counts"
            ),
        },
    }


def _load_tenx_h5(
    path: Path,
    *,
    expression_source: Optional[str],
    expression_scale: Optional[str],
) -> Dict[str, Any]:
    try:
        import h5py
    except ImportError as error:
        raise ImportError("h5py is required for 10x HDF5 annotation input") from error
    with h5py.File(path, "r") as handle:
        if "matrix" not in handle:
            raise ValueError("HDF5 input is not a supported 10x matrix")
        group = handle["matrix"]
        required = {"data", "indices", "indptr", "shape", "barcodes", "features"}
        missing = sorted(required - set(group.keys()))
        if missing:
            raise ValueError(
                "10x HDF5 matrix is missing required entries: {}".format(missing)
            )
        shape = tuple(map(int, group["shape"][:]))
        feature_by_observation = sparse.csc_matrix(
            (
                np.asarray(group["data"][:]),
                np.asarray(group["indices"][:]),
                np.asarray(group["indptr"][:]),
            ),
            shape=shape,
        )
        matrix = feature_by_observation.T.tocsr()
        observation_ids = _decode(group["barcodes"][:])
        feature_group = group["features"]
        fields = [key for key in feature_group.keys() if not key.startswith("_")]
        feature_metadata = pd.DataFrame(
            {key: _decode(feature_group[key][:]) for key in fields}
        )
        feature_ids = tuple(
            feature_metadata["name"].astype(str)
            if "name" in feature_metadata
            else feature_metadata["id"].astype(str)
        )
    candidate = _candidate_record("matrix", matrix, feature_ids)
    selected, scale_selection = _select_candidate(
        [candidate],
        expression_source=expression_source,
        expression_scale=expression_scale,
    )
    inspection = {
        "input_format": "10x_h5",
        "path": str(path),
        "n_obs": int(matrix.shape[0]),
        "n_vars": int(matrix.shape[1]),
        "obs_names_uniqueness": len(set(observation_ids)) == len(observation_ids),
        "var_names_uniqueness": len(set(feature_ids)) == len(feature_ids),
        "X_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "X_storage_type": candidate["storage_type"],
        "X_numeric_summary": candidate["profile"],
        "available_layers": [],
        "raw_availability": False,
        "raw_shape": None,
        "obs_columns": [],
        "var_columns": fields,
        "raw_var_columns": [],
        "obsm_keys": [],
        "obsp_keys": [],
        "uns_keys": [],
        "candidate_expression_sources": [
            {key: value for key, value in candidate.items() if key != "profile"}
        ],
        "candidate_feature_identity_columns": fields,
        "candidate_spatial_coordinates": [],
        "selected_spatial_coordinates": None,
        "candidate_label_leakage_columns": [],
        "allowed_context_fields": [],
        "evaluation_only_fields": [],
        "forbidden_annotation_fields": [],
    }
    return {
        "matrix": matrix,
        "observation_ids": observation_ids,
        "feature_ids": feature_ids,
        "feature_metadata": feature_metadata,
        "observation_metadata": pd.DataFrame(index=list(observation_ids)),
        "inspection": inspection,
        "candidates": [candidate],
        "selected": selected,
        "scale_selection": scale_selection,
        "spatial_coordinates": None,
        "spatial_key": None,
        "qc_metadata": {
            "precomputed_obs_qc": None,
            "recognized_obs_qc_columns": [],
            "counts_available_from_selected_source": True,
        },
    }


def _apply_id_transform(
    values: Sequence[str],
    specification: Optional[Mapping[str, str]],
) -> Tuple[str, ...]:
    if specification is None:
        return tuple(map(str, values))
    if set(specification) != {"pattern", "replacement"}:
        raise ValueError(
            "observation_id_transform requires exactly pattern and replacement"
        )
    pattern = re.compile(str(specification["pattern"]))
    transformed = tuple(
        pattern.sub(str(specification["replacement"]), str(value)) for value in values
    )
    if len(set(transformed)) != len(transformed):
        raise ValueError("observation_id_transform does not produce unique IDs")
    return transformed


def load_annotation_inputs(
    *,
    output_dir: PathLike,
    expression_h5ad: Optional[PathLike] = None,
    expression_h5: Optional[PathLike] = None,
    assignments_csv: Optional[PathLike] = None,
    lstar_consensus_csv: Optional[PathLike] = None,
    id_col: Optional[str] = None,
    consensus_id_col: Optional[str] = None,
    consensus_domain_col: Optional[str] = None,
    method_columns: Optional[Sequence[str]] = None,
    expression_source: Optional[str] = None,
    expression_scale: Optional[str] = None,
    allow_partial_observations: bool = False,
    observation_id_transform: Optional[Mapping[str, str]] = None,
    allowed_context_fields: Sequence[str] = (),
    evaluation_only_fields: Sequence[str] = (),
    forbidden_annotation_fields: Sequence[str] = (),
    run_manifest: Optional[PathLike] = None,
) -> AnnotationInputs:
    """Load H5AD or 10x HDF5 expression and align L-STAR observations."""
    if not isinstance(allow_partial_observations, (bool, np.bool_)):
        raise ValueError("allow_partial_observations must be Boolean")
    for name, fields in (
        ("allowed_context_fields", allowed_context_fields),
        ("evaluation_only_fields", evaluation_only_fields),
        ("forbidden_annotation_fields", forbidden_annotation_fields),
    ):
        if isinstance(fields, (str, bytes)):
            raise ValueError("{} must be a sequence of field names".format(name))
        if any(not isinstance(field, str) or not field.strip() for field in fields):
            raise ValueError("{} must contain non-empty strings".format(name))
    conflicts = set(allowed_context_fields) & (
        set(evaluation_only_fields) | set(forbidden_annotation_fields)
    )
    if conflicts:
        raise ValueError(
            "allowed_context_fields conflict with excluded fields: {}".format(
                sorted(conflicts)
            )
        )
    if (expression_h5ad is None) == (expression_h5 is None):
        raise ValueError("Provide exactly one of expression_h5ad or expression_h5")
    output_path = Path(output_dir).expanduser().resolve()
    (
        assignment_path,
        consensus_path,
        assignment_id_col,
        selected_consensus_id_col,
        methods,
        manifest,
    ) = _resolve_tabular_inputs(
        output_dir=output_path,
        assignments_csv=assignments_csv,
        lstar_consensus_csv=lstar_consensus_csv,
        id_col=id_col,
        consensus_id_col=consensus_id_col,
        consensus_domain_col=consensus_domain_col,
        method_columns=method_columns,
        run_manifest=run_manifest,
    )
    selected_domain_col = str(manifest["consensus_domain_col"])
    assignments = pd.read_csv(
        assignment_path,
        usecols=[assignment_id_col, *methods],
        dtype={assignment_id_col: str},
    )
    consensus = pd.read_csv(
        consensus_path,
        usecols=[selected_consensus_id_col, selected_domain_col],
        dtype={selected_consensus_id_col: str, selected_domain_col: str},
    )
    if assignments[assignment_id_col].duplicated().any():
        raise ValueError("Duplicate observation identifiers in assignments_csv")
    if consensus[selected_consensus_id_col].duplicated().any():
        raise ValueError("Duplicate observation identifiers in consensus assignment")

    if expression_h5ad is not None:
        expression_path = _path(expression_h5ad, "expression_h5ad")
        if expression_path.suffix.lower() != ".h5ad":
            raise ValueError("expression_h5ad must point to a .h5ad file")
        loaded = _load_h5ad(
            expression_path,
            expression_source=expression_source,
            expression_scale=expression_scale,
            allowed_context_fields=allowed_context_fields,
            evaluation_only_fields=evaluation_only_fields,
            forbidden_annotation_fields=forbidden_annotation_fields,
        )
        input_format = "h5ad"
    else:
        expression_path = _path(expression_h5, "expression_h5")
        if expression_path.suffix.lower() not in {".h5", ".hdf5"}:
            raise ValueError("expression_h5 must point to a .h5 or .hdf5 file")
        if allowed_context_fields:
            raise ValueError(
                "allowed_context_fields requires H5AD observation metadata"
            )
        loaded = _load_tenx_h5(
            expression_path,
            expression_source=expression_source,
            expression_scale=expression_scale,
        )
        input_format = "10x_h5"

    h5_ids = tuple(map(str, loaded["observation_ids"]))
    if len(set(h5_ids)) != len(h5_ids):
        raise ValueError("Duplicate observation identifiers in expression input")
    assignment_original = tuple(assignments[assignment_id_col].astype(str))
    consensus_original = tuple(consensus[selected_consensus_id_col].astype(str))
    assignment_ids = _apply_id_transform(assignment_original, observation_id_transform)
    consensus_ids = _apply_id_transform(consensus_original, observation_id_transform)
    assignments = assignments.copy()
    consensus = consensus.copy()
    assignments[assignment_id_col] = assignment_ids
    consensus[selected_consensus_id_col] = consensus_ids

    h5_set = set(h5_ids)
    consensus_set = set(consensus_ids)
    assignment_set = set(assignment_ids)
    h5_only = h5_set - consensus_set
    consensus_only = consensus_set - h5_set
    if not allow_partial_observations and (h5_only or consensus_only):
        raise ValueError(
            "H5 observation IDs and consensus IDs must match exactly; "
            "h5_only={}, consensus_only={}".format(len(h5_only), len(consensus_only))
        )
    final_ids = tuple(value for value in h5_ids if value in consensus_set)
    if not final_ids:
        raise ValueError("No expression observations align with consensus assignments")
    missing_assignments = set(final_ids) - assignment_set
    if missing_assignments:
        raise ValueError(
            "Consensus-aligned observations are absent from assignments_csv"
        )

    h5_position = {value: index for index, value in enumerate(h5_ids)}
    consensus_position = {value: index for index, value in enumerate(consensus_ids)}
    assignment_position = {value: index for index, value in enumerate(assignment_ids)}
    expression_positions = [h5_position[value] for value in final_ids]
    matrix = loaded["matrix"][expression_positions, :]
    assignments = assignments.set_index(assignment_id_col, drop=False).loc[
        list(final_ids)
    ].reset_index(drop=True)
    consensus = consensus.set_index(selected_consensus_id_col, drop=False).loc[
        list(final_ids)
    ].reset_index(drop=True)
    consensus = consensus.rename(
        columns={selected_consensus_id_col: assignment_id_col, selected_domain_col: CONSENSUS_COLUMN}
    )
    observation_metadata = loaded["observation_metadata"].reindex(list(final_ids))
    coordinates = loaded["spatial_coordinates"]
    if coordinates is not None:
        coordinates = np.asarray(coordinates)[expression_positions, :]

    qc_metadata = dict(loaded["qc_metadata"])
    if qc_metadata.get("precomputed_obs_qc") is not None:
        qc_metadata["precomputed_obs_qc"] = qc_metadata["precomputed_obs_qc"].reindex(
            list(final_ids)
        )

    union_ids = list(h5_ids) + [value for value in consensus_ids if value not in h5_set]
    final_id_set = set(final_ids)
    alignment_table = pd.DataFrame(
        {
            "observation_id": union_ids,
            "h5ad_position": [h5_position.get(value) for value in union_ids],
            "consensus_position": [consensus_position.get(value) for value in union_ids],
            "assignment_position": [assignment_position.get(value) for value in union_ids],
            "in_expression": [value in h5_set for value in union_ids],
            "in_consensus": [value in consensus_set for value in union_ids],
            "in_assignments": [value in assignment_set for value in union_ids],
            "retained_for_annotation": [value in final_id_set for value in union_ids],
        }
    )
    transform_record = (
        None if observation_id_transform is None else dict(observation_id_transform)
    )
    alignment_summary = {
        "total_h5ad_observations": len(h5_ids),
        "total_consensus_observations": len(consensus_ids),
        "matched_observations": len(h5_set & consensus_set),
        "h5ad_only_observations": len(h5_only),
        "consensus_only_observations": len(consensus_only),
        "duplicate_identifiers": 0,
        "applied_identifier_transform": transform_record,
        "partial_scope_explicitly_allowed": bool(allow_partial_observations),
        "final_annotation_population": len(final_ids),
        "final_order_source": "expression observation order",
    }
    expression_manifest = {
        "input_format": input_format,
        "expression_data_path": str(expression_path),
        "selected_expression_source": loaded["selected"]["source"],
        "selected_expression_scale": loaded["scale_selection"]["selected_scale"],
        "scale_selection": dict(loaded["scale_selection"]),
        "operation_sources": {
            "detection_fraction": loaded["selected"]["source"],
            "marker_statistics": loaded["selected"]["source"],
        },
        "selection_evidence": list(loaded["selected"]["scale_evidence"]),
        "candidate_sources": [
            {key: value for key, value in row.items() if key != "profile"}
            for row in loaded["candidates"]
        ],
        "full_matrix_sent_to_llm": False,
    }
    return AnnotationInputs(
        expression_matrix=matrix,
        observation_ids=final_ids,
        feature_ids=tuple(map(str, loaded["feature_ids"])),
        feature_metadata=loaded["feature_metadata"],
        observation_metadata=observation_metadata,
        assignments=assignments,
        consensus=consensus,
        id_col=assignment_id_col,
        consensus_id_col=selected_consensus_id_col,
        consensus_domain_col=selected_domain_col,
        method_columns=methods,
        expression_data_path=expression_path,
        input_format=input_format,
        assignments_csv=assignment_path,
        lstar_consensus_csv=consensus_path,
        manifest=manifest,
        inspection_report=loaded["inspection"],
        alignment_table=alignment_table,
        alignment_summary=alignment_summary,
        expression_source_manifest=expression_manifest,
        spatial_coordinates=coordinates,
        spatial_key=loaded["spatial_key"],
        qc_metadata=qc_metadata,
    )


__all__ = ["AnnotationInputs", "load_annotation_inputs"]
