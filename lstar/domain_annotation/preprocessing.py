"""Deterministic, auditable preprocessing for domain annotation.

Normalization is delegated to scanpy (``scanpy.pp.normalize_total`` and
``scanpy.pp.log1p``), a widely used and independently validated
implementation, rather than a custom normalization routine. No imputation is
performed anywhere in this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

if __package__:
    from .feature_resolution import FeatureResolutionResult
else:
    from feature_resolution import FeatureResolutionResult


PREPROCESSING_RULES_VERSION = "3.0.0"
SUPPORTED_EXPRESSION_SCALES = frozenset(
    {
        "raw_counts",
        "normalized_linear",
        "log1p_normalized",
        "scaled",
        "corrected",
        "imputed",
        "binary_detection",
        "unknown",
    }
)
SUPPORTED_ASSAY_TYPES = frozenset(
    {"auto", "targeted", "transcriptome_wide"}
)
SUPPORTED_OBSERVATION_UNITS = frozenset(
    {"cell", "nucleus", "spot", "spatial_bin"}
)

@dataclass(frozen=True)
class PreprocessingResult:
    """Expression matrix and provenance used by marker calculation."""

    marker_values: Any
    analysis_genes: Tuple[str, ...]
    feature_metadata: pd.DataFrame
    original_feature_metadata: pd.DataFrame
    expression_profile: Mapping[str, Any]
    identifier_profile: Mapping[str, Any]
    normalization: Mapping[str, Any]
    assay_profile: Mapping[str, Any]
    quality_gates: Mapping[str, Any]
    qc_filtering: Mapping[str, Any] = field(default_factory=dict)


def _finite_percentiles(values: np.ndarray, probabilities: Sequence[float]) -> List[float]:
    if values.size == 0:
        return [0.0 for _ in probabilities]
    return [float(value) for value in np.quantile(values, probabilities)]


def profile_expression_matrix(values: Any) -> Dict[str, Any]:
    """Profile a dense or sparse expression matrix without densifying it."""
    is_sparse = sparse.issparse(values)
    matrix = values.tocsr(copy=True) if is_sparse else np.asarray(values)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("Expression matrix must be non-empty and two-dimensional")
    if np.issubdtype(matrix.dtype, np.bool_) or not np.issubdtype(
        matrix.dtype, np.number
    ):
        raise ValueError("Expression matrix must be numeric and non-Boolean")
    finite_values = matrix.data if is_sparse else matrix
    if not np.all(np.isfinite(finite_values)):
        raise ValueError("Expression matrix must contain only finite values")

    number_of_values = int(matrix.size)
    if is_sparse:
        matrix.eliminate_zeros()
        zero_count = number_of_values - int(matrix.nnz)
    else:
        zero_count = int(np.count_nonzero(matrix == 0.0))
    nonzero_count = number_of_values - zero_count
    if is_sparse:
        noninteger_nonzero_count = int(
            np.count_nonzero(np.abs(matrix.data - np.rint(matrix.data)) > 1e-8)
        )
    else:
        noninteger_nonzero_count = 0
        batch_size = 256
        for start in range(0, matrix.shape[1], batch_size):
            batch = np.asarray(matrix[:, start : start + batch_size], dtype=float)
            nonzero = batch != 0.0
            noninteger_nonzero_count += int(
                np.count_nonzero(nonzero & (np.abs(batch - np.rint(batch)) > 1e-8))
            )

    row_totals = np.asarray(matrix.sum(axis=1, dtype=np.float64), dtype=float).ravel()
    row_mean = float(np.mean(row_totals))
    row_cv = float(np.std(row_totals) / row_mean) if row_mean else 0.0
    prevalence = (
        np.asarray(matrix.getnnz(axis=0), dtype=float).ravel() / matrix.shape[0]
        if is_sparse
        else np.mean(matrix != 0.0, axis=0)
    )
    if is_sparse:
        minimum = float(min(0.0, np.min(matrix.data) if matrix.nnz else 0.0))
        maximum = float(max(0.0, np.max(matrix.data) if matrix.nnz else 0.0))
    else:
        minimum = float(np.min(matrix))
        maximum = float(np.max(matrix))
    noninteger_fraction = (
        float(noninteger_nonzero_count / nonzero_count) if nonzero_count else 0.0
    )
    if is_sparse:
        nonzero_values = np.asarray(matrix.data, dtype=float)
    else:
        nonzero_values = np.asarray(matrix[matrix != 0.0], dtype=float)
    if nonzero_values.size > 1_000_000:
        sample_positions = np.linspace(
            0, nonzero_values.size - 1, 1_000_000, dtype=int
        )
        nonzero_values = nonzero_values[sample_positions]

    detected_scale = "unknown"
    detection_confidence = "Low"
    alternatives: List[str] = []
    diagnostics: List[str] = []
    if minimum < 0.0:
        gene_means = np.asarray(matrix.mean(axis=0), dtype=float).ravel()
        near_centered = float(np.mean(np.abs(gene_means) < 0.25))
        if near_centered >= 0.5:
            detected_scale = "scaled"
            detection_confidence = "Medium"
            diagnostics.append("Negative values and approximately centered genes")
        else:
            detected_scale = "corrected"
            detection_confidence = "Low"
            diagnostics.append("Negative values without a clear z-score pattern")
            alternatives.append("scaled")
    elif noninteger_fraction <= 0.001 and maximum <= 1.0:
        detected_scale = "binary_detection"
        detection_confidence = "High"
        diagnostics.append("Nonnegative matrix contains only binary detection values")
    elif noninteger_fraction <= 0.001:
        detected_scale = "raw_counts"
        detection_confidence = "High"
        diagnostics.append("Nonnegative matrix with nearly all nonzero values integral")
    elif zero_count / number_of_values < 0.01:
        detected_scale = "imputed"
        detection_confidence = "Low"
        diagnostics.append(
            "Continuous nonnegative matrix has almost no zeros and may be imputed"
        )
    elif maximum <= 30.0 and zero_count > 0:
        detected_scale = "log1p_normalized"
        detection_confidence = "Medium" if row_cv > 0.35 else "High"
        diagnostics.append(
            "Nonnegative continuous matrix with zeros and compressed maximum"
        )
        alternatives.append("normalized_linear")
    elif row_cv <= 0.15:
        detected_scale = "normalized_linear"
        detection_confidence = "Medium"
        diagnostics.append("Continuous matrix with approximately stable row totals")
        alternatives.append("log1p_normalized")
    else:
        diagnostics.append("Continuous scale does not match a supported pattern reliably")
        alternatives.extend(["normalized_linear", "log1p_normalized"])

    panel_size = int(matrix.shape[1])
    return {
        "value_type": (
            "integer" if noninteger_fraction <= 0.001 else "continuous"
        ),
        "minimum": minimum,
        "maximum": maximum,
        "has_negative_values": minimum < 0.0,
        "zero_fraction": float(zero_count / number_of_values),
        "nonzero_noninteger_fraction": noninteger_fraction,
        "observation_count": int(matrix.shape[0]),
        "gene_count": panel_size,
        "row_total_percentiles": _finite_percentiles(
            row_totals, (0.0, 0.25, 0.5, 0.75, 1.0)
        ),
        "nonzero_value_percentiles": _finite_percentiles(
            nonzero_values, (0.0, 0.25, 0.5, 0.75, 1.0)
        ),
        "row_total_cv": row_cv,
        "gene_prevalence_percentiles": _finite_percentiles(
            prevalence, (0.0, 0.25, 0.5, 0.75, 1.0)
        ),
        "detected_scale": detected_scale,
        "detection_confidence": detection_confidence,
        "alternative_scales": alternatives,
        "diagnostics": diagnostics,
        "storage_type": (
            "sparse_{}".format(matrix.getformat()) if is_sparse else "dense"
        ),
        "rules_version": PREPROCESSING_RULES_VERSION,
    }


def _scale_is_consistent(scale: str, profile: Mapping[str, Any]) -> bool:
    minimum = float(profile["minimum"])
    noninteger_fraction = float(profile["nonzero_noninteger_fraction"])
    if scale == "raw_counts":
        return minimum >= 0.0 and noninteger_fraction <= 0.001
    if scale in {"normalized_linear", "log1p_normalized"}:
        return minimum >= 0.0
    if scale == "binary_detection":
        return minimum >= 0.0 and float(profile["maximum"]) <= 1.0
    if scale in {"scaled", "corrected", "imputed", "unknown"}:
        return True
    return False


def select_expression_scale(
    profile: Mapping[str, Any],
    declared_scale: Optional[str],
) -> Dict[str, Any]:
    """Resolve declared and detected scales with explicit conflict reporting."""
    if declared_scale is not None:
        if declared_scale not in SUPPORTED_EXPRESSION_SCALES:
            raise ValueError(
                "expression_scale must be one of {}".format(
                    sorted(SUPPORTED_EXPRESSION_SCALES)
                )
            )
        declaration_consistent = _scale_is_consistent(declared_scale, profile)
        if not declaration_consistent:
            raise ValueError(
                "Declared expression_scale {!r} conflicts with deterministic "
                "matrix diagnostics".format(declared_scale)
            )
        selected = declared_scale
        confidence = "High"
    else:
        declaration_consistent = None
        selected = str(profile["detected_scale"])
        confidence = str(profile["detection_confidence"])

    return {
        "declared_scale": declared_scale,
        "detected_scale": profile["detected_scale"],
        "declaration_consistent": declaration_consistent,
        "selected_scale": selected,
        "selection_confidence": confidence,
        "alternative_scales": list(profile["alternative_scales"]),
        "user_confirmation_required": confidence == "Low",
    }


def _unique_analysis_name(base: str, seen: Dict[str, int]) -> str:
    safe = base.strip() or "feature"
    count = seen.get(safe, 0)
    seen[safe] = count + 1
    return safe if count == 0 else "{}__feature{}".format(safe, count + 1)


def build_analysis_features(
    values: Any,
    metadata: pd.DataFrame,
    *,
    selected_scale: str,
) -> Tuple[Any, Tuple[str, ...], pd.DataFrame, Dict[str, Any]]:
    """Apply an explicit duplicate policy and create unique analysis features."""
    is_sparse = sparse.issparse(values)
    matrix = (
        values.astype(np.float32).tocsr()
        if is_sparse
        else np.asarray(values, dtype=np.float32)
    )
    if matrix.shape[1] != len(metadata):
        raise ValueError("Feature metadata does not match expression columns")

    merge_counts = selected_scale == "raw_counts"
    groups: List[List[int]] = []
    used = set()
    if merge_counts:
        grouped_positions: Dict[str, List[int]] = {}
        for _, row in metadata.iterrows():
            key = str(row["biological_gene_id"])
            if bool(row["annotation_eligible"]):
                grouped_positions.setdefault(key, []).append(int(row["feature_position"]))
        for positions in grouped_positions.values():
            groups.append(positions)
            used.update(positions)
    for position in range(matrix.shape[1]):
        if position not in used:
            groups.append([position])
    groups.sort(key=lambda positions: min(positions))

    analysis_rows = []
    seen_names: Dict[str, int] = {}
    merged_group_count = 0
    inconsistent_symbol_groups: List[Dict[str, Any]] = []
    for positions in groups:
        group_metadata = metadata.iloc[positions]
        representative = group_metadata.iloc[0]
        if len(positions) > 1:
            merged_group_count += 1
            aggregation = "sum_raw_counts"
            # biological_gene_id upper-cases the symbol, so two features whose
            # symbols differ only by case are merged as one gene. That is
            # correct for reference builds that list one gene at several loci
            # (identical symbols), but not when a distinct gene was resolved to
            # a case variant of another gene's symbol. Report the difference
            # without changing the merge.
            distinct_symbols = sorted(
                {
                    str(value)
                    for value in group_metadata["annotation_symbol"]
                    if pd.notna(value) and str(value).strip()
                }
            )
            if len(distinct_symbols) > 1:
                inconsistent_symbol_groups.append(
                    {
                        "biological_gene_id": str(
                            representative["biological_gene_id"]
                        ),
                        "annotation_symbols": distinct_symbols,
                        "original_gene_ids": group_metadata["original_gene_id"]
                        .astype(str)
                        .tolist(),
                        "feature_positions": [int(value) for value in positions],
                    }
                )
        else:
            aggregation = "retained_separately"
        annotation_symbol = representative["annotation_symbol"]
        base_name = (
            str(annotation_symbol)
            if pd.notna(annotation_symbol) and str(annotation_symbol).strip()
            else str(representative["original_gene_id"])
        )
        analysis_gene = _unique_analysis_name(base_name, seen_names)
        analysis_rows.append(
            {
                "analysis_gene": analysis_gene,
                "original_gene_ids": json.dumps(
                    group_metadata["original_gene_id"].astype(str).tolist(),
                    ensure_ascii=False,
                ),
                "source_symbol": representative["source_symbol"],
                "canonical_symbol": representative["canonical_symbol"],
                "ortholog_symbol": representative["ortholog_symbol"],
                "annotation_symbol": representative["annotation_symbol"],
                "annotation_label": representative["annotation_label"],
                "annotation_label_source": representative[
                    "annotation_label_source"
                ],
                "annotation_label_confidence": float(
                    group_metadata["annotation_label_confidence"].min()
                ),
                "source_feature_label": representative["source_feature_label"],
                "source_label_candidates": representative[
                    "source_label_candidates"
                ],
                "source_label_candidate_count": int(
                    representative["source_label_candidate_count"]
                ),
                "source_label_parse_rule": representative[
                    "source_label_parse_rule"
                ],
                "source_label_status": representative["source_label_status"],
                "identity_value": representative["identity_value"],
                "identity_type": representative["identity_type"],
                "identity_status": representative["identity_status"],
                "identity_confidence": float(
                    group_metadata["identity_confidence"].min()
                ),
                "feature_evidence_tier": representative[
                    "feature_evidence_tier"
                ],
                "annotation_usable_for_context": bool(
                    group_metadata["annotation_usable_for_context"].all()
                ),
                "annotation_usable_for_naming": bool(
                    group_metadata["annotation_usable_for_naming"].all()
                ),
                "direct_confidence_eligible": bool(
                    group_metadata["direct_confidence_eligible"].all()
                ),
                "inference_confidence_eligible": bool(
                    group_metadata["inference_confidence_eligible"].all()
                ),
                "ambiguity_reason": representative["ambiguity_reason"],
                "species": representative["species"],
                "mapping_type": representative["mapping_type"],
                "mapping_confidence": float(
                    group_metadata["mapping_confidence"].min()
                ),
                "biological_gene_id": representative["biological_gene_id"],
                "duplicate_group": representative["duplicate_group"],
                "annotation_eligible": bool(
                    group_metadata["annotation_eligible"].all()
                ),
                "aggregation": aggregation,
            }
        )
    if is_sparse:
        source_positions = []
        target_positions = []
        for target, positions in enumerate(groups):
            source_positions.extend(positions)
            target_positions.extend([target] * len(positions))
        aggregation_matrix = sparse.csr_matrix(
            (
                np.ones(len(source_positions), dtype=np.float32),
                (source_positions, target_positions),
            ),
            shape=(matrix.shape[1], len(groups)),
        )
        analysis_matrix = (matrix @ aggregation_matrix).tocsr()
    else:
        columns = [
            np.sum(matrix[:, positions], axis=1, dtype=np.float32)
            if len(positions) > 1
            else matrix[:, positions[0]]
            for positions in groups
        ]
        analysis_matrix = np.column_stack(columns).astype(np.float32, copy=False)
    analysis_metadata = pd.DataFrame(analysis_rows)
    duplicate_policy = {
        "selected_scale": selected_scale,
        "raw_count_duplicate_groups_merged": int(merged_group_count),
        "aggregation_rule": (
            "sum compatible canonical duplicates before normalization"
            if merge_counts
            else "retain separately and collapse by biological_gene_id during evidence selection"
        ),
        "merged_groups_with_inconsistent_symbols": (
            len(inconsistent_symbol_groups)
        ),
        "inconsistent_symbol_warnings": inconsistent_symbol_groups,
        "inconsistent_symbol_note": (
            "Merged features whose annotation_symbol values are not identical "
            "were grouped only because biological_gene_id upper-cases the "
            "symbol. Each entry may be one gene annotated inconsistently, or "
            "distinct genes wrongly summed. The merge is applied either way; "
            "review these groups before relying on their markers."
        ),
    }
    return (
        analysis_matrix,
        tuple(analysis_metadata["analysis_gene"].astype(str)),
        analysis_metadata,
        duplicate_policy,
    )


def normalize_expression(
    values: Any,
    *,
    selected_scale: str,
    target_library_size: Optional[float] = None,
    unsupported_scale_policy: str = "stop",
) -> Tuple[Any, Dict[str, Any]]:
    """Apply one supported normalization policy using scanpy.

    Three expression scales are supported:

    - ``raw_counts``: ``scanpy.pp.normalize_total`` followed by
      ``scanpy.pp.log1p``. ``target_library_size=None`` (the default) omits
      ``target_sum`` from the ``normalize_total`` call entirely, so scanpy's
      own default applies: each cell is scaled to the median total count
      across cells in this dataset, rather than to a fixed constant. Passing
      an explicit value sets ``target_sum`` to that fixed target instead.
      Omitting rather than forwarding ``None`` keeps the audit honest — the
      run did not choose a target, scanpy did.
    - ``normalized_linear``: ``scanpy.pp.log1p`` applied directly to the
      already library-size-normalized nonnegative values.
    - ``log1p_normalized``: used as supplied, with no further transformation.

    No imputation is performed for any scale; sparse input stays sparse.
    """
    import scanpy as sc
    from anndata import AnnData

    try:
        import importlib.metadata

        scanpy_version = importlib.metadata.version("scanpy")
    except Exception:
        scanpy_version = str(sc.__version__)
    is_sparse = sparse.issparse(values)
    # `copy=True` is required, not just defensive: scanpy.pp.normalize_total
    # and scanpy.pp.log1p mutate their AnnData's X in place, and
    # np.asarray(..., dtype=np.float32) silently returns the caller's own
    # array (no copy) when it is already float32. Without an explicit copy
    # here, this function would corrupt the caller's original expression
    # matrix as a side effect.
    matrix = (
        values.astype(np.float32, copy=True).tocsr()
        if is_sparse
        else np.array(values, dtype=np.float32, copy=True)
    )
    if selected_scale == "raw_counts":
        adata = AnnData(X=matrix)
        # Passing nothing when ``target_library_size`` is None is the point:
        # scanpy then applies its own default rather than a value we picked.
        normalize_kwargs = (
            {}
            if target_library_size is None
            else {"target_sum": float(target_library_size)}
        )
        sc.pp.normalize_total(adata, **normalize_kwargs)
        sc.pp.log1p(adata)
        normalized = adata.X
        method = "scanpy_normalize_total_then_log1p"
        log_transform = "log1p"
    elif selected_scale == "normalized_linear":
        if np.any(matrix.data < 0.0) if is_sparse else np.any(matrix < 0.0):
            raise ValueError("normalized_linear expression must be nonnegative")
        adata = AnnData(X=matrix)
        sc.pp.log1p(adata)
        normalized = adata.X
        method = "scanpy_log1p_of_declared_linear_normalized_values"
        log_transform = "log1p"
    elif selected_scale == "log1p_normalized":
        if np.any(matrix.data < 0.0) if is_sparse else np.any(matrix < 0.0):
            raise ValueError("log1p_normalized expression must be nonnegative")
        normalized = matrix.copy()
        method = "use_as_supplied"
        log_transform = "already_log1p"
    else:
        if unsupported_scale_policy != "stop":
            raise ValueError(
                "Only unsupported_scale_policy='stop' is currently reproducible"
            )
        raise ValueError(
            "Expression scale {!r} is not supported for prevalence-based marker "
            "analysis. Supply a compatible unscaled matrix or explicit supported "
            "scale.".format(selected_scale)
        )
    normalized = (
        normalized.tocsr().astype(np.float32)
        if sparse.issparse(normalized)
        else np.asarray(normalized, dtype=np.float32)
    )
    if selected_scale == "raw_counts":
        target_sum_policy = (
            "scanpy_default_median_library_size"
            if target_library_size is None
            else "fixed_target_sum"
        )
    else:
        target_sum_policy = None

    return normalized, {
        "input_scale": selected_scale,
        "method": method,
        "engine": "scanpy",
        "scanpy_version": scanpy_version,
        "target_library_size": (
            float(target_library_size)
            if selected_scale == "raw_counts" and target_library_size is not None
            else None
        ),
        "target_sum_policy": target_sum_policy,
        "log_transform": log_transform,
        "marker_matrix_scale": "log1p_normalized",
        "original_matrix_retained": True,
        "imputation": "none",
        "rules_version": PREPROCESSING_RULES_VERSION,
    }


def preprocess_expression(
    expression: Any,
    *,
    id_col: Optional[str],
    feature_ids: Optional[Sequence[str]] = None,
    observation_unit: str,
    feature_resolution: FeatureResolutionResult,
    species: Optional[str] = None,
    expression_scale: Optional[str] = None,
    assay_type: str = "auto",
    target_library_size: Optional[float] = None,
    input_evidence_quality: float = 1.0,
    qc_cell_mask: Optional[Sequence[bool]] = None,
    qc_audit: Optional[Mapping[str, Any]] = None,
    min_cells_per_gene: int = 0,
) -> PreprocessingResult:
    """Profile, resolve, aggregate, and normalize expression deterministically.

    ``qc_cell_mask`` and ``min_cells_per_gene`` apply quality-control
    filtering produced upstream (typically by
    :func:`qc_filtering.apply_quality_control`). Scale profiling, scale
    selection, and the feature-scope/order check against
    ``feature_resolution`` always run against the full, unfiltered
    observation and feature population; cell-level QC only subsets rows
    before duplicate-feature merging, and gene-level QC only subsets the
    merged analysis columns afterward. QC never changes which observations
    or features are covered by the final domain-assignment output — it only
    changes what feeds marker computation.

    Args:
        qc_cell_mask: Optional Boolean mask, aligned to ``expression`` rows,
            selecting the observations that pass cell-level QC. When
            ``None``, no cell-level QC is applied.
        qc_audit: Optional JSON-serializable audit record describing how
            ``qc_cell_mask`` was produced (as returned by
            ``QCFilterResult.audit``). Stored verbatim under
            ``PreprocessingResult.qc_filtering["cell_qc"]``.
        min_cells_per_gene: Minimum number of QC-passed cells in which an
            analysis feature must be detected (nonzero) to be retained. ``0``
            disables gene-level QC.
    """
    if observation_unit not in SUPPORTED_OBSERVATION_UNITS:
        raise ValueError(
            "observation_unit must be one of {}".format(
                sorted(SUPPORTED_OBSERVATION_UNITS)
            )
        )
    if assay_type not in SUPPORTED_ASSAY_TYPES:
        raise ValueError(
            "assay_type must be one of {}".format(sorted(SUPPORTED_ASSAY_TYPES))
        )
    if not 0.0 <= float(input_evidence_quality) <= 1.0:
        raise ValueError("input_evidence_quality must be in [0, 1]")
    if isinstance(expression, pd.DataFrame):
        if id_col is None:
            raise ValueError("id_col is required for DataFrame expression input")
        gene_names = [column for column in expression.columns if column != id_col]
        raw_values = expression.loc[:, gene_names].to_numpy(copy=False)
    else:
        if feature_ids is None:
            raise ValueError("feature_ids are required for matrix expression input")
        gene_names = list(map(str, feature_ids))
        raw_values = expression
        if getattr(raw_values, "ndim", None) != 2:
            raise ValueError("matrix expression input must be two-dimensional")
        if raw_values.shape[1] != len(gene_names):
            raise ValueError("feature_ids do not match expression columns")
    expression_profile = profile_expression_matrix(raw_values)
    scale_selection = select_expression_scale(expression_profile, expression_scale)
    feature_metadata = feature_resolution.compatibility_metadata.copy()
    if feature_metadata["original_gene_id"].astype(str).tolist() != list(
        map(str, gene_names)
    ):
        raise ValueError(
            "Feature-resolution output changed feature scope or order"
        )
    identifier_profile = dict(feature_resolution.qc_report)
    identifier_profile.update(
        {
            "resolved_fraction": float(
                feature_resolution.mapping_table["annotation_eligible"].mean()
            ),
            "context_usable_fraction": float(
                feature_resolution.mapping_table[
                    "annotation_usable_for_context"
                ].mean()
            ),
            "direct_confidence_eligible_fraction": float(
                feature_resolution.mapping_table[
                    "direct_confidence_eligible"
                ].mean()
            ),
            "inference_confidence_eligible_fraction": float(
                feature_resolution.mapping_table[
                    "inference_confidence_eligible"
                ].mean()
            ),
            "feature_evidence_tier_counts": {
                str(key): int(value)
                for key, value in feature_resolution.mapping_table[
                    "feature_evidence_tier"
                ].value_counts().items()
            },
            "ambiguous_fraction": float(
                (
                    feature_resolution.mapping_table["mapping_status"]
                    == "resolved_multiple"
                ).mean()
            ),
            "ortholog_fraction": float(
                (feature_resolution.mapping_table["mapping_type"] == "ortholog").mean()
            ),
            "unresolved_fraction": float(
                (
                    ~feature_resolution.mapping_table["annotation_eligible"].astype(bool)
                ).mean()
            ),
            "duplicate_canonical_group_count": int(
                feature_metadata["duplicate_group"].dropna().nunique()
            ),
            "mapping_type_counts": {
                str(key): int(value)
                for key, value in feature_resolution.mapping_table[
                    "mapping_type"
                ].value_counts().items()
            },
            "rules_version": PREPROCESSING_RULES_VERSION,
        }
    )
    if qc_cell_mask is not None:
        cell_mask = np.asarray(qc_cell_mask, dtype=bool)
        if cell_mask.ndim != 1 or cell_mask.shape[0] != raw_values.shape[0]:
            raise ValueError(
                "qc_cell_mask length must equal the number of expression-matrix "
                "rows ({})".format(raw_values.shape[0])
            )
        if not np.any(cell_mask):
            raise ValueError("qc_cell_mask must retain at least one observation")
        analysis_input_values = raw_values[cell_mask]
    else:
        analysis_input_values = raw_values

    analysis_values, analysis_genes, analysis_metadata, duplicate_policy = (
        build_analysis_features(
            analysis_input_values,
            feature_metadata,
            selected_scale=str(scale_selection["selected_scale"]),
        )
    )

    # Captured before gene-level QC so a gene filter can never flip the
    # targeted/transcriptome-wide assay resolution below.
    resolved_assay_feature_count = len(analysis_genes)
    gene_filter_record = {
        "min_cells_per_gene": int(min_cells_per_gene),
        "total_analysis_features": int(len(analysis_genes)),
        "removed_below_min_cells": 0,
        "retained": int(len(analysis_genes)),
    }
    if min_cells_per_gene and int(min_cells_per_gene) > 0:
        is_sparse_analysis = sparse.issparse(analysis_values)
        if is_sparse_analysis:
            detected_counts = np.asarray(
                (analysis_values > 0).sum(axis=0), dtype=int
            ).ravel()
        else:
            detected_counts = np.asarray(
                np.sum(analysis_values > 0.0, axis=0), dtype=int
            ).ravel()
        keep_genes = detected_counts >= int(min_cells_per_gene)
        if not np.any(keep_genes):
            raise ValueError(
                "min_cells_per_gene removed every analysis feature; no genes "
                "were detected in at least {} cells".format(min_cells_per_gene)
            )
        removed = int(len(analysis_genes) - int(np.sum(keep_genes)))
        if removed:
            analysis_values = analysis_values[:, keep_genes]
            analysis_genes = tuple(
                gene for gene, keep in zip(analysis_genes, keep_genes) if keep
            )
            analysis_metadata = analysis_metadata.loc[keep_genes].reset_index(
                drop=True
            )
            gene_filter_record.update(
                {
                    "removed_below_min_cells": removed,
                    "retained": int(len(analysis_genes)),
                }
            )

    marker_values, normalization = normalize_expression(
        analysis_values,
        selected_scale=str(scale_selection["selected_scale"]),
        target_library_size=target_library_size,
    )
    normalization = {**scale_selection, **normalization, "duplicate_policy": duplicate_policy}

    resolved_assay = (
        "targeted"
        if assay_type == "auto" and resolved_assay_feature_count <= 2000
        else "transcriptome_wide" if assay_type == "auto" else assay_type
    )
    assay_profile = {
        "declared_assay_type": assay_type,
        "resolved_assay_type": resolved_assay,
        "observation_unit": observation_unit,
        "panel_gene_count": int(len(analysis_genes)),
        "targeted_panel": resolved_assay == "targeted",
        "annotation_coverage_score": (
            0.60 if resolved_assay == "targeted" else 1.0
        ),
        "resolution_guidance": (
            "Targeted panels support only identities represented by measured genes; "
            "subtype-level absence is not evidence of biological absence."
            if resolved_assay == "targeted"
            else "Transcriptome-wide input supports lineage-to-subtype evaluation "
            "subject to marker and identifier quality."
        ),
    }
    resolved_fraction = float(identifier_profile["resolved_fraction"])
    scale_confidence = str(scale_selection["selection_confidence"])
    quality_gates = {
        "species_resolved": bool(
            feature_resolution.recipe.get("source_taxon_id")
            or (isinstance(species, str) and species.strip())
        ),
        "namespace_resolved": bool(
            feature_resolution.recipe.get("annotation_representation") != "none"
        ),
        "identifier_resolution_sufficient": resolved_fraction >= 0.5,
        "inference_context_available": bool(
            feature_resolution.mapping_table[
                "inference_confidence_eligible"
            ].astype(bool).any()
        ),
        "expression_scale_sufficient": scale_confidence != "Low",
        "global_identifier_quality": resolved_fraction,
        "global_expression_scale_quality": {
            "High": 1.0,
            "Medium": 0.7,
            "Low": 0.3,
        }[scale_confidence],
        "global_input_evidence_quality": float(input_evidence_quality),
    }
    return PreprocessingResult(
        marker_values=marker_values,
        analysis_genes=analysis_genes,
        feature_metadata=analysis_metadata,
        original_feature_metadata=feature_metadata,
        expression_profile=expression_profile,
        identifier_profile=identifier_profile,
        normalization=normalization,
        assay_profile=assay_profile,
        quality_gates=quality_gates,
        qc_filtering={
            "cell_qc": dict(qc_audit) if qc_audit is not None else None,
            "gene_filter": gene_filter_record,
        },
    )


__all__ = [
    "PREPROCESSING_RULES_VERSION",
    "PreprocessingResult",
    "preprocess_expression",
    "profile_expression_matrix",
    "select_expression_scale",
]
