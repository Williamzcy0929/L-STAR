"""Platform-adaptive, scanpy-based quality control for domain annotation.

Cell-level QC only affects which observations feed marker computation. It
never removes observations from the final domain-assignment output: every
aligned observation still receives a domain label regardless of QC outcome.
No imputation is performed anywhere in this module.

Two platforms determine the threshold set, based on the number of features
presented to :func:`apply_quality_control` (mirroring the ``targeted`` vs.
``transcriptome_wide`` assay-resolution rule in
:func:`preprocessing.preprocess_expression`):

- ``targeted_panel`` (<= 2000 features): ``min_counts=50``, ``min_genes=20``,
  no mitochondrial gate. Standard transcriptome-wide thresholds would remove
  most or all cells from small targeted panels.
- ``transcriptome_wide`` (> 2000 features): ``min_genes=200``, a
  mitochondrial-fraction gate at 20% when mitochondrial features are
  identifiable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

if __package__:
    from .markers import is_mitochondrial_gene
else:
    from markers import is_mitochondrial_gene


QC_RULES_VERSION = "1.1.0"
TARGETED_PANEL_MAX_FEATURES = 2000
TRANSCRIPTOME_MIN_GENES = 200
TRANSCRIPTOME_MAX_PCT_MT = 20.0
TARGETED_MIN_COUNTS = 50.0
TARGETED_MIN_GENES = 20
MIN_CELLS_PER_GENE = 3

_PRECOMPUTED_GENES_COLUMNS = ("n_genes_by_counts", "n_genes")
_PRECOMPUTED_COUNTS_COLUMNS = ("total_counts", "n_counts")
_PRECOMPUTED_MITO_COLUMNS = ("pct_counts_mt", "pct_counts_mito")


@dataclass(frozen=True)
class QCFilterResult:
    """Cell-level quality-control outcome for domain-annotation marker computation."""

    cell_mask: np.ndarray
    audit: Dict[str, Any]


def _platform(n_features: int) -> str:
    return (
        "targeted_panel"
        if n_features <= TARGETED_PANEL_MAX_FEATURES
        else "transcriptome_wide"
    )


def _threshold_dict(platform: str) -> Dict[str, Any]:
    if platform == "targeted_panel":
        return {
            "min_genes": TARGETED_MIN_GENES,
            "min_counts": TARGETED_MIN_COUNTS,
            "max_pct_counts_mt": None,
            "min_cells_per_gene": MIN_CELLS_PER_GENE,
        }
    return {
        "min_genes": TRANSCRIPTOME_MIN_GENES,
        "min_counts": None,
        "max_pct_counts_mt": TRANSCRIPTOME_MAX_PCT_MT,
        "min_cells_per_gene": MIN_CELLS_PER_GENE,
    }


def _first_present_column(
    frame: pd.DataFrame, candidates: Sequence[str]
) -> Optional[str]:
    for name in candidates:
        if name in frame.columns:
            return name
    return None


def _mito_mask_from_labels(feature_mito_labels: Sequence[Optional[str]]) -> np.ndarray:
    return np.asarray(
        [is_mitochondrial_gene(label) for label in feature_mito_labels],
        dtype=bool,
    )


def _finalize(
    *,
    mode: str,
    platform: str,
    panel_feature_count: int,
    selected_scale: str,
    scanpy_version: str,
    thresholds: Dict[str, Any],
    n_genes: np.ndarray,
    total_counts: Optional[np.ndarray],
    pct_counts_mt: Optional[np.ndarray],
    mito_gate_applied: bool,
    n_mito_features: int,
    mito_skip_reason: Optional[str],
) -> QCFilterResult:
    total = int(n_genes.shape[0])
    failed_min_genes = np.zeros(total, dtype=bool)
    failed_min_counts = np.zeros(total, dtype=bool)
    failed_pct_mt = np.zeros(total, dtype=bool)

    if thresholds["min_genes"] is not None:
        failed_min_genes = n_genes < float(thresholds["min_genes"])
    if thresholds["min_counts"] is not None and total_counts is not None:
        failed_min_counts = total_counts < float(thresholds["min_counts"])
    if mito_gate_applied and pct_counts_mt is not None:
        failed_pct_mt = pct_counts_mt >= float(thresholds["max_pct_counts_mt"])

    retained_mask = ~(failed_min_genes | failed_min_counts | failed_pct_mt)

    audit = {
        "rules_version": QC_RULES_VERSION,
        "engine": "scanpy",
        "scanpy_version": scanpy_version,
        "mode": mode,
        "platform": platform,
        "panel_feature_count": int(panel_feature_count),
        "selected_scale": selected_scale,
        "thresholds": thresholds,
        "mito": {
            "gate_applied": bool(mito_gate_applied),
            "n_mito_features": int(n_mito_features),
            "reason": mito_skip_reason,
        },
        "cells": {
            "total": total,
            "failed_min_genes": int(np.sum(failed_min_genes)),
            "failed_min_counts": int(np.sum(failed_min_counts)),
            "failed_pct_mt": int(np.sum(failed_pct_mt)),
            "retained": int(np.sum(retained_mask)),
        },
        "imputation": "none",
        "scope_note": (
            "QC affects marker computation only; L_STAR_domain_assignment.csv "
            "covers all aligned observations regardless of QC outcome."
        ),
    }
    return QCFilterResult(cell_mask=retained_mask, audit=audit)


def apply_quality_control(
    expression_values: Any,
    *,
    selected_scale: str,
    feature_mito_labels: Sequence[Optional[str]],
    precomputed_obs_qc: Optional[pd.DataFrame] = None,
) -> QCFilterResult:
    """Apply platform-adaptive, deterministic cell-level quality control.

    Three modes are chosen deterministically depending on what the input
    provides:

    - ``counts_scanpy``: ``selected_scale == "raw_counts"``. Uses
      ``scanpy.pp.calculate_qc_metrics`` on the supplied counts matrix, at
      its own defaults except for ``percent_top``, which is clamped to the
      entries at or below the panel's feature count (scanpy's default,
      ``(50, 100, 200, 500)``, otherwise raises ``IndexError`` on panels with
      fewer than 500 features).
    - ``precomputed_obs``: the expression is not raw counts, but
      ``precomputed_obs_qc`` carries recognized QC columns (for example, an
      H5AD whose ``obs`` already stores ``n_genes_by_counts`` /
      ``total_counts`` from an upstream scanpy run). Thresholds are applied
      to those columns directly.
    - ``no_counts_no_metrics``: neither counts nor usable precomputed columns
      are available. Falls back to a detected-gene count from the supplied
      matrix (nonzero entries), with no counts gate and no mitochondrial
      gate.

    No imputation is performed. Filtering only ever affects which
    observations feed marker computation; it never removes observations from
    the final domain-assignment output.

    Args:
        expression_values: Aligned, cell-by-feature matrix for the selected
            expression source (dense or sparse), in the scale named by
            ``selected_scale``.
        selected_scale: The expression scale chosen for this run (one of the
            values from :data:`preprocessing.SUPPORTED_EXPRESSION_SCALES`).
        feature_mito_labels: One label per feature column, used to identify
            mitochondrial genes via
            :func:`markers.is_mitochondrial_gene`. Callers typically pass
            the resolved ``annotation_symbol`` (falling back to the original
            feature identifier) so that species-specific mitochondrial
            naming conventions resolved during feature resolution are used.
        precomputed_obs_qc: Optional per-observation QC metadata aligned to
            the same row order as ``expression_values``, as produced by
            :func:`dataset_input.load_annotation_inputs`
            (``AnnotationInputs.qc_metadata["precomputed_obs_qc"]``).

    Returns:
        A :class:`QCFilterResult` with the retained-cell mask and a
        JSON-serializable audit record.

    Raises:
        ValueError: If ``expression_values`` is not a two-dimensional matrix
            or ``feature_mito_labels`` does not match its column count.
    """
    import scanpy as sc

    try:
        import importlib.metadata

        scanpy_version = importlib.metadata.version("scanpy")
    except Exception:
        scanpy_version = str(sc.__version__)

    is_sparse = sparse.issparse(expression_values)
    matrix = expression_values.tocsr() if is_sparse else np.asarray(expression_values)
    if matrix.ndim != 2:
        raise ValueError("expression_values must be a two-dimensional matrix")
    n_observations, n_features = matrix.shape
    mito_labels = list(feature_mito_labels)
    if len(mito_labels) != n_features:
        raise ValueError(
            "feature_mito_labels length must equal the number of expression "
            f"columns ({n_features})"
        )

    platform = _platform(n_features)
    thresholds = _threshold_dict(platform)
    mito_mask = _mito_mask_from_labels(mito_labels)
    n_mito_features = int(np.sum(mito_mask))

    if selected_scale == "raw_counts":
        from anndata import AnnData

        adata = AnnData(X=matrix.astype(np.float32))
        mito_gate_applied = n_mito_features >= 1 and platform == "transcriptome_wide"
        if n_mito_features == 0:
            mito_skip_reason = "no mitochondrial features identifiable"
        elif platform == "targeted_panel":
            mito_skip_reason = "mitochondrial gate is not applied to targeted panels"
        else:
            mito_skip_reason = None
        adata.var["mt"] = mito_mask
        qc_vars = ["mt"] if mito_gate_applied else []
        # scanpy's percent_top default, (50, 100, 200, 500), indexes into the
        # feature axis and raises IndexError once an entry exceeds the panel
        # size (observed on <500-feature targeted panels, e.g. a 351-gene
        # seqFISH panel). Entries above n_features are dropped instead of
        # substituting a different fixed list, so panels large enough for the
        # full scanpy default still get it unchanged; this is the one
        # deliberate deviation from scanpy's own defaults in this module.
        percent_top = tuple(n for n in (50, 100, 200, 500) if n <= n_features) or None
        sc.pp.calculate_qc_metrics(
            adata, qc_vars=qc_vars, percent_top=percent_top, inplace=True
        )
        n_genes = adata.obs["n_genes_by_counts"].to_numpy(dtype=float)
        total_counts = adata.obs["total_counts"].to_numpy(dtype=float)
        pct_counts_mt = (
            adata.obs["pct_counts_mt"].to_numpy(dtype=float)
            if mito_gate_applied
            else None
        )
        return _finalize(
            mode="counts_scanpy",
            platform=platform,
            panel_feature_count=n_features,
            selected_scale=selected_scale,
            scanpy_version=scanpy_version,
            thresholds=thresholds,
            n_genes=n_genes,
            total_counts=total_counts,
            pct_counts_mt=pct_counts_mt,
            mito_gate_applied=mito_gate_applied,
            n_mito_features=n_mito_features,
            mito_skip_reason=mito_skip_reason,
        )

    if precomputed_obs_qc is not None and len(precomputed_obs_qc) == n_observations:
        genes_col = _first_present_column(precomputed_obs_qc, _PRECOMPUTED_GENES_COLUMNS)
        counts_col = _first_present_column(precomputed_obs_qc, _PRECOMPUTED_COUNTS_COLUMNS)
        mito_col = _first_present_column(precomputed_obs_qc, _PRECOMPUTED_MITO_COLUMNS)
        if genes_col is not None or counts_col is not None:
            n_genes = (
                precomputed_obs_qc[genes_col].to_numpy(dtype=float)
                if genes_col is not None
                else np.asarray((matrix > 0).sum(axis=1), dtype=float).ravel()
            )
            total_counts = (
                precomputed_obs_qc[counts_col].to_numpy(dtype=float)
                if counts_col is not None
                else None
            )
            pct_counts_mt = None
            mito_gate_applied = False
            mito_skip_reason = "no mitochondrial features identifiable"
            if platform == "transcriptome_wide" and mito_col is not None:
                candidate = precomputed_obs_qc[mito_col].to_numpy(dtype=float)
                if np.any(np.isfinite(candidate)):
                    pct_counts_mt = candidate
                    mito_gate_applied = True
                    mito_skip_reason = None
                else:
                    mito_skip_reason = (
                        f"precomputed column {mito_col!r} has no finite values"
                    )
            elif platform == "targeted_panel":
                mito_skip_reason = "mitochondrial gate is not applied to targeted panels"
            thresholds_precomputed = dict(thresholds)
            if counts_col is None:
                thresholds_precomputed["min_counts"] = None
            return _finalize(
                mode="precomputed_obs",
                platform=platform,
                panel_feature_count=n_features,
                selected_scale=selected_scale,
                scanpy_version=scanpy_version,
                thresholds=thresholds_precomputed,
                n_genes=n_genes,
                total_counts=total_counts,
                pct_counts_mt=pct_counts_mt,
                mito_gate_applied=mito_gate_applied,
                n_mito_features=n_mito_features,
                mito_skip_reason=mito_skip_reason,
            )

    # no_counts_no_metrics: fall back to a detected-gene count from the
    # supplied matrix; no counts gate, no mitochondrial gate.
    if is_sparse:
        n_genes = np.asarray(matrix.getnnz(axis=1), dtype=float)
    else:
        n_genes = np.asarray(np.sum(matrix > 0.0, axis=1), dtype=float)
    thresholds_fallback = {
        "min_genes": thresholds["min_genes"],
        "min_counts": None,
        "max_pct_counts_mt": None,
        "min_cells_per_gene": thresholds["min_cells_per_gene"],
    }
    return _finalize(
        mode="no_counts_no_metrics",
        platform=platform,
        panel_feature_count=n_features,
        selected_scale=selected_scale,
        scanpy_version=scanpy_version,
        thresholds=thresholds_fallback,
        n_genes=n_genes,
        total_counts=None,
        pct_counts_mt=None,
        mito_gate_applied=False,
        n_mito_features=n_mito_features,
        mito_skip_reason="counts are unavailable for this expression scale",
    )


__all__ = [
    "MIN_CELLS_PER_GENE",
    "QC_RULES_VERSION",
    "TARGETED_MIN_COUNTS",
    "TARGETED_MIN_GENES",
    "TARGETED_PANEL_MAX_FEATURES",
    "TRANSCRIPTOME_MAX_PCT_MT",
    "TRANSCRIPTOME_MIN_GENES",
    "QCFilterResult",
    "apply_quality_control",
]
