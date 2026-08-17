"""Marker-gene statistics for L-STAR domain annotation.

This module intentionally contains no network or reference-database access.

Differential expression is delegated to ``scanpy.tl.rank_genes_groups``, a
widely used and independently validated implementation, rather than a custom
statistical engine. The module follows scanpy's own workflow as closely as
the task allows: ``method`` is left unset so scanpy picks the test, and
markers are taken from the top of the ranking scanpy returns, with no
threshold filter — ``filter_rank_genes_groups`` is an opt-in step scanpy does
not invoke. Only positive (enriched) markers are retained; there is no negative-marker or
contradiction-evidence pathway in this module.

Two kinds of gene are deliberately *not* removed here, because removing them
would mean imposing a judgement scanpy does not make:

- No hand-curated "technical gene" list (housekeeping, ribosomal, stress).
  Deciding what counts as a housekeeping gene has no principled cutoff.
- No mitochondrial-gene removal. scanpy provides no such step: it computes a
  mitochondrial *fraction* per cell for quality control and leaves the genes
  in the matrix.

With no threshold filter, mitochondrial and other broadly expressed genes can
therefore reach the ranked marker list. They are handled where a human
annotator handles them — at interpretation time, by the naming prompt in
``skills/domain_annotation/annotation.prompt.md``, which instructs the model
to read them as cell-state or capture-quality signals rather than as evidence
of domain identity. :func:`is_mitochondrial_gene` exists only for the
mitochondrial-fraction quality-control gate in :mod:`qc_filtering`.
"""

import math
from numbers import Integral, Real
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import sparse


MARKER_RULES_VERSION = "4.1.0"
MARKER_STATISTICS_ENGINE = "scanpy.tl.rank_genes_groups"
# ``None`` means "do not pass ``method`` to rank_genes_groups at all", so
# scanpy's own default applies (it resolves an unset ``method`` to "t-test").
# Naming the value ``None`` rather than "t-test" keeps the manifest honest:
# it records that we did not choose a test, not that we chose scanpy's.
MARKER_STATISTICS_METHOD = None
# scanpy's resolution of an unset ``method``, recorded for provenance.
SCANPY_DEFAULT_MARKER_TEST_METHOD = "t-test"
# The prevalence defaults published by scanpy.tl.filter_rank_genes_groups,
# adopted verbatim so the cutoffs are the upstream authors' rather than ours.
# Its third criterion, min_fold_change, is not adopted -- see
# select_positive_markers for why.
SCANPY_MIN_IN_GROUP_FRACTION = 0.25
SCANPY_MAX_OUT_GROUP_FRACTION = 0.5
# The values scanpy.tl.rank_genes_groups accepts for ``method``.
SUPPORTED_MARKER_TEST_METHODS = frozenset(
    {"logreg", "t-test", "wilcoxon", "t-test_overestim_var"}
)

_MARKER_COLUMNS = (
    "gene",
    "score",
    "effect_size",
    "avg_log2FC",
    "pct_in",
    "pct_out",
    "delta_pct",
    "adjusted_p_value",
)


def _validate_unit_interval(name: str, value: Real) -> None:
    """Validate a finite real value in the closed unit interval."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number in [0, 1]")
    if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be a finite real number in [0, 1]")


def _validate_count(name: str, value: Integral, minimum: int = 0) -> None:
    """Validate an integer count."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")
    if int(value) < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")


def is_mitochondrial_gene(gene: str) -> bool:
    """Return whether a gene name marks a mitochondrially encoded transcript.

    Detection is by the ``MT-`` symbol prefix alone, matched
    case-insensitively and ignoring surrounding whitespace. This is used only
    by :func:`qc_filtering.apply_quality_control` for the standard
    mitochondrial-fraction quality-control gate; it never removes a gene from
    naming evidence.
    """
    if not isinstance(gene, str):
        return False
    return gene.strip().upper().startswith("MT-")


def _validate_gene_names(gene_names: Sequence[str], number_of_genes: int) -> Sequence[str]:
    """Return validated gene names without changing their spelling or order."""
    if isinstance(gene_names, (str, bytes)):
        raise ValueError("gene_names must be a sequence of gene-name strings")

    try:
        names = list(gene_names)
    except TypeError as exc:
        raise ValueError("gene_names must be a sequence of gene-name strings") from exc

    if len(names) != number_of_genes:
        raise ValueError(
            "gene_names length must equal the number of expression-matrix columns "
            f"({number_of_genes})"
        )
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("gene_names must contain only non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("gene_names must be unique")
    return names


def _validate_group_labels(
    group_labels: Sequence[str], number_of_cells: int
) -> pd.Series:
    """Return validated per-cell group labels as a string Series."""
    if isinstance(group_labels, (str, bytes)):
        raise ValueError("group_labels must be a sequence of label strings")
    labels = pd.Series(list(group_labels)).astype(str)
    if len(labels) != number_of_cells:
        raise ValueError(
            "group_labels length must equal the number of expression-matrix rows "
            f"({number_of_cells})"
        )
    return labels


def compute_marker_statistics(
    expression_values: Any,
    gene_names: Sequence[str],
    group_labels: Sequence[str],
    *,
    groups: Optional[Sequence[str]] = None,
    method: Optional[str] = MARKER_STATISTICS_METHOD,
) -> Dict[str, pd.DataFrame]:
    """Compute one-vs-rest positive-marker statistics for every requested group.

    Differential expression is computed with a single call to
    ``scanpy.tl.rank_genes_groups``, covering every requested group at once,
    with Benjamini-Hochberg correction applied independently within each
    group's family of tests. ``method`` is the only statistical argument this
    module passes explicitly; every other ``rank_genes_groups`` parameter
    (``reference``, ``corr_method``, ``n_genes``, ``tie_correct``) is left at
    its scanpy default. In particular ``pts`` is left at its default of
    ``False`` — this function computes ``pct_in``/``pct_out`` itself, from the
    same nonzero-cell fraction definition scanpy's ``pts`` uses, so the
    returned columns are unaffected. Expression values must already be
    log1p-normalized (the ``marker_matrix_scale`` produced by
    :func:`preprocessing.normalize_expression`). No genes are removed from
    the returned tables, including non-significant ones; ranking and
    selection and ranking happen in :func:`select_positive_markers`.

    Args:
        expression_values: Dense or sparse cell-by-gene, nonnegative,
            log1p-normalized expression matrix.
        gene_names: Unique, non-empty gene names matching matrix columns.
        group_labels: One domain or method label per cell. At least two
            distinct labels must be present.
        groups: Labels to compute statistics for. Defaults to every distinct
            label in first-seen order. Each requested label must have at
            least two in-group cells and at least one out-group cell
            (scanpy cannot rank a single-cell group).
        method: Differential-expression test passed to
            ``scanpy.tl.rank_genes_groups``; one of
            :data:`SUPPORTED_MARKER_TEST_METHODS`. Defaults to
            :data:`MARKER_STATISTICS_METHOD` (``None``), which leaves the
            argument unset so scanpy picks the test. ``score`` then holds that test's statistic, so the
            downstream score-based ranking follows the chosen test.

    Returns:
        A mapping from group label to a DataFrame in the original gene order
        with the test statistic, fold-change, prevalence, and raw and
        BH-adjusted p-values.

    Raises:
        ValueError: If any input violates the matrix, name, label, or method
            contract.
    """
    import scanpy as sc
    from anndata import AnnData

    if method is not None and method not in SUPPORTED_MARKER_TEST_METHODS:
        raise ValueError(
            "method must be None or one of: {}".format(
                ", ".join(sorted(SUPPORTED_MARKER_TEST_METHODS))
            )
        )
    is_sparse = sparse.issparse(expression_values)
    try:
        values = (
            expression_values.tocsr()
            if is_sparse
            else np.asarray(expression_values)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("expression_values must be a numeric matrix") from exc
    if np.issubdtype(values.dtype, np.bool_):
        raise ValueError("expression_values must not contain Boolean values")
    try:
        if not np.issubdtype(values.dtype, np.number):
            values = values.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("expression_values must be a dense numeric matrix") from exc

    if values.ndim != 2:
        raise ValueError("expression_values must be a two-dimensional dense matrix")
    number_of_cells, number_of_genes = values.shape
    if number_of_cells == 0:
        raise ValueError("expression_values must contain at least one cell")
    if number_of_genes == 0:
        raise ValueError("expression_values must contain at least one gene")
    finite_values = values.data if is_sparse else values
    if not np.all(np.isfinite(finite_values)):
        raise ValueError("expression_values must contain only finite values")
    if np.any(finite_values < 0.0):
        raise ValueError("expression_values must contain only nonnegative values")

    names = list(_validate_gene_names(gene_names, number_of_genes))
    labels = _validate_group_labels(group_labels, number_of_cells)
    labels_array = labels.to_numpy()
    distinct_labels = list(dict.fromkeys(labels.tolist()))
    if len(distinct_labels) < 2:
        raise ValueError("group_labels must contain at least two distinct labels")

    requested_groups = list(groups) if groups is not None else distinct_labels
    if not requested_groups:
        raise ValueError("groups must be non-empty when provided")
    unknown_groups = sorted(set(requested_groups) - set(distinct_labels))
    if unknown_groups:
        raise ValueError(
            f"groups contains labels absent from group_labels: {unknown_groups}"
        )
    for group in requested_groups:
        in_group_count = int(np.sum(labels_array == group))
        # scanpy's Wilcoxon test requires at least two in-group samples; a
        # single-cell group cannot be ranked and must be excluded by the
        # caller (typically routed to a degenerate/abstention path) instead
        # of being passed here.
        if in_group_count < 2:
            raise ValueError(
                f"group {group!r} has fewer than two in-group cells "
                f"({in_group_count}); at least two are required"
            )
        if in_group_count == number_of_cells:
            raise ValueError(f"group {group!r} leaves no out-group cells")

    matrix = values.astype(np.float32)
    adata = AnnData(X=matrix)
    adata.var_names = pd.Index(names)
    adata.obs["group"] = pd.Categorical(labels_array, categories=distinct_labels)
    # Declares the matrix is already log1p-transformed so scanpy's
    # fold-change calculation (expm1-based) is well-defined.
    adata.uns["log1p"] = {"base": None}

    # Passing nothing when ``method`` is None is the point: scanpy then
    # applies its own default rather than a value we picked.
    rank_kwargs = {} if method is None else {"method": method}
    sc.tl.rank_genes_groups(
        adata,
        "group",
        groups=list(requested_groups),
        **rank_kwargs,
    )
    result = adata.uns["rank_genes_groups"]

    frames: Dict[str, pd.DataFrame] = {}
    for group in requested_groups:
        group_mask = labels_array == group
        if is_sparse:
            mean_in = np.asarray(matrix[group_mask].mean(axis=0), dtype=float).ravel()
            mean_out = np.asarray(matrix[~group_mask].mean(axis=0), dtype=float).ravel()
            # scanpy's own ``pts`` is a nonzero-cell fraction computed via
            # ``getnnz`` for sparse matrices; matched here since ``pts=True``
            # is no longer passed to rank_genes_groups (see docstring).
            nnz_in = matrix[group_mask].getnnz(axis=0).astype(np.float64)
            nnz_out = matrix[~group_mask].getnnz(axis=0).astype(np.float64)
        else:
            mean_in = np.mean(matrix[group_mask], axis=0, dtype=np.float64)
            mean_out = np.mean(matrix[~group_mask], axis=0, dtype=np.float64)
            nnz_in = np.count_nonzero(matrix[group_mask], axis=0).astype(np.float64)
            nnz_out = np.count_nonzero(matrix[~group_mask], axis=0).astype(np.float64)
        mean_in_by_gene = pd.Series(mean_in, index=names)
        mean_out_by_gene = pd.Series(mean_out, index=names)
        n_in_group = float(np.sum(group_mask))
        n_out_group = float(np.sum(~group_mask))
        pct_in_by_gene = pd.Series(nnz_in / n_in_group, index=names)
        pct_out_by_gene = pd.Series(nnz_out / n_out_group, index=names)

        ranked_names = np.asarray(result["names"][group])
        ranked = pd.DataFrame(
            {
                "score": np.asarray(result["scores"][group], dtype=float),
                "p_value": np.asarray(result["pvals"][group], dtype=float),
                "adjusted_p_value": np.asarray(
                    result["pvals_adj"][group], dtype=float
                ),
                "avg_log2FC": np.asarray(
                    result["logfoldchanges"][group], dtype=float
                ),
            },
            index=ranked_names,
        )
        # Restore the original gene order (scanpy ranks by score).
        ranked = ranked.reindex(names)

        score = np.where(np.isfinite(ranked["score"]), ranked["score"], 0.0)
        p_value = np.where(np.isfinite(ranked["p_value"]), ranked["p_value"], 1.0)
        adjusted_p_value = np.where(
            np.isfinite(ranked["adjusted_p_value"]), ranked["adjusted_p_value"], 1.0
        )
        avg_log2fc = np.where(
            np.isfinite(ranked["avg_log2FC"]), ranked["avg_log2FC"], 0.0
        )
        pct_in = pct_in_by_gene.to_numpy(dtype=float)
        pct_out = pct_out_by_gene.to_numpy(dtype=float)

        frames[group] = pd.DataFrame(
            {
                "gene": names,
                "score": score,
                "effect_size": (mean_in_by_gene - mean_out_by_gene).to_numpy(
                    dtype=float
                ),
                "effect_definition": "difference in mean log1p-normalized expression",
                "mean_in": mean_in_by_gene.to_numpy(dtype=float),
                "mean_out": mean_out_by_gene.to_numpy(dtype=float),
                "avg_log2FC": avg_log2fc,
                "pct_in": pct_in,
                "pct_out": pct_out,
                "delta_pct": pct_in - pct_out,
                "p_value": p_value,
                "adjusted_p_value": adjusted_p_value,
            },
            columns=[
                "gene",
                "score",
                "effect_size",
                "effect_definition",
                "mean_in",
                "mean_out",
                "avg_log2FC",
                "pct_in",
                "pct_out",
                "delta_pct",
                "p_value",
                "adjusted_p_value",
            ],
        )

    return frames


def _validate_marker_table(marker_statistics: pd.DataFrame) -> None:
    """Validate the columns and scalar values needed for marker filtering."""
    if not isinstance(marker_statistics, pd.DataFrame):
        raise ValueError("marker_statistics must be a pandas DataFrame")
    missing_columns = [
        column for column in _MARKER_COLUMNS if column not in marker_statistics.columns
    ]
    if missing_columns:
        raise ValueError(
            "marker_statistics is missing required columns: " + ", ".join(missing_columns)
        )

    genes = marker_statistics["gene"]
    valid_genes = genes.map(
        lambda gene: isinstance(gene, str) and bool(gene.strip())
    )
    if genes.isna().any() or not valid_genes.all():
        raise ValueError("marker_statistics gene values must be non-empty strings")

    numeric_columns = [column for column in _MARKER_COLUMNS if column != "gene"]
    for column in numeric_columns:
        try:
            values = marker_statistics[column].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"marker_statistics column {column!r} must be numeric") from exc
        if not np.all(np.isfinite(values)):
            raise ValueError(f"marker_statistics column {column!r} must contain finite values")

    for column in ("pct_in", "pct_out", "adjusted_p_value"):
        values = marker_statistics[column].to_numpy(dtype=float)
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"marker_statistics column {column!r} must lie in [0, 1]")


def _stable_marker_sort(markers: pd.DataFrame) -> pd.DataFrame:
    """Sort a filtered positive-marker table deterministically."""
    return markers.sort_values(
        by=["score", "avg_log2FC", "adjusted_p_value", "gene"],
        ascending=[False, False, True, True],
        kind="mergesort",
    )


def select_positive_markers(
    marker_statistics: pd.DataFrame,
    min_pct_in: Optional[float] = SCANPY_MIN_IN_GROUP_FRACTION,
    max_pct_out: Optional[float] = SCANPY_MAX_OUT_GROUP_FRACTION,
    max_positive_markers: Optional[int] = 25,
) -> Dict[str, pd.DataFrame]:
    """Select the positive (enriched) markers and keep the top of the ranking.

    A marker qualifies when it is enriched (``avg_log2FC > 0``) and passes two
    prevalence criteria taken from ``scanpy.tl.filter_rank_genes_groups``,
    with that function's own default values and its strict comparisons:
    detected in more than ``min_pct_in`` of the domain's observations, and in
    fewer than ``max_pct_out`` of the rest.

    Its third criterion, ``min_fold_change``, is deliberately not applied.
    That threshold asks for a greater-than-twofold difference, which suits
    discrete cell-type clusters but not spatial domains, whose neighbours
    differ by gradients: on cortical data it starved a genuine layer-6 domain
    down to three incoherent genes by rejecting TBR1 at a fold change of
    0.87. The prevalence criteria carry the discriminative work — a gene
    expressed throughout the tissue fails ``max_pct_out`` regardless of how
    consistently it differs — so dropping the fold-change floor widens the
    evidence without admitting ubiquitous genes.

    Ranking is by ``score`` descending — the statistic
    ``rank_genes_groups`` itself ranks by — with fold change, adjusted
    p-value, and gene name as deterministic tie-breaks. Passing ``None`` for
    either prevalence threshold switches that criterion off. The complete
    input table is always returned under ``all_markers``, never truncated or
    reordered.

    Args:
        marker_statistics: Full marker-statistics table, as returned by
            :func:`compute_marker_statistics` for one group.
        min_pct_in: Detection fraction within the domain must exceed this.
            ``None`` disables the criterion.
        max_pct_out: Detection fraction outside the domain must fall below
            this. ``None`` disables the criterion.
        max_positive_markers: Maximum number of positive markers returned, or
            ``None`` for no truncation.

    Returns:
        A dictionary containing independent DataFrames under ``all_markers``
        and ``positive_markers``.

    Raises:
        ValueError: If the table or any parameter is invalid.
    """
    for name, value in (("min_pct_in", min_pct_in), ("max_pct_out", max_pct_out)):
        if value is not None:
            _validate_unit_interval(name, value)
    if max_positive_markers is not None:
        _validate_count("max_positive_markers", max_positive_markers)
    _validate_marker_table(marker_statistics)

    all_markers = marker_statistics.copy(deep=True)
    selected = all_markers["avg_log2FC"] > 0.0
    if min_pct_in is not None:
        selected &= all_markers["pct_in"] > float(min_pct_in)
    if max_pct_out is not None:
        selected &= all_markers["pct_out"] < float(max_pct_out)

    positive_markers = _stable_marker_sort(all_markers.loc[selected].copy())
    if max_positive_markers is not None:
        positive_markers = positive_markers.head(int(max_positive_markers))

    return {
        "all_markers": all_markers,
        "positive_markers": positive_markers,
    }


__all__ = [
    "MARKER_RULES_VERSION",
    "MARKER_STATISTICS_ENGINE",
    "SCANPY_MAX_OUT_GROUP_FRACTION",
    "SCANPY_MIN_IN_GROUP_FRACTION",
    "MARKER_STATISTICS_METHOD",
    "SCANPY_DEFAULT_MARKER_TEST_METHOD",
    "SUPPORTED_MARKER_TEST_METHODS",
    "compute_marker_statistics",
    "select_positive_markers",
    "is_mitochondrial_gene",
]
