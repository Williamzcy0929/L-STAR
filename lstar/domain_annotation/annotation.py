"""Generalizable post-consensus biological annotation for L-STAR domains.

The implementation preserves expression-observation order and numerical L-STAR
membership for the exactly aligned or explicitly declared partial population.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

if __package__:
    from .global_annotation import (
        GLOBAL_ANNOTATION_RULES_VERSION,
        load_global_annotation_resources,
        run_global_annotation,
    )
    from .feature_resolution import (
        FeatureAnnotationEvidenceError,
        FEATURE_RESOLUTION_RULES_VERSION,
        discover_supporting_files,
        load_feature_resolution_resources,
        replan_feature_resolution,
        resolve_features,
    )
    from .dataset_input import AnnotationInputs, load_annotation_inputs
    from .llm_annotation import (
        DEFAULT_ANNOTATION_MODEL,
        DEFAULT_ANNOTATION_REASONING_EFFORT,
        SUPPORTED_REASONING_EFFORTS,
        create_openai_client,
    )
    from .markers import (
        MARKER_RULES_VERSION,
        SCANPY_DEFAULT_MARKER_TEST_METHOD,
        SCANPY_MAX_OUT_GROUP_FRACTION,
        SCANPY_MIN_IN_GROUP_FRACTION,
        SUPPORTED_MARKER_TEST_METHODS,
        compute_marker_statistics,
        select_positive_markers,
    )
    from .output_utils import (
        domain_artifact_stem,
        remove_stale_managed_artifacts,
        validate_final_assignment,
        write_dataframe_atomic,
        write_json_atomic,
    )
    from .preprocessing import (
        PREPROCESSING_RULES_VERSION,
        PreprocessingResult,
        preprocess_expression,
    )
    from .qc_filtering import QC_RULES_VERSION, apply_quality_control
    from .sampling_strategy import resolve_sampling_level
    from .visual_context import build_visual_context
else:
    from global_annotation import (
        GLOBAL_ANNOTATION_RULES_VERSION,
        load_global_annotation_resources,
        run_global_annotation,
    )
    from feature_resolution import (
        FeatureAnnotationEvidenceError,
        FEATURE_RESOLUTION_RULES_VERSION,
        discover_supporting_files,
        load_feature_resolution_resources,
        replan_feature_resolution,
        resolve_features,
    )
    from dataset_input import AnnotationInputs, load_annotation_inputs
    from llm_annotation import (
        DEFAULT_ANNOTATION_MODEL,
        DEFAULT_ANNOTATION_REASONING_EFFORT,
        SUPPORTED_REASONING_EFFORTS,
        create_openai_client,
    )
    from markers import (
        MARKER_RULES_VERSION,
        SCANPY_DEFAULT_MARKER_TEST_METHOD,
        SCANPY_MAX_OUT_GROUP_FRACTION,
        SCANPY_MIN_IN_GROUP_FRACTION,
        SUPPORTED_MARKER_TEST_METHODS,
        compute_marker_statistics,
        select_positive_markers,
    )
    from output_utils import (
        domain_artifact_stem,
        remove_stale_managed_artifacts,
        validate_final_assignment,
        write_dataframe_atomic,
        write_json_atomic,
    )
    from preprocessing import (
        PREPROCESSING_RULES_VERSION,
        PreprocessingResult,
        preprocess_expression,
    )
    from qc_filtering import QC_RULES_VERSION, apply_quality_control
    from sampling_strategy import resolve_sampling_level
    from visual_context import build_visual_context


PathLike = Union[str, Path]
PRIMARY_OUTPUT_FILENAME = "L_STAR_domain_assignment.csv"
DEFAULT_ARTIFACTS_DIRNAME = "annotation_artifacts"
INTERNAL_USE_PALO = True
INTERNAL_MAX_BACKGROUND_IMAGE_DIMENSION = 1600
MIN_INTERPRETABLE_POSITIVE_GENES = 2
ALLOWED_RESOLUTION_RESOURCE_SUFFIXES = frozenset({".h5ad", ".h5", ".hdf5"})
logger = logging.getLogger(__name__)


def _marker_statistics_description(marker_test_method: Optional[str]) -> str:
    """Describe the differential-expression call for the run manifest."""
    if marker_test_method is None:
        return (
            "scanpy rank_genes_groups with every parameter at its scanpy "
            "default, including method, which scanpy resolves to {!r}: "
            "per-domain Benjamini-Hochberg, positive markers only".format(
                SCANPY_DEFAULT_MARKER_TEST_METHOD
            )
        )
    return (
        "scanpy rank_genes_groups(method={!r}), every other parameter at its "
        "scanpy default: per-domain Benjamini-Hochberg, positive markers "
        "only. scanpy's own default when method is unset is {!r}.".format(
            marker_test_method, SCANPY_DEFAULT_MARKER_TEST_METHOD
        )
    )


def _first_seen(values: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(str(value) for value in values))


def _mito_labels_for_qc(feature_resolution: Any) -> List[str]:
    """Return one mitochondrial-detection label per original feature.

    Prefers the resolved ``annotation_symbol`` (which captures
    species-specific naming conventions established during feature
    resolution, e.g. ``MT-`` vs. ``mt-``) and falls back to the original
    feature identifier when no symbol was resolved.
    """
    metadata = feature_resolution.compatibility_metadata
    symbols = metadata["annotation_symbol"]
    originals = metadata["original_gene_id"].astype(str)
    return [
        str(symbol) if pd.notna(symbol) and str(symbol).strip() else str(original)
        for symbol, original in zip(symbols, originals)
    ]


def _path(value: PathLike, name: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError("{} must be a string or pathlib.Path".format(name))
    if isinstance(value, str) and not value.strip():
        raise ValueError("{} must not be empty".format(name))
    return Path(value).expanduser()


def _marker_filter_settings(
    max_positive_markers: Optional[int] = 25,
) -> Dict[str, Any]:
    """Return the fixed marker policy used by the annotation pipeline.

    Markers are the top of the ranking ``scanpy.tl.rank_genes_groups``
    returns, restricted to positive (enriched) genes that pass the two
    prevalence criteria of ``scanpy.tl.filter_rank_genes_groups`` at that
    function's own default values. Its fold-change criterion is not applied:
    a greater-than-twofold floor suits discrete cell-type clusters but starves
    spatial domains, whose neighbours differ by gradients. The prevalence
    criteria are what keep ubiquitously expressed genes out, so dropping the
    fold-change floor widens the evidence without letting them in. There is
    no negative-marker or contradiction-evidence policy.
    This dict is recorded in the evidence card and the run manifest.
    """
    return {
        "min_pct_in": SCANPY_MIN_IN_GROUP_FRACTION,
        "max_pct_out": SCANPY_MAX_OUT_GROUP_FRACTION,
        "min_log2fc": None,
        "max_positive_markers": max_positive_markers,
        "marker_direction_policy": "positive_only",
        "ranking": "scanpy_rank_genes_groups_score_desc",
        "threshold_semantics": (
            "pct_in and pct_out use scanpy.tl.filter_rank_genes_groups' own "
            "default values and strict comparisons (pct_in > 0.25, "
            "pct_out < 0.5); its min_fold_change criterion is deliberately "
            "not applied, because a >2-fold floor starves spatial domains "
            "whose neighbours differ by gradients. No adjusted-p-value "
            "criterion is applied, matching that function"
        ),
    }


def _normalized_entropy(values: Sequence[str]) -> float:
    counts = pd.Series(list(values), dtype=str).value_counts().to_numpy(dtype=float)
    if len(counts) <= 1:
        return 0.0
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log(probabilities)) / np.log(len(counts)))


def _build_input_contexts(
    inputs: AnnotationInputs,
    lstar_labels: np.ndarray,
    domain_ids: Sequence[str],
) -> tuple:
    """Summarize allowed obs metadata and optional spatial neighborhoods."""
    contexts: Dict[str, Dict[str, Any]] = {
        str(domain_id): {
            "expression_source": inputs.expression_source_manifest[
                "selected_expression_source"
            ],
            "expression_scale": inputs.expression_source_manifest[
                "selected_expression_scale"
            ],
            "input_format": inputs.input_format,
            "allowed_observation_context": {},
            "spatial": {"available": False},
        }
        for domain_id in domain_ids
    }
    for domain_id in domain_ids:
        mask = lstar_labels == str(domain_id)
        for column in inputs.observation_metadata.columns:
            values = inputs.observation_metadata.loc[mask, column].astype(str)
            counts = values.value_counts(dropna=False)
            contexts[str(domain_id)]["allowed_observation_context"][str(column)] = {
                "distinct_values": int(len(counts)),
                "top_values": [
                    {
                        "value": str(value),
                        "count": int(count),
                        "fraction": float(count / len(values)),
                    }
                    for value, count in counts.head(10).items()
                ],
            }

    relationship_rows = []
    coordinates = inputs.spatial_coordinates
    if coordinates is not None and len(coordinates) >= 2:
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] < 2:
            raise ValueError("Selected spatial coordinates must be an n_obs-by-2 matrix")
        if not np.all(np.isfinite(coordinates[:, :2])):
            raise ValueError("Selected spatial coordinates contain non-finite values")
        k = min(7, len(coordinates))
        _, neighbors = cKDTree(coordinates[:, :2]).query(coordinates[:, :2], k=k)
        if neighbors.ndim == 1:
            neighbors = neighbors[:, None]
        adjacency: Dict[tuple, int] = {}
        for source_index, row in enumerate(neighbors):
            source_domain = str(lstar_labels[source_index])
            for target_index in row[1:]:
                target_domain = str(lstar_labels[int(target_index)])
                adjacency[(source_domain, target_domain)] = (
                    adjacency.get((source_domain, target_domain), 0) + 1
                )
        for domain_id in domain_ids:
            mask = lstar_labels == str(domain_id)
            selected = coordinates[mask, :2]
            centroid = np.mean(selected, axis=0)
            dispersion = np.sqrt(np.sum((selected - centroid) ** 2, axis=1))
            neighbors_for_domain = sorted(
                [
                    (target, count)
                    for (source, target), count in adjacency.items()
                    if source == str(domain_id) and target != str(domain_id)
                ],
                key=lambda item: (-item[1], item[0]),
            )
            total_external = sum(count for _, count in neighbors_for_domain)
            contexts[str(domain_id)]["spatial"] = {
                "available": True,
                "coordinate_key": inputs.spatial_key,
                "centroid": [float(value) for value in centroid],
                "median_distance_to_centroid": float(np.median(dispersion)),
                "neighbor_domains": [
                    {
                        "domain_id": target,
                        "directed_edge_count": int(count),
                        "external_edge_fraction": (
                            float(count / total_external) if total_external else 0.0
                        ),
                    }
                    for target, count in neighbors_for_domain[:10]
                ],
            }
        for (source, target), count in sorted(adjacency.items()):
            relationship_rows.append(
                {
                    "source_domain": source,
                    "target_domain": target,
                    "directed_neighbor_edges": int(count),
                    "same_domain": source == target,
                    "spatial_coordinate_key": inputs.spatial_key,
                }
            )
    relationships = pd.DataFrame(
        relationship_rows,
        columns=[
            "source_domain",
            "target_domain",
            "directed_neighbor_edges",
            "same_domain",
            "spatial_coordinate_key",
        ],
    )
    return contexts, relationships


def _add_molecular_relationships(
    contexts: Mapping[str, Dict[str, Any]],
    spatial_relationships: pd.DataFrame,
    domain_means: Mapping[str, np.ndarray],
    domain_ids: Sequence[str],
) -> pd.DataFrame:
    """Add domain-to-domain molecular similarity to the global context."""
    spatial_lookup = {
        (str(row.source_domain), str(row.target_domain)): row
        for row in spatial_relationships.itertuples(index=False)
    }
    rows = []
    for source in domain_ids:
        source_vector = np.asarray(domain_means[str(source)], dtype=float)
        source_norm = float(np.linalg.norm(source_vector))
        molecular_neighbors = []
        for target in domain_ids:
            target_vector = np.asarray(domain_means[str(target)], dtype=float)
            denominator = source_norm * float(np.linalg.norm(target_vector))
            similarity = (
                float(np.dot(source_vector, target_vector) / denominator)
                if denominator else 0.0
            )
            spatial = spatial_lookup.get((str(source), str(target)))
            rows.append(
                {
                    "source_domain": str(source),
                    "target_domain": str(target),
                    "molecular_cosine_similarity": similarity,
                    "directed_neighbor_edges": (
                        int(spatial.directed_neighbor_edges) if spatial is not None else 0
                    ),
                    "same_domain": str(source) == str(target),
                    "spatial_coordinate_key": (
                        spatial.spatial_coordinate_key if spatial is not None else None
                    ),
                }
            )
            if str(source) != str(target):
                molecular_neighbors.append((str(target), similarity))
        contexts[str(source)]["molecular_neighbor_domains"] = [
            {"domain_id": target, "cosine_similarity": float(similarity)}
            for target, similarity in sorted(
                molecular_neighbors, key=lambda item: (-item[1], item[0])
            )[:10]
        ]
    return pd.DataFrame(rows)


def _compact_marker_records(markers: pd.DataFrame) -> List[Dict[str, Any]]:
    """Serialize marker evidence without promoting inference labels to genes."""

    def label_candidates(value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        if not isinstance(value, str) or not value.strip():
            return []
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return (
            [str(item) for item in parsed if str(item).strip()]
            if isinstance(parsed, list)
            else []
        )

    confidence_labels = {
        "A": "validated_biological_identity",
        "B1": "exact_identity_single_source_candidate",
        "B2": "identity_with_multiple_or_unresolved_source_candidates",
        "C": "exact_feature_identity_unresolved_biology",
    }
    records = []
    for row in markers.to_dict(orient="records"):
        feature_label = str(
            row.get("source_feature_label")
            or row.get("annotation_label")
            or row.get("analysis_gene")
        ).strip()
        if not feature_label or feature_label.lower() in {"nan", "none", "null"}:
            raise FeatureAnnotationEvidenceError(
                "A selected marker has no serializable feature label"
            )
        tier = str(row.get("feature_evidence_tier", "C"))
        direct_eligible = bool(row.get("direct_confidence_eligible", False))
        inference_eligible = bool(
            row.get("inference_confidence_eligible", False)
        )
        evidence_id = str(
            row.get("annotation_label") if direct_eligible else feature_label
        ).strip()
        if not evidence_id or evidence_id.lower() in {"nan", "none", "null"}:
            raise FeatureAnnotationEvidenceError(
                "A selected marker has no valid annotation evidence identifier"
            )
        mapping_score = float(row.get("mapping_confidence", 0.0))
        if not np.isfinite(mapping_score):
            raise FeatureAnnotationEvidenceError(
                "A selected marker has non-finite mapping evidence"
            )
        record = {
            "gene": evidence_id,
            "feature_label": feature_label,
            "resolved_identity": str(row.get("identity_value") or feature_label),
            "label_provenance": str(
                row.get("annotation_label_source") or "opaque_identifier"
            ),
            "label_candidates": label_candidates(
                row.get("source_label_candidates")
            ),
            "label_parsing_rule": str(
                row.get("source_label_parse_rule")
                or "exact_identity_subtraction_and_generic_delimiter_split_v1"
            ),
            "feature_evidence_tier": tier,
            "mapping_confidence": confidence_labels.get(
                tier, "unresolved_feature_evidence"
            ),
            "mapping_score": mapping_score,
            "direct_confidence_eligible": direct_eligible,
            "inference_confidence_eligible": inference_eligible,
            "annotation_usable_for_naming": bool(
                row.get("annotation_usable_for_naming", False)
            ),
            "analysis_gene": str(row["analysis_gene"]),
            "original_gene_ids": str(row["original_gene_ids"]),
            "mapping_type": str(row["mapping_type"]),
        }
        for field in (
            "effect_size",
            "mean_in",
            "mean_out",
            "pct_in",
            "pct_out",
            "delta_pct",
            "score",
            "domain_specificity",
            "adjusted_p_value",
        ):
            value = float(row[field])
            if not np.isfinite(value):
                raise FeatureAnnotationEvidenceError(
                    "A selected marker has non-finite {}".format(field)
                )
            record[field] = value
        records.append(record)
    return records


def _deduplicate_biological_markers(
    markers: pd.DataFrame,
    *,
    maximum: int,
) -> pd.DataFrame:
    """De-duplicate and cap the positive markers sent to the LLM.

    Ranking is by the ``score`` column — the statistic
    ``scanpy.tl.rank_genes_groups`` itself ranks by — so the order here
    matches the order scanpy returned. De-duplication collapses features that
    resolve to the same biological gene, so one gene measured by several
    probes occupies one slot rather than several.
    """
    if markers.empty:
        return markers.copy()
    context_column = (
        "annotation_usable_for_context"
        if "annotation_usable_for_context" in markers.columns
        else "annotation_eligible"
    )
    retained = markers.loc[markers[context_column].astype(bool)].copy()
    if "direct_confidence_eligible" not in retained.columns:
        retained["direct_confidence_eligible"] = retained[
            "annotation_eligible"
        ].astype(bool)
    if "inference_confidence_eligible" not in retained.columns:
        retained["inference_confidence_eligible"] = False
    if "identity_value" not in retained.columns:
        retained["identity_value"] = retained["biological_gene_id"].astype(str)
    # A marker must carry at least one identity trustworthy enough to name a
    # domain from — either a direct resolved identity or one the feature
    # resolver has explicitly cleared for restricted inference (e.g. an
    # ortholog-mapped or context-inferred symbol). This eligibility gate
    # comes from feature resolution, not from domain-annotation confidence,
    # and is unrelated to (and unaffected by) the v3 single-call rewrite.
    retained = retained.loc[
        retained["direct_confidence_eligible"].astype(bool)
        | retained["inference_confidence_eligible"].astype(bool)
    ].copy()
    retained = retained.sort_values(
        ["score", "avg_log2FC", "adjusted_p_value", "gene"],
        ascending=[False, False, True, True],
        kind="mergesort",
    )
    retained["_evidence_dedup_key"] = np.where(
        retained["direct_confidence_eligible"].astype(bool),
        "direct:" + retained["biological_gene_id"].astype(str),
        "inference:" + retained["identity_value"].astype(str),
    )
    retained = retained.drop_duplicates("_evidence_dedup_key", keep="first")
    selected = retained.head(maximum)
    return selected.drop(columns=["_evidence_dedup_key"]).reset_index(drop=True)


def _add_feature_provenance(
    statistics: pd.DataFrame,
    preprocessing: PreprocessingResult,
    *,
    domain_specificity: np.ndarray,
) -> pd.DataFrame:
    metadata = preprocessing.feature_metadata.rename(
        columns={"analysis_gene": "feature_analysis_gene"}
    )
    merged = statistics.merge(
        metadata,
        left_on="gene",
        right_on="feature_analysis_gene",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if merged["feature_evidence_tier"].isna().any():
        missing = merged.loc[
            merged["feature_evidence_tier"].isna(), "gene"
        ].astype(str).head(5).tolist()
        raise FeatureAnnotationEvidenceError(
            "Marker records could not be linked to resolved input features: {}".format(
                ", ".join(missing)
            )
        )
    merged["analysis_gene"] = merged["gene"].astype(str)
    merged["domain_specificity"] = domain_specificity
    return merged.drop(columns=["feature_analysis_gene"])


def _prepare_domain_evidence(
    lstar_labels: np.ndarray,
    domain_id: str,
    domain_means: Mapping[str, np.ndarray],
    marker_statistics: Optional[pd.DataFrame],
    *,
    preprocessing: PreprocessingResult,
    dataset_context: str,
    biological_context: Mapping[str, Optional[str]],
    notes: Optional[str],
    sampling_level: str,
    settings: Mapping[str, Any],
    marker_test_method: str,
    input_context: Mapping[str, Any],
    n_observations_prefilter: int,
) -> Dict[str, Any]:
    """Build one domain's positive-marker evidence card.

    ``lstar_labels``, ``assignments``, and ``domain_means`` must all be
    QC-scoped (aligned to the same QC-passed observation order). There is no
    negative-marker or contradiction-evidence pathway anywhere in this
    function. ``marker_statistics`` is ``None`` when the domain was excluded
    from the shared scanpy rank call upstream (fewer than two QC-passed
    cells, or QC left no out-group) and this function takes a
    degenerate/abstention path instead.
    """
    domain_mask = lstar_labels == str(domain_id)
    in_group_count = int(np.sum(domain_mask))
    total_count = int(domain_mask.shape[0])
    is_degenerate = marker_statistics is None

    if is_degenerate:
        if in_group_count == 0:
            degenerate_reason = "No observations in this domain passed expression QC."
        elif in_group_count == total_count:
            degenerate_reason = (
                "The L-STAR consensus has one domain, so no domain-versus-rest "
                "comparison exists."
            )
        else:
            degenerate_reason = (
                "Only {} observation(s) in this domain passed expression QC; "
                "at least two are required for marker statistics.".format(
                    in_group_count
                )
            )
        marker_statistics = pd.DataFrame(
            {"gene": list(preprocessing.analysis_genes)}
        )
        for column in (
            "effect_size", "mean_in", "mean_out", "avg_log2FC", "pct_in",
            "pct_out", "delta_pct", "p_value", "adjusted_p_value", "score",
        ):
            marker_statistics[column] = 0.0
        marker_statistics["effect_definition"] = (
            "not evaluable without an out-group"
        )
    else:
        degenerate_reason = None

    other_means = [
        mean for key, mean in domain_means.items() if str(key) != str(domain_id)
    ]
    specificity = (
        domain_means[str(domain_id)] - np.max(np.vstack(other_means), axis=0)
        if other_means
        else np.zeros(len(preprocessing.analysis_genes), dtype=float)
    )
    marker_artifact = _add_feature_provenance(
        marker_statistics,
        preprocessing,
        domain_specificity=specificity,
    )

    if is_degenerate:
        positive_all = marker_artifact.iloc[0:0].copy()
    else:
        filtered = select_positive_markers(
            marker_artifact,
            min_pct_in=settings["min_pct_in"],
            max_pct_out=settings["max_pct_out"],
            max_positive_markers=None,
        )
        positive_all = filtered["positive_markers"].copy()

    positive_genes = set(positive_all["gene"].astype(str))
    marker_artifact["marker_direction"] = np.where(
        marker_artifact["gene"].astype(str).isin(positive_genes),
        "positive",
        "not_retained",
    )
    positive_all = marker_artifact.loc[
        marker_artifact["marker_direction"] == "positive"
    ].copy()
    positive_for_naming = _deduplicate_biological_markers(
        positive_all, maximum=settings["max_positive_markers"]
    )

    eligible_positive = positive_all.loc[
        positive_all["direct_confidence_eligible"].astype(bool)
    ]
    ambiguous_fraction = (
        float((positive_all["feature_evidence_tier"] == "B2").mean())
        if len(positive_all)
        else 0.0
    )
    distinct_biological_genes = int(
        eligible_positive["biological_gene_id"].nunique()
    )
    selected_markers = positive_for_naming
    selected_direct = selected_markers.loc[
        selected_markers["direct_confidence_eligible"].astype(bool)
    ]
    selected_inference = selected_markers.loc[
        ~selected_markers["direct_confidence_eligible"].astype(bool)
        & selected_markers["inference_confidence_eligible"].astype(bool)
    ]
    direct_positive_selected = positive_for_naming.loc[
        positive_for_naming["direct_confidence_eligible"].astype(bool)
    ]
    inference_positive_selected = positive_for_naming.loc[
        ~positive_for_naming["direct_confidence_eligible"].astype(bool)
        & positive_for_naming["inference_confidence_eligible"].astype(bool)
    ]
    evidence_mode = (
        "direct_and_inference"
        if len(selected_direct) and len(selected_inference)
        else "direct"
        if len(selected_direct)
        else "inference_only"
        if len(selected_inference)
        else "empty"
    )
    abstention_reasons = []
    evidence_warnings = []
    gates = preprocessing.quality_gates
    if is_degenerate:
        abstention_reasons.append(degenerate_reason)
    if not gates["species_resolved"]:
        evidence_warnings.append("Species is unresolved.")
    if not gates["namespace_resolved"]:
        evidence_warnings.append("Gene identifier namespace is unresolved.")
    if not gates["expression_scale_sufficient"]:
        abstention_reasons.append("Expression scale quality is insufficient.")
    if ambiguous_fraction >= 0.5:
        evidence_warnings.append("Ambiguous source labels dominate positive evidence.")
    if distinct_biological_genes < MIN_INTERPRETABLE_POSITIVE_GENES:
        evidence_warnings.append(
            "Fewer than two distinct direct biological positive genes remain."
        )
    if len(selected_markers) == 0:
        abstention_reasons.append("No marker evidence can be serialized safely.")
    elif evidence_mode == "inference_only":
        evidence_warnings.append(
            "No direct biological labels remain; restricted context inference is active."
        )

    evidence_card = {
        "domain_id": str(domain_id),
        "dataset_context": dataset_context,
        "biological_context": dict(biological_context),
        "notes": notes,
        "sampling_level": sampling_level,
        "evidence_mode": evidence_mode,
        "n_observations": in_group_count,
        "positive_markers": _compact_marker_records(positive_for_naming),
        "input_context": dict(input_context),
        "evidence_warnings": evidence_warnings,
        "marker_filter_settings": dict(settings),
        "abstention_gate": {
            "required": bool(abstention_reasons),
            "reasons": abstention_reasons,
        },
    }
    return {
        "domain_id": str(domain_id),
        "n_cells": in_group_count,
        "n_cells_prefilter": int(n_observations_prefilter),
        "evidence_card": evidence_card,
        "marker_artifact": marker_artifact,
        "marker_counts": {
            "positive_passing_filters": int(len(positive_all)),
            # Kept as a distinct key because the preflight contract reads it as
            # R_domain. Since the hand-curated technical-gene exclusion was
            # removed, it now equals positive_passing_filters.
            "biologically_filtered_marker_records": int(len(positive_all)),
            "positive_sent_to_llm": int(len(positive_for_naming)),
            "direct_positive_sent_to_llm": int(len(direct_positive_selected)),
            "inference_positive_sent_to_llm": int(
                len(inference_positive_selected)
            ),
            "feature_evidence_tier_counts_sent_to_llm": {
                str(key): int(value)
                for key, value in selected_markers[
                    "feature_evidence_tier"
                ].value_counts().items()
            },
            "distinct_interpretable_positive_genes": distinct_biological_genes,
        },
    }


def _annotation_evidence_preflight(
    feature_resolution: Any,
    prepared_domains: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Summarize collection and marker evidence before annotation API calls."""
    mapping = feature_resolution.mapping_table
    tier_counts = {
        tier: int((mapping["feature_evidence_tier"] == tier).sum())
        for tier in ("A", "B1", "B2", "C")
    }
    domain_records = []
    for prepared in prepared_domains:
        counts = prepared["marker_counts"]
        direct_count = int(counts["direct_positive_sent_to_llm"])
        inference_count = int(counts["inference_positive_sent_to_llm"])
        domain_records.append(
            {
                "domain_id": str(prepared["domain_id"]),
                "R_domain": int(counts["biologically_filtered_marker_records"]),
                "D_domain": direct_count,
                "I_domain": inference_count,
                "P_domain": direct_count + inference_count,
                "evidence_mode": prepared["evidence_card"]["evidence_mode"],
                "restricted_inference": bool(
                    direct_count == 0 and inference_count > 0
                ),
            }
        )
    raw_total = sum(record["R_domain"] for record in domain_records)
    direct_total = sum(record["D_domain"] for record in domain_records)
    inference_total = sum(record["I_domain"] for record in domain_records)
    payload_total = direct_total + inference_total
    return {
        "collection_level": {
            "total_features": int(len(mapping)),
            "exact_identities": int(
                (mapping["identity_status"] == "resolved_unique").sum()
            ),
            "feature_evidence_tier_counts": tier_counts,
            "context_usable_features": int(
                mapping["annotation_usable_for_context"].astype(bool).sum()
            ),
            "direct_confidence_eligible_features": int(
                mapping["direct_confidence_eligible"].astype(bool).sum()
            ),
            "inference_confidence_eligible_features": int(
                mapping["inference_confidence_eligible"].astype(bool).sum()
            ),
            "source_label_fallback_active": bool(
                mapping["fallback_applied"].astype(bool).any()
            ),
            "execution_replan_used": bool(
                feature_resolution.qc_report.get("execution_replan_used", False)
            ),
        },
        "marker_level": {
            "R_total": int(raw_total),
            "D_total": int(direct_total),
            "I_total": int(inference_total),
            "P_total": int(payload_total),
            "restricted_inference_active": bool(
                direct_total == 0 and inference_total > 0
            ),
            "domains": domain_records,
        },
        "rules": {
            "direct_tiers": ["A", "B1"],
            "inference_only_tiers": ["B2", "C"],
            "empty_payload_is_technical_failure_only_when_raw_markers_exist": True,
            "marker_direction_policy": "positive_only",
        },
    }


def _validate_annotation_evidence_preflight(
    preflight: Mapping[str, Any],
) -> None:
    marker_level = preflight["marker_level"]
    raw_total = int(marker_level["R_total"])
    payload_total = int(marker_level["P_total"])
    if raw_total > 0 and payload_total == 0:
        raise FeatureAnnotationEvidenceError(
            "Marker construction found biological evidence, but no marker record "
            "could be linked and serialized for annotation"
        )


def _validate_output_target(output_path: Path, inputs: AnnotationInputs) -> None:
    if output_path.resolve() in {
        inputs.expression_data_path.resolve(),
        inputs.assignments_csv.resolve(),
        inputs.lstar_consensus_csv.resolve(),
    }:
        raise ValueError("output_csv must not overwrite an annotation input file")


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_artifact_targets(
    output_path: Path,
    artifact_root: Path,
    inputs: AnnotationInputs,
) -> None:
    resolved_output = output_path.resolve()
    resolved_root = artifact_root.resolve()
    if artifact_root.exists() and not artifact_root.is_dir():
        raise ValueError("artifacts_dir exists but is not a directory")
    if _is_within(resolved_output, resolved_root) or _is_within(
        resolved_root, resolved_output
    ):
        raise ValueError("output_csv and artifacts_dir must be separate paths")
    input_paths = {
        inputs.expression_data_path.resolve(),
        inputs.assignments_csv.resolve(),
        inputs.lstar_consensus_csv.resolve(),
    }
    if any(_is_within(path, resolved_root) for path in input_paths):
        raise ValueError("Annotation inputs must not be stored inside artifacts_dir")
    managed_directories = (
        "domain_markers", "evidence_cards",
        "preprocessing", "feature_resolution", "visual_context",
    )
    for name in managed_directories:
        candidate = artifact_root / name
        if candidate.is_symlink():
            raise ValueError("Managed artifact paths must not be symbolic links")
        if candidate.exists() and not candidate.is_dir():
            raise ValueError("Managed artifact path is not a directory: {}".format(candidate))


def annotate_domain(
    expression_h5ad: Optional[PathLike] = None,
    *,
    expression_h5: Optional[PathLike] = None,
    expression_source: Optional[str] = None,
    dataset_context: Optional[str] = None,
    species: Optional[str] = None,
    tissue: Optional[str] = None,
    notes: Optional[str] = None,
    sampling_level: Optional[str] = None,
    expression_scale: Optional[str] = None,
    max_positive_markers: Optional[int] = 25,
    marker_test_method: Optional[str] = None,
    resolution_resource_paths: Sequence[PathLike] = (),
    auto_discover_resolution_resources: bool = True,
    output_dir: PathLike = "lstar_output",
    assignments_csv: Optional[PathLike] = None,
    lstar_consensus_csv: Optional[PathLike] = None,
    run_manifest: Optional[PathLike] = None,
    id_col: Optional[str] = None,
    allow_partial_observations: bool = False,
    observation_id_transform: Optional[Mapping[str, str]] = None,
    allowed_context_fields: Sequence[str] = (),
    evaluation_only_fields: Sequence[str] = (),
    forbidden_annotation_fields: Sequence[str] = (),
    model_name: Optional[str] = None,
    reasoning_effort: str = DEFAULT_ANNOTATION_REASONING_EFFORT,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    he_image_path: Optional[PathLike] = None,
    include_visual_background: bool = True,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Annotate each L-STAR domain by naming it from its ranked marker genes.

    Domain naming is one call: every evaluable domain and its ranked
    positive markers are sent to the model together, and it returns a name
    per domain, with no separate candidate-generation or reconciliation
    stage and no annotation-confidence output. This follows the single-call,
    marker-list-to-name pattern validated for single-cell cluster annotation
    by GPTCelltype (Hou & Ji, *Nature Methods*, 2024); see
    ``skills/domain_annotation/annotation.prompt.md`` for the full naming
    prompt and its granularity and evidence rules. A domain with no usable
    positive-marker evidence, or that fails a deterministic upstream evidence
    gate, is named ``"Unknown"`` without spending a call on it; if the model's
    response never passes validation after repair attempts, the affected
    domains are named ``"Unknown"`` deterministically so the run always
    completes. The output CSV has exactly three columns: the observation ID,
    ``L-STAR``, and ``domain_name``.

    H5AD or a 10x HDF5 matrix plus an L-STAR consensus assignment is the
    expression-input contract. CSV expression matrices are not accepted.
    Automatically discovered and explicitly supplied feature-resolution files
    are likewise restricted to H5AD or H5/HDF5 containers. When spatial
    coordinates are available, the optional visual background contains only a
    point-and-domain-color map, plus H&E when available.
    ``dataset_context`` and the user-declared ``sampling_level`` (``spot`` or
    ``cell``) are mandatory. Sampling level is the only platform sampling input
    and controls how mixture and label scope are interpreted by the
    annotation LLM. Taxon ID, identifier namespace, and assay type are resolved
    internally rather than supplied as public arguments. Species and optional
    tissue context are supplied as biological priors for annotation.
    ``notes`` is a separate, optional free-text field for structural context the
    biological priors do not capture, for example that the tissue is known to
    have a layered organization; the annotation LLM uses it only to choose
    naming conventions and to break ties among names the markers already
    support, never to override marker evidence. An LLM plans one
    collection-level feature-resolution recipe, and deterministic primitives
    execute it without platform adapters.
    ``marker_test_method`` selects the differential-expression test passed to
    ``scanpy.tl.rank_genes_groups``. It defaults to ``None``, which means
    ``method`` is not passed to ``rank_genes_groups`` at all, so scanpy's own
    default applies: scanpy resolves an unset ``method`` to ``"t-test"``.
    Pass an explicit value (``"wilcoxon"``, for instance) to depart from that
    default. The chosen test, and whether it was scanpy's default, are
    recorded in the run manifest.
    ``show_progress=True`` prints stage and per-domain status without prompt,
    expression, model-response, or credential contents.
    The final table preserves original expression-observation order for the
    validated consensus-aligned population.
    """
    settings = _marker_filter_settings(max_positive_markers)
    if not isinstance(dataset_context, str) or not dataset_context.strip():
        raise ValueError("dataset_context is required and must be non-empty")
    if species is not None and (
        not isinstance(species, str) or not species.strip()
    ):
        raise ValueError("species must be a non-empty string when provided")
    if tissue is not None and (
        not isinstance(tissue, str) or not tissue.strip()
    ):
        raise ValueError("tissue must be a non-empty string when provided")
    biological_context = {
        "species": species.strip() if isinstance(species, str) else None,
        "tissue": tissue.strip() if isinstance(tissue, str) else None,
    }
    if (
        marker_test_method is not None
        and marker_test_method not in SUPPORTED_MARKER_TEST_METHODS
    ):
        raise ValueError(
            "marker_test_method must be None or one of: {}".format(
                ", ".join(sorted(SUPPORTED_MARKER_TEST_METHODS))
            )
        )
    if notes is not None and (not isinstance(notes, str) or not notes.strip()):
        raise ValueError("notes must be a non-empty string when provided")
    notes = notes.strip() if isinstance(notes, str) else None
    if not isinstance(sampling_level, str) or not sampling_level.strip():
        raise ValueError("sampling_level is required and must be non-empty")
    sampling_level = resolve_sampling_level(sampling_level)
    if not isinstance(include_visual_background, (bool, np.bool_)):
        raise ValueError("include_visual_background must be Boolean")
    if not isinstance(show_progress, (bool, np.bool_)):
        raise ValueError("show_progress must be Boolean")
    if reasoning_effort not in SUPPORTED_REASONING_EFFORTS:
        raise ValueError(
            "reasoning_effort must be one of: {}".format(
                ", ".join(sorted(SUPPORTED_REASONING_EFFORTS))
            )
        )

    def progress(message: str = "") -> None:
        if show_progress:
            logger.info(message)

    def progress_detail(label: str, value: Any, *, level: int = 1) -> None:
        progress("{}- {}: {}".format("  " * level, label, value))

    def progress_section(title: str) -> None:
        progress("\n" + "=" * 60)
        progress(title)
        progress("=" * 60)

    progress_section("STARTING L-STAR DOMAIN ANNOTATION")
    progress("Dataset context: {}".format(dataset_context.strip()))
    if tissue is not None:
        progress("Tissue: {}".format(tissue.strip()))
    if notes is not None:
        progress("Notes: {}".format(notes))
    progress(
        "Sampling level: {}".format(sampling_level)
    )
    progress_section("STEP 1: Load and Align Inputs")

    output_root = _path(output_dir, "output_dir")
    inputs = load_annotation_inputs(
        output_dir=output_root,
        expression_h5ad=expression_h5ad,
        expression_h5=expression_h5,
        assignments_csv=assignments_csv,
        lstar_consensus_csv=lstar_consensus_csv,
        run_manifest=run_manifest,
        id_col=id_col,
        consensus_id_col=None,
        consensus_domain_col=None,
        method_columns=None,
        expression_source=expression_source,
        expression_scale=expression_scale,
        allow_partial_observations=allow_partial_observations,
        observation_id_transform=observation_id_transform,
        allowed_context_fields=allowed_context_fields,
        evaluation_only_fields=evaluation_only_fields,
        forbidden_annotation_fields=forbidden_annotation_fields,
    )
    output_path = output_root / PRIMARY_OUTPUT_FILENAME
    artifact_root = output_root / DEFAULT_ARTIFACTS_DIRNAME
    _validate_output_target(output_path, inputs)
    _validate_artifact_targets(output_path, artifact_root, inputs)

    input_domain_ids = _first_seen(
        inputs.consensus["L-STAR"].astype(str).tolist()
    )
    progress_detail("Expression input", inputs.expression_data_path)
    progress_detail("Expression format", inputs.input_format)
    progress_detail("Aligned observations", len(inputs.observation_ids))
    progress_detail("Input features", len(inputs.gene_columns))
    progress_detail("L-STAR domains", len(input_domain_ids))
    progress_detail(
        "Consensus methods",
        ", ".join(inputs.method_columns) if inputs.method_columns else "none",
    )

    active_model = model_name or DEFAULT_ANNOTATION_MODEL
    active_client = create_openai_client(api_key, api_base)
    explicit_resource_paths = (
        [resolution_resource_paths]
        if isinstance(resolution_resource_paths, (str, Path))
        else list(resolution_resource_paths)
    )
    unsupported_resource_paths = [
        str(Path(path).expanduser())
        for path in explicit_resource_paths
        if Path(path).suffix.lower() not in ALLOWED_RESOLUTION_RESOURCE_SUFFIXES
    ]
    if unsupported_resource_paths:
        raise ValueError(
            "resolution_resource_paths only supports H5AD or H5/HDF5 files; "
            "unsupported paths: {}".format(", ".join(unsupported_resource_paths))
        )
    if auto_discover_resolution_resources:
        discovered_resources = tuple(
            path
            for path in discover_supporting_files(
                inputs.expression_data_path.parent,
                excluded_paths=[
                    inputs.expression_data_path,
                    inputs.assignments_csv,
                    inputs.lstar_consensus_csv,
                    output_path,
                ],
            )
            if path.suffix.lower() in ALLOWED_RESOLUTION_RESOURCE_SUFFIXES
        )
    else:
        discovered_resources = tuple()
    implicit_feature_resources = [inputs.expression_data_path]
    combined_resource_paths = list(
        dict.fromkeys(
            str(Path(path).expanduser().resolve())
            for path in (
                implicit_feature_resources
                + explicit_resource_paths
                + list(discovered_resources)
            )
        )
    )
    progress_section("STEP 2: Resolve Features and Preprocess Expression")
    progress_detail("Feature-resolution model", active_model)
    progress_detail("Reasoning effort", reasoning_effort)
    progress_detail("H5/H5AD feature resources", len(combined_resource_paths))
    progress("Planning one dataset-level feature-resolution workflow...")
    feature_resolution = resolve_features(
        inputs.gene_columns,
        species=species.strip() if isinstance(species, str) else None,
        source_taxon_id=None,
        source_assembly=None,
        identifier_namespace=None,
        feature_level=None,
        resource_paths=combined_resource_paths,
        recipe_json=None,
        model_name=active_model,
        reasoning_effort=reasoning_effort,
        client=active_client,
    )
    eligible_features = int(
        feature_resolution.mapping_table["annotation_eligible"].astype(bool).sum()
    )
    collection_tiers = {
        tier: int(
            (feature_resolution.mapping_table["feature_evidence_tier"] == tier).sum()
        )
        for tier in ("A", "B1", "B2", "C")
    }
    progress("Feature resolution complete.")
    progress_detail(
        "Annotation-eligible features",
        "{}/{}".format(eligible_features, len(feature_resolution.mapping_table)),
    )
    progress_detail("Resolution recipe", feature_resolution.recipe["recipe_id"])
    progress_detail(
        "Exact feature identities",
        int(
            (
                feature_resolution.mapping_table["identity_status"]
                == "resolved_unique"
            ).sum()
        ),
    )
    progress_detail(
        "Feature evidence tiers",
        "A (canonical)={A}; B1 (anchored source)={B1}; "
        "B2 (ambiguous source)={B2}; C (opaque)={C}".format(**collection_tiers),
    )
    progress_detail(
        "Source-label fallback",
        (
            "active"
            if bool(
                feature_resolution.mapping_table["fallback_applied"].astype(bool).any()
            )
            else "inactive"
        ),
    )
    if feature_resolution.planner_rule_adjustments:
        adjustment_summary = "; ".join(
            "{}: {:.12g} -> {:.12g}".format(
                adjustment["rule"],
                float(adjustment["requested_value"]),
                float(adjustment["effective_value"]),
            )
            for adjustment in feature_resolution.planner_rule_adjustments
        )
        progress_detail("Validation rule calibration", adjustment_summary)
    progress("Applying quality control to expression observations...")
    qc_result = apply_quality_control(
        inputs.expression_matrix,
        selected_scale=str(
            inputs.expression_source_manifest["selected_expression_scale"]
        ),
        feature_mito_labels=_mito_labels_for_qc(feature_resolution),
        precomputed_obs_qc=inputs.qc_metadata.get("precomputed_obs_qc"),
    )
    qc_cells = qc_result.audit["cells"]
    progress("Quality control complete.")
    progress_detail(
        "Observations retained for marker computation",
        "{}/{}".format(qc_cells["retained"], qc_cells["total"]),
    )
    progress_detail("QC mode", qc_result.audit["mode"])
    progress_detail(
        "Mitochondrial gate",
        "applied" if qc_result.audit["mito"]["gate_applied"] else "skipped ({})".format(
            qc_result.audit["mito"]["reason"]
        ),
    )
    progress("Normalizing expression and preparing the marker matrix...")
    preprocessing = preprocess_expression(
        inputs.expression_matrix,
        id_col=inputs.id_col,
        feature_ids=inputs.feature_ids,
        observation_unit=sampling_level,
        feature_resolution=feature_resolution,
        species=species.strip() if isinstance(species, str) else None,
        expression_scale=str(
            inputs.expression_source_manifest["selected_expression_scale"]
        ),
        assay_type="auto",
        input_evidence_quality={"h5ad": 1.0, "10x_h5": 0.85}[
            inputs.input_format
        ],
        qc_cell_mask=qc_result.cell_mask,
        qc_audit=qc_result.audit,
        min_cells_per_gene=3,
    )
    progress("Expression preprocessing complete.")
    progress_detail("Analysis features", len(preprocessing.analysis_genes))
    progress_detail(
        "Marker matrix scale", preprocessing.normalization["marker_matrix_scale"]
    )
    inconsistent_symbol_groups = preprocessing.normalization.get(
        "duplicate_policy", {}
    ).get("inconsistent_symbol_warnings", [])
    if inconsistent_symbol_groups:
        progress_detail(
            "Merged groups with inconsistent symbols",
            "{}; review preprocessing/normalization.json: {}".format(
                len(inconsistent_symbol_groups),
                "; ".join(
                    "/".join(group["annotation_symbols"])
                    for group in inconsistent_symbol_groups[:5]
                ),
            ),
        )
    # `lstar_labels` and `inputs` stay full-population: they cover every
    # aligned observation and feed the final domain-assignment output,
    # and spatial/visual context regardless of QC
    # outcome. `qc_labels`/`qc_assignments` are the QC-passed subset used by
    # the marker-evidence pipeline only.
    lstar_labels = inputs.consensus["L-STAR"].astype(str).to_numpy()
    domain_ids = _first_seen(lstar_labels.tolist())
    qc_labels = lstar_labels[qc_result.cell_mask]
    qc_assignments = inputs.assignments.loc[qc_result.cell_mask].reset_index(
        drop=True
    )
    domain_means = {}
    for domain_id in domain_ids:
        selected = preprocessing.marker_values[qc_labels == domain_id]
        if selected.shape[0] == 0:
            domain_means[domain_id] = np.zeros(
                len(preprocessing.analysis_genes), dtype=float
            )
            continue
        mean = selected.mean(axis=0)
        domain_means[domain_id] = np.asarray(mean, dtype=float).ravel()
    progress_section("STEP 3: Prepare Spatial and Visual Background")
    if include_visual_background:
        progress("Resolving aligned spatial coordinates and available H&E...")
    visual_context = (
        build_visual_context(
            expression_path=inputs.expression_data_path,
            observation_ids=inputs.observation_ids,
            domain_labels=lstar_labels,
            existing_coordinates=inputs.spatial_coordinates,
            existing_spatial_key=inputs.spatial_key,
            output_dir=artifact_root / "visual_context",
            he_image_path=he_image_path,
            use_palo=INTERNAL_USE_PALO,
            max_image_dimension=INTERNAL_MAX_BACKGROUND_IMAGE_DIMENSION,
        )
        if include_visual_background
        else None
    )
    if visual_context is None:
        progress("Visual background disabled.")
    else:
        visual_audit = visual_context.audit
        progress_detail(
            "Domain visualization",
            "available"
            if visual_audit.get("domain_visualization_available", False)
            else "unavailable",
        )
        if visual_audit.get("domain_visualization_available", False):
            progress_detail(
                "Visualization renderer",
                visual_audit.get("domain_visualization_renderer", "unknown"),
            )
        progress_detail(
            "H&E background",
            "available"
            if visual_audit.get("histology_available", False)
            else "unavailable",
        )
        progress_detail(
            "Images prepared for annotation requests",
            len(visual_context.prompt_images),
        )
    if (
        visual_context is not None
        and inputs.spatial_coordinates is None
        and visual_context.spatial_coordinates is not None
    ):
        inputs = replace(
            inputs,
            spatial_coordinates=visual_context.spatial_coordinates,
            spatial_key=visual_context.spatial_key,
        )
    input_contexts, domain_relationships = _build_input_contexts(
        inputs, lstar_labels, domain_ids
    )
    domain_relationships = _add_molecular_relationships(
        input_contexts, domain_relationships, domain_means, domain_ids
    )
    progress_section("STEP 4: Build Domain Marker Evidence")
    progress("Computing domain-versus-rest positive markers...")
    def prepare_all_domains(
        active_preprocessing: PreprocessingResult,
        active_domain_means: Mapping[str, np.ndarray],
        active_qc_labels: np.ndarray,
        active_qc_assignments: pd.DataFrame,
        *,
        announce: bool,
    ) -> List[Dict[str, Any]]:
        # A domain qualifies for the shared scanpy rank call only when it has
        # at least two QC-passed cells and leaves at least one out-group
        # cell; everything else (empty, single-cell, or all-cells domains)
        # takes the degenerate/abstention path inside _prepare_domain_evidence.
        valid_groups = [
            domain_id
            for domain_id in domain_ids
            if 2 <= int(np.sum(active_qc_labels == domain_id)) < len(active_qc_labels)
        ]
        marker_frames = (
            compute_marker_statistics(
                active_preprocessing.marker_values,
                active_preprocessing.analysis_genes,
                active_qc_labels,
                groups=valid_groups,
                method=marker_test_method,
            )
            if valid_groups
            else {}
        )
        prepared_records = []
        for index, domain_id in enumerate(domain_ids, start=1):
            if announce:
                progress(
                    "[{}/{}] Domain {}: Building marker evidence...".format(
                        index, len(domain_ids), domain_id
                    )
                )
            prepared = _prepare_domain_evidence(
                active_qc_labels,
                domain_id,
                active_domain_means,
                marker_frames.get(domain_id),
                preprocessing=active_preprocessing,
                dataset_context=dataset_context.strip(),
                biological_context=biological_context,
                notes=notes,
                sampling_level=sampling_level,
                settings=settings,
                marker_test_method=marker_test_method,
                input_context=input_contexts[domain_id],
                n_observations_prefilter=int(np.sum(lstar_labels == domain_id)),
            )
            prepared_records.append(prepared)
            if announce:
                marker_counts = prepared["marker_counts"]
                tier_counts = marker_counts[
                    "feature_evidence_tier_counts_sent_to_llm"
                ]
                progress_detail(
                    "Observations",
                    "{} QC-passed / {} pre-filter".format(
                        prepared["n_cells"], prepared["n_cells_prefilter"]
                    ),
                )
                progress_detail(
                    "Positive markers",
                    "{} passed filters; {} sent to LLM".format(
                        marker_counts["positive_passing_filters"],
                        marker_counts["positive_sent_to_llm"],
                    ),
                )
                progress_detail(
                    "Evidence tiers sent to LLM",
                    "A={}; B1={}; B2={}; C={}".format(
                        int(tier_counts.get("A", 0)),
                        int(tier_counts.get("B1", 0)),
                        int(tier_counts.get("B2", 0)),
                        int(tier_counts.get("C", 0)),
                    )
                )
        return prepared_records

    prepared_domains = prepare_all_domains(
        preprocessing, domain_means, qc_labels, qc_assignments, announce=True
    )
    preflight = _annotation_evidence_preflight(
        feature_resolution, prepared_domains
    )
    marker_preflight = preflight["marker_level"]
    progress("Marker evidence payload validation complete.")
    progress_detail("Raw markers", marker_preflight["R_total"])
    progress_detail("Direct evidence", marker_preflight["D_total"])
    progress_detail("Inference context", marker_preflight["I_total"])
    progress_detail("Sendable marker evidence", marker_preflight["P_total"])

    marker_replan_audit: Dict[str, Any] = {
        "attempted": False,
        "accepted": False,
        "reason": None,
    }
    if (
        int(marker_preflight["R_total"]) > 0
        and int(marker_preflight["D_total"]) == 0
        and not bool(
            preflight["collection_level"].get("execution_replan_used", False)
        )
    ):
        marker_replan_audit["attempted"] = True
        progress(
            "No direct marker labels remain; running the one marker-guided "
            "feature-resolution replan..."
        )
        marker_feature_labels = []
        for prepared in prepared_domains:
            marker_rows = prepared["marker_artifact"]
            retained = marker_rows.loc[
                marker_rows["marker_direction"] == "positive"
            ]
            marker_feature_labels.extend(
                retained["source_feature_label"].astype(str).tolist()
            )
        try:
            revised_resolution = replan_feature_resolution(
                inputs.gene_columns,
                previous_result=feature_resolution,
                species=species.strip() if isinstance(species, str) else None,
                source_taxon_id=None,
                source_assembly=None,
                identifier_namespace=None,
                feature_level=None,
                resource_paths=combined_resource_paths,
                marker_feature_labels=marker_feature_labels,
                model_name=active_model,
                reasoning_effort=reasoning_effort,
                client=active_client,
            )
            # Mitochondrial-gene identification can change with the revised
            # resolution, so QC is recomputed against it rather than reusing
            # the original qc_result (a stale mito gate would otherwise
            # silently persist through the replan).
            revised_qc_result = apply_quality_control(
                inputs.expression_matrix,
                selected_scale=str(
                    inputs.expression_source_manifest["selected_expression_scale"]
                ),
                feature_mito_labels=_mito_labels_for_qc(revised_resolution),
                precomputed_obs_qc=inputs.qc_metadata.get("precomputed_obs_qc"),
            )
            revised_preprocessing = preprocess_expression(
                inputs.expression_matrix,
                id_col=inputs.id_col,
                feature_ids=inputs.feature_ids,
                observation_unit=sampling_level,
                feature_resolution=revised_resolution,
                species=species.strip() if isinstance(species, str) else None,
                expression_scale=str(
                    inputs.expression_source_manifest["selected_expression_scale"]
                ),
                assay_type="auto",
                input_evidence_quality={"h5ad": 1.0, "10x_h5": 0.85}[
                    inputs.input_format
                ],
                qc_cell_mask=revised_qc_result.cell_mask,
                qc_audit=revised_qc_result.audit,
                min_cells_per_gene=3,
            )
            revised_qc_labels = lstar_labels[revised_qc_result.cell_mask]
            revised_qc_assignments = inputs.assignments.loc[
                revised_qc_result.cell_mask
            ].reset_index(drop=True)
            revised_domain_means = {}
            for domain_id in domain_ids:
                selected = revised_preprocessing.marker_values[
                    revised_qc_labels == domain_id
                ]
                if selected.shape[0] == 0:
                    revised_domain_means[domain_id] = np.zeros(
                        len(revised_preprocessing.analysis_genes), dtype=float
                    )
                    continue
                revised_domain_means[domain_id] = np.asarray(
                    selected.mean(axis=0), dtype=float
                ).ravel()
            revised_prepared = prepare_all_domains(
                revised_preprocessing,
                revised_domain_means,
                revised_qc_labels,
                revised_qc_assignments,
                announce=False,
            )
            revised_preflight = _annotation_evidence_preflight(
                revised_resolution, revised_prepared
            )
            if int(revised_preflight["marker_level"]["P_total"]) > 0:
                feature_resolution = revised_resolution
                preprocessing = revised_preprocessing
                qc_result = revised_qc_result
                qc_labels = revised_qc_labels
                qc_assignments = revised_qc_assignments
                domain_means = revised_domain_means
                prepared_domains = revised_prepared
                preflight = revised_preflight
                marker_preflight = preflight["marker_level"]
                input_contexts, domain_relationships = _build_input_contexts(
                    inputs, lstar_labels, domain_ids
                )
                domain_relationships = _add_molecular_relationships(
                    input_contexts,
                    domain_relationships,
                    domain_means,
                    domain_ids,
                )
                marker_replan_audit["accepted"] = True
                marker_replan_audit["reason"] = (
                    "revised payload retained serializable evidence"
                )
                progress("Rebuilt domain marker evidence after the accepted replan:")
                for index, prepared in enumerate(prepared_domains, start=1):
                    counts = prepared["marker_counts"]
                    tier_counts = counts[
                        "feature_evidence_tier_counts_sent_to_llm"
                    ]
                    progress(
                        "[{}/{}] Domain {}: Marker evidence rebuilt".format(
                            index,
                            len(prepared_domains),
                            prepared["domain_id"],
                        )
                    )
                    progress_detail(
                        "Positive markers",
                        "{} passed filters; {} sent to LLM".format(
                            counts["positive_passing_filters"],
                            counts["positive_sent_to_llm"],
                        ),
                    )
                    progress_detail(
                        "Evidence tiers sent to LLM",
                        "A={}; B1={}; B2={}; C={}".format(
                            int(tier_counts.get("A", 0)),
                            int(tier_counts.get("B1", 0)),
                            int(tier_counts.get("B2", 0)),
                            int(tier_counts.get("C", 0)),
                        ),
                    )
            else:
                marker_replan_audit["reason"] = (
                    "revised plan removed all serializable marker evidence"
                )
        except (FeatureAnnotationEvidenceError, RuntimeError, ValueError) as error:
            marker_replan_audit["reason"] = "{}: {}".format(
                type(error).__name__, str(error)
            )
        preflight["marker_guided_replan"] = marker_replan_audit
        progress_detail(
            "Marker-guided replan",
            "accepted" if marker_replan_audit["accepted"] else "not accepted",
        )
        progress_detail("Direct evidence", marker_preflight["D_total"])
        progress_detail("Inference context", marker_preflight["I_total"])
        progress_detail("Sendable marker evidence", marker_preflight["P_total"])
    else:
        preflight["marker_guided_replan"] = marker_replan_audit

    artifact_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        artifact_root / "annotation_evidence_preflight.json", preflight
    )
    _validate_annotation_evidence_preflight(preflight)
    if bool(marker_preflight["restricted_inference_active"]):
        progress_detail(
            "Restricted context-inference path",
            "active; these domains have no direct biological markers, only "
            "context-inferred evidence",
        )
    restricted_domain_ids = [
        record["domain_id"]
        for record in marker_preflight["domains"]
        if record["restricted_inference"]
    ]
    if restricted_domain_ids:
        progress_detail(
            "Restricted context-inference domains",
            "{}".format(", ".join(restricted_domain_ids)),
        )

    progress_section("STEP 5: Name Domains")
    decision = run_global_annotation(
        [prepared["evidence_card"] for prepared in prepared_domains],
        dataset_context=dataset_context.strip(),
        api_key=api_key,
        api_base=api_base,
        model_name=active_model,
        reasoning_effort=reasoning_effort,
        client=active_client,
        background_images=(
            visual_context.prompt_images if visual_context is not None else ()
        ),
        progress_callback=progress,
    )
    if visual_context is not None:
        annotation_request_count = len(decision.raw_responses)
        updated_visual_audit = dict(visual_context.audit)
        updated_visual_audit["annotation_requests_with_images"] = int(
            annotation_request_count if visual_context.prompt_images else 0
        )
        updated_visual_audit["images_transmitted"] = (
            [item["label"] for item in visual_context.prompt_images]
            if annotation_request_count
            else []
        )
        visual_context = replace(visual_context, audit=updated_visual_audit)
    domain_map = pd.DataFrame(list(decision.domains)).loc[
        :, ["domain_id", "domain_name"]
    ].rename(columns={"domain_id": "L-STAR"})
    domain_map["L-STAR"] = domain_map["L-STAR"].astype(str)
    if domain_map["L-STAR"].tolist() != domain_ids:
        raise RuntimeError("Global annotation changed domain order or scope")

    final_assignment = inputs.consensus.loc[:, [inputs.id_col, "L-STAR"]].copy()
    final_assignment["L-STAR"] = final_assignment["L-STAR"].astype(str)
    domain_name_lookup = domain_map.set_index("L-STAR")["domain_name"].to_dict()
    final_assignment["domain_name"] = final_assignment["L-STAR"].map(domain_name_lookup)
    validate_final_assignment(
        final_assignment,
        id_col=inputs.id_col,
        expected_ids=inputs.assignments[inputs.id_col].astype(str).tolist(),
        expected_lstar_labels=lstar_labels.tolist(),
    )
    progress_section("STEP 6: Write Annotation Results and Audit Artifacts")
    progress(
        "Writing the {}-level annotation table and audit bundle...".format(
            sampling_level
        )
    )

    timestamp = datetime.now(timezone.utc).isoformat()
    subdirectories = {
        name: artifact_root / name
        for name in (
            "domain_markers", "evidence_cards",
            "preprocessing",
            "feature_resolution",
            "visual_context",
        )
    }
    managed_artifact_files = [
        "annotation_manifest.json", "annotation_audit.json",
        "h5ad_inspection.json", "observation_alignment.csv",
        "expression_source_manifest.json", "feature_resolution_recipe.json",
        "feature_resolution.csv", "feature_resolution_summary.json",
        "annotation_label_resolution.csv", "annotation_evidence_preflight.json",
        "domain_evidence.csv",
        "domain_relationships.csv", "annotation_prompt_payload.json",
        "annotation_response.json",
        "annotation_run_manifest.json",
        "domain_annotation_map.csv", "preprocessing/feature_mapping.csv",
        "preprocessing/analysis_feature_mapping.csv",
        "preprocessing/expression_profile.json", "preprocessing/normalization.json",
        "preprocessing/qc_filtering.json",
        "feature_resolution/feature_profile.json",
        "feature_resolution/resource_manifest.json",
        "feature_resolution/pilot_results.json",
        "feature_resolution/planner_response.json",
        "feature_resolution/resolution_recipe.json",
        "feature_resolution/feature_mapping.csv",
        "feature_resolution/quality_control.json",
        "feature_resolution/llm_responses.json",
        "visual_context/visual_context.json",
    ]
    if visual_context is not None and visual_context.audit.get(
        "domain_visualization_available", False
    ):
        managed_artifact_files.append(
            "visual_context/lstar_domain_assignment.png"
        )
    if visual_context is not None and visual_context.audit.get(
        "histology_available", False
    ):
        managed_artifact_files.append("visual_context/histology_background.jpg")
    for domain_id in domain_ids:
        stem = domain_artifact_stem(domain_id)
        managed_artifact_files.extend(
            [
                "domain_markers/{}_markers.csv".format(stem),
                "evidence_cards/{}.json".format(stem),
            ]
        )

    for directory in subdirectories.values():
        directory.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        artifact_root / "h5ad_inspection.json", inputs.inspection_report
    )
    write_dataframe_atomic(
        artifact_root / "observation_alignment.csv", inputs.alignment_table
    )
    write_json_atomic(
        artifact_root / "expression_source_manifest.json",
        inputs.expression_source_manifest,
    )
    write_json_atomic(
        subdirectories["visual_context"] / "visual_context.json",
        (
            visual_context.audit
            if visual_context is not None
            else {
                "policy": "Visual background disabled by include_visual_background=False.",
                "domain_visualization_available": False,
                "histology_available": False,
                "images_transmitted": [],
            }
        ),
    )
    prepared_by_domain = {
        prepared["domain_id"]: prepared for prepared in prepared_domains
    }
    for domain_id in domain_ids:
        prepared = prepared_by_domain[domain_id]
        stem = domain_artifact_stem(domain_id)
        write_dataframe_atomic(
            subdirectories["domain_markers"] / "{}_markers.csv".format(stem),
            prepared["marker_artifact"],
        )
        write_json_atomic(
            subdirectories["evidence_cards"] / "{}.json".format(stem),
            prepared["evidence_card"],
        )

    write_json_atomic(
        artifact_root / "annotation_prompt_payload.json",
        {
            "dataset_context": dataset_context.strip(),
            "biological_context": biological_context,
            "notes": notes,
            "sampling_level": sampling_level,
            "payload": decision.payload,
            "deterministic_unknown_domain_ids": list(
                decision.deterministic_unknown_domain_ids
            ),
            "full_expression_matrix_included": False,
            "visual_background": (
                visual_context.audit
                if visual_context is not None
                else {"images_transmitted": []}
            ),
            "label_leakage_columns_excluded": inputs.inspection_report.get(
                "candidate_label_leakage_columns", []
            ),
        },
    )
    write_json_atomic(
        artifact_root / "annotation_response.json",
        {
            "validated_response": decision.response,
            "final_domains": list(decision.domains),
            "raw_responses": list(decision.raw_responses),
            "validation_errors": list(decision.validation_errors),
            "fallback_used": decision.fallback_used,
        },
    )
    write_dataframe_atomic(
        subdirectories["preprocessing"] / "feature_mapping.csv",
        preprocessing.original_feature_metadata,
    )
    write_dataframe_atomic(
        subdirectories["preprocessing"] / "analysis_feature_mapping.csv",
        preprocessing.feature_metadata,
    )
    write_json_atomic(
        subdirectories["preprocessing"] / "expression_profile.json",
        {
            "expression_profile": preprocessing.expression_profile,
            "identifier_profile": preprocessing.identifier_profile,
            "assay_profile": preprocessing.assay_profile,
            "quality_gates": preprocessing.quality_gates,
        },
    )
    write_json_atomic(
        subdirectories["preprocessing"] / "normalization.json",
        preprocessing.normalization,
    )
    write_json_atomic(
        subdirectories["preprocessing"] / "qc_filtering.json",
        preprocessing.qc_filtering,
    )
    marker_rows = pd.concat(
        [prepared["marker_artifact"] for prepared in prepared_domains],
        ignore_index=True,
    )
    positive_marker_rows = marker_rows.loc[
        marker_rows["marker_direction"] == "positive"
    ]
    marker_weighted_resolution_rate = (
        float(positive_marker_rows["annotation_eligible"].astype(bool).mean())
        if len(positive_marker_rows)
        else None
    )
    marker_weighted_context_rate = (
        float(
            positive_marker_rows["annotation_usable_for_context"]
            .astype(bool)
            .mean()
        )
        if len(positive_marker_rows)
        else None
    )
    marker_weighted_inference_rate = (
        float(
            positive_marker_rows["inference_confidence_eligible"]
            .astype(bool)
            .mean()
        )
        if len(positive_marker_rows)
        else None
    )
    resolution_qc = dict(feature_resolution.qc_report)
    resolution_qc["marker_weighted_resolution_rate"] = (
        marker_weighted_resolution_rate
    )
    resolution_qc["marker_weighted_context_usable_rate"] = (
        marker_weighted_context_rate
    )
    resolution_qc["marker_weighted_inference_eligible_rate"] = (
        marker_weighted_inference_rate
    )
    write_json_atomic(
        subdirectories["feature_resolution"] / "feature_profile.json",
        feature_resolution.feature_profile,
    )
    write_json_atomic(
        subdirectories["feature_resolution"] / "resource_manifest.json",
        {"resources": list(feature_resolution.resource_manifest)},
    )
    write_json_atomic(
        subdirectories["feature_resolution"] / "pilot_results.json",
        {"pilots": list(feature_resolution.pilot_results)},
    )
    write_json_atomic(
        subdirectories["feature_resolution"] / "planner_response.json",
        feature_resolution.planner_response,
    )
    write_json_atomic(
        subdirectories["feature_resolution"] / "resolution_recipe.json",
        {
            "planner_response": feature_resolution.planner_response,
            "resource_manifest": list(feature_resolution.resource_manifest),
            "validation_rule_adjustments": list(
                feature_resolution.planner_rule_adjustments
            ),
        },
    )
    write_dataframe_atomic(
        subdirectories["feature_resolution"] / "feature_mapping.csv",
        feature_resolution.mapping_table,
    )
    write_json_atomic(
        subdirectories["feature_resolution"] / "quality_control.json",
        resolution_qc,
    )
    write_json_atomic(
        subdirectories["feature_resolution"] / "llm_responses.json",
        {
            "raw_responses": list(feature_resolution.raw_planner_responses),
            "validation_errors": list(
                feature_resolution.planner_validation_errors
            ),
            "validation_rule_adjustments": list(
                feature_resolution.planner_rule_adjustments
            ),
        },
    )
    write_json_atomic(
        artifact_root / "feature_resolution_recipe.json",
        {
            "planner_response": feature_resolution.planner_response,
            "resource_manifest": list(feature_resolution.resource_manifest),
            "validation_rule_adjustments": list(
                feature_resolution.planner_rule_adjustments
            ),
        },
    )
    write_dataframe_atomic(
        artifact_root / "feature_resolution.csv",
        feature_resolution.mapping_table,
    )
    label_resolution_columns = [
        "feature_position",
        "original_feature_id",
        "identity_value",
        "identity_type",
        "identity_status",
        "identity_confidence",
        "canonical_annotation_symbol",
        "source_feature_label",
        "source_label_candidates",
        "source_label_candidate_count",
        "source_label_parse_rule",
        "source_label_status",
        "annotation_label",
        "annotation_label_source",
        "annotation_label_confidence",
        "feature_evidence_tier",
        "annotation_usable_for_context",
        "annotation_usable_for_naming",
        "direct_confidence_eligible",
        "inference_confidence_eligible",
        "fallback_applied",
        "ambiguity_reason",
    ]
    write_dataframe_atomic(
        artifact_root / "annotation_label_resolution.csv",
        feature_resolution.mapping_table.loc[:, label_resolution_columns],
    )
    write_json_atomic(
        artifact_root / "annotation_evidence_preflight.json", preflight
    )
    write_json_atomic(
        artifact_root / "feature_resolution_summary.json", resolution_qc
    )
    domain_evidence_rows = []
    for domain_id in domain_ids:
        prepared = prepared_by_domain[domain_id]
        card = prepared["evidence_card"]
        domain_evidence_rows.append(
            {
                "domain_id": domain_id,
                "sampling_level": sampling_level,
                "n_observations": int(prepared["n_cells"]),
                "n_observations_prefilter": int(prepared["n_cells_prefilter"]),
                "positive_markers": json.dumps(
                    [row["gene"] for row in card["positive_markers"]],
                    ensure_ascii=False,
                ),
                "positive_sent_to_llm": prepared["marker_counts"][
                    "positive_sent_to_llm"
                ],
                "evidence_mode": card["evidence_mode"],
                "direct_positive_sent_to_llm": prepared["marker_counts"][
                    "direct_positive_sent_to_llm"
                ],
                "inference_positive_sent_to_llm": prepared["marker_counts"][
                    "inference_positive_sent_to_llm"
                ],
                "selected_expression_source": inputs.expression_source_manifest[
                    "selected_expression_source"
                ],
                "selected_expression_scale": inputs.expression_source_manifest[
                    "selected_expression_scale"
                ],
                "abstention_required": bool(card["abstention_gate"]["required"]),
            }
        )
    write_dataframe_atomic(
        artifact_root / "domain_evidence.csv", pd.DataFrame(domain_evidence_rows)
    )
    write_dataframe_atomic(
        artifact_root / "domain_relationships.csv", domain_relationships
    )
    write_dataframe_atomic(artifact_root / "domain_annotation_map.csv", domain_map)
    write_dataframe_atomic(output_path, final_assignment)

    removed_stale_artifacts = list(
        remove_stale_managed_artifacts(artifact_root, managed_artifact_files)
    )
    domain_sizes_prefilter = {
        domain_id: int(np.sum(lstar_labels == domain_id)) for domain_id in domain_ids
    }
    domain_sizes_qc_passed = {
        str(prepared["domain_id"]): int(prepared["n_cells"])
        for prepared in prepared_domains
    }
    resources = load_global_annotation_resources()
    resolution_prompt, resolution_schema, resolution_skill_version = (
        load_feature_resolution_resources()
    )
    manifest = {
        "timestamp": timestamp,
        "expression_data_path": str(inputs.expression_data_path),
        "expression_input_format": inputs.input_format,
        "expression_source_manifest": dict(inputs.expression_source_manifest),
        "assignments_csv": str(inputs.assignments_csv),
        "lstar_consensus_csv": str(inputs.lstar_consensus_csv),
        "id_col": inputs.id_col,
        "consensus_id_col": inputs.consensus_id_col,
        "consensus_domain_col": inputs.consensus_domain_col,
        "observation_alignment": dict(inputs.alignment_summary),
        "method_columns": list(inputs.method_columns),
        "consensus_method_source": "lstar_run_manifest",
        "dataset_context": dataset_context.strip(),
        "biological_context": biological_context,
        "notes": notes,
        "auto_discover_resolution_resources": bool(
            auto_discover_resolution_resources
        ),
        "sampling_level": sampling_level,
        "expression_profile": preprocessing.expression_profile,
        "identifier_profile": preprocessing.identifier_profile,
        "normalization": preprocessing.normalization,
        "assay_profile": preprocessing.assay_profile,
        "quality_gates": preprocessing.quality_gates,
        "qc_filtering": preprocessing.qc_filtering,
        "feature_resolution": {
            "recipe_id": feature_resolution.recipe["recipe_id"],
            "recipe": feature_resolution.recipe,
            "quality_control": resolution_qc,
            "resource_manifest": list(feature_resolution.resource_manifest),
            "validation_rule_adjustments": list(
                feature_resolution.planner_rule_adjustments
            ),
            "rules_version": FEATURE_RESOLUTION_RULES_VERSION,
            "skill_version": resolution_skill_version,
            "planner_prompt_hash_sha256": hashlib.sha256(
                resolution_prompt.encode("utf-8")
            ).hexdigest(),
            "planner_schema_id": resolution_schema.get("$id"),
        },
        "marker_filter_parameters": settings,
        "marker_statistics": {
            "method": _marker_statistics_description(marker_test_method),
            "marker_test_method": marker_test_method,
            "marker_test_method_is_scanpy_default": marker_test_method is None,
            "scanpy_default_marker_test_method": SCANPY_DEFAULT_MARKER_TEST_METHOD,
            "engine": "scanpy",
            "scanpy_version": preprocessing.normalization.get("scanpy_version"),
            "multiple_testing": "Benjamini-Hochberg across all genes per domain",
            "primary_effect": "mean_in - mean_out on log1p-normalized values",
            "rank_effects": ["rank_genes_groups_score"],
            "specificity": "domain mean minus maximum mean across other domains",
            "marker_rules_version": MARKER_RULES_VERSION,
            "direction_policy": "positive_only",
            "technical_gene_exclusion": (
                "none; no hand-curated housekeeping/ribosomal/stress list and "
                "no mitochondrial-gene removal, matching scanpy, which "
                "provides neither. Broadly expressed genes are discounted by "
                "the naming prompt at interpretation time instead"
            ),
        },
        "domain_sizes_prefilter": domain_sizes_prefilter,
        "domain_sizes_qc_passed": domain_sizes_qc_passed,
        "llm_model_name": active_model,
        "llm_reasoning_effort": reasoning_effort,
        "annotation_skill_version": resources.version,
        "annotation_prompt_hash_sha256": hashlib.sha256(
            resources.annotation_prompt.encode("utf-8")
        ).hexdigest(),
        "annotation_schema_id": resources.annotation_schema.get("$id"),
        "preprocessing_rules_version": PREPROCESSING_RULES_VERSION,
        "qc_rules_version": QC_RULES_VERSION,
        "global_annotation_rules_version": GLOBAL_ANNOTATION_RULES_VERSION,
        "feature_resolution_rules_version": FEATURE_RESOLUTION_RULES_VERSION,
        "output_csv": str(output_path.resolve()),
        "artifacts_dir": str(artifact_root.resolve()),
        "managed_artifact_files": managed_artifact_files,
        "removed_stale_artifacts": removed_stale_artifacts,
        "raw_expression_transmitted_to_llm": False,
        "visual_background": (
            visual_context.audit
            if visual_context is not None
            else {"images_transmitted": []}
        ),
        "automatic_resolution_resource_suffixes": [".h5", ".hdf5", ".h5ad"],
        "observation_membership_modified": bool(
            inputs.alignment_summary["h5ad_only_observations"]
            or inputs.alignment_summary["consensus_only_observations"]
        ),
    }
    annotation_audit = {
        "timestamp": timestamp,
        "sampling_level": sampling_level,
        "observation_count": len(final_assignment),
        "observation_order_preserved": True,
        "lstar_membership_preserved": True,
        "domain_count": len(domain_ids),
        "domain_order": domain_ids,
        "domains": [
            {
                "domain_id": domain_id,
                "n_cells_or_spots": prepared_by_domain[domain_id]["n_cells"],
                "n_cells_or_spots_prefilter": prepared_by_domain[domain_id][
                    "n_cells_prefilter"
                ],
                "marker_counts": prepared_by_domain[domain_id]["marker_counts"],
                "abstention_gate": prepared_by_domain[domain_id]["evidence_card"][
                    "abstention_gate"
                ],
                "final": next(
                    row for row in decision.domains
                    if str(row["domain_id"]) == domain_id
                ),
            }
            for domain_id in domain_ids
        ],
        "primary_output_validated": True,
        "raw_expression_transmitted_to_llm": False,
        "annotation_evidence_preflight": preflight,
    }
    write_json_atomic(artifact_root / "annotation_audit.json", annotation_audit)
    write_json_atomic(artifact_root / "annotation_manifest.json", manifest)
    write_json_atomic(artifact_root / "annotation_run_manifest.json", manifest)

    final_assignment.attrs["output_csv"] = str(output_path.resolve())
    final_assignment.attrs["artifacts_dir"] = str(artifact_root.resolve())
    final_assignment.attrs["annotation_manifest"] = manifest
    progress_section("L-STAR DOMAIN ANNOTATION COMPLETE")
    progress("Output directory: {}".format(output_root.resolve()))
    progress_detail("Annotated observations", len(final_assignment))
    progress_detail("Annotated domains", len(domain_ids))
    progress_detail("Annotation CSV", output_path.resolve())
    progress_detail("Audit artifacts", artifact_root.resolve())
    return final_assignment


# Compatibility alias for code written against the standalone prototype. The
# package-level public API is the singular ``annotate_domain`` function.
annotate_domains = annotate_domain


__all__ = ["annotate_domain"]
