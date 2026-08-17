"""LLM-directed, adapter-free feature identity and ortholog resolution.

The LLM plans one declarative workflow from collection-level evidence. Generic
primitives then execute that recipe deterministically for every feature.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

if __package__:
    from .llm_annotation import (
        DEFAULT_ANNOTATION_MODEL,
        DEFAULT_ANNOTATION_REASONING_EFFORT,
        call_model,
        create_openai_client,
    )
else:
    from llm_annotation import (
        DEFAULT_ANNOTATION_MODEL,
        DEFAULT_ANNOTATION_REASONING_EFFORT,
        call_model,
        create_openai_client,
    )


PathLike = Union[str, Path]
FEATURE_RESOLUTION_RULES_VERSION = "1.7.0"
RESOURCE_DIR = Path(__file__).resolve().parent / "skills" / "feature_resolution"
FEATURE_EVIDENCE_TIERS = frozenset({"A", "B1", "B2", "C"})
ALLOWED_OUTPUT_FIELDS = frozenset(
    {
        "parsed_feature_id",
        "original_feature_version",
        "selected_namespace",
        "feature_level",
        "source_taxon_id",
        "source_assembly",
        "source_database",
        "source_database_version",
        "source_gene_id",
        "source_gene_symbol",
        "source_transcript_id",
        "parent_gene_id",
        "canonical_source_gene_id",
        "target_gene_id",
        "target_gene_symbol",
        "target_taxon_id",
        "homology_type",
        "orthology_confidence",
        "orthology_evidence_source",
        "chrom",
        "start",
        "end",
        "strand",
        "sequence_sha256",
    }
)
MAPPING_STATUSES = frozenset(
    {
        "resolved_unique",
        "resolved_multiple",
        "unresolved",
        "unresolved_taxon",
        "unresolved_assembly",
        "non_gene_feature",
        "invalid_feature",
    }
)
DECLARED_IDENTITY_METHOD_TO_OUTPUT_FIELD = {
    "gene_symbol": "source_gene_symbol",
    "gene_id": "source_gene_id",
    "transcript_id": "source_transcript_id",
    "non_gene": "feature_level",
}
_DECLARED_IDENTITY_FIELD_TO_METHOD = {
    output_field: method
    for method, output_field in DECLARED_IDENTITY_METHOD_TO_OUTPUT_FIELD.items()
    if method != "non_gene"
}
_DECLARED_IDENTITY_METHOD_ALIASES = {
    "symbol": "gene_symbol",
    "source_gene_symbol": "gene_symbol",
    "source_gene_id": "gene_id",
    "source_transcript_id": "transcript_id",
    "transcript": "transcript_id",
}
DISCOVERABLE_RESOURCE_SUFFIXES = frozenset(
    {
        ".csv", ".tsv", ".tab", ".json", ".gtf", ".gff", ".gff3",
        ".fa", ".fasta", ".fna", ".faa", ".h5ad", ".h5", ".hdf5",
        ".txt", ".md", ".html", ".htm",
    }
)


@dataclass(frozen=True)
class ResolutionResource:
    resource_id: str
    path: Path
    resource_type: str
    checksum_sha256: str
    table_name: str
    table: pd.DataFrame
    evidence_strength: float
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class FeatureResolutionResult:
    mapping_table: pd.DataFrame
    compatibility_metadata: pd.DataFrame
    feature_profile: Mapping[str, Any]
    resource_manifest: Tuple[Mapping[str, Any], ...]
    pilot_results: Tuple[Mapping[str, Any], ...]
    planner_response: Mapping[str, Any]
    recipe: Mapping[str, Any]
    qc_report: Mapping[str, Any]
    raw_planner_responses: Tuple[str, ...]
    planner_validation_errors: Tuple[str, ...]
    planner_rule_adjustments: Tuple[Mapping[str, Any], ...]


class FeatureAnnotationEvidenceError(RuntimeError):
    """Feature evidence could not be linked or serialized safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "na", "none", "null"}:
        return None
    return text


_SOURCE_LABEL_MISSING_VALUES = frozenset(
    {
        "-",
        ".",
        "missing",
        "na",
        "nan",
        "none",
        "null",
        "unknown",
        "unmapped",
        "unannotated",
    }
)


def _normalized_source_candidate(value: Any) -> Optional[str]:
    """Normalize one readable source-label segment without asserting identity."""
    text = _safe_text(value)
    if text is None:
        return None
    text = re.sub(r"(?:\[[^\[\]]{1,24}\])+$", "", text).strip()
    text = re.sub(r"\s+", " ", text)
    if not text or text.lower() in _SOURCE_LABEL_MISSING_VALUES:
        return None
    if not re.search(r"[A-Za-z0-9]", text):
        return None
    return text


def _identity_fields(row: Mapping[str, Any]) -> Tuple[Tuple[str, str], ...]:
    """Return exact identity fields already established by recipe execution."""
    fields = (
        "canonical_source_gene_id",
        "source_gene_id",
        "source_transcript_id",
        "parent_gene_id",
        "target_gene_id",
        "parsed_feature_id",
    )
    result = []
    seen = set()
    for field in fields:
        value = _safe_text(row.get(field))
        if value is None or value in seen:
            continue
        seen.add(value)
        result.append((field, value))
    return tuple(result)


def _source_label_candidates(
    source_label: Any,
    identity_values: Sequence[str],
) -> Tuple[str, ...]:
    """Extract label candidates while preserving unresolved biological ambiguity."""
    label = _safe_text(source_label)
    if label is None:
        return tuple()
    segments = re.split(r"\s*(?:\||;|/|,\s+)\s*", label)
    candidates = []
    seen = set()
    for segment in segments:
        residual = str(segment)
        for identity in identity_values:
            identity_text = str(identity).strip()
            if not identity_text:
                continue
            if residual.strip().casefold() == identity_text.casefold():
                residual = ""
                break
            residual = re.sub(
                r"(?<![A-Za-z0-9]){}(?![A-Za-z0-9])".format(
                    re.escape(identity_text)
                ),
                " ",
                residual,
                flags=re.I,
            )
        candidate = _normalized_source_candidate(residual)
        if candidate is None:
            continue
        key = candidate.casefold()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)
    return tuple(candidates)


def _add_feature_evidence_contract(mapping: pd.DataFrame) -> pd.DataFrame:
    """Classify canonical, anchored, ambiguous, and opaque feature evidence."""
    result = mapping.copy()
    records = []
    for _, row in result.iterrows():
        row_map = row.to_dict()
        source_label = str(row_map["original_feature_id"])
        identity_fields = _identity_fields(row_map)
        identity_type = identity_fields[0][0] if identity_fields else "feature_record"
        identity_value = (
            identity_fields[0][1] if identity_fields else source_label
        )
        canonical_symbol = _safe_text(row_map.get("annotation_symbol"))
        original_mapping_type = str(row_map.get("mapping_type") or "unresolved")
        original_confidence = float(row_map.get("mapping_confidence") or 0.0)
        mapping_status = str(row_map.get("mapping_status") or "unresolved")
        excluded = mapping_status in {"non_gene_feature", "invalid_feature"}
        exact_identity = mapping_status == "resolved_unique"
        candidates = (
            tuple()
            if canonical_symbol is not None and exact_identity
            else _source_label_candidates(
                source_label,
                [value for _, value in identity_fields],
            )
        )

        if canonical_symbol is not None and exact_identity and not excluded:
            tier = "A"
            label = canonical_symbol
            label_source = (
                "validated_ortholog"
                if original_mapping_type == "ortholog"
                else "canonical_symbol"
            )
            label_status = "resolved_direct"
            usable_for_context = True
            usable_for_naming = True
            direct_eligible = True
            inference_eligible = True
            mapping_type = original_mapping_type
            label_confidence = original_confidence
        elif exact_identity and len(candidates) == 1 and not excluded:
            tier = "B1"
            label = source_label
            label_source = "anchored_source_label"
            label_status = "anchored_unique"
            usable_for_context = True
            usable_for_naming = True
            direct_eligible = True
            inference_eligible = True
            mapping_type = "anchored_source_label"
            label_confidence = max(original_confidence, 0.65)
        elif candidates and not excluded:
            tier = "B2"
            label = source_label
            label_source = "ambiguous_source_label"
            label_status = "anchored_multiple" if exact_identity else "unresolved_multiple"
            usable_for_context = True
            usable_for_naming = False
            direct_eligible = False
            inference_eligible = True
            mapping_type = "ambiguous_source_label"
            label_confidence = min(max(original_confidence, 0.30), 0.50)
        else:
            tier = "C"
            label = source_label
            label_source = "opaque_identifier"
            label_status = "excluded" if excluded else "opaque"
            usable_for_context = not excluded and _safe_text(source_label) is not None
            usable_for_naming = False
            direct_eligible = False
            inference_eligible = usable_for_context
            mapping_type = "opaque_identifier"
            label_confidence = 0.0

        records.append(
            {
                "identity_value": identity_value,
                "identity_type": identity_type,
                "identity_status": mapping_status,
                "identity_confidence": original_confidence,
                "identity_mapping_type": original_mapping_type,
                "canonical_annotation_symbol": canonical_symbol,
                "source_feature_label": source_label,
                "source_label_candidates": json.dumps(
                    list(candidates), ensure_ascii=False
                ),
                "source_label_candidate_count": int(len(candidates)),
                "source_label_parse_rule": (
                    "exact_identity_subtraction_and_generic_delimiter_split_v1"
                ),
                "source_label_status": label_status,
                "annotation_label": label,
                "annotation_label_source": label_source,
                "annotation_label_confidence": label_confidence,
                "annotation_usable_for_context": usable_for_context,
                "annotation_usable_for_naming": usable_for_naming,
                "direct_confidence_eligible": direct_eligible,
                "inference_confidence_eligible": inference_eligible,
                "feature_evidence_tier": tier,
                "fallback_applied": tier in {"B1", "B2", "C"},
                "mapping_type": mapping_type,
                "annotation_symbol": label if direct_eligible else None,
                "annotation_eligible": direct_eligible,
                "ambiguity_reason": (
                    "multiple source-label candidates"
                    if tier == "B2"
                    else row_map.get("ambiguity_reason")
                ),
            }
        )
    evidence_table = pd.DataFrame(records, index=result.index)
    for column in evidence_table:
        result[column] = evidence_table[column]
    return result


def profile_feature_collection(feature_ids: Sequence[str]) -> Dict[str, Any]:
    """Describe identifiers as generic strings without assigning a namespace."""
    values = [str(value) for value in feature_ids]
    if not values or any(not value.strip() for value in values):
        raise ValueError("feature_ids must contain non-empty strings")
    lengths = np.asarray([len(value) for value in values], dtype=float)
    presence: Dict[str, int] = {}
    maximum_occurrences: Dict[str, int] = {}
    for value in values:
        counts: Dict[str, int] = {}
        for character in value:
            if not character.isalnum():
                counts[character] = counts.get(character, 0) + 1
        for character, count in counts.items():
            presence[character] = presence.get(character, 0) + 1
            if count > maximum_occurrences.get(character, 0):
                maximum_occurrences[character] = count
    delimiter_profiles = sorted(
        (
            {
                "delimiter": character,
                "feature_fraction": float(count / len(values)),
                "maximum_occurrences": int(maximum_occurrences[character]),
            }
            for character, count in presence.items()
        ),
        key=lambda item: (-item["feature_fraction"], item["delimiter"]),
    )[:30]
    prefixes: Dict[str, int] = {}
    suffixes: Dict[str, int] = {}
    for value in values:
        for width in (1, 2, 3, 4):
            if len(value) >= width:
                prefixes[value[:width]] = prefixes.get(value[:width], 0) + 1
                suffixes[value[-width:]] = suffixes.get(value[-width:], 0) + 1
    top = lambda counts: [
        {"token": token, "count": int(count), "fraction": float(count / len(values))}
        for token, count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )[:20]
    ]
    sample_positions = np.linspace(
        0, len(values) - 1, min(50, len(values)), dtype=int
    )
    return {
        "feature_count": len(values),
        "duplicate_identifier_count": int(pd.Series(values).duplicated().sum()),
        "length_percentiles": [
            float(value) for value in np.quantile(lengths, [0, 0.25, 0.5, 0.75, 1])
        ],
        "numeric_only_fraction": float(
            np.mean([bool(re.fullmatch(r"\d+", value)) for value in values])
        ),
        "terminal_version_fraction": float(
            np.mean([bool(re.search(r"\.\d+$", value)) for value in values])
        ),
        "coordinate_like_fraction": float(
            np.mean(
                [
                    bool(re.search(r"[^:]+:\d+(?:-|\.\.)\d+", value))
                    for value in values
                ]
            )
        ),
        "punctuation_profiles": delimiter_profiles,
        "top_prefixes": top(prefixes),
        "top_suffixes": top(suffixes),
        "representative_features": [values[index] for index in sample_positions],
        "rules_version": FEATURE_RESOLUTION_RULES_VERSION,
    }


def _decode_series(values: Iterable[Any]) -> List[str]:
    decoded = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8", errors="replace"))
        else:
            decoded.append(str(value))
    return decoded


def discover_supporting_files(
    directory: PathLike,
    *,
    excluded_paths: Sequence[PathLike] = (),
    maximum_table_bytes: int = 50 * 1024 * 1024,
) -> Tuple[Path, ...]:
    """Discover generic local evidence without recognizing a platform name."""
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("supporting-file discovery root must be a directory")
    excluded = {Path(path).expanduser().resolve() for path in excluded_paths}
    discovered = []
    for path in sorted(root.iterdir(), key=lambda value: value.name):
        if path.resolve() in excluded or not path.is_file() or path.is_symlink():
            continue
        suffix = path.suffix.lower()
        if suffix not in DISCOVERABLE_RESOURCE_SUFFIXES:
            continue
        if suffix in {".csv", ".tsv", ".tab", ".json", ".txt", ".md", ".html", ".htm"} and path.stat().st_size > maximum_table_bytes:
            continue
        discovered.append(path.resolve())
    return tuple(discovered)


def _read_delimited_resource(path: Path) -> List[Tuple[str, pd.DataFrame, float]]:
    separator = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    table = pd.read_csv(
        path,
        sep=separator,
        engine="c",
        dtype=str,
        keep_default_na=False,
    )
    return [(path.stem, table, 0.85)]


def _parse_gtf_attributes(value: str) -> Dict[str, str]:
    attributes = {}
    for entry in str(value).strip().strip(";").split(";"):
        entry = entry.strip()
        if not entry:
            continue
        match = re.match(r"([^=\s]+)(?:=|\s+)\"?([^\"]+)\"?$", entry)
        if match:
            attributes[match.group(1)] = match.group(2).strip().strip('"')
    return attributes


def _read_gtf_gff_resource(path: Path) -> List[Tuple[str, pd.DataFrame, float]]:
    base_columns = [
        "chrom", "source", "feature_type", "start", "end", "score", "strand",
        "phase", "attributes",
    ]
    table = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        names=base_columns,
        dtype=str,
        keep_default_na=False,
    )
    parsed = table["attributes"].map(_parse_gtf_attributes)
    attribute_names = sorted({key for record in parsed for key in record})
    for name in attribute_names:
        table[name] = parsed.map(lambda record: record.get(name, ""))
    return [(path.stem, table, 0.95)]


def _read_fasta_resource(path: Path) -> List[Tuple[str, pd.DataFrame, float]]:
    records = []
    identifier = None
    description = None
    sequence_parts: List[str] = []

    def append_record() -> None:
        if identifier is None:
            return
        sequence = "".join(sequence_parts).upper()
        records.append(
            {
                "record_id": identifier,
                "description": description,
                "sequence_length": len(sequence),
                "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
            }
        )

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if text.startswith(">"):
                append_record()
                description = text[1:]
                identifier = description.split()[0] if description else ""
                sequence_parts = []
            elif text:
                sequence_parts.append(text)
    append_record()
    return [(path.stem, pd.DataFrame(records), 0.90)]


def _read_h5ad_resource(path: Path) -> List[Tuple[str, pd.DataFrame, float]]:
    try:
        import anndata as ad
    except ImportError as error:
        raise ImportError("anndata is required to inspect H5AD feature metadata") from error
    data = ad.read_h5ad(path, backed="r")
    try:
        tables = []
        table = data.var.copy()
        table.insert(0, "feature_index", data.var_names.astype(str))
        table = table.reset_index(drop=True).astype(str)
        tables.append(("var", table, 0.90))
        if data.raw is not None:
            raw_table = data.raw.var.copy()
            raw_table.insert(0, "feature_index", data.raw.var_names.astype(str))
            raw_table = raw_table.reset_index(drop=True).astype(str)
            tables.append(("raw_var", raw_table, 0.90))
    finally:
        data.file.close()
    return tables


def _read_hdf5_resource(path: Path) -> List[Tuple[str, pd.DataFrame, float]]:
    try:
        import h5py
    except ImportError as error:
        raise ImportError("h5py is required to inspect HDF5 feature metadata") from error
    tables = []
    with h5py.File(path, "r") as handle:
        groups: Dict[Tuple[str, int], Dict[str, List[str]]] = {}

        def visit(name: str, item: Any) -> None:
            if not isinstance(item, h5py.Dataset) or item.ndim != 1:
                return
            field_name = Path(name).name
            if field_name.startswith("_"):
                return
            if item.shape[0] == 0 or item.shape[0] > 2_000_000:
                return
            if item.dtype.kind not in {"S", "U", "O"}:
                return
            path_parts = tuple(part.lower() for part in Path(name).parts)
            feature_metadata_path = any(
                part in {"feature", "features", "gene", "genes", "var"}
                for part in path_parts[:-1]
            )
            if not feature_metadata_path:
                return
            parent = str(Path(name).parent)
            groups.setdefault((parent, int(item.shape[0])), {})[
                field_name
            ] = _decode_series(item[:])

        handle.visititems(visit)
        for (parent, _), columns in groups.items():
            if columns:
                table_name = parent.replace("/", "_").strip("_") or "root"
                tables.append((table_name, pd.DataFrame(columns), 0.90))
    return tables


def _read_json_resource(path: Path) -> List[Tuple[str, pd.DataFrame, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list) and payload and all(isinstance(item, dict) for item in payload):
        return [(path.stem, pd.DataFrame(payload).astype(str), 0.85)]
    if isinstance(payload, dict):
        tables = []
        for key, value in payload.items():
            if isinstance(value, list) and value and all(
                isinstance(item, dict) for item in value
            ):
                tables.append((str(key), pd.DataFrame(value).astype(str), 0.85))
        scalar_values = {
            str(key): value
            for key, value in payload.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }
        if scalar_values:
            tables.append(("metadata", pd.DataFrame([scalar_values]).astype(str), 0.75))
        return tables
    return []


def load_resolution_resources(
    resource_paths: Sequence[PathLike],
) -> Tuple[Tuple[ResolutionResource, ...], Tuple[Mapping[str, Any], ...]]:
    """Load generic local resources without dispatching by biological namespace."""
    paths = []
    for value in resource_paths:
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError("resolution resource not found: {}".format(path))
        if path not in paths:
            paths.append(path)
    resources = []
    manifests = []
    for path_index, path in enumerate(paths, start=1):
        suffix = path.suffix.lower()
        if suffix in {".csv", ".tsv", ".tab"}:
            tables = _read_delimited_resource(path)
            resource_type = "mapping_table"
        elif suffix in {".gtf", ".gff", ".gff3"}:
            tables = _read_gtf_gff_resource(path)
            resource_type = "annotation_table"
        elif suffix in {".fa", ".fasta", ".fna", ".faa"}:
            tables = _read_fasta_resource(path)
            resource_type = "fasta"
        elif suffix == ".h5ad":
            tables = _read_h5ad_resource(path)
            resource_type = "h5ad_feature_metadata"
        elif suffix in {".h5", ".hdf5"}:
            tables = _read_hdf5_resource(path)
            resource_type = "hdf5_metadata"
        elif suffix == ".json":
            tables = _read_json_resource(path)
            resource_type = "json_records"
        else:
            tables = []
            resource_type = "unparsed_supporting_file"
        checksum = _sha256(path)
        if not tables:
            record = {
                "path": str(path),
                "checksum_sha256": checksum,
                "resource_type": resource_type,
                "parsed": False,
            }
            if suffix in {".txt", ".md", ".html", ".htm"}:
                record["untrusted_text_excerpt"] = path.read_text(
                    encoding="utf-8", errors="replace"
                )[:4000]
            manifests.append(record)
        for table_index, (table_name, table, strength) in enumerate(tables, start=1):
            resource_id = "resource_{:03d}_table_{:02d}".format(path_index, table_index)
            resource = ResolutionResource(
                resource_id=resource_id,
                path=path,
                resource_type=resource_type,
                checksum_sha256=checksum,
                table_name=table_name,
                table=table,
                evidence_strength=strength,
                metadata={},
            )
            resources.append(resource)
            manifests.append(
                {
                    "resource_id": resource_id,
                    "path": str(path),
                    "checksum_sha256": checksum,
                    "resource_type": resource_type,
                    "table_name": table_name,
                    "row_count": int(len(table)),
                    "columns": list(map(str, table.columns)),
                    "sample_rows": [
                        {
                            str(key): str(value)[:200]
                            for key, value in row.items()
                        }
                        for row in table.iloc[
                            np.unique(
                                np.linspace(
                                    0, len(table) - 1, min(5, len(table)), dtype=int
                                )
                            )
                        ].to_dict(orient="records")
                    ],
                    "parsed": True,
                }
            )
    return tuple(resources), tuple(manifests)


def _candidate_feature_transforms(
    feature_ids: Sequence[str],
    feature_profile: Mapping[str, Any],
) -> Mapping[str, Mapping[str, Any]]:
    values = list(map(str, feature_ids))
    transforms: Dict[str, Mapping[str, Any]] = {
        "original": {
            "values": values,
            "recipe": {"method": "copy"},
        },
        "strip_terminal_version": {
            "values": [re.sub(r"\.\d+$", "", value) for value in values],
            "recipe": {"method": "strip_terminal_version"},
        },
        "strip_terminal_integer_suffix": {
            "values": [re.sub(r"-\d+$", "", value) for value in values],
            "recipe": {"method": "strip_terminal_integer_suffix"},
        },
    }
    passing_profiles = sorted(
        (
            item
            for item in feature_profile.get("punctuation_profiles", [])
            if float(item["feature_fraction"]) >= 0.05
        ),
        key=lambda item: (-float(item["feature_fraction"]), str(item["delimiter"])),
    )
    delimiters = [item["delimiter"] for item in passing_profiles]
    for delimiter in delimiters[:8]:
        transforms["first_token_{}".format(ord(delimiter[0]))] = {
            "values": [value.split(delimiter)[0].strip() for value in values],
            "recipe": {
                "method": "split_token",
                "delimiter": delimiter,
                "token_index": 0,
            },
        }
        transforms["last_token_{}".format(ord(delimiter[0]))] = {
            "values": [value.split(delimiter)[-1].strip() for value in values],
            "recipe": {
                "method": "split_token",
                "delimiter": delimiter,
                "token_index": -1,
            },
        }
    return transforms


def pilot_resolution_workflows(
    feature_ids: Sequence[str],
    feature_profile: Mapping[str, Any],
    resources: Sequence[ResolutionResource],
) -> Tuple[Mapping[str, Any], ...]:
    """Test exact collection-level matches against every available field."""
    transforms = _candidate_feature_transforms(feature_ids, feature_profile)
    pilots = []
    for resource in resources:
        for column in resource.table.columns:
            positional_values = resource.table[column].map(_safe_text)
            resource_values = positional_values
            resource_values = resource_values[resource_values.notna()].astype(str)
            if resource_values.empty:
                continue
            counts = resource_values.value_counts()
            value_set = set(counts.index)
            for transform_name, transform in transforms.items():
                transformed = transform["values"]
                if len(positional_values) == len(transformed):
                    positional_match = np.asarray(
                        [
                            resource_value is not None
                            and feature_value == str(resource_value)
                            for feature_value, resource_value in zip(
                                transformed, positional_values
                            )
                        ]
                    )
                    positional_rate = float(np.mean(positional_match))
                    if positional_rate >= 0.80:
                        pilots.append(
                            {
                                "pilot_id": "pilot_{:05d}".format(len(pilots) + 1),
                                "operation": "positional_alignment",
                                "feature_transform": transform_name,
                                "transform_recipe": transform["recipe"],
                                "resource_id": resource.resource_id,
                                "resource_field": str(column),
                                "exact_match_rate": positional_rate,
                                "unique_mapping_rate": positional_rate,
                                "ambiguous_mapping_rate": 0.0,
                                "unmatched_rate": 1.0 - positional_rate,
                                "evidence_score": float(
                                    positional_rate
                                    * resource.evidence_strength
                                    * 0.95
                                ),
                            }
                        )
                matched = np.asarray([value in value_set for value in transformed])
                if not matched.any():
                    continue
                ambiguous = np.asarray(
                    [counts.get(value, 0) > 1 for value in transformed]
                ) & matched
                match_rate = float(np.mean(matched))
                ambiguous_rate = float(np.mean(ambiguous))
                unique_rate = match_rate - ambiguous_rate
                pilots.append(
                    {
                        "pilot_id": "pilot_{:05d}".format(len(pilots) + 1),
                        "operation": "exact_join",
                        "feature_transform": transform_name,
                        "transform_recipe": transform["recipe"],
                        "resource_id": resource.resource_id,
                        "resource_field": str(column),
                        "exact_match_rate": match_rate,
                        "unique_mapping_rate": unique_rate,
                        "ambiguous_mapping_rate": ambiguous_rate,
                        "unmatched_rate": 1.0 - match_rate,
                        "evidence_score": float(
                            unique_rate * resource.evidence_strength
                            - ambiguous_rate * 0.5
                        ),
                    }
                )
    pilots.sort(
        key=lambda row: (
            -float(row["evidence_score"]),
            -float(row["exact_match_rate"]),
            str(row["resource_id"]),
            str(row["resource_field"]),
        )
    )
    for index, row in enumerate(pilots, start=1):
        row["pilot_id"] = "pilot_{:05d}".format(index)
    return tuple(pilots[:100])


def _schema_validate(payload: Any, schema: Mapping[str, Any]) -> None:
    try:
        import jsonschema
    except ImportError:
        jsonschema = None
    if jsonschema is not None:
        try:
            jsonschema.validate(payload, dict(schema))
        except jsonschema.exceptions.ValidationError as error:
            raise ValueError("planner schema validation failed: {}".format(error))
    if not isinstance(payload, dict):
        raise ValueError("planner response must be one JSON object")


def _identity_method_from_selected_hypothesis(
    payload: Mapping[str, Any],
) -> Optional[str]:
    """Resolve one identity method only from an unambiguous selected hypothesis."""
    selected_id = payload.get("selected_hypothesis_id")
    selected = next(
        (
            hypothesis
            for hypothesis in payload.get("hypotheses", [])
            if isinstance(hypothesis, Mapping)
            and hypothesis.get("hypothesis_id") == selected_id
        ),
        None,
    )
    if selected is None:
        return None
    try:
        confidence = float(selected.get("confidence", 0.0))
    except (TypeError, ValueError):
        return None
    if confidence < 0.90:
        return None

    feature_level = str(selected.get("feature_level", "")).lower()
    feature_type = str(selected.get("feature_type", "")).lower()
    candidates = set()
    if feature_level == "transcript":
        candidates.add("transcript_id")
    elif feature_level == "gene":
        if re.search(r"\bsymbols?\b", feature_type):
            candidates.add("gene_symbol")
        if "gene" in feature_type and re.search(
            r"\b(?:ids?|identifiers?|accessions?|ensembl)\b",
            feature_type,
        ):
            candidates.add("gene_id")
    elif feature_level in {"peak", "antibody", "control"}:
        candidates.add("non_gene")
    return next(iter(candidates)) if len(candidates) == 1 else None


def _canonicalize_declared_identity_methods(payload: Any) -> Any:
    """Normalize legacy wide steps without choosing biological identity."""
    if not isinstance(payload, Mapping):
        return payload
    normalized = copy.deepcopy(dict(payload))
    recipe = normalized.get("recipe")
    if not isinstance(recipe, Mapping):
        return normalized
    steps = recipe.get("resolution_steps", [])
    if not isinstance(steps, list):
        return normalized
    hypothesis_method = _identity_method_from_selected_hypothesis(normalized)
    for step in steps:
        if (
            not isinstance(step, dict)
            or step.get("operation") != "set_declared_identity"
        ):
            continue
        method = step.get("method")
        if (
            isinstance(method, str)
            and method in DECLARED_IDENTITY_METHOD_TO_OUTPUT_FIELD
        ):
            continue
        method_text = _safe_text(method)
        canonical_method = (
            _DECLARED_IDENTITY_METHOD_ALIASES.get(method_text.lower())
            if method_text is not None
            else None
        )
        generic_copy = method is None or method == "copy"
        if canonical_method is None and generic_copy:
            canonical_method = _DECLARED_IDENTITY_FIELD_TO_METHOD.get(
                str(step.get("output_field"))
            ) or _DECLARED_IDENTITY_FIELD_TO_METHOD.get(
                str(step.get("input_field"))
            )
        if canonical_method is None and generic_copy:
            canonical_method = hypothesis_method
        if canonical_method is not None:
            step["method"] = canonical_method
    common_fields = {"step_id", "operation", "reason"}
    for step in steps:
        if not isinstance(step, dict):
            continue
        operation = step.get("operation")
        if operation == "derive_field":
            allowed = common_fields | {
                "input_field",
                "output_field",
                "method",
                "preserve_version",
            }
            if step.get("method") == "regex_capture":
                allowed |= {"pattern", "capture_group"}
            elif step.get("method") == "split_token":
                allowed |= {"delimiter", "token_index"}
            step.setdefault("preserve_version", False)
        elif operation == "set_declared_identity":
            allowed = common_fields | {"input_field", "method"}
            if not isinstance(step.get("input_field"), str) or not str(
                step.get("input_field")
            ).strip():
                step["input_field"] = "original_feature_id"
        elif operation == "join_resource":
            allowed = common_fields | {
                "input_field",
                "resource_id",
                "resource_field",
                "output_mappings",
            }
        elif operation == "join_resource_by_position":
            allowed = common_fields | {
                "resource_id",
                "resource_field",
                "output_mappings",
            }
        elif operation == "interval_overlap_resource":
            allowed = common_fields | {
                "resource_id",
                "output_mappings",
                "coordinate_fields",
            }
        elif operation == "mark_unresolved":
            allowed = common_fields
        else:
            continue
        for field in list(step):
            if field not in allowed:
                del step[field]
    return normalized


def _validate_safe_regex(pattern: str) -> None:
    if len(pattern) > 500:
        raise ValueError("recipe regex is too long")
    if re.search(r"\\[1-9]|\(\?<[=!]|\(\?P=", pattern):
        raise ValueError("recipe regex uses disallowed backreferences or lookbehind")
    if re.search(r"\([^)]*[+*][^)]*\)\s*[+*{]", pattern):
        raise ValueError("recipe regex contains a nested repetition")
    re.compile(pattern)


def load_feature_resolution_resources() -> Tuple[str, Mapping[str, Any], str]:
    prompt_path = RESOURCE_DIR / "planner.prompt.md"
    schema_path = RESOURCE_DIR / "planner.schema.json"
    version_path = RESOURCE_DIR / "VERSION"
    for path in (prompt_path, schema_path, version_path):
        if not path.is_file():
            raise FileNotFoundError("missing feature-resolution resource: {}".format(path))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    direct_fields = set(schema["$defs"]["outputField"]["enum"])
    output_mapping = schema["$defs"]["outputMapping"]
    mapped_reference = output_mapping["properties"]["output_field"].get("$ref")
    if (
        direct_fields != ALLOWED_OUTPUT_FIELDS
        or mapped_reference != "#/$defs/outputField"
    ):
        raise RuntimeError(
            "Feature-resolution schema output fields do not match the executor contract"
        )
    return (
        prompt_path.read_text(encoding="utf-8").strip(),
        schema,
        version_path.read_text(encoding="utf-8").strip(),
    )


def _bind_resource_contract_to_schema(
    schema: Mapping[str, Any],
    resources: Sequence[ResolutionResource],
) -> Mapping[str, Any]:
    """Bind each resource ID to only its own literal field contract."""
    bounded = copy.deepcopy(dict(schema))
    variants = bounded["properties"]["recipe"]["properties"][
        "resolution_steps"
    ]["items"]["anyOf"]
    non_resource_variants = [
        variant
        for variant in variants
        if "resource_id" not in variant["properties"]
    ]
    resource_variants = [
        variant
        for variant in variants
        if "resource_id" in variant["properties"]
    ]
    if not resources:
        bounded["properties"]["recipe"]["properties"][
            "resolution_steps"
        ]["items"]["anyOf"] = non_resource_variants
        return bounded

    expanded_variants = list(non_resource_variants)
    all_resource_fields = sorted(
        {
            str(column)
            for resource in resources
            for column in resource.table.columns
        }
    )
    for base_variant in resource_variants:
        for resource in resources:
            resource_fields = sorted(map(str, resource.table.columns))
            if not resource_fields:
                continue
            variant = copy.deepcopy(base_variant)
            properties = variant["properties"]
            properties["resource_id"] = {
                "type": "string",
                "const": resource.resource_id,
            }
            if "resource_field" in properties:
                properties["resource_field"]["enum"] = resource_fields
            if "output_mappings" in properties:
                output_mapping = copy.deepcopy(
                    bounded["$defs"]["outputMapping"]
                )
                output_mapping["properties"]["resource_field"][
                    "enum"
                ] = resource_fields
                properties["output_mappings"]["items"] = output_mapping
            if "coordinate_fields" in properties:
                coordinates = copy.deepcopy(
                    bounded["$defs"]["coordinateFields"]
                )
                coordinate_properties = coordinates["properties"]
                for field_name in (
                    "resource_chrom",
                    "resource_start",
                    "resource_end",
                ):
                    coordinate_properties[field_name]["enum"] = resource_fields
                coordinate_properties["resource_strand"]["enum"] = [
                    None,
                    *resource_fields,
                ]
                properties["coordinate_fields"] = coordinates
            expanded_variants.append(variant)
    bounded["properties"]["recipe"]["properties"]["resolution_steps"][
        "items"
    ]["anyOf"] = expanded_variants
    bounded["$defs"]["outputMapping"]["properties"]["resource_field"][
        "enum"
    ] = all_resource_fields
    return bounded


def _resource_lookup(resources: Sequence[ResolutionResource]) -> Dict[str, ResolutionResource]:
    return {resource.resource_id: resource for resource in resources}


def _format_resource_fields(resource: ResolutionResource) -> str:
    fields = sorted(map(str, resource.table.columns))
    return ", ".join(fields) if fields else "<none>"


def _any_resource_value_matches(
    resources: Sequence[ResolutionResource],
    *,
    equals: Optional[str] = None,
    substrings: Sequence[str] = (),
) -> bool:
    """Scan every resource table cell for an exact or substring match.

    Vectorized per column with an early exit on the first match, so this
    covers the whole table rather than a fixed row prefix.
    """
    for resource in resources:
        for column in resource.table.columns:
            values = resource.table[column].astype(str)
            if equals is not None and bool((values == equals).any()):
                return True
            for token in substrings:
                if bool(values.str.contains(token, regex=False).any()):
                    return True
    return False


def _best_recipe_pilots(
    payload: Mapping[str, Any],
    pilots: Sequence[Mapping[str, Any]],
) -> Tuple[Mapping[str, Any], ...]:
    selected_ids = set(map(str, payload.get("selected_pilot_ids", [])))
    selected = [
        pilot
        for pilot in pilots
        if str(pilot.get("pilot_id")) in selected_ids
    ]
    best_by_id: Dict[str, Mapping[str, Any]] = {}
    for step in payload.get("recipe", {}).get("resolution_steps", []):
        expected_operation = {
            "join_resource": "exact_join",
            "join_resource_by_position": "positional_alignment",
        }.get(step.get("operation"))
        if expected_operation is None:
            continue
        matching = [
            pilot
            for pilot in selected
            if pilot.get("operation") == expected_operation
            and pilot.get("resource_id") == step.get("resource_id")
            and pilot.get("resource_field") == step.get("resource_field")
        ]
        if not matching:
            continue
        best = max(
            matching,
            key=lambda pilot: (
                float(pilot.get("evidence_score", 0.0)),
                float(pilot.get("exact_match_rate", 0.0)),
                float(pilot.get("unique_mapping_rate", 0.0)),
                -float(pilot.get("ambiguous_mapping_rate", 1.0)),
            ),
        )
        best_by_id[str(best["pilot_id"])] = best
    return tuple(best_by_id.values())


def _calibrate_pilot_validation_rules(
    payload: Mapping[str, Any],
    pilots: Sequence[Mapping[str, Any]],
    declarations: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], Tuple[Mapping[str, Any], ...]]:
    """Derive validation rules from observed pilots and operation semantics."""
    used_pilots = _best_recipe_pilots(payload, pilots)
    calibrated = copy.deepcopy(dict(payload))
    recipe = calibrated["recipe"]
    rules = recipe["validation_rules"]
    pilot_ids = [str(pilot["pilot_id"]) for pilot in used_pilots]
    steps = recipe["resolution_steps"]
    declared_identity = any(
        step.get("operation") == "set_declared_identity" for step in steps
    )
    produced_fields = {
        str(mapping.get("output_field"))
        for step in steps
        for mapping in step.get("output_mappings", [])
    } | {
        str(step.get("output_field"))
        for step in steps
        if step.get("operation") == "derive_field"
    }
    canonical_label_expected = (
        recipe.get("annotation_representation") == "source_gene"
        and (
            "source_gene_symbol" in produced_fields
            or any(
                step.get("operation") == "set_declared_identity"
                and step.get("method") == "gene_symbol"
                for step in steps
            )
        )
    ) or (
        recipe.get("annotation_representation") == "target_ortholog"
        and "target_gene_symbol" in produced_fields
    )
    if used_pilots:
        exact_rate = min(
            float(pilot["exact_match_rate"]) for pilot in used_pilots
        )
        unique_rate = min(
            float(pilot["unique_mapping_rate"]) for pilot in used_pilots
        )
        ambiguity_rate = max(
            float(pilot["ambiguous_mapping_rate"]) for pilot in used_pilots
        )
        annotation_rate = unique_rate if canonical_label_expected else 0.0
    elif declared_identity:
        exact_rate = 1.0
        unique_rate = 1.0
        ambiguity_rate = 0.0
        annotation_rate = 1.0 if canonical_label_expected else 0.0
    else:
        exact_rate = 0.0
        unique_rate = 0.0
        ambiguity_rate = 0.0
        annotation_rate = 0.0
    derived_values = {
        "minimum_exact_match_rate": exact_rate,
        "minimum_unique_mapping_rate": unique_rate,
        "minimum_annotation_eligible_rate": annotation_rate,
        "maximum_ambiguous_mapping_rate": ambiguity_rate,
        "require_taxon_consistency": bool(
            declarations.get("source_taxon_id")
            or declarations.get("species")
            or recipe.get("source_taxon_id")
        ),
        "require_assembly_consistency": bool(
            recipe.get("input_feature_level") in {"transcript", "peak"}
        ),
    }
    adjustments = []
    for rule_name, effective_value in derived_values.items():
        requested_value = rules[rule_name]
        if requested_value == effective_value:
            continue
        rules[rule_name] = effective_value
        adjustments.append(
            {
                "rule": rule_name,
                "requested_value": requested_value,
                "effective_value": effective_value,
                "basis_pilot_ids": pilot_ids,
                "reason": (
                    "Derived from selected pilot observations and declared "
                    "operation semantics; planner-authored thresholds are not used."
                ),
            }
        )
    return calibrated, tuple(adjustments)


def validate_resolution_plan(
    payload: Mapping[str, Any],
    *,
    declarations: Mapping[str, Any],
    resources: Sequence[ResolutionResource],
    pilots: Sequence[Mapping[str, Any]],
) -> None:
    hypotheses = payload.get("hypotheses")
    if not isinstance(hypotheses, list) or not hypotheses:
        raise ValueError("planner must provide at least one hypothesis")
    hypothesis_ids = [item.get("hypothesis_id") for item in hypotheses]
    if len(hypothesis_ids) != len(set(hypothesis_ids)):
        raise ValueError("hypothesis IDs must be unique")
    selected = payload.get("selected_hypothesis_id")
    if selected is not None and selected not in hypothesis_ids:
        raise ValueError("selected_hypothesis_id is absent from hypotheses")
    selected_hypothesis = next(
        (
            item
            for item in hypotheses
            if item.get("hypothesis_id") == selected
        ),
        None,
    )
    pilot_lookup = {str(row["pilot_id"]): row for row in pilots}
    unknown_pilots = [
        value for value in payload.get("selected_pilot_ids", [])
        if value not in pilot_lookup
    ]
    if unknown_pilots:
        raise ValueError("planner selected unknown pilot IDs")
    recipe = payload["recipe"]
    selected_pilots = [
        pilot_lookup[value] for value in payload.get("selected_pilot_ids", [])
    ]
    rules = recipe["validation_rules"]
    declared_taxon = declarations.get("source_taxon_id")
    if (
        declared_taxon is not None
        and recipe.get("source_taxon_id") is not None
        and int(recipe["source_taxon_id"]) != int(declared_taxon)
    ):
        raise ValueError("recipe taxon conflicts with explicit declaration")
    if declared_taxon is None and recipe.get("source_taxon_id") is not None:
        taxon_text = str(recipe["source_taxon_id"])
        taxon_supported = _any_resource_value_matches(
            resources,
            equals=taxon_text,
            substrings=(
                "NCBITaxon:{}".format(taxon_text),
                "taxon_id={}".format(taxon_text),
            ),
        )
        if not taxon_supported:
            raise ValueError(
                "recipe taxon lacks explicit declaration or resource evidence"
            )
    declared_assembly = _safe_text(declarations.get("source_assembly"))
    if (
        declared_assembly
        and _safe_text(recipe.get("source_assembly"))
        and recipe["source_assembly"] != declared_assembly
    ):
        raise ValueError("recipe assembly conflicts with explicit declaration")
    if not declared_assembly and _safe_text(recipe.get("source_assembly")):
        assembly = str(recipe["source_assembly"])
        assembly_supported = _any_resource_value_matches(
            resources, substrings=(assembly,)
        )
        if not assembly_supported:
            raise ValueError(
                "recipe assembly lacks explicit declaration or resource evidence"
            )

    resource_lookup = _resource_lookup(resources)
    available_working_fields = {"original_feature_id"}
    field_origins = {"original_feature_id": "input"}
    used_pilots: Dict[str, Mapping[str, Any]] = {}
    for step in recipe["resolution_steps"]:
        operation = step["operation"]
        step_id = step.get("step_id")
        input_field = step.get("input_field")
        if operation in {"derive_field", "join_resource"} and (
            not isinstance(input_field, str) or not input_field.strip()
        ):
            raise ValueError(
                "step {!r} operation {!r} requires a non-empty input_field; "
                "available working fields: {}".format(
                    step_id,
                    operation,
                    ", ".join(sorted(available_working_fields)),
                )
            )
        if input_field is not None and input_field not in available_working_fields:
            raise ValueError(
                "step {!r} operation {!r} references unavailable input_field "
                "{!r}; available working fields: {}".format(
                    step_id,
                    operation,
                    input_field,
                    ", ".join(sorted(available_working_fields)),
                )
            )
        if operation == "derive_field":
            if step.get("output_field") not in ALLOWED_OUTPUT_FIELDS:
                raise ValueError(
                    "derive_field output {!r} is not allowed; use one of: {}".format(
                        step.get("output_field"),
                        ", ".join(sorted(ALLOWED_OUTPUT_FIELDS)),
                    )
                )
            if step.get("output_field") in {
                "source_gene_id",
                "source_gene_symbol",
                "source_transcript_id",
                "target_gene_id",
                "target_gene_symbol",
            } and not declarations.get("identifier_namespace"):
                raise ValueError(
                    "string parsing cannot create biological identity without an "
                    "explicit namespace/format declaration"
                )
            method = step.get("method")
            if method not in {
                "copy",
                "strip_terminal_version",
                "strip_terminal_integer_suffix",
                "regex_capture",
                "split_token",
            }:
                raise ValueError(
                    "step {!r} derive_field uses invalid method {!r}; allowed "
                    "methods: copy, strip_terminal_version, "
                    "strip_terminal_integer_suffix, regex_capture, "
                    "split_token".format(step_id, method)
                )
            if method == "regex_capture":
                if not step.get("pattern"):
                    raise ValueError(
                        "step {!r} regex_capture requires pattern".format(step_id)
                    )
                if step.get("capture_group") is None:
                    raise ValueError(
                        "step {!r} regex_capture requires capture_group".format(
                            step_id
                        )
                    )
                _validate_safe_regex(str(step["pattern"]))
            if method == "split_token" and (
                not isinstance(step.get("delimiter"), str)
                or not step["delimiter"]
            ):
                raise ValueError(
                    "step {!r} split_token requires a non-empty delimiter".format(
                        step_id
                    )
                )
            available_working_fields.add(str(step["output_field"]))
            field_origins[str(step["output_field"])] = "derived_string"
        elif operation == "set_declared_identity":
            if step.get("method") not in DECLARED_IDENTITY_METHOD_TO_OUTPUT_FIELD:
                raise ValueError(
                    "step {!r} set_declared_identity uses invalid method {!r}; "
                    "allowed methods: {}".format(
                        step_id,
                        step.get("method"),
                        ", ".join(DECLARED_IDENTITY_METHOD_TO_OUTPUT_FIELD),
                    )
                )
            declared_input_field = input_field or "original_feature_id"
            if declared_input_field not in available_working_fields:
                raise ValueError(
                    "step {!r} set_declared_identity references unavailable "
                    "input_field {!r}; available working fields: {}".format(
                        step_id,
                        declared_input_field,
                        ", ".join(sorted(available_working_fields)),
                    )
                )
            if not declarations.get("identifier_namespace"):
                hypothesis_text = str(
                    (selected_hypothesis or {}).get("feature_type", "")
                ).lower()
                method = str(step.get("method"))
                identity_supported = {
                    "gene_symbol": "symbol" in hypothesis_text,
                    "gene_id": (
                        "gene" in hypothesis_text
                        and any(
                            token in hypothesis_text
                            for token in ("id", "identifier", "ensembl")
                        )
                    ),
                    "transcript_id": "transcript" in hypothesis_text,
                    "non_gene": (
                        (selected_hypothesis or {}).get("feature_level")
                        in {"peak", "antibody", "control"}
                    ),
                }[method]
                if (
                    selected_hypothesis is None
                    or float(selected_hypothesis.get("confidence", 0.0)) < 0.90
                    or not identity_supported
                ):
                    raise ValueError(
                        "set_declared_identity without a namespace requires a "
                        "high-confidence selected collection-level identity hypothesis"
                    )
            output = DECLARED_IDENTITY_METHOD_TO_OUTPUT_FIELD[step["method"]]
            available_working_fields.add(output)
            field_origins[output] = "declaration"
        elif operation in {
            "join_resource",
            "join_resource_by_position",
            "interval_overlap_resource",
        }:
            resource_id = step.get("resource_id")
            if resource_id not in resource_lookup:
                raise ValueError(
                    "step {!r} references unknown resource {!r}; available "
                    "resources: {}".format(
                        step.get("step_id"),
                        resource_id,
                        ", ".join(sorted(resource_lookup)) or "<none>",
                    )
                )
            resource = resource_lookup[resource_id]
            if not step.get("output_mappings"):
                raise ValueError(
                    "step {!r} operation {!r} requires at least one "
                    "output_mapping".format(step_id, operation)
                )
            if operation in {
                "join_resource",
                "join_resource_by_position",
            } and step.get("resource_field") not in resource.table.columns:
                raise ValueError(
                    "step {!r} operation {!r} references absent field {!r} in "
                    "resource {!r}; available fields: {}. resource_field must "
                    "be copied exactly from that resource's manifest entry or "
                    "from a selected pilot for the same resource".format(
                        step.get("step_id"),
                        operation,
                        step.get("resource_field"),
                        resource_id,
                        _format_resource_fields(resource),
                    )
                )
            exact_pilots = [
                pilot
                for pilot in selected_pilots
                if pilot["operation"] == "exact_join"
                and pilot["resource_id"] == resource_id
                and pilot["resource_field"] == step.get("resource_field")
            ]
            if (
                operation == "join_resource"
                and not exact_pilots
                and field_origins.get(str(step.get("input_field"))) != "resource"
            ):
                raise ValueError(
                    "step {!r} join_resource requires a selected exact_join "
                    "pilot for resource {!r} field {!r}".format(
                        step_id,
                        resource_id,
                        step.get("resource_field"),
                    )
                )
            if operation == "join_resource" and exact_pilots:
                best_exact_pilot = max(
                    exact_pilots,
                    key=lambda pilot: (
                        float(pilot.get("evidence_score", 0.0)),
                        float(pilot["exact_match_rate"]),
                        float(pilot["unique_mapping_rate"]),
                        -float(pilot["ambiguous_mapping_rate"]),
                    ),
                )
                used_pilots[str(best_exact_pilot["pilot_id"])] = best_exact_pilot
            if operation == "join_resource_by_position":
                positional_pilots = [
                    pilot
                    for pilot in selected_pilots
                    if pilot["operation"] == "positional_alignment"
                    and pilot["resource_id"] == resource_id
                    and pilot["resource_field"] == step.get("resource_field")
                ]
                if not positional_pilots:
                    raise ValueError(
                        "join_resource_by_position requires a selected positional "
                        "pilot for the same resource field"
                    )
                best_positional_pilot = max(
                    positional_pilots,
                    key=lambda pilot: (
                        float(pilot.get("evidence_score", 0.0)),
                        float(pilot["exact_match_rate"]),
                    ),
                )
                used_pilots[
                    str(best_positional_pilot["pilot_id"])
                ] = best_positional_pilot
                if int(declarations.get("feature_count", -1)) != len(resource.table):
                    raise ValueError(
                        "join_resource_by_position requires identical collection lengths"
                    )
            if operation == "interval_overlap_resource":
                coordinate_fields = step.get("coordinate_fields")
                if not coordinate_fields:
                    raise ValueError(
                        "interval_overlap_resource requires coordinate_fields"
                    )
                for name in (
                    "input_chrom", "input_start", "input_end"
                ):
                    if coordinate_fields[name] not in available_working_fields:
                        raise ValueError(
                            "interval overlap references unavailable input coordinates"
                        )
                input_strand = coordinate_fields.get("input_strand")
                if (
                    input_strand is not None
                    and input_strand not in available_working_fields
                ):
                    raise ValueError(
                        "step {!r} interval overlap references unavailable "
                        "input_strand {!r}; available working fields: {}".format(
                            step_id,
                            input_strand,
                            ", ".join(sorted(available_working_fields)),
                        )
                    )
                for name in (
                    "resource_chrom", "resource_start", "resource_end"
                ):
                    if coordinate_fields[name] not in resource.table.columns:
                        raise ValueError(
                            "step {!r} interval coordinate {!r} references absent "
                            "field {!r} in resource {!r}; available fields: {}".format(
                                step.get("step_id"),
                                name,
                                coordinate_fields[name],
                                resource_id,
                                _format_resource_fields(resource),
                            )
                        )
                resource_strand = coordinate_fields.get("resource_strand")
                if (
                    resource_strand is not None
                    and resource_strand not in resource.table.columns
                ):
                    raise ValueError(
                        "step {!r} interval resource_strand references absent "
                        "field {!r} in resource {!r}; available fields: {}".format(
                            step_id,
                            resource_strand,
                            resource_id,
                            _format_resource_fields(resource),
                        )
                    )
            for mapping_index, mapping in enumerate(
                step.get("output_mappings", []), start=1
            ):
                if mapping["resource_field"] not in resource.table.columns:
                    raise ValueError(
                        "step {!r} output mapping {} references absent field {!r} "
                        "in resource {!r}; available fields: {}. "
                        "output_mappings.resource_field must remain the literal "
                        "resource column; only output_field uses a canonical "
                        "executor name".format(
                            step.get("step_id"),
                            mapping_index,
                            mapping["resource_field"],
                            resource_id,
                            _format_resource_fields(resource),
                        )
                    )
                if mapping["output_field"] not in ALLOWED_OUTPUT_FIELDS:
                    raise ValueError(
                        "output mapping field {!r} is not allowed; use one of: {}".format(
                            mapping["output_field"],
                            ", ".join(sorted(ALLOWED_OUTPUT_FIELDS)),
                        )
                    )
                available_working_fields.add(mapping["output_field"])
                field_origins[mapping["output_field"]] = "resource"
        elif operation == "mark_unresolved":
            continue
        else:
            raise ValueError("unsupported resolution operation")

    for pilot in used_pilots.values():
        comparisons = (
            (
                "exact_match_rate",
                float(pilot["exact_match_rate"]),
                ">=",
                "minimum_exact_match_rate",
                float(rules["minimum_exact_match_rate"]),
            ),
            (
                "unique_mapping_rate",
                float(pilot["unique_mapping_rate"]),
                ">=",
                "minimum_unique_mapping_rate",
                float(rules["minimum_unique_mapping_rate"]),
            ),
            (
                "ambiguous_mapping_rate",
                float(pilot["ambiguous_mapping_rate"]),
                "<=",
                "maximum_ambiguous_mapping_rate",
                float(rules["maximum_ambiguous_mapping_rate"]),
            ),
        )
        failures = [
            "{}={:.12g} must be {} {}={:.12g}".format(
                metric_name,
                observed,
                operator,
                threshold_name,
                threshold,
            )
            for metric_name, observed, operator, threshold_name, threshold in comparisons
            if (operator == ">=" and observed < threshold)
            or (operator == "<=" and observed > threshold)
        ]
        if failures:
            raise ValueError(
                "recipe uses pilot {!r} ({!r}, resource {!r}, field {!r}) but "
                "its validation thresholds are infeasible: {}. Select a pilot "
                "that meets the rules, remove unused pilot IDs, or set the "
                "thresholds no stricter than the supplied pilot metrics".format(
                    pilot["pilot_id"],
                    pilot["operation"],
                    pilot["resource_id"],
                    pilot["resource_field"],
                    "; ".join(failures),
                )
            )


def plan_feature_resolution(
    *,
    feature_profile: Mapping[str, Any],
    declarations: Mapping[str, Any],
    resource_manifest: Sequence[Mapping[str, Any]],
    resources: Sequence[ResolutionResource],
    pilot_results: Sequence[Mapping[str, Any]],
    client: Any,
    model_name: str,
    reasoning_effort: str = DEFAULT_ANNOTATION_REASONING_EFFORT,
    recipe_override: Optional[Mapping[str, Any]] = None,
    execution_review: Optional[Mapping[str, Any]] = None,
) -> Tuple[
    Mapping[str, Any],
    Tuple[str, ...],
    Tuple[str, ...],
    Tuple[Mapping[str, Any], ...],
]:
    """Generate or validate one dataset-specific declarative recipe."""
    prompt, schema, _ = load_feature_resolution_resources()
    schema = _bind_resource_contract_to_schema(schema, resources)
    if recipe_override is not None:
        payload = dict(recipe_override)
        if "recipe" not in payload:
            payload = {
                "hypotheses": [
                    {
                        "hypothesis_id": "hypothesis_1",
                        "feature_type": "user_supplied_recipe",
                        "feature_level": payload.get("input_feature_level", "unknown"),
                        "possible_origin": "validated recipe override",
                        "confidence": 1.0,
                        "required_evidence": [],
                    }
                ],
                "selected_hypothesis_id": "hypothesis_1",
                "workflow_hypothesis_confidence": 1.0,
                "selected_pilot_ids": [],
                "recipe": payload,
                "decision_reason": "User supplied a reusable declarative recipe.",
                "rejected_hypotheses": [],
                "required_resources": [],
            }
        payload = _canonicalize_declared_identity_methods(payload)
        _schema_validate(payload, schema)
        payload, rule_adjustments = _calibrate_pilot_validation_rules(
            payload, pilot_results, declarations
        )
        validate_resolution_plan(
            payload,
            declarations=declarations,
            resources=resources,
            pilots=pilot_results,
        )
        return payload, tuple(), tuple(), rule_adjustments

    planner_input = {
        "feature_profile": feature_profile,
        "explicit_declarations": declarations,
        "resource_manifest": list(resource_manifest),
        "valid_resource_field_contract": [
            {
                "resource_id": resource.resource_id,
                "resource_fields": list(map(str, resource.table.columns)),
            }
            for resource in resources
        ],
        "pilot_results": list(pilot_results),
        "valid_pilot_join_targets": [
            {
                "pilot_id": pilot["pilot_id"],
                "operation": pilot["operation"],
                "resource_id": pilot["resource_id"],
                "resource_field": pilot["resource_field"],
                "feature_transform": pilot["feature_transform"],
                "transform_recipe": pilot["transform_recipe"],
                "exact_match_rate": pilot["exact_match_rate"],
                "unique_mapping_rate": pilot["unique_mapping_rate"],
                "ambiguous_mapping_rate": pilot["ambiguous_mapping_rate"],
                "evidence_score": pilot["evidence_score"],
            }
            for pilot in pilot_results
        ],
        "available_operations": [
            "derive_field", "join_resource", "join_resource_by_position",
            "interval_overlap_resource",
            "set_declared_identity", "mark_unresolved",
        ],
        "operation_method_contracts": {
            "derive_field": {
                "allowed_methods": [
                    "copy",
                    "strip_terminal_version",
                    "strip_terminal_integer_suffix",
                    "regex_capture",
                    "split_token",
                ],
            },
            "set_declared_identity": {
                "allowed_methods": list(
                    DECLARED_IDENTITY_METHOD_TO_OUTPUT_FIELD
                ),
                "method_to_output_field": dict(
                    DECLARED_IDENTITY_METHOD_TO_OUTPUT_FIELD
                ),
            },
            "join_resource": {"method": None},
            "join_resource_by_position": {"method": None},
            "interval_overlap_resource": {"method": None},
            "mark_unresolved": {"method": None},
        },
        "allowed_output_fields": sorted(ALLOWED_OUTPUT_FIELDS),
        "working_field_contract": {
            "initial_available_fields": ["original_feature_id"],
            "field_producing_operations": {
                "derive_field": "adds its output_field",
                "set_declared_identity": (
                    "adds the canonical field its method maps to"
                ),
                "join_resource": "adds every output_mapping output_field",
                "join_resource_by_position": (
                    "adds every output_mapping output_field"
                ),
                "interval_overlap_resource": (
                    "adds every output_mapping output_field"
                ),
            },
            "rule": (
                "input_field may only name a field that is already available: "
                "original_feature_id, or a field produced by an earlier step. "
                "allowed_output_fields lists fields a step may produce, not "
                "fields available to consume. A pilot whose transform_recipe "
                "method is not 'copy' therefore requires an earlier "
                "derive_field step that applies that exact recipe and writes a "
                "working field, which the join then consumes as its "
                "input_field."
            ),
        },
        "instruction": (
            "Plan once at collection level. Do not map individual features in "
            "the response. Every resource_id/resource_field pair must occur "
            "verbatim in valid_resource_field_contract. For a pilot-backed "
            "join, copy resource_id and resource_field verbatim from "
            "valid_pilot_join_targets, and reproduce that pilot's "
            "transform_recipe as a preceding derive_field step whenever its "
            "method is not 'copy'; the join's input_field must be that "
            "derive_field's output_field."
        ),
        "execution_review": execution_review,
    }
    user_prompt = "FEATURE_RESOLUTION_PLANNING_INPUT\n{}".format(
        json.dumps(planner_input, ensure_ascii=False, allow_nan=False)
    )
    raw_responses: List[str] = []
    validation_errors: List[str] = []
    prior_raw = ""
    for attempt in range(3):
        active_prompt = user_prompt
        if attempt:
            active_prompt += (
                "\n\nREPAIR_REQUEST\nPrevious response invalid: {}. Return a corrected "
                "recipe using only supplied evidence. Previous response:\n{}"
            ).format(validation_errors[-1], prior_raw)
        raw = call_model(
            client,
            model_name=model_name,
            system_prompt=prompt,
            user_prompt=active_prompt,
            schema=schema,
            reasoning_effort=reasoning_effort,
        )
        prior_raw = raw
        raw_responses.append(raw)
        try:
            payload = json.loads(raw)
            payload = _canonicalize_declared_identity_methods(payload)
            _schema_validate(payload, schema)
            payload, rule_adjustments = _calibrate_pilot_validation_rules(
                payload, pilot_results, declarations
            )
            validate_resolution_plan(
                payload,
                declarations=declarations,
                resources=resources,
                pilots=pilot_results,
            )
            return (
                payload,
                tuple(raw_responses),
                tuple(validation_errors),
                rule_adjustments,
            )
        except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            validation_errors.append(str(error))
    raise RuntimeError(
        "Feature-resolution plan remained invalid after three attempts: {}".format(
            validation_errors[-1]
        )
    )


def _derive_values(series: pd.Series, step: Mapping[str, Any]) -> pd.Series:
    method = step["method"]
    values = series.astype(str)
    if method == "copy":
        return values
    if method == "strip_terminal_version":
        return values.str.replace(r"\.([0-9]+)$", "", regex=True)
    if method == "strip_terminal_integer_suffix":
        return values.str.replace(r"-([0-9]+)$", "", regex=True)
    if method == "split_token":
        delimiter = str(step["delimiter"])
        index = int(step.get("token_index") or 0)
        return values.map(
            lambda value: value.split(delimiter)[index].strip()
            if -len(value.split(delimiter)) <= index < len(value.split(delimiter))
            else ""
        )
    if method == "regex_capture":
        pattern = re.compile(str(step["pattern"]))
        group = step.get("capture_group")
        group = int(group) if isinstance(group, int) or str(group).isdigit() else str(group)
        return values.map(
            lambda value: (
                str(pattern.search(value).group(group)).strip()
                if pattern.search(value) is not None
                else ""
            )
        )
    raise ValueError("unsupported derive_field method")


def _apply_exact_join(
    working: pd.DataFrame,
    step: Mapping[str, Any],
    resource: ResolutionResource,
    evidence: List[List[Mapping[str, Any]]],
    ambiguous_candidates: List[List[Mapping[str, Any]]],
) -> None:
    input_field = str(step["input_field"])
    resource_field = str(step["resource_field"])
    table = resource.table.copy()
    table[resource_field] = table[resource_field].astype(str)
    groups = {key: group for key, group in table.groupby(resource_field, sort=False)}
    for row_index, input_value in enumerate(working[input_field].astype(str)):
        group = groups.get(input_value)
        if group is None:
            continue
        candidate_records = []
        for _, match in group.iterrows():
            candidate = {
                mapping["output_field"]: _safe_text(match[mapping["resource_field"]])
                for mapping in step["output_mappings"]
            }
            candidate_records.append(candidate)
        unique_candidates = [
            dict(item)
            for item in {
                tuple(sorted(candidate.items())) for candidate in candidate_records
            }
        ]
        evidence[row_index].append(
            {
                "step_id": step["step_id"],
                "operation": "exact_join",
                "resource_id": resource.resource_id,
                "resource_checksum": resource.checksum_sha256,
                "match_count": len(unique_candidates),
                "evidence_strength": resource.evidence_strength,
            }
        )
        if len(unique_candidates) == 1:
            for output_field, value in unique_candidates[0].items():
                if value is not None:
                    working.at[row_index, output_field] = value
        else:
            ambiguous_candidates[row_index].extend(unique_candidates)


def _apply_interval_overlap(
    working: pd.DataFrame,
    step: Mapping[str, Any],
    resource: ResolutionResource,
    evidence: List[List[Mapping[str, Any]]],
    ambiguous_candidates: List[List[Mapping[str, Any]]],
) -> None:
    fields = step["coordinate_fields"]
    table = resource.table.copy()
    for column in (fields["resource_start"], fields["resource_end"]):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table = table.dropna(subset=[fields["resource_chrom"], fields["resource_start"], fields["resource_end"]])
    by_chrom = {
        str(chrom): group for chrom, group in table.groupby(fields["resource_chrom"], sort=False)
    }
    for row_index, row in working.iterrows():
        chrom = _safe_text(row.get(fields["input_chrom"]))
        try:
            start = float(row.get(fields["input_start"]))
            end = float(row.get(fields["input_end"]))
        except (TypeError, ValueError):
            continue
        candidates = by_chrom.get(str(chrom))
        if candidates is None:
            continue
        overlap = candidates.loc[
            (candidates[fields["resource_start"]] <= end)
            & (candidates[fields["resource_end"]] >= start)
        ]
        if fields.get("input_strand") and fields.get("resource_strand"):
            strand = _safe_text(row.get(fields["input_strand"]))
            if strand:
                overlap = overlap.loc[overlap[fields["resource_strand"]].astype(str) == strand]
        candidate_records = [
            {
                mapping["output_field"]: _safe_text(match[mapping["resource_field"]])
                for mapping in step["output_mappings"]
            }
            for _, match in overlap.iterrows()
        ]
        unique_candidates = [
            dict(item)
            for item in {
                tuple(sorted(candidate.items())) for candidate in candidate_records
            }
        ]
        if not unique_candidates:
            continue
        evidence[row_index].append(
            {
                "step_id": step["step_id"],
                "operation": "interval_overlap",
                "resource_id": resource.resource_id,
                "resource_checksum": resource.checksum_sha256,
                "match_count": len(unique_candidates),
                "evidence_strength": min(resource.evidence_strength, 0.75),
            }
        )
        if len(unique_candidates) == 1:
            for output_field, value in unique_candidates[0].items():
                if value is not None:
                    working.at[row_index, output_field] = value
        else:
            ambiguous_candidates[row_index].extend(unique_candidates)


def _apply_positional_join(
    working: pd.DataFrame,
    step: Mapping[str, Any],
    resource: ResolutionResource,
    pilot: Mapping[str, Any],
    evidence: List[List[Mapping[str, Any]]],
) -> None:
    """Transfer resource fields only after collection-level positional validation."""
    if len(working) != len(resource.table):
        raise ValueError("positional join collection lengths changed after validation")
    agreement = float(pilot["exact_match_rate"])
    evidence_strength = min(resource.evidence_strength, 0.85) * agreement
    table = resource.table.reset_index(drop=True)
    for row_index, match in table.iterrows():
        mapped_count = 0
        for mapping in step["output_mappings"]:
            value = _safe_text(match[mapping["resource_field"]])
            if value is not None:
                working.at[row_index, mapping["output_field"]] = value
                mapped_count += 1
        evidence[row_index].append(
            {
                "step_id": step["step_id"],
                "operation": "positional_join",
                "resource_id": resource.resource_id,
                "resource_checksum": resource.checksum_sha256,
                "pilot_id": pilot["pilot_id"],
                "whole_collection_agreement": agreement,
                "match_count": 1 if mapped_count else 0,
                "evidence_strength": evidence_strength,
            }
        )


def execute_resolution_recipe(
    feature_ids: Sequence[str],
    *,
    planner_response: Mapping[str, Any],
    declarations: Mapping[str, Any],
    resources: Sequence[ResolutionResource],
    pilot_results: Sequence[Mapping[str, Any]] = (),
) -> pd.DataFrame:
    """Apply one validated recipe uniformly and retain every ambiguity."""
    validate_resolution_plan(
        planner_response,
        declarations=declarations,
        resources=resources,
        pilots=pilot_results,
    )
    recipe = planner_response["recipe"]
    working = pd.DataFrame(
        {
            "feature_position": np.arange(len(feature_ids), dtype=int),
            "original_feature_id": list(map(str, feature_ids)),
        }
    )
    for field in ALLOWED_OUTPUT_FIELDS:
        if field not in working:
            working[field] = None
    evidence: List[List[Mapping[str, Any]]] = [[] for _ in feature_ids]
    ambiguous: List[List[Mapping[str, Any]]] = [[] for _ in feature_ids]
    forced_unresolved = [False for _ in feature_ids]
    resource_lookup = _resource_lookup(resources)
    selected_pilot_ids = set(planner_response.get("selected_pilot_ids", []))
    selected_pilots = [
        pilot
        for pilot in pilot_results
        if pilot.get("pilot_id") in selected_pilot_ids
    ]

    for step in recipe["resolution_steps"]:
        operation = step["operation"]
        if operation == "derive_field":
            output_field = step["output_field"]
            input_field = step["input_field"]
            if input_field not in working.columns:
                raise ValueError(
                    "validated derive_field step {!r} has unavailable "
                    "input_field {!r}".format(step.get("step_id"), input_field)
                )
            input_values = working[input_field]
            working[output_field] = _derive_values(input_values, step)
            if step["method"] == "strip_terminal_version" and bool(
                step.get("preserve_version")
            ):
                versions = input_values.astype(str).str.extract(
                    r"\.([0-9]+)$", expand=False
                )
                working["original_feature_version"] = working[
                    "original_feature_version"
                ].where(
                    working["original_feature_version"].notna(), versions
                )
            for index, value in enumerate(working[output_field]):
                if _safe_text(value):
                    evidence[index].append(
                        {
                            "step_id": step["step_id"],
                            "operation": step["method"],
                            "evidence_strength": 0.65,
                        }
                    )
        elif operation == "set_declared_identity":
            input_field = str(step.get("input_field") or "original_feature_id")
            method = step["method"]
            output_field = DECLARED_IDENTITY_METHOD_TO_OUTPUT_FIELD[method]
            if method == "non_gene":
                working[output_field] = "non_gene"
            else:
                working[output_field] = working[input_field].astype(str)
            namespace_declared = bool(declarations.get("identifier_namespace"))
            for index in range(len(working)):
                evidence[index].append(
                    {
                        "step_id": step["step_id"],
                        "operation": (
                            "explicit_declaration"
                            if namespace_declared
                            else "collection_level_hypothesis"
                        ),
                        "declared_namespace": declarations.get("identifier_namespace"),
                        "evidence_strength": 0.80 if namespace_declared else 0.65,
                    }
                )
        elif operation == "join_resource":
            _apply_exact_join(
                working, step, resource_lookup[str(step["resource_id"])], evidence, ambiguous
            )
        elif operation == "join_resource_by_position":
            matching_pilot = next(
                (
                    pilot
                    for pilot in selected_pilots
                    if pilot.get("operation") == "positional_alignment"
                    and pilot.get("resource_id") == step.get("resource_id")
                    and pilot.get("resource_field") == step.get("resource_field")
                ),
                None,
            )
            if matching_pilot is None:
                raise ValueError(
                    "positional execution requires its validated pilot results"
                )
            _apply_positional_join(
                working,
                step,
                resource_lookup[str(step["resource_id"])],
                matching_pilot,
                evidence,
            )
        elif operation == "interval_overlap_resource":
            _apply_interval_overlap(
                working, step, resource_lookup[str(step["resource_id"])], evidence, ambiguous
            )
        elif operation == "mark_unresolved":
            forced_unresolved = [True for _ in feature_ids]

    working["source_taxon_id"] = working["source_taxon_id"].where(
        working["source_taxon_id"].notna(), recipe.get("source_taxon_id")
    )
    working["source_assembly"] = working["source_assembly"].where(
        working["source_assembly"].notna(), recipe.get("source_assembly")
    )
    working["feature_level"] = working["feature_level"].where(
        working["feature_level"].notna(), recipe.get("input_feature_level")
    )
    selected_hypothesis_id = planner_response.get("selected_hypothesis_id")
    selected_hypothesis = next(
        (
            hypothesis
            for hypothesis in planner_response.get("hypotheses", [])
            if hypothesis.get("hypothesis_id") == selected_hypothesis_id
        ),
        None,
    )
    selected_namespace = (
        selected_hypothesis.get("feature_type")
        if selected_hypothesis
        else declarations.get("identifier_namespace")
    )
    working["selected_namespace"] = working["selected_namespace"].where(
        working["selected_namespace"].notna(), selected_namespace
    )

    mapping_status = []
    mapping_confidence = []
    annotation_symbols = []
    mapping_types = []
    for index, row in working.iterrows():
        source_identity = (
            _safe_text(row.get("canonical_source_gene_id"))
            or _safe_text(row.get("source_gene_id"))
            or _safe_text(row.get("source_gene_symbol"))
            or _safe_text(row.get("parent_gene_id"))
        )
        target_identity = (
            _safe_text(row.get("target_gene_id"))
            or _safe_text(row.get("target_gene_symbol"))
        )
        representation = recipe["annotation_representation"]
        annotation_symbol = (
            _safe_text(row.get("target_gene_symbol"))
            if representation == "target_ortholog"
            else _safe_text(row.get("source_gene_symbol"))
            if representation == "source_gene"
            else None
        )
        if not _safe_text(row["original_feature_id"]):
            status = "invalid_feature"
        elif str(row.get("feature_level")) in {"non_gene", "control", "antibody", "peak"}:
            status = "non_gene_feature"
        elif forced_unresolved[index]:
            status = "unresolved"
        elif ambiguous[index]:
            status = "resolved_multiple"
        elif (
            recipe.get("source_taxon_id") is None
            and not _safe_text(declarations.get("species"))
        ):
            status = "unresolved_taxon"
        elif recipe.get("source_assembly") is None and recipe["input_feature_level"] in {"transcript", "peak"}:
            status = "unresolved_assembly"
        elif source_identity or target_identity:
            status = "resolved_unique"
        else:
            status = "unresolved"
        strengths = [float(item.get("evidence_strength", 0.0)) for item in evidence[index]]
        confidence = max(strengths, default=0.0)
        if status == "resolved_multiple":
            confidence = min(confidence, 0.30)
        elif status != "resolved_unique":
            confidence = 0.0
        if representation == "target_ortholog" and not _safe_text(row.get("homology_type")):
            annotation_symbol = None
            confidence = min(confidence, 0.30)
        mapping_status.append(status)
        mapping_confidence.append(confidence)
        annotation_symbols.append(annotation_symbol)
        mapping_types.append(
            "ortholog" if representation == "target_ortholog" and annotation_symbol
            else "resource" if any(item.get("resource_id") for item in evidence[index])
            else "collection_hypothesis" if any(
                item.get("operation") == "collection_level_hypothesis"
                for item in evidence[index]
            )
            else "declared" if strengths
            else "unresolved"
        )

    working["parsed_tokens"] = working.apply(
        lambda row: json.dumps(
            {
                field: _safe_text(row[field])
                for field in ALLOWED_OUTPUT_FIELDS
                if field in row and _safe_text(row[field])
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        axis=1,
    )
    candidate_namespaces = [
        str(hypothesis.get("feature_type"))
        for hypothesis in planner_response.get("hypotheses", [])
        if hypothesis.get("feature_type")
    ]
    working["candidate_namespaces"] = json.dumps(
        candidate_namespaces, ensure_ascii=False
    )
    working["ortholog_candidates"] = [
        json.dumps(candidates, ensure_ascii=False, sort_keys=True)
        for candidates in ambiguous
    ]
    working["selected_ortholog_id"] = working["target_gene_id"]
    working["mapping_status"] = mapping_status
    working["mapping_confidence"] = mapping_confidence
    working["workflow_hypothesis_confidence"] = float(
        planner_response["workflow_hypothesis_confidence"]
    )
    working["mapping_evidence"] = [
        json.dumps(items, ensure_ascii=False, sort_keys=True) for items in evidence
    ]
    working["ambiguity_reason"] = [
        "multiple evidence-supported candidates" if items else None
        for items in ambiguous
    ]
    working["annotation_symbol"] = annotation_symbols
    working["annotation_eligible"] = (
        (working["mapping_status"] == "resolved_unique")
        & working["annotation_symbol"].notna()
    )
    working["recipe_id"] = recipe["recipe_id"]
    working["mapping_type"] = mapping_types
    return _add_feature_evidence_contract(working)


def _compatibility_metadata(mapping: pd.DataFrame, species: Optional[str]) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "feature_position": mapping["feature_position"].astype(int),
            "original_gene_id": mapping["original_feature_id"].astype(str),
            "parsed_gene_id": mapping["parsed_feature_id"].map(_safe_text).fillna(
                mapping["original_feature_id"].astype(str)
            ),
            "source_symbol": mapping["source_gene_symbol"].map(_safe_text),
            "canonical_symbol": mapping["source_gene_symbol"].map(_safe_text),
            "ortholog_symbol": mapping["target_gene_symbol"].map(_safe_text),
            "annotation_symbol": mapping["annotation_symbol"].map(_safe_text),
            "species": species,
            "mapping_type": mapping["mapping_type"].astype(str),
            "mapping_confidence": mapping["mapping_confidence"].astype(float),
            "annotation_eligible": mapping["annotation_eligible"].astype(bool),
            "identity_value": mapping["identity_value"].astype(str),
            "identity_type": mapping["identity_type"].astype(str),
            "identity_status": mapping["identity_status"].astype(str),
            "identity_confidence": mapping["identity_confidence"].astype(float),
            "identity_mapping_type": mapping["identity_mapping_type"].astype(str),
            "canonical_annotation_symbol": mapping[
                "canonical_annotation_symbol"
            ].map(_safe_text),
            "source_feature_label": mapping["source_feature_label"].astype(str),
            "source_label_candidates": mapping["source_label_candidates"].astype(str),
            "source_label_candidate_count": mapping[
                "source_label_candidate_count"
            ].astype(int),
            "source_label_parse_rule": mapping[
                "source_label_parse_rule"
            ].astype(str),
            "source_label_status": mapping["source_label_status"].astype(str),
            "annotation_label": mapping["annotation_label"].astype(str),
            "annotation_label_source": mapping[
                "annotation_label_source"
            ].astype(str),
            "annotation_label_confidence": mapping[
                "annotation_label_confidence"
            ].astype(float),
            "annotation_usable_for_context": mapping[
                "annotation_usable_for_context"
            ].astype(bool),
            "annotation_usable_for_naming": mapping[
                "annotation_usable_for_naming"
            ].astype(bool),
            "direct_confidence_eligible": mapping[
                "direct_confidence_eligible"
            ].astype(bool),
            "inference_confidence_eligible": mapping[
                "inference_confidence_eligible"
            ].astype(bool),
            "feature_evidence_tier": mapping["feature_evidence_tier"].astype(str),
            "fallback_applied": mapping["fallback_applied"].astype(bool),
            "ambiguity_reason": mapping["ambiguity_reason"].map(_safe_text),
        }
    )
    def biological_gene_id(row: pd.Series) -> str:
        tier = str(row["feature_evidence_tier"])
        if tier == "A" and _safe_text(row.get("annotation_symbol")):
            return str(row["annotation_symbol"]).upper()
        if tier == "B1":
            try:
                candidates = json.loads(str(row["source_label_candidates"]))
            except json.JSONDecodeError:
                candidates = []
            if isinstance(candidates, list) and len(candidates) == 1:
                return "SOURCE_LABEL:{}".format(str(candidates[0]).upper())
        return "UNRESOLVED:{}".format(int(row["feature_position"]))

    table["biological_gene_id"] = table.apply(biological_gene_id, axis=1)
    sizes = table.groupby("biological_gene_id")["biological_gene_id"].transform("size")
    table["duplicate_group"] = np.where(
        (sizes > 1) & table["annotation_eligible"],
        table["biological_gene_id"],
        None,
    )
    return table


def _quality_control_report(mapping: pd.DataFrame, recipe: Mapping[str, Any]) -> Dict[str, Any]:
    status_counts = mapping["mapping_status"].value_counts().to_dict()
    total = len(mapping)
    unique = int(status_counts.get("resolved_unique", 0))
    multiple = int(status_counts.get("resolved_multiple", 0))
    unresolved = total - unique - multiple
    eligible = mapping["annotation_eligible"].astype(bool)
    context_eligible = mapping["annotation_usable_for_context"].astype(bool)
    direct_eligible = mapping["direct_confidence_eligible"].astype(bool)
    inference_eligible = mapping["inference_confidence_eligible"].astype(bool)
    ortholog = mapping["mapping_type"] == "ortholog"
    tier_counts = {
        str(key): int(value)
        for key, value in mapping["feature_evidence_tier"].value_counts().items()
    }
    reasons = mapping.loc[
        mapping["mapping_status"].isin(
            ["unresolved", "unresolved_taxon", "unresolved_assembly", "invalid_feature"]
        ),
        "mapping_status",
    ].value_counts().to_dict()
    return {
        "total_feature_count": total,
        "feature_level": recipe["input_feature_level"],
        "overall_resolution_rate": float((unique + multiple) / total),
        "unique_resolution_rate": float(unique / total),
        "ambiguity_rate": float(multiple / total),
        "unresolved_rate": float(unresolved / total),
        "annotation_eligible_rate": float(eligible.mean()),
        "context_usable_rate": float(context_eligible.mean()),
        "direct_confidence_eligible_rate": float(direct_eligible.mean()),
        "inference_confidence_eligible_rate": float(inference_eligible.mean()),
        "feature_evidence_tier_counts": tier_counts,
        "source_label_fallback_active": bool(
            mapping["feature_evidence_tier"].isin(["B1", "B2", "C"]).any()
        ),
        "gene_level_resolution_rate": float(
            (
                mapping["source_gene_id"].map(_safe_text).notna()
                | mapping["source_gene_symbol"].map(_safe_text).notna()
            ).mean()
        ),
        "transcript_to_parent_gene_resolution_rate": float(
            mapping["parent_gene_id"].map(_safe_text).notna().mean()
        ),
        "ortholog_coverage": float(ortholog.mean()),
        "non_gene_feature_fraction": float(
            (mapping["mapping_status"] == "non_gene_feature").mean()
        ),
        "marker_weighted_resolution_rate": None,
        "source_taxon_id": recipe.get("source_taxon_id"),
        "source_assembly": recipe.get("source_assembly"),
        "unresolved_reasons": {str(key): int(value) for key, value in reasons.items()},
        "rules_version": FEATURE_RESOLUTION_RULES_VERSION,
    }


def _recipe_validation_report(
    qc: Mapping[str, Any],
    recipe: Mapping[str, Any],
) -> Dict[str, Any]:
    rules = recipe["validation_rules"]
    checks = {
        "minimum_exact_match_rate": (
            float(qc["overall_resolution_rate"])
            >= float(rules["minimum_exact_match_rate"])
        ),
        "minimum_unique_mapping_rate": (
            float(qc["unique_resolution_rate"])
            >= float(rules["minimum_unique_mapping_rate"])
        ),
        "minimum_annotation_eligible_rate": (
            float(qc["annotation_eligible_rate"])
            >= float(rules["minimum_annotation_eligible_rate"])
        ),
        "maximum_ambiguous_mapping_rate": (
            float(qc["ambiguity_rate"])
            <= float(rules["maximum_ambiguous_mapping_rate"])
        ),
        "taxon_consistency": (
            not bool(rules["require_taxon_consistency"])
            or qc.get("taxon_state") in {"verified", "user_declared"}
        ),
        "assembly_consistency": (
            not bool(rules["require_assembly_consistency"])
            or bool(_safe_text(recipe.get("source_assembly")))
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
    }


def resolve_features(
    feature_ids: Sequence[str],
    *,
    species: Optional[str],
    source_taxon_id: Optional[int],
    source_assembly: Optional[str],
    identifier_namespace: Optional[str],
    feature_level: Optional[str],
    resource_paths: Sequence[PathLike] = (),
    recipe_json: Optional[PathLike] = None,
    model_name: Optional[str] = None,
    reasoning_effort: str = DEFAULT_ANNOTATION_REASONING_EFFORT,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    client: Any = None,
) -> FeatureResolutionResult:
    """Profile, plan, execute, and audit one dataset-level resolution workflow."""
    feature_profile = profile_feature_collection(feature_ids)
    resources, resource_manifest = load_resolution_resources(resource_paths)
    pilots = pilot_resolution_workflows(feature_ids, feature_profile, resources)
    declarations = {
        "feature_count": len(feature_ids),
        "species": _safe_text(species),
        "source_taxon_id": source_taxon_id,
        "source_assembly": _safe_text(source_assembly),
        "identifier_namespace": _safe_text(identifier_namespace),
        "feature_level": _safe_text(feature_level),
    }
    recipe_override = None
    if recipe_json is not None:
        path = Path(recipe_json).expanduser()
        if not path.is_file():
            raise FileNotFoundError("feature_resolution_recipe_json not found: {}".format(path))
        saved = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(saved, dict) and "planner_response" in saved:
            expected_resources = {
                str(item.get("resource_id")): str(item.get("checksum_sha256"))
                for item in saved.get("resource_manifest", [])
                if item.get("resource_id")
            }
            observed_resources = {
                str(item.get("resource_id")): str(item.get("checksum_sha256"))
                for item in resource_manifest
                if item.get("resource_id")
            }
            if expected_resources != observed_resources:
                raise ValueError(
                    "Reusable feature-resolution recipe resource checksums do not match"
                )
            recipe_override = saved["planner_response"]
        else:
            recipe_override = saved
    active_client = client
    if recipe_override is None:
        active_client = active_client or create_openai_client(api_key, api_base)
    (
        planner_response,
        raw,
        validation_errors,
        rule_adjustments,
    ) = plan_feature_resolution(
        feature_profile=feature_profile,
        declarations=declarations,
        resource_manifest=resource_manifest,
        resources=resources,
        pilot_results=pilots,
        client=active_client,
        model_name=model_name or DEFAULT_ANNOTATION_MODEL,
        reasoning_effort=reasoning_effort,
        recipe_override=recipe_override,
    )
    mapping = execute_resolution_recipe(
        feature_ids,
        planner_response=planner_response,
        declarations=declarations,
        resources=resources,
        pilot_results=pilots,
    )
    qc = _quality_control_report(mapping, planner_response["recipe"])
    qc["taxon_state"] = (
        "verified"
        if planner_response["recipe"].get("source_taxon_id") is not None
        else "user_declared"
        if declarations.get("species")
        else "unknown"
    )
    execution_validation = _recipe_validation_report(
        qc, planner_response["recipe"]
    )
    direct_evidence_missing = (
        float(qc["direct_confidence_eligible_rate"]) == 0.0
    )
    execution_replan_used = False
    if (
        (not execution_validation["passed"] or direct_evidence_missing)
        and recipe_override is None
    ):
        execution_replan_used = True
        (
            revised_response,
            revised_raw,
            revised_errors,
            revised_adjustments,
        ) = plan_feature_resolution(
            feature_profile=feature_profile,
            declarations=declarations,
            resource_manifest=resource_manifest,
            resources=resources,
            pilot_results=pilots,
            client=active_client,
            model_name=model_name or DEFAULT_ANNOTATION_MODEL,
            reasoning_effort=reasoning_effort,
            execution_review={
                "previous_plan": planner_response,
                "execution_quality_control": qc,
                "execution_validation": execution_validation,
                "direct_evidence_missing": direct_evidence_missing,
                "feature_evidence_tier_counts": qc.get(
                    "feature_evidence_tier_counts", {}
                ),
                "representative_unresolved_source_labels": mapping.loc[
                    ~mapping["direct_confidence_eligible"].astype(bool),
                    "source_feature_label",
                ].astype(str).head(20).tolist(),
                "instruction": (
                    "Revise the recipe only if supported evidence can fix the failed "
                    "checks or recover direct biological labels. Do not invent "
                    "feature mappings. If direct labels remain unavailable, preserve "
                    "the exact feature identities for restricted inference."
                ),
            },
        )
        planner_response = revised_response
        raw = tuple(list(raw) + list(revised_raw))
        validation_errors = tuple(
            list(validation_errors) + list(revised_errors)
        )
        rule_adjustments = tuple(
            list(rule_adjustments) + list(revised_adjustments)
        )
        mapping = execute_resolution_recipe(
            feature_ids,
            planner_response=planner_response,
            declarations=declarations,
            resources=resources,
            pilot_results=pilots,
        )
        qc = _quality_control_report(mapping, planner_response["recipe"])
        qc["taxon_state"] = (
            "verified"
            if planner_response["recipe"].get("source_taxon_id") is not None
            else "user_declared"
            if declarations.get("species")
            else "unknown"
        )
        execution_validation = _recipe_validation_report(
            qc, planner_response["recipe"]
        )
    if not execution_validation["passed"]:
        structural_failures = set(execution_validation["failed_checks"]) - {
            "minimum_annotation_eligible_rate"
        }
        if structural_failures:
            raise FeatureAnnotationEvidenceError(
                "Feature-resolution execution failed structural validation: {}".format(
                    ", ".join(sorted(structural_failures))
                )
            )
    if not bool(mapping["annotation_usable_for_context"].astype(bool).any()):
        raise FeatureAnnotationEvidenceError(
            "Feature resolution produced no serializable feature evidence"
        )
    qc = {
        **qc,
        "recipe_execution_validation": execution_validation,
        "execution_replan_used": execution_replan_used,
    }
    compatibility = _compatibility_metadata(mapping, species)
    return FeatureResolutionResult(
        mapping_table=mapping,
        compatibility_metadata=compatibility,
        feature_profile=feature_profile,
        resource_manifest=resource_manifest,
        pilot_results=pilots,
        planner_response=planner_response,
        recipe=planner_response["recipe"],
        qc_report=qc,
        raw_planner_responses=raw,
        planner_validation_errors=validation_errors,
        planner_rule_adjustments=rule_adjustments,
    )


def replan_feature_resolution(
    feature_ids: Sequence[str],
    *,
    previous_result: FeatureResolutionResult,
    species: Optional[str],
    source_taxon_id: Optional[int],
    source_assembly: Optional[str],
    identifier_namespace: Optional[str],
    feature_level: Optional[str],
    resource_paths: Sequence[PathLike],
    marker_feature_labels: Sequence[str],
    model_name: Optional[str] = None,
    reasoning_effort: str = DEFAULT_ANNOTATION_REASONING_EFFORT,
    client: Any = None,
) -> FeatureResolutionResult:
    """Use the one allowed execution replan with marker-level evidence loss."""
    if bool(previous_result.qc_report.get("execution_replan_used", False)):
        raise ValueError("The feature-resolution execution replan was already used")
    feature_profile = profile_feature_collection(feature_ids)
    resources, resource_manifest = load_resolution_resources(resource_paths)
    if tuple(resource_manifest) != tuple(previous_result.resource_manifest):
        raise ValueError(
            "Feature-resolution resources changed before marker-guided replan"
        )
    pilots = pilot_resolution_workflows(feature_ids, feature_profile, resources)
    declarations = {
        "feature_count": len(feature_ids),
        "species": _safe_text(species),
        "source_taxon_id": source_taxon_id,
        "source_assembly": _safe_text(source_assembly),
        "identifier_namespace": _safe_text(identifier_namespace),
        "feature_level": _safe_text(feature_level),
    }
    active_client = client or create_openai_client(None, None)
    planner_response, raw, errors, adjustments = plan_feature_resolution(
        feature_profile=feature_profile,
        declarations=declarations,
        resource_manifest=resource_manifest,
        resources=resources,
        pilot_results=pilots,
        client=active_client,
        model_name=model_name or DEFAULT_ANNOTATION_MODEL,
        reasoning_effort=reasoning_effort,
        execution_review={
            "previous_plan": previous_result.planner_response,
            "execution_quality_control": previous_result.qc_report,
            "marker_level_direct_evidence_missing": True,
            "representative_marker_feature_labels": list(
                dict.fromkeys(map(str, marker_feature_labels))
            )[:50],
            "instruction": (
                "The marker payload contains valid features but no Tier A/B1 "
                "direct labels. Revise the recipe only when supplied resource "
                "evidence can recover direct biological labels. Never invent a "
                "mapping. Preserve unresolved features for Tier B2/C inference."
            ),
        },
    )
    mapping = execute_resolution_recipe(
        feature_ids,
        planner_response=planner_response,
        declarations=declarations,
        resources=resources,
        pilot_results=pilots,
    )
    qc = _quality_control_report(mapping, planner_response["recipe"])
    qc["taxon_state"] = (
        "verified"
        if planner_response["recipe"].get("source_taxon_id") is not None
        else "user_declared"
        if declarations.get("species")
        else "unknown"
    )
    execution_validation = _recipe_validation_report(qc, planner_response["recipe"])
    structural_failures = set(execution_validation["failed_checks"]) - {
        "minimum_annotation_eligible_rate"
    }
    if structural_failures:
        raise FeatureAnnotationEvidenceError(
            "Marker-guided feature replan failed structural validation: {}".format(
                ", ".join(sorted(structural_failures))
            )
        )
    if not bool(mapping["annotation_usable_for_context"].astype(bool).any()):
        raise FeatureAnnotationEvidenceError(
            "Marker-guided feature replan produced no serializable feature evidence"
        )
    qc = {
        **qc,
        "recipe_execution_validation": execution_validation,
        "execution_replan_used": True,
        "marker_evidence_guided_replan": True,
    }
    return FeatureResolutionResult(
        mapping_table=mapping,
        compatibility_metadata=_compatibility_metadata(mapping, species),
        feature_profile=feature_profile,
        resource_manifest=resource_manifest,
        pilot_results=pilots,
        planner_response=planner_response,
        recipe=planner_response["recipe"],
        qc_report=qc,
        raw_planner_responses=tuple(
            list(previous_result.raw_planner_responses) + list(raw)
        ),
        planner_validation_errors=tuple(
            list(previous_result.planner_validation_errors) + list(errors)
        ),
        planner_rule_adjustments=tuple(
            list(previous_result.planner_rule_adjustments) + list(adjustments)
        ),
    )


__all__ = [
    "FEATURE_EVIDENCE_TIERS",
    "FEATURE_RESOLUTION_RULES_VERSION",
    "FeatureAnnotationEvidenceError",
    "FeatureResolutionResult",
    "execute_resolution_recipe",
    "discover_supporting_files",
    "load_resolution_resources",
    "pilot_resolution_workflows",
    "plan_feature_resolution",
    "profile_feature_collection",
    "replan_feature_resolution",
    "resolve_features",
    "validate_resolution_plan",
]
