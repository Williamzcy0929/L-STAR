"""Safe artifact writing and final annotation-output validation."""

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


_MANAGED_DOMAIN_ARTIFACT_PATTERNS = (
    # Still written every run in v3.
    re.compile(r"domain_markers/domain_[A-Za-z0-9._-]+_markers\.csv"),
    re.compile(r"evidence_cards/domain_[A-Za-z0-9._-]+\.json"),
    re.compile(r"cross_method_support/domain_[A-Za-z0-9._-]+\.json"),
    # Legacy cleanup only: the v3 single dataset-level call no longer writes
    # per-domain candidate, LLM-response, or confidence files, but a v3 run
    # over an existing v2 output directory must still be able to purge them
    # rather than leave them orphaned.
    re.compile(r"llm_responses/domain_[A-Za-z0-9._-]+\.json"),
    re.compile(r"domain_candidates/domain_[A-Za-z0-9._-]+\.json"),
    re.compile(r"confidence_components/domain_[A-Za-z0-9._-]+\.json"),
)
# Fixed-name (not per-domain) artifacts from the two-stage candidate/
# reconciliation flow that v3 no longer writes. Legacy cleanup only, same
# rationale as the second half of _MANAGED_DOMAIN_ARTIFACT_PATTERNS above.
LEGACY_TOP_LEVEL_ARTIFACTS = (
    "candidate_score_matrix.csv",
    "dataset_reconciliation.json",
    "annotation_reconciliation.json",
)


def domain_name_contains_annotation_confidence(name: str) -> bool:
    """Return whether a domain label encodes annotation confidence.

    High, medium, and low can be legitimate biological qualifiers, as in
    ``high-glycolytic`` or ``medium-sized bile duct``.  They are rejected only
    when they are explicitly presented as annotation confidence or as a
    detached confidence-style suffix.
    """
    normalized = str(name).strip()
    if re.search(r"\bconfidence\b", normalized, flags=re.I):
        return True
    if re.fullmatch(r"(?:high|medium|low)", normalized, flags=re.I):
        return True
    if re.search(
        r"(?:\(|\[)\s*(?:high|medium|low)\s*(?:\)|\])",
        normalized,
        flags=re.I,
    ):
        return True
    return bool(
        re.search(
            r"\s(?:-|:|\|)\s*(?:high|medium|low)\s*$",
            normalized,
            flags=re.I,
        )
    )


def domain_name_references_assignment_id(name: str, domain_id: str) -> bool:
    """Return whether a label explicitly uses its L-STAR assignment ID.

    Digits embedded in biological nomenclature, such as ``Krt18``, ``Hoxb9``,
    ``AP2``, or ``E8.5``, are not assignment-ID references.  Only explicit
    bookkeeping labels such as ``domain 18`` or ``cluster_18`` are rejected.
    """
    normalized_id = str(domain_id).strip()
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", normalized_id) is None:
        return False

    escaped_id = re.escape(normalized_id)
    numeric_end = r"(?![A-Za-z0-9.])"
    explicit_prefix = (
        r"(?<![A-Za-z0-9])"
        r"(?:l[\s-]*star(?:\s+domain)?|"
        r"domain(?:\s*[_-]?\s*id)?|"
        r"cluster(?:\s*[_-]?\s*id)?|assignment)"
        r"\s*(?:[#:=_-]\s*)?"
        + escaped_id
        + numeric_end
    )
    hash_reference = (
        r"(?<![A-Za-z0-9])#\s*" + escaped_id + numeric_end
    )
    id_first = (
        r"(?<![A-Za-z0-9.])"
        + escaped_id
        + r"\s+(?:domain|cluster)\b"
    )
    if re.search(explicit_prefix, str(name), flags=re.I):
        return True
    if re.search(hash_reference, str(name), flags=re.I):
        return True
    if re.search(id_first, str(name), flags=re.I):
        return True
    return re.fullmatch(
        r"\s*" + escaped_id + r"\s*",
        str(name),
        flags=re.I,
    ) is not None


def domain_artifact_stem(domain_id: str) -> str:
    """Create a filesystem-safe stem without changing the stored domain ID."""
    raw = str(domain_id)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-")
    if safe and safe == raw and safe not in {".", ".."}:
        return "domain_{}".format(safe)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
    return "domain_{}_{}".format(safe or "value", digest)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError("Object of type {} is not JSON serializable".format(type(value).__name__))


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON through a temporary file in the destination directory."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(destination.parent),
            prefix=".{}.".format(destination.name),
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(
                dict(payload),
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
                default=_json_default,
            )
            handle.write("\n")
        os.replace(str(temp_path), str(destination))
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def write_dataframe_atomic(path: Path, dataframe: pd.DataFrame) -> None:
    """Write a CSV atomically while preserving the DataFrame's row order."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=str(destination.parent),
            prefix=".{}.".format(destination.name),
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            dataframe.to_csv(handle, index=False)
        os.replace(str(temp_path), str(destination))
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def remove_stale_managed_artifacts(
    artifact_root: Path,
    active_relative_paths: Sequence[str],
) -> Sequence[str]:
    """Remove obsolete files owned by a preceding annotation run.

    Eligible paths are listed by the preceding annotation manifest and either
    match one of the fixed per-domain artifact patterns or are exactly one of
    ``LEGACY_TOP_LEVEL_ARTIFACTS``. No directory is ever removed, and
    arbitrary manifest paths are ignored.
    """
    root = Path(artifact_root)
    manifest_path = root / "annotation_manifest.json"
    if not manifest_path.is_file():
        return []

    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            previous_manifest = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return []
    if not isinstance(previous_manifest, dict):
        return []

    previous_paths = previous_manifest.get("managed_artifact_files")
    if not isinstance(previous_paths, list):
        return []

    active = set(active_relative_paths)
    removed = []
    resolved_root = root.resolve()
    for relative_path in previous_paths:
        if not isinstance(relative_path, str) or relative_path in active:
            continue
        if not (
            relative_path in LEGACY_TOP_LEVEL_ARTIFACTS
            or any(
                pattern.fullmatch(relative_path)
                for pattern in _MANAGED_DOMAIN_ARTIFACT_PATTERNS
            )
        ):
            continue
        target = root / relative_path
        try:
            target.parent.resolve().relative_to(resolved_root)
        except (OSError, ValueError):
            continue
        if target.parent.is_symlink():
            continue
        if target.is_dir():
            continue
        if target.exists() or target.is_symlink():
            target.unlink()
            removed.append(relative_path)
    return removed


def validate_final_assignment(
    dataframe: pd.DataFrame,
    *,
    id_col: str,
    expected_ids: Sequence[str],
    expected_lstar_labels: Sequence[str],
) -> None:
    """Enforce the primary three-column observation-assignment contract."""
    expected_columns = [id_col, "L-STAR", "domain_name"]
    if list(dataframe.columns) != expected_columns:
        raise ValueError(
            "Final assignment columns must be exactly {}".format(expected_columns)
        )
    if len(dataframe) != len(expected_ids):
        raise ValueError("Final assignment row count changed")
    if dataframe[id_col].astype(str).tolist() != list(map(str, expected_ids)):
        raise ValueError("Final identifier values or row order changed")
    if dataframe["L-STAR"].astype(str).tolist() != list(
        map(str, expected_lstar_labels)
    ):
        raise ValueError("Final L-STAR labels or row order changed")
    if dataframe[id_col].duplicated().any():
        raise ValueError("Final assignment contains duplicate identifiers")
    if dataframe["domain_name"].isna().any() or (
        dataframe["domain_name"].astype(str).str.strip() == ""
    ).any():
        raise ValueError("Every cell must have a non-empty domain_name")
    if (dataframe["domain_name"] == "Uncharacterized domain").any():
        raise ValueError("Uncharacterized domain is not allowed in final output")
    grouped = dataframe.groupby("L-STAR", sort=False, dropna=False)
    if (grouped["domain_name"].nunique(dropna=False) != 1).any():
        raise ValueError("Each L-STAR domain must map to exactly one domain_name")
    for domain_id, group in grouped:
        name = str(group["domain_name"].iloc[0])
        if domain_name_references_assignment_id(name, str(domain_id)):
            raise ValueError(
                "A domain_name explicitly references its L-STAR assignment ID: "
                "{!r}".format(name)
            )
        if domain_name_contains_annotation_confidence(name):
            raise ValueError(
                "A domain_name encodes annotation confidence: {!r}".format(name)
            )
