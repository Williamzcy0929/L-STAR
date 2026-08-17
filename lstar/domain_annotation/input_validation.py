"""Validation helpers for L-STAR assignment and consensus tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union


PathLike = Union[str, Path]

MANIFEST_FILENAME = "lstar_run_manifest.json"
CANONICAL_CONSENSUS_FILENAME = "L_STAR_consensus.csv"
CONSENSUS_COLUMN = "L-STAR"
MANIFEST_KEYS = (
    "id_col",
    "assignments_csv",
    "method_columns",
    "lstar_consensus_csv",
    "selected_models",
    "selected_model_to_assignment_column",
    "output_dir",
    "consensus_column",
    "selection_mode",
    "assignment_source_mode",
)


def _as_path(value: PathLike, field_name: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError(
            f"{field_name} must be a string or pathlib.Path, got "
            f"{type(value).__name__}."
        )
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{field_name} must not be empty.")
    return Path(value).expanduser()


def _require_file(path: Path, field_name: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{field_name} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{field_name} must be a file: {path}")
    return path.resolve()


def resolve_run_manifest_path(
    output_dir: PathLike,
    run_manifest: Optional[PathLike] = None,
) -> Optional[Path]:
    """Return the run-manifest path to read, or ``None`` when there is none.

    ``run_manifest`` names the manifest explicitly, which is required when the
    annotation writes somewhere other than the directory holding it — for
    example an ablation that keeps its artifacts in a subdirectory. It may name
    the file itself or the directory containing it, and it must exist. Without
    it, the manifest is read from ``output_dir`` when present, as before.
    """
    if run_manifest is not None:
        manifest_path = _as_path(run_manifest, "run_manifest")
        if manifest_path.is_dir():
            manifest_path = manifest_path / MANIFEST_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"run_manifest not found: {manifest_path}"
            )
        return manifest_path
    output_path = _as_path(output_dir, "output_dir")
    candidate = output_path / MANIFEST_FILENAME
    return candidate if candidate.exists() else None


def load_run_manifest(
    output_dir: PathLike,
    run_manifest: Optional[PathLike] = None,
) -> Dict[str, Any]:
    """Load the fixed L-STAR run-context keys when a manifest is present."""
    manifest_path = resolve_run_manifest_path(output_dir, run_manifest)
    if manifest_path is None:
        return {}
    if not manifest_path.is_file():
        raise ValueError(f"L-STAR run manifest must be a file: {manifest_path}")

    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read valid JSON from L-STAR run manifest "
            f"{manifest_path}: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(
            f"L-STAR run manifest must contain a JSON object: {manifest_path}"
        )
    return {key: payload[key] for key in MANIFEST_KEYS if key in payload}


def read_csv_header(csv_path: PathLike) -> Tuple[str, ...]:
    """Read and validate an unmangled assignment or consensus CSV header."""
    path = _require_file(_as_path(csv_path, "csv_path"), "CSV file")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise ValueError(f"CSV file is empty: {path}") from exc
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"Could not read CSV header from {path}: {exc}") from exc

    if not header:
        raise ValueError(f"CSV header is empty: {path}")
    empty_positions = [index + 1 for index, name in enumerate(header) if not name.strip()]
    if empty_positions:
        raise ValueError(
            f"CSV header in {path} contains empty column name(s) at "
            f"position(s): {empty_positions}."
        )

    positions: Dict[str, List[int]] = {}
    for index, name in enumerate(header):
        positions.setdefault(name, []).append(index + 1)
    duplicates = {
        name: indexes for name, indexes in positions.items() if len(indexes) > 1
    }
    if duplicates:
        details = ", ".join(
            f"{name!r} at positions {indexes}"
            for name, indexes in duplicates.items()
        )
        raise ValueError(f"CSV header in {path} has duplicate columns: {details}.")
    return tuple(header)


def _validate_method_column_list(
    values: Any,
    source_name: str,
) -> Tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{source_name} must be a non-empty sequence of strings.")
    columns = tuple(values)
    if not columns:
        raise ValueError(f"{source_name} must contain at least one method column.")
    invalid = [value for value in columns if not isinstance(value, str) or not value.strip()]
    if invalid:
        raise ValueError(
            f"{source_name} contains empty or non-string method names: {invalid!r}."
        )
    duplicates = list(dict.fromkeys(
        column for index, column in enumerate(columns) if column in columns[:index]
    ))
    if duplicates:
        raise ValueError(
            f"{source_name} contains duplicate method columns: {duplicates!r}."
        )
    return columns


def resolve_method_columns(
    assignments_header: Sequence[str],
    id_col: str,
    output_dir: PathLike,
    *,
    method_columns: Optional[Sequence[str]] = None,
    manifest: Optional[Mapping[str, Any]] = None,
) -> Tuple[str, ...]:
    """Resolve methods selected by consensus, with an explicit legacy path."""
    manifest_methods = None
    if manifest is not None and "method_columns" in manifest:
        manifest_methods = _validate_method_column_list(
            manifest["method_columns"], "manifest method_columns"
        )

    if manifest_methods is not None:
        if method_columns is not None:
            explicit_methods = _validate_method_column_list(
                method_columns, "method_columns"
            )
            if explicit_methods != manifest_methods:
                raise ValueError(
                    "method_columns must match the methods recorded as participating "
                    "in the L-STAR consensus run manifest. Omit method_columns to use "
                    "the recorded consensus methods automatically."
                )
        resolved = manifest_methods
    elif method_columns is not None:
        resolved = _validate_method_column_list(
            method_columns, "legacy method_columns"
        )
    else:
        raise ValueError(
            "Consensus methods could not be resolved automatically. Rerun L-STAR "
            f"to create {_as_path(output_dir, 'output_dir') / MANIFEST_FILENAME}. "
            "The public annotation pipeline does not accept method_columns overrides."
        )

    if id_col in resolved:
        raise ValueError(f"id_col {id_col!r} cannot also be a method column.")
    missing = [column for column in resolved if column not in assignments_header]
    if missing:
        raise ValueError(
            f"Resolved method column(s) {missing!r} are absent from assignments. "
            f"Available columns: {list(assignments_header)}."
        )
    return resolved


__all__ = [
    "CANONICAL_CONSENSUS_FILENAME",
    "CONSENSUS_COLUMN",
    "MANIFEST_FILENAME",
    "MANIFEST_KEYS",
    "load_run_manifest",
    "resolve_run_manifest_path",
    "read_csv_header",
    "resolve_method_columns",
]
