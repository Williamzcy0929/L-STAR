"""All-wise LLM comparison and position-score ranking."""

import json
import logging
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Union

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from lstar.config import (
    DEFAULT_HE_BASENAME,
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PAIRWISE_REASONING_EFFORT,
    DEFAULT_PAIRWISE_TEMPERATURE,
    DEFAULT_REPS,
    RANKING_CSV_NAME,
)
from lstar.io_utils import append_jsonl
from lstar.pairwise import (
    ask_llm_with_retries,
    discover_models,
    file_to_data_url,
)
from lstar.types import ComparisonResult


ALLWISE_SUBDIR = "allwise"

logger = logging.getLogger(__name__)


def build_comparison_labels(model_ids: Sequence[str]) -> Dict[str, str]:
    """Map stable internal comparison labels to model identifiers."""
    return {f"model_{index:02d}": model_id for index, model_id in enumerate(model_ids, 1)}


def build_all_way_messages(
    he_url: Optional[str],
    model_label_to_id: Dict[str, str],
    model_urls: Dict[str, str],
    simple_mode: bool = True,
    dataset_name: str = "the dataset",
) -> list:
    """Build one prompt containing every candidate method image."""
    if len(model_label_to_id) < 2:
        raise ValueError("Expected at least 2 model outputs for all-wise comparison")

    labels = list(model_label_to_id)
    rank_schema = json.dumps(labels)
    if simple_mode:
        system_content = (
            "You are an expert model evaluator for spatial transcriptomics layer "
            "identification. Compare all provided model outputs for the same tissue "
            "section and return a strict ranking."
        )
        criteria = (
            "Rank all provided models by performance for identifying biologically "
            "plausible spatial layers or domains. Return JSON only, with exactly this "
            f'schema: {{"rank": {rank_schema}, "reasoning": "brief explanation"}}. '
            "List the model labels from best to worst, do not use ties, and use every "
            "model label exactly once."
        )
    else:
        system_content = (
            "You are an expert model evaluator for spatial transcriptomics layer "
            "identification. Prioritize biological plausibility based on the H&E image "
            "when provided. Do not prefer a model merely because its boundaries are "
            "smoother or cleaner. Fragmented clusters can be correct when they reflect "
            "biological structure."
        )
        criteria = (
            "Rank all provided model outputs from best to worst using biological "
            "plausibility, layer alignment, spatial coherence, and avoidance of clear "
            "over- or under-segmentation. Return JSON only, with exactly this schema: "
            f'{{"rank": {rank_schema}, "reasoning": "brief explanation"}}. '
            "Do not use ties, and use every model label exactly once."
        )

    dataset_context = (
        f"The slices belong to {dataset_name}. Compare the model performance for "
        "identifying the spatial layers or domains in the images."
    )
    messages = [{"role": "system", "content": system_content}]
    context_content = [{"type": "text", "text": dataset_context}]
    if he_url:
        context_content.append({"type": "image_url", "image_url": {"url": he_url}})
    messages.append({"role": "user", "content": context_content})

    comparison_content = [{"type": "text", "text": criteria}]
    for label, model_id in model_label_to_id.items():
        comparison_content.extend(
            [
                {"type": "text", "text": label},
                {
                    "type": "image_url",
                    "image_url": {"url": model_urls[model_id]},
                },
            ]
        )
    messages.append({"role": "user", "content": comparison_content})
    return messages


def parse_rank(
    text: str,
    expected_labels: Sequence[str],
) -> Optional[List[str]]:
    """Parse a complete strict ranking, rejecting omissions and duplicates."""
    expected_labels = list(expected_labels)
    expected = set(expected_labels)
    if len(expected) != len(expected_labels):
        raise ValueError("Expected comparison labels must be unique")
    label_lookup = {label.lower(): label for label in expected_labels}

    raw = (text or "").strip()
    json_candidates = [raw]
    fenced = re.search(
        r"```(?:json)?\s*(.*?)```",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced:
        json_candidates.insert(0, fenced.group(1).strip())
    object_match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if object_match:
        json_candidates.insert(0, object_match.group(0))

    for candidate in json_candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        rank_values = parsed.get("rank") if isinstance(parsed, dict) else None
        if not isinstance(rank_values, list):
            continue
        normalized = [
            label_lookup.get(str(value).strip().lower(), str(value).strip())
            for value in rank_values
        ]
        if len(normalized) == len(expected_labels) and set(normalized) == expected:
            return normalized

    matches = re.findall(r"\bmodel_\d{2}\b", raw, flags=re.IGNORECASE)
    normalized_matches = [label_lookup.get(match.lower(), match) for match in matches]
    if len(normalized_matches) == len(expected_labels) and set(normalized_matches) == expected:
        return normalized_matches
    return None


def rank_scores_for_models(
    model_ids: Sequence[str],
    ranked_models: Sequence[str],
) -> Dict[str, float]:
    """Assign m-q points to each model at one-indexed rank position q."""
    model_ids = list(model_ids)
    ranked_models = list(ranked_models)
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("Model identifiers must be unique")
    if (
        len(ranked_models) != len(model_ids)
        or len(set(ranked_models)) != len(ranked_models)
        or set(ranked_models) != set(model_ids)
    ):
        raise ValueError("Rank must contain every model exactly once")

    model_count = len(model_ids)
    return {
        model_id: float(model_count - position)
        for position, model_id in enumerate(ranked_models, 1)
    }


def run_single_all_way_comparison(
    client: OpenAI,
    he_url: Optional[str],
    model_ids: Sequence[str],
    model_label_to_id: Dict[str, str],
    model_urls: Dict[str, str],
    rep: int,
    simple_mode: bool,
    model_name: str,
    temperature: float,
    reasoning_effort: str,
    dataset_name: str,
) -> dict:
    """Run and validate one all-wise LLM comparison."""
    messages = build_all_way_messages(
        he_url=he_url,
        model_label_to_id=model_label_to_id,
        model_urls=model_urls,
        simple_mode=simple_mode,
        dataset_name=dataset_name,
    )
    output_text = ask_llm_with_retries(
        client,
        messages,
        model_name,
        temperature,
        reasoning_effort,
    )
    rank_labels = parse_rank(output_text, list(model_label_to_id))
    if rank_labels is None:
        raise ValueError(
            "Could not parse a complete strict all-wise ranking from the LLM "
            f"response: {output_text}"
        )

    ranked_models = [model_label_to_id[label] for label in rank_labels]
    scores = rank_scores_for_models(model_ids, ranked_models)
    return {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_labels": list(model_ids),
        "comparison_labels": dict(model_label_to_id),
        "rank_labels": rank_labels,
        "ranked_models": ranked_models,
        "scores": scores,
        "gpt_output": output_text,
        "repetition": rep,
    }


def compute_allwise_ranking(
    result_files: Sequence[Path],
    output_csv: Path,
) -> pd.DataFrame:
    """Aggregate all-wise position scores and write a sortable ranking."""
    comparisons = defaultdict(int)
    points = defaultdict(float)
    expected_models = None

    for result_file in result_files:
        if not result_file.exists():
            raise FileNotFoundError(f"All-wise result file not found: {result_file}")
        with result_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                result = json.loads(line)
                model_ids = result.get("model_labels") or []
                ranked_models = result.get("ranked_models") or []
                row_scores = rank_scores_for_models(model_ids, ranked_models)
                if expected_models is None:
                    expected_models = set(model_ids)
                elif set(model_ids) != expected_models:
                    raise ValueError(f"Inconsistent model set in all-wise result: {result_file}")
                for model_id in model_ids:
                    comparisons[model_id] += 1
                    points[model_id] += row_scores[model_id]

    if not comparisons:
        raise ValueError("No usable all-wise comparison results were found")

    rows = []
    for model_id in sorted(comparisons):
        comparison_count = comparisons[model_id]
        point_count = points[model_id]
        rows.append(
            {
                "model": model_id,
                "comparisons": comparison_count,
                "points": round(point_count, 4),
                "score": round(point_count / comparison_count, 6),
            }
        )
    ranking_df = pd.DataFrame(rows).sort_values(
        ["score", "points", "model"],
        ascending=[False, False, False],
        kind="stable",
        ignore_index=True,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    ranking_df.to_csv(output_csv, index=False)
    logger.info("Saved all-wise ranking to: %s", output_csv)
    return ranking_df


def compute_all_way_winning_rates(
    result_files: Sequence[Path],
    output_csv: Path,
) -> pd.DataFrame:
    """Compatibility name for all-wise position-score aggregation."""
    return compute_allwise_ranking(result_files, output_csv)


def run_allwise_comparisons(
    image_dir: Union[str, Path],
    dataset_name: str,
    *,
    reps: int = DEFAULT_REPS,
    he_basename: str = DEFAULT_HE_BASENAME,
    skip_comparisons: bool = False,
    simple_mode: bool = True,
    output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
    force_rerun: bool = False,
    model_name: str = DEFAULT_MODEL_NAME,
    pairwise_temperature: float = DEFAULT_PAIRWISE_TEMPERATURE,
    pairwise_reasoning_effort: Literal[
        "minimal", "medium", "high"
    ] = DEFAULT_PAIRWISE_REASONING_EFFORT,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> ComparisonResult:
    """Run one simultaneous all-wise comparison per repetition."""
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    artifact_dir = output_dir / ALLWISE_SUBDIR
    ranking_csv_path = output_dir / RANKING_CSV_NAME
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if skip_comparisons:
        if not ranking_csv_path.exists():
            raise FileNotFoundError(
                "Cannot skip comparisons: ranking CSV not found at " f"{ranking_csv_path}"
            )
        ranking_df = pd.read_csv(ranking_csv_path)
        return ComparisonResult(
            ranking_df=ranking_df,
            ranking_csv_path=ranking_csv_path,
            artifact_dir=artifact_dir,
            score_column="score",
        )

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key not provided. Use api_key parameter or set "
            "OPENAI_API_KEY environment variable."
        )

    existing_results = sorted(artifact_dir.glob("allwise_results_rep*.jsonl"))
    if existing_results and not force_rerun:
        raise FileExistsError(
            f"Existing all-wise results found in {artifact_dir}. Use "
            "force_rerun=True to replace them or skip_comparisons=True to reuse "
            "ranking.csv."
        )
    if force_rerun:
        for result_file in existing_results:
            result_file.unlink()
        if ranking_csv_path.exists():
            ranking_csv_path.unlink()

    client = OpenAI(api_key=api_key, base_url=api_base) if api_base else OpenAI(api_key=api_key)
    he_path, model_images = discover_models(image_dir, he_basename)
    he_url = file_to_data_url(he_path) if he_path else None
    model_urls = {model_id: file_to_data_url(path) for model_id, path in model_images.items()}
    model_ids = sorted(model_images)
    model_label_to_id = build_comparison_labels(model_ids)

    result_files = []
    progress = tqdm(
        total=reps,
        desc="All-Wise Comparisons",
        unit="comparison",
        position=0,
        leave=True,
    )
    try:
        for rep in range(1, reps + 1):
            result_file = artifact_dir / f"allwise_results_rep{rep:02d}.jsonl"
            result = run_single_all_way_comparison(
                client=client,
                he_url=he_url,
                model_ids=model_ids,
                model_label_to_id=model_label_to_id,
                model_urls=model_urls,
                rep=rep,
                simple_mode=simple_mode,
                model_name=model_name,
                temperature=pairwise_temperature,
                reasoning_effort=pairwise_reasoning_effort,
                dataset_name=dataset_name,
            )
            append_jsonl(result_file, result)
            result_files.append(result_file)
            progress.update(1)
    finally:
        progress.close()

    ranking_df = compute_allwise_ranking(result_files, ranking_csv_path)
    return ComparisonResult(
        ranking_df=ranking_df,
        ranking_csv_path=ranking_csv_path,
        artifact_dir=artifact_dir,
        score_column="score",
    )
