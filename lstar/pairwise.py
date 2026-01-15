"""Pairwise LLM-based model comparisons and ranking."""

import os
import time
import json
import base64
import re
import csv
import logging
from pathlib import Path
from itertools import combinations
from typing import Tuple, Dict, Optional, List, Literal, Sequence, Union
from collections import defaultdict
import hashlib

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from lstar.config import (
    DEFAULT_MODEL_NAME,
    DEFAULT_PAIRWISE_TEMPERATURE,
    DEFAULT_PAIRWISE_REASONING_EFFORT,
    DEFAULT_OUTPUT_DIR,
    PAIRWISE_SUBDIR,
    RANKING_CSV_NAME,
    DEFAULT_HE_BASENAME,
    DEFAULT_REPS,
    DEFAULT_TOP_K,
)
from lstar.io_utils import append_jsonl

logger = logging.getLogger(__name__)


def file_to_data_url(path: Path) -> str:
    """Convert image file to data URL.
    
    Supports multiple formats: png, jpg, jpeg
    """
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    
    path_lower = str(path).lower()
    if path_lower.endswith(".png"):
        mime = "image/png"
    elif path_lower.endswith((".jpg", ".jpeg")):
        mime = "image/jpeg"
    else:
        # Default to jpeg for unknown extensions
        mime = "image/jpeg"
        logger.warning(f"Unknown image format for {path}, defaulting to image/jpeg")
    
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def discover_models(
    image_dir: Path,
    he_basename: str = DEFAULT_HE_BASENAME,
) -> Tuple[Optional[Path], Dict[str, Path]]:
    """
    Discover H&E image and model output images in a directory.
    
    Returns:
        Tuple of (he_path, model_images) where:
        - he_path: Path to H&E image if exists, else None
        - model_images: Dict mapping model_id -> Path
    """
    logger.info(f"Discovering models in: {image_dir}")
    
    img_dir = Path(image_dir)
    if not img_dir.is_dir():
        raise NotADirectoryError(f"Image directory not found: {image_dir}")
    
    # Supported image extensions (same for H&E and model images)
    image_extensions = [".png", ".jpg", ".jpeg"]
    
    # Try to find H&E image with different extensions
    he_path = None
    he_basename_str = str(he_basename)
    
    if "." in he_basename_str:
        # Treat as a full filename with extension
        candidate = img_dir / he_basename_str
        if candidate.exists():
            he_path = candidate
            logger.info(f"Found H&E image: {he_path}")
        else:
            logger.warning(f"H&E image not found at {candidate}, will proceed without it")
    else:
        # Treat as basename without extension; try multiple extensions
        for ext in image_extensions:
            candidate = img_dir / f"{he_basename_str}{ext}"
            if candidate.exists():
                he_path = candidate
                logger.info(f"Found H&E image: {he_path}")
                break
        
        if he_path is None:
            logger.warning(
                f"H&E image not found for basename '{he_basename_str}' "
                f"with supported extensions: {image_extensions}; will proceed without it"
            )
    
    # Discover model images
    model_images: Dict[str, Path] = {}
    for ext in image_extensions:
        for img_file in img_dir.glob(f"*{ext}"):
            # Skip the H&E image by comparing full path resolution
            if he_path is not None and img_file.resolve() == he_path.resolve():
                continue
            
            model_id = img_file.stem
            # If we already found this model with a different extension, skip
            # (prioritize png > jpg/jpeg)
            if model_id in model_images:
                existing_ext = model_images[model_id].suffix.lower()
                current_ext = img_file.suffix.lower()
                priority = {".png": 0, ".jpg": 1, ".jpeg": 1}
                if priority.get(current_ext, 99) >= priority.get(existing_ext, 99):
                    continue
            model_images[model_id] = img_file
    
    logger.info(f"Found {len(model_images)} model images: {list(model_images.keys())}")
    
    if len(model_images) < 2:
        raise ValueError(f"Need at least 2 model images, found {len(model_images)}")
    
    return he_path, model_images


def build_pairwise_messages(
    he_url: Optional[str],
    img1_url: str,
    img2_url: str,
    simple_mode: bool = True,
    dataset_name: str = "the dataset",
) -> list:
    """Build messages for pairwise comparison."""
    if simple_mode:
        system_content = (
            "You are an expert model evaluator for spatial transcriptomics layer identification. "
            "Always start with EXACTLY one word: 'first' or 'second', then provide two short paragraphs "
            "in the form: First Model: reasoning Second Model: reasoning"
        )
        
        messages = [{"role": "system", "content": system_content}]
        
        if he_url:
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"The slices belong to {dataset_name}. Based on the information, please compare the model performance of identifying the layers of the slice provided in the next few messages."
                    },
                    {"type": "image_url", "image_url": {"url": he_url}},
                ],
            })
            
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "These two pictures are two identification results from two different models on this slice. Please compare which model performed better. Start your answer with EXACTLY ONE WORD: either 'first' or 'second'; then give a brief but structured justification in one paragraph for each model in the format of 'First Model: [reasoning] Second Model: [reasoning]', highlighting the reasons for your choice."
                    },
                    {"type": "image_url", "image_url": {"url": img1_url}},
                    {"type": "image_url", "image_url": {"url": img2_url}},
                ],
            })
        else:
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"The slices belong to {dataset_name}. Based on the information, please compare the model performance of identifying the layers of the slice provided in the next few messages."
                    }
                ],
            })
            
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "These two pictures are two identification results from two different models on this slice. Please compare which model performed better. Start your answer with EXACTLY ONE WORD: either 'first' or 'second'; then give a brief but structured justification in one paragraph for each model in the format of 'First Model: [reasoning] Second Model: [reasoning]', highlighting the reasons for your choice."
                    },
                    {"type": "image_url", "image_url": {"url": img1_url}},
                    {"type": "image_url", "image_url": {"url": img2_url}},
                ],
            })
        
        return messages
    
    else:
        # Complex mode
        system_content = (
            "You are an expert model evaluator for spatial transcriptomics layer identification. "
            "Your comparison MUST prioritize biological plausibility based on the H&E image (if H&E image is provided). "
            "\n\n"
            "CRITICAL BIAS WARNING: Do NOT prefer a model just because its boundaries are 'smoother' or 'cleaner'. "
            "Biological structures are complex. 'Fragmented' or 'patchy' clusters are GOOD if they "
            "accurately reflect structures seen in the H&E image (if H&E image is provided) (e.g., mixed cell populations, sparse layers). "
            "A smooth boundary that incorrectly cuts through a clear H&E layer is BAD. "
            "Your choice must be defensible by the H&E reference (if H&E reference is provided), not by visual aesthetics. "
            "\n\n"
            "OUTPUT FORMAT: Always start with EXACTLY one word: 'first' or 'second', then provide two short paragraphs in the form: "
            "'First Model: reasoning Second Model: reasoning'. "
            "\n\n"
            "BREVITY REQUIREMENT: Keep your response under 200 words total. Be concise but precise in your reasoning."
        )
        
        messages = [{"role": "system", "content": system_content}]
        
        if he_url:
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Based on the information, please compare the model performance of identifying the layers of the slice provided in the next few messages."
                    },
                    {"type": "image_url", "image_url": {"url": he_url}},
                ],
            })
            
            user_text = (
                "These two pictures are two identification results from two different models on this slice. "
                "Please compare which model performed better based *only* on the H&E reference. "
                "\n\n"
                "**Reminder:** Do not penalize 'fragmented' clusters if they match the biology in the H&E slide (if H&E slide is provided). "
                "Prioritize accuracy over 'smoothness'. "
                "\n\n"
                "Start your answer with EXACTLY ONE WORD: either 'first' or 'second'; then give a brief but "
                "structured justification in one paragraph for each model in the format of "
                "'First Model: [reasoning] Second Model: [reasoning]', highlighting the reasons for your choice."
                "\n\n"
                "IMPORTANT: Keep your total response under 200 words. Be concise."
            )
        else:
            user_text = (
                "These two pictures are two identification results from two different models on this slice. "
                "Please compare which model performed better. "
                "\n\n"
                "Start your answer with EXACTLY ONE WORD: either 'first' or 'second'; then give a brief but "
                "structured justification in one paragraph for each model in the format of "
                "'First Model: [reasoning] Second Model: [reasoning]', highlighting the reasons for your choice."
                "\n\n"
                "IMPORTANT: Keep your total response under 200 words. Be concise."
            )
        
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": img1_url}},
                {"type": "image_url", "image_url": {"url": img2_url}},
            ],
        })
        
        return messages


def messages_to_responses_input(messages: list) -> list:
    """Convert messages to OpenAI responses API input format."""
    converted = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, str):
            converted.append({
                "role": role,
                "content": [
                    {"type": "input_text", "text": content}
                ],
            })
        elif isinstance(content, list):
            new_content = []
            for part in content:
                if part.get("type") == "text":
                    new_content.append({"type": "input_text", "text": part.get("text", "")})
                elif part.get("type") == "image_url":
                    image_url_obj = part.get("image_url", {})
                    if isinstance(image_url_obj, dict):
                        url_string = image_url_obj.get("url", "")
                    else:
                        url_string = image_url_obj
                    new_content.append({"type": "input_image", "image_url": url_string})
            converted.append({"role": role, "content": new_content})
    return converted


def ask_llm_with_retries(
    client: OpenAI,
    messages: list,
    model_name: str,
    temperature: float,
    reasoning_effort: str,
    max_retries: int = 3,
    backoff_base: float = 1.5,
) -> str:
    """Call LLM API with retry logic."""
    last_err = None
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.debug(f"Calling {model_name} (attempt {attempt}) with temperature={temperature}, reasoning_effort={reasoning_effort}")
            
            input_messages = messages_to_responses_input(messages)
            
            api_params = {
                "model": model_name,
                "input": input_messages,
                "temperature": temperature,
            }
            
            if reasoning_effort:
                api_params["reasoning"] = {"effort": reasoning_effort}
            
            resp = client.responses.create(**api_params)
            out = resp.output_text if hasattr(resp, 'output_text') else ""
            
            if hasattr(resp, 'usage') and hasattr(resp.usage, 'output_tokens_details'):
                reasoning_tokens = resp.usage.output_tokens_details.reasoning_tokens
                logger.debug(f"Received response (chars={len(out)}, reasoning_tokens={reasoning_tokens})")
            else:
                logger.debug(f"Received response (chars={len(out)})")
            
            return out or ""
        except Exception as e:
            last_err = e
            wait_s = backoff_base ** attempt
            logger.warning(f"API error: {e}. Backing off {wait_s:.1f}s ...")
            time.sleep(wait_s)
    
    logger.error("Exhausted retries.")
    raise last_err


def parse_choice(text: str) -> Optional[str]:
    """Parse 'first' or 'second' from LLM response."""
    m = re.search(r"\b(first|second)\b", (text or "").strip(), flags=re.IGNORECASE)
    return m.group(1).lower() if m else None


def get_pairwise_cache_key(
    model1: str,
    model2: str,
    rep: int,
    simple_mode: bool,
    model_name: str,
    temperature: float,
    reasoning_effort: str,
) -> str:
    """Generate cache key for a pairwise comparison."""
    key_parts = [
        model1,
        model2,
        str(rep),
        "simple" if simple_mode else "complex",
        model_name,
        str(temperature),
        reasoning_effort,
    ]
    key_str = "_".join(key_parts)
    return hashlib.md5(key_str.encode()).hexdigest()


def get_cached_pairwise_result(cache_path: Path) -> Optional[dict]:
    """Read cached pairwise result if it exists."""
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read cache file {cache_path}: {e}")
            return None
    return None


def cache_pairwise_result(cache_path: Path, result: dict) -> None:
    """Write pairwise result to cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def run_single_pairwise_comparison(
    client: OpenAI,
    he_url: Optional[str],
    model1_id: str,
    model2_id: str,
    img1_url: str,
    img2_url: str,
    rep: int,
    simple_mode: bool,
    model_name: str,
    temperature: float,
    reasoning_effort: str,
    dataset_name: str,
    cache_path: Optional[Path] = None,
    force_rerun: bool = False,
    disable_cache: bool = False,
    in_memory_cache: Optional[Dict[str, dict]] = None,
) -> dict:
    """Run a single pairwise comparison, using cache if available."""
    # Check in-memory cache first
    if in_memory_cache is not None and not force_rerun:
        cache_key = get_pairwise_cache_key(
            model1_id, model2_id, rep, simple_mode,
            model_name, temperature, reasoning_effort
        )
        if cache_key in in_memory_cache:
            return in_memory_cache[cache_key]
    
    # Check file cache (only if not disabled)
    if cache_path and not disable_cache and not force_rerun:
        cached = get_cached_pairwise_result(cache_path)
        if cached is not None:
            # Store in in-memory cache if available
            if in_memory_cache is not None:
                cache_key = get_pairwise_cache_key(
                    model1_id, model2_id, rep, simple_mode,
                    model_name, temperature, reasoning_effort
                )
                in_memory_cache[cache_key] = cached
            return cached
    
    # Run LLM call
    messages = build_pairwise_messages(he_url, img1_url, img2_url, simple_mode, dataset_name)
    out_text = ask_llm_with_retries(client, messages, model_name, temperature, reasoning_effort)
    choice = parse_choice(out_text)
    
    result = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model1_label": model1_id,
        "model2_label": model2_id,
        "gpt_choice": choice,
        "gpt_output": out_text,
        "repetition": rep,
    }
    
    # Store in in-memory cache if available
    if in_memory_cache is not None:
        cache_key = get_pairwise_cache_key(
            model1_id, model2_id, rep, simple_mode,
            model_name, temperature, reasoning_effort
        )
        in_memory_cache[cache_key] = result
    
    # Cache result to file (only if not disabled)
    if cache_path and not disable_cache:
        cache_pairwise_result(cache_path, result)
    
    return result


def compute_winning_rates(
    pairwise_files: List[Path],
    output_csv: Path,
) -> pd.DataFrame:
    """Compute winning rates from pairwise comparison JSONL files."""
    logger.info("Computing winning rates from pairwise comparisons")
    
    games = defaultdict(int)
    wins = defaultdict(int)
    losses = defaultdict(int)
    ties = defaultdict(int)
    points = defaultdict(float)
    
    total_rows = 0
    used_rows = 0
    
    for jsonl_path in pairwise_files:
        if not jsonl_path.exists():
            logger.warning(f"Pairwise file not found: {jsonl_path}")
            continue
        
        logger.debug(f"Processing: {jsonl_path}")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                total_rows += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                m1 = obj.get("model1_label")
                m2 = obj.get("model2_label")
                if not m1 or not m2:
                    continue
                
                choice = (obj.get("gpt_choice") or "").strip().lower()
                
                games[m1] += 1
                games[m2] += 1
                
                if choice == "first":
                    wins[m1] += 1
                    losses[m2] += 1
                    points[m1] += 1.0
                    used_rows += 1
                elif choice == "second":
                    wins[m2] += 1
                    losses[m1] += 1
                    points[m2] += 1.0
                    used_rows += 1
                else:
                    ties[m1] += 1
                    ties[m2] += 1
                    points[m1] += 0.5
                    points[m2] += 0.5
                    used_rows += 1
    
    logger.info(f"Rows total: {total_rows} | rows used: {used_rows}")
    
    models = sorted(games.keys())
    table = []
    for m in models:
        g = games[m]
        w = wins[m]
        l = losses[m]
        t = ties[m]
        pts = points[m]
        win_rate = (pts / g) if g else 0.0
        table.append({
            "model": m,
            "games": g,
            "wins": w,
            "losses": l,
            "ties": t,
            "points": round(pts, 4),
            "win_rate": round(win_rate, 6),
        })
    
    ranked = sorted(
        table,
        key=lambda r: (r["win_rate"], r["points"], r["games"], r["model"]),
        reverse=True,
    )
    
    logger.info("\n=== Ranking by Win Rate (win=1, tie=0.5) ===")
    for i, r in enumerate(ranked, 1):
        logger.info(f"{i:2d}. {r['model']:20s}  win_rate={r['win_rate']:.3f}  "
                   f"points={r['points']:.3f}  games={r['games']:3d}  "
                   f"W-L-T={r['wins']}-{r['losses']}-{r['ties']}")
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(ranked)
    df.to_csv(output_csv, index=False)
    
    logger.info(f"Saved winning rates to: {output_csv}")
    
    return df


def select_top_models(
    ranking_df: pd.DataFrame,
    top_k: int = DEFAULT_TOP_K,
    mode: Literal["fixed", "elbow"] = "fixed",
) -> List[str]:
    """Select top K models from ranking DataFrame."""
    logger.info(f"Selecting top models (mode={mode}, top_k={top_k})")
    
    # Get win_rate column
    if "win_rate" in ranking_df.columns:
        win_col = "win_rate"
    elif "winning_rate" in ranking_df.columns:
        win_col = "winning_rate"
    elif "WinningRate" in ranking_df.columns:
        win_col = "WinningRate"
    else:
        win_col = ranking_df.columns[-1]
    
    # Get model column
    if "model" in ranking_df.columns:
        model_col = "model"
    elif "Model" in ranking_df.columns:
        model_col = "Model"
    else:
        model_col = ranking_df.columns[0]
    
    models = ranking_df.sort_values(win_col, ascending=False)
    model_list = models[model_col].tolist()
    win_rates = models[win_col].tolist()
    
    if mode == "fixed":
        k = min(top_k, len(model_list))
        top_models = model_list[:k]
        logger.info(f"Selected top {k} models (fixed): {top_models}")
        return top_models
    
    elif mode == "elbow":
        if len(model_list) <= 2:
            k = len(model_list)
        else:
            drops = []
            for i in range(len(model_list) - 1):
                drop = win_rates[i] - win_rates[i+1]
                drops.append((i+1, drop))
            
            elbow_idx = max(drops, key=lambda x: x[1])[0]
            k = max(3, min(elbow_idx, top_k))
        
        top_models = model_list[:k]
        logger.info(f"Selected top {k} models (elbow): {top_models}")
        return top_models
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


def run_pairwise_comparisons(
    image_dir: Union[str, Path],
    dataset_name: str,
    *,
    reps: int = DEFAULT_REPS,
    top_k: int = DEFAULT_TOP_K,
    top_k_mode: Literal["fixed", "elbow"] = "fixed",
    he_basename: str = DEFAULT_HE_BASENAME,
    skip_pairwise: bool = False,
    simple_mode: bool = True,
    output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
    force_rerun: bool = False,
    model_name: str = DEFAULT_MODEL_NAME,
    pairwise_temperature: float = DEFAULT_PAIRWISE_TEMPERATURE,
    pairwise_reasoning_effort: Literal["minimal", "medium", "high"] = DEFAULT_PAIRWISE_REASONING_EFFORT,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    disable_cache: bool = False,
) -> Tuple[pd.DataFrame, Path, Path]:
    """
    Run LLM-based pairwise comparisons to rank models.
    
    Parameters
    ----------
    image_dir : str or Path
        Directory containing H&E and model output images
    
    dataset_name : str
        Background information about the dataset name provided in LLM prompts.
        This helps the LLM understand the context of the spatial transcriptomics data.
        Example: "DLPFC (from 10X Visium Human Brain)" or "STARmap Mouse Ventricular Cardiomyocytes"
    
    reps : int
        Number of pairwise comparison repetitions
    
    top_k : int
        Number of top models to consider (used for ranking)
    
    top_k_mode : {"fixed", "elbow"}
        Mode for top-k selection in ranking
    
    he_basename : str
        Basename of H&E image file
    
    skip_pairwise : bool
        If True, skip pairwise comparisons and reuse existing ranking
    
    simple_mode : bool
        If True, use simple prompts for pairwise comparisons
    
    output_dir : str or Path
        Output directory for pairwise results and ranking CSV
    
    force_rerun : bool
        If True, ignore cache and recompute all comparisons
    
    model_name : str
        LLM model name to use
    
    pairwise_temperature : float
        Temperature for pairwise LLM calls
    
    pairwise_reasoning_effort : {"minimal", "medium", "high"}
        Reasoning effort level for pairwise comparisons
    
    api_key : str, optional
        OpenAI API key (or set OPENAI_API_KEY environment variable)
    
    api_base : str, optional
        Custom API base URL
    
    disable_cache : bool, default False
        If True, disable writing individual cache JSON files.
        Results will still be written to JSONL files, but individual
        cache_{hash}.json files will not be created.
    
    Returns
    -------
    ranking_df : pd.DataFrame
        DataFrame version of the ranking CSV
    pairwise_dir : Path
        Directory where pairwise JSON/JSONL files are written
    ranking_csv_path : Path
        Path to the ranking CSV file
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    pairwise_dir = output_dir / PAIRWISE_SUBDIR
    ranking_csv_path = output_dir / RANKING_CSV_NAME
    
    # Set up API client
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not provided. Use api_key parameter or set OPENAI_API_KEY environment variable.")
    
    client = OpenAI(api_key=api_key, base_url=api_base) if api_base else OpenAI(api_key=api_key)
    
    # Discover models
    he_path, model_images = discover_models(image_dir, he_basename)
    
    # Encode images once
    logger.info("Encoding images...")
    he_url = file_to_data_url(he_path) if he_path else None
    model_urls = {model_id: file_to_data_url(path) for model_id, path in model_images.items()}
    
    model_ids = sorted(model_images.keys())
    pairs = list(combinations(model_ids, 2))
    logger.info(f"Total pairs to compare: {len(pairs)}")
    
    if skip_pairwise:
        if not ranking_csv_path.exists():
            raise FileNotFoundError(
                f"Cannot skip pairwise: ranking CSV not found at {ranking_csv_path}. "
                f"Please run with skip_pairwise=False first to generate pairwise results."
            )
        logger.info("Skipping pairwise comparisons, using existing ranking")
        ranking_df = pd.read_csv(ranking_csv_path)
        return ranking_df, pairwise_dir, ranking_csv_path
    
    # Run pairwise comparisons with caching
    pairwise_dir.mkdir(parents=True, exist_ok=True)
    jsonl_files = []
    
    # In-memory cache to avoid repeated file I/O
    # Always use in-memory cache for performance, even when file cache is enabled
    in_memory_cache: Dict[str, dict] = {}
    
    # Overall progress bar for all repetitions
    total_comparisons = reps * len(pairs)
    overall_pbar = tqdm(
        total=total_comparisons,
        desc="Pairwise Comparisons",
        unit="comparison",
        position=0,
        leave=True
    )
    
    for rep in range(1, reps + 1):
        logger.info(f"\n=== Repetition {rep}/{reps} ===")
        
        jsonl_path = pairwise_dir / f"pairwise_results_rep{rep:02d}.jsonl"
        if force_rerun and jsonl_path.exists():
            jsonl_path.unlink()
        
        # Progress bar for current repetition
        rep_pbar = tqdm(
            total=len(pairs),
            desc=f"Round {rep}/{reps}",
            unit="pair",
            position=1,
            leave=False
        )
        
        for idx, (model1_id, model2_id) in enumerate(pairs, start=1):
            # Generate cache path (only if not disabled)
            cache_path = None
            if not disable_cache:
                cache_key = get_pairwise_cache_key(
                    model1_id, model2_id, rep, simple_mode,
                    model_name, pairwise_temperature, pairwise_reasoning_effort
                )
                cache_path = pairwise_dir / f"cache_{cache_key}.json"
            
            img1_url = model_urls[model1_id]
            img2_url = model_urls[model2_id]
            
            try:
                result = run_single_pairwise_comparison(
                    client, he_url, model1_id, model2_id, img1_url, img2_url,
                    rep, simple_mode, model_name, pairwise_temperature,
                    pairwise_reasoning_effort, dataset_name, cache_path, force_rerun,
                    disable_cache, in_memory_cache
                )
                
                # Append to JSONL
                append_jsonl(jsonl_path, result)
                
                # Update progress bars
                rep_pbar.set_postfix({"current": f"{model1_id} vs {model2_id}"})
                rep_pbar.update(1)
                overall_pbar.update(1)
                
            except Exception as e:
                logger.error(f"Error in pairwise comparison {model1_id} vs {model2_id}: {e}")
                err_row = {
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "model1_label": model1_id,
                    "model2_label": model2_id,
                    "gpt_choice": None,
                    "gpt_output": f"[API ERROR] {e}",
                    "repetition": rep,
                }
                append_jsonl(jsonl_path, err_row)
                rep_pbar.update(1)
                overall_pbar.update(1)
        
        rep_pbar.close()
        jsonl_files.append(jsonl_path)
        logger.info(f"Completed repetition {rep}, saved to: {jsonl_path}")
    
    overall_pbar.close()
    
    # Compute winning rates
    ranking_df = compute_winning_rates(jsonl_files, ranking_csv_path)
    
    return ranking_df, pairwise_dir, ranking_csv_path

