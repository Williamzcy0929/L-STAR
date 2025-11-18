import os
import sys
import time
import json
import base64
import re
import csv
import argparse
from pathlib import Path
from itertools import combinations
from typing import Tuple, Dict, Optional, List
from collections import defaultdict

from openai import OpenAI

# API configuration (will be set from arguments or environment)
API_KEY = None
MODEL_NAME = "gpt-5"
DATASET_NAME = "DATASET_NAME"

# ============================================================================
# LLM HYPERPARAMETERS
# ============================================================================

# Hyperparameters for pairwise comparisons
PAIRWISE_TEMPERATURE = 1.0
PAIRWISE_REASONING_EFFORT = "medium"

# Hyperparameters for second-round reasoning (HIGHER reasoning effort for careful analysis)
SECOND_ROUND_TEMPERATURE = 1.0
SECOND_ROUND_REASONING_EFFORT = "high"
# ============================================================================

client = None

def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def print_step(msg: str) -> None:
    print(f"[{now()}] {msg}")
    sys.stdout.flush()


def file_to_data_url(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    mime = "image/png" if str(path).lower().endswith(".png") else "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def discover_models(
    image_dir: str,
    he_basename: str = "he.png",
) -> Tuple[Optional[Path], Dict[str, Path]]:
    """
    Discover H&E image and model output images in a directory.
    
    Args:
        image_dir: Directory containing images
        he_basename: Filename of H&E image (default: "he.png")
    
    Returns:
        Tuple of (he_path, model_images) where:
        - he_path: Path to H&E image if exists, else None
        - model_images: Dict mapping model_id -> Path
    """
    print_step(f"Discovering models in: {image_dir}")
    
    img_dir = Path(image_dir)
    if not img_dir.is_dir():
        raise NotADirectoryError(f"Image directory not found: {image_dir}")
    
    he_path = img_dir / he_basename
    if not he_path.exists():
        print_step(f"Warning: H&E image not found at {he_path}, will proceed without it")
        he_path = None
    else:
        print_step(f"Found H&E image: {he_path}")
    
    model_images = {}
    for img_file in img_dir.glob("*.png"):
        if img_file.name == he_basename:
            continue
        model_id = img_file.stem
        model_images[model_id] = img_file
    
    print_step(f"Found {len(model_images)} model images: {list(model_images.keys())}")
    
    if len(model_images) < 2:
        raise ValueError(f"Need at least 2 model images, found {len(model_images)}")
    
    return he_path, model_images


def build_pairwise_messages(he_url: Optional[str], img1_url: str, img2_url: str, simple_mode: bool = True) -> list:
    """Build messages for pairwise comparison with updated prompts.
    
    Args:
        he_url: Data URL for H&E image (optional)
        img1_url: Data URL for first model image
        img2_url: Data URL for second model image
        simple_mode: If True, use simple prompts; if False, use complex prompts with bias warnings
    
    Returns:
        List of messages for LLM API
    """
    if simple_mode:
        # Simple mode prompts
        system_content = (
            "You are an expert model evaluator for spatial transcriptomics layer identification. "
            "Always start with EXACTLY one word: 'first' or 'second', then provide two short paragraphs "
            "in the form: First Model: reasoning Second Model: reasoning"
        )
        
        messages = [{"role": "system", "content": system_content}]
        
        if he_url:
            # Simple mode with H&E
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"The slices belong to the {DATASET_NAME}. Based on the information, please compare the model performance of identifying the layers of the slice provided in the next few messages."
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
            # Simple mode without H&E
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"The slices belong to the {DATASET_NAME}. Based on the information, please compare the model performance of identifying the layers of the slice provided in the next few messages."
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
        # Complex mode prompts (original)
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


def ask_llm_with_retries(messages: list, max_retries: int = 3, backoff_base: float = 1.5,
                         temperature: float = PAIRWISE_TEMPERATURE,
                         reasoning_effort: str = PAIRWISE_REASONING_EFFORT) -> str:
    """Call LLM API with retry logic and specified hyperparameters."""
    last_err = None
    
    for attempt in range(1, max_retries + 1):
        try:
            print_step(f"Calling {MODEL_NAME} (attempt {attempt}) with temperature={temperature}, reasoning_effort={reasoning_effort}...")
            
            # Convert messages to responses API input format (preserves images)
            input_messages = messages_to_responses_input(messages)
            
            # Build API call parameters for GPT-5 responses.create()
            api_params = {
                "model": MODEL_NAME,
                "input": input_messages,
                "temperature": temperature,
            }
            
            # Add reasoning parameter for GPT-5 (format: reasoning={"effort": "high"})
            if reasoning_effort:
                api_params["reasoning"] = {"effort": reasoning_effort}
            
            resp = client.responses.create(**api_params)
            out = resp.output_text if hasattr(resp, 'output_text') else ""
            
            # Log reasoning tokens if available
            if hasattr(resp, 'usage') and hasattr(resp.usage, 'output_tokens_details'):
                reasoning_tokens = resp.usage.output_tokens_details.reasoning_tokens
                print_step(f"Received response from {MODEL_NAME} (chars={len(out)}, reasoning_tokens={reasoning_tokens}).")
            else:
                print_step(f"Received response from {MODEL_NAME} (chars={len(out)}).")
            
            return out or ""
        except Exception as e:
            last_err = e
            wait_s = backoff_base ** attempt
            print_step(f"API error: {e}. Backing off {wait_s:.1f}s ...")
            time.sleep(wait_s)
    print_step("Exhausted retries.")
    raise last_err


def parse_choice(text: str) -> Optional[str]:
    """Parse 'first' or 'second' from LLM response."""
    m = re.search(r"\b(first|second)\b", (text or "").strip(), flags=re.IGNORECASE)
    return m.group(1).lower() if m else None


def append_jsonl(path: Path, obj: dict) -> None:
    """Append a JSON object to a JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def run_pairwise_comparisons_for_reps(
    he_path: Optional[Path],
    model_images: Dict[str, Path],
    reps: int,
    pairwise_dir: Path,
    simple_mode: bool = True,
) -> List[Path]:
    """
    Run pairwise comparisons for multiple repetitions.
    
    Args:
        he_path: Path to H&E image (optional)
        model_images: Dict mapping model_id -> image_path
        reps: Number of repetitions
        pairwise_dir: Directory for pairwise results
        simple_mode: If True, use simple prompts; if False, use complex prompts
    
    Returns:
        List of paths to JSONL files (one per repetition)
    """
    print_step(f"Running {reps} repetitions of pairwise comparisons")
    print_step(f"Prompt mode: {'SIMPLE' if simple_mode else 'COMPLEX (with bias warnings)'}")
    
    pairwise_dir.mkdir(parents=True, exist_ok=True)
    print_step(f"Pairwise results directory: {pairwise_dir}")
    
    # Encode images once
    print_step("Encoding images...")
    he_url = file_to_data_url(he_path) if he_path else None
    model_urls = {model_id: file_to_data_url(path) for model_id, path in model_images.items()}
    
    model_ids = sorted(model_images.keys())
    pairs = list(combinations(model_ids, 2))
    print_step(f"Total pairs to compare: {len(pairs)}")
    
    jsonl_files = []
    
    for rep in range(1, reps + 1):
        print_step(f"\n=== Repetition {rep}/{reps} ===")
        
        jsonl_path = pairwise_dir / f"pairwise_results_rep{rep:02d}.jsonl"
        if jsonl_path.exists():
            jsonl_path.unlink()
        
        for idx, (model1_id, model2_id) in enumerate(pairs, start=1):
            print_step(f"[Rep {rep}, Pair {idx}/{len(pairs)}] {model1_id} vs {model2_id}")
            
            img1_url = model_urls[model1_id]
            img2_url = model_urls[model2_id]
            
            try:
                messages = build_pairwise_messages(he_url, img1_url, img2_url, simple_mode=simple_mode)
                out_text = ask_llm_with_retries(messages)
                choice = parse_choice(out_text)
                
                print_step(f"Model chose: {choice!r}")
                
                row = {
                    "ts": now(),
                    "model1_label": model1_id,
                    "model2_label": model2_id,
                    "gpt_choice": choice,
                    "gpt_output": out_text,
                    "repetition": rep,
                }
                append_jsonl(jsonl_path, row)
                print_step("Logged JSONL row.")
                
            except Exception as e:
                err_row = {
                    "ts": now(),
                    "model1_label": model1_id,
                    "model2_label": model2_id,
                    "gpt_choice": None,
                    "gpt_output": f"[API ERROR] {e}",
                    "repetition": rep,
                }
                append_jsonl(jsonl_path, err_row)
                print_step(f"Logged API error: {e}")
        
        jsonl_files.append(jsonl_path)
        print_step(f"Completed repetition {rep}, saved to: {jsonl_path}")
    
    return jsonl_files


def compute_winning_rates_from_reps(
    pairwise_files: List[Path],
    output_csv: Path,
) -> Path:
    """
    Compute winning rates from pairwise comparison JSONL files.
    Reuses logic from Ranking.py.
    
    Args:
        pairwise_files: List of JSONL file paths
        output_csv: Output CSV file path
    
    Returns:
        Path to output CSV
    """
    print_step("Computing winning rates from pairwise comparisons")
    
    games = defaultdict(int)
    wins = defaultdict(int)
    losses = defaultdict(int)
    ties = defaultdict(int)
    points = defaultdict(float)
    
    total_rows = 0
    used_rows = 0
    
    for jsonl_path in pairwise_files:
        print_step(f"Processing: {jsonl_path}")
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
                    # Treat unknown as tie
                    ties[m1] += 1
                    ties[m2] += 1
                    points[m1] += 0.5
                    points[m2] += 0.5
                    used_rows += 1
    
    print_step(f"Rows total: {total_rows} | rows used: {used_rows}")
    
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
    
    # Sort by win_rate descending
    ranked = sorted(
        table,
        key=lambda r: (r["win_rate"], r["points"], r["games"], r["model"]),
        reverse=True,
    )
    
    print_step("\n=== Ranking by Win Rate (win=1, tie=0.5) ===")
    for i, r in enumerate(ranked, 1):
        print_step(f"{i:2d}. {r['model']:20s}  win_rate={r['win_rate']:.3f}  "
                  f"points={r['points']:.3f}  games={r['games']:3d}  "
                  f"W-L-T={r['wins']}-{r['losses']}-{r['ties']}")
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model","games","wins","losses","ties","points","win_rate"]
        )
        writer.writeheader()
        writer.writerows(ranked)
    
    print_step(f"Saved winning rates to: {output_csv}")
    
    return output_csv


def select_top_models(
    winning_rate_csv: Path,
    top_k: int = 5,
    mode: str = "fixed",
) -> List[str]:
    """
    Select top K models from winning rate CSV.
    
    Args:
        winning_rate_csv: Path to winning rate CSV
        top_k: Number of top models (used if mode="fixed")
        mode: "fixed" or "elbow"
    
    Returns:
        List of top model IDs
    """
    print_step(f"Selecting top models (mode={mode}, top_k={top_k})")
    
    models = []
    with winning_rate_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            models.append({
                "model": row["model"],
                "win_rate": float(row["win_rate"]),
            })
    
    if mode == "fixed":
        k = min(top_k, len(models))
        top_models = [m["model"] for m in models[:k]]
        print_step(f"Selected top {k} models (fixed): {top_models}")
        return top_models
    
    elif mode == "elbow":
        # Simple elbow detection: find largest drop in win_rate
        if len(models) <= 2:
            k = len(models)
        else:
            drops = []
            for i in range(len(models) - 1):
                drop = models[i]["win_rate"] - models[i+1]["win_rate"]
                drops.append((i+1, drop))
            
            # Find elbow as position with largest drop
            elbow_idx = max(drops, key=lambda x: x[1])[0]
            k = max(3, min(elbow_idx, top_k))  # At least 3, at most top_k
        
        top_models = [m["model"] for m in models[:k]]
        print_step(f"Selected top {k} models (elbow): {top_models}")
        return top_models
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


def build_second_round_messages(
    he_url: Optional[str],
    top_model_urls: Dict[str, str],
    top_model_stats: Dict[str, float],
    pairwise_summary: str,
) -> Tuple[list, Dict[str, str]]:
    """Build messages for second-round reasoning."""
    
    system_content = """You are an expert in spatial transcriptomics and histopathology.

You will be given:
- One H&E reference image of a tissue section (if H&E reference is provided).
- K clustering result images from different models applied to the same section.
- Quantitative context from pairwise comparisons (winning rates and detailed comparison results).

YOUR ROLE:
You are performing a "second round reasoning" audit to detect and remove potential "poison pill" models before consensus clustering.

KEY OBJECTIVE:
You are NOT scoring or ranking models.
You are deciding which models should be kept for consensus clustering as a coherent, biologically plausible ensemble.

CRITICAL RULES:

1. Focus on CONSISTENCY, not aesthetics.
   - Smooth, contiguous, or visually "pretty" cluster shapes are NOT inherently better.
   - Fragmented or irregular clusters can be biologically meaningful.
   - Do NOT favor models just because their clusters look smoother or less fragmented.

2. Use the H&E image as biological ground (if H&E image is provided).
   - Check whether the main tissue structures and boundaries in the H&E (if H&E image is provided) (e.g., layers, regions, sharp tissue edges, white-matter/grey-matter boundaries, anatomical compartments) are reasonably respected by each model's clustering.
   - A model is suspicious ("poison pill") if it clearly ignores major boundaries that are obvious in the H&E, especially when other models agree with each other.

3. Evaluate MODELS AS A GROUP.
   - First, understand the common "macro-structure" implied by the ensemble of models.
     For example, how many major regions there are, where the main boundaries are, and how they align with the H&E (if H&E image is provided).
   - Then, identify models that are fundamentally inconsistent with this shared macro-structure.
   - A model is a "poison pill" if:
     - It substantially disagrees with almost all other models on key boundaries or large regions, AND
     - Its disagreements are not strongly supported by the H&E image (if H&E image is provided).

4. Poison pill decision logic:
   - If one model is clearly an outlier (strong conflicts with H&E (if H&E image is provided) and with all others), EXCLUDE it.
   - If two or more models form a small minority that strongly disagree with the majority AND appear less supported by the H&E (if H&E image is provided), you may EXCLUDE these minority models.
   - If all models are broadly consistent with each other and with the H&E (if H&E image is provided), you may keep all of them.
   - When in doubt between "keep" and "exclude", give priority to removing clearly problematic outliers, not to maximizing the number of models.

5. Output format:
   - You MUST respond with a single JSON object ONLY.
   - Do NOT include any extra text outside the JSON.
   - The JSON must contain:
       - "final_keep_ids": an array of model IDs to be used for consensus clustering.
       - "per_model_notes": an object mapping model_id → short textual justification.
   - BREVITY: Each justification in "per_model_notes" should be 1-3 sentences (under 50 words each).

6. ID handling:
   - Use ONLY the model IDs exactly as provided in the input.
   - Do NOT invent new IDs.
   - Do NOT omit any model in "per_model_notes".

Think carefully about:
- The common macro-structure implied by all models.
- Which models clearly violate that structure and the H&E evidence (if H&E image is provided).
Then return the JSON result as specified.

IMPORTANT: Keep your response concise. Aim for under 500 words total in all justifications combined."""
    
    # Build generic model IDs and mapping
    generic_to_real = {}
    model_list_text = []
    
    for idx, (real_id, url) in enumerate(sorted(top_model_urls.items()), start=1):
        generic_id = f"model_{idx}"
        generic_to_real[generic_id] = real_id
        model_list_text.append(f"   - id: \"{generic_id}\"\n"
                               f"     description: \"Top-K model {idx} clustering result ({real_id})\"\n"
                               f"     image: <MODEL_{idx}_IMAGE_HERE>")
    
    # Build quantitative context section
    quant_context_lines = ["[Quantitative context from pairwise comparisons]", ""]
    quant_context_lines.append("Winning rates for the Top-K models (from pairwise comparisons):")
    quant_context_lines.append("")
    for real_id in sorted(top_model_urls.keys()):
        win_rate = top_model_stats.get(real_id, 0.0)
        quant_context_lines.append(f"  - {real_id}: win_rate = {win_rate:.3f}")
    quant_context_lines.append("")
    quant_context_lines.append("Detailed pairwise comparison results:")
    quant_context_lines.append(pairwise_summary)
    quant_context_lines.append("")
    quant_context_lines.append("Use these values only as soft context. Your primary criteria remain:")
    quant_context_lines.append("(1) consistency across models in macro-structure, and")
    quant_context_lines.append("(2) agreement with the H&E image for major boundaries and regions.")
    quant_context_lines.append("")
    
    quant_context_text = "\n".join(quant_context_lines)
    
    user_content_text = f"""You will now perform the second-round "poison pill" screening for consensus clustering.

Below is the input for a SINGLE dataset.

[Task]
- One H&E image (if H&E image is provided): the reference histology for this tissue section.
- {len(top_model_urls)} model outputs (Top-K models from previous pairwise comparisons) applied to the same section.
- Your job: decide which models should be KEPT as a coherent ensemble for consensus clustering, and which (if any) should be treated as "poison pill" outliers and excluded.

[Images]

1. H&E reference image:
   - id: "he_image"
   - role: "reference"
   - image: <HE_IMAGE_HERE>

2. Model clustering outputs (Top-{len(top_model_urls)}):
   - Each model has an ID and an image.
   - All images show cluster labels over the same tissue section as the H&E (if H&E image is provided).

   Models:
{chr(10).join(model_list_text)}

{quant_context_text}

[What you must check]

1. Macro-structure consistency:
   - Do these {len(top_model_urls)} models broadly agree on the major tissue regions (e.g., layers, lobes, compartments)?
   - Are the main boundaries and region shapes roughly similar across most models?

2. Agreement with H&E (if H&E image is provided):
   - Looking at the H&E (if H&E image is provided) image, which boundaries are clearly visible or biologically plausible?
   - For each model, does it respect these boundaries, or does it place large clusters across obvious anatomical borders?

3. Poison pill detection:
   - Identify any model(s) that:
     - Strongly disagree with almost all others on major regions or boundaries, AND
     - Are not well supported by the H&E appearance (if H&E image is provided).
   - These models should be treated as potential "poison pill" models and EXCLUDED from the final ensemble.

[Important warning]

- We DO NOT care about "smoothness" or "prettiness" of clusters.
- Do NOT penalize models for having more fragmented or irregular clusters if they are plausible and H&E-consistent (if H&E image is provided).
- A model that looks visually smooth but ignores clear H&E boundaries (if H&E image is provided), or strongly disagrees with all others, is likely a poison pill.

[Required output]

Return ONLY a JSON object with the following structure:

{{
  "final_keep_ids": [ ... ],
  "per_model_notes": {{
{chr(10).join(f'    "{generic_id}": "...",' for generic_id in sorted(generic_to_real.keys()))}
  }}
}}

- "final_keep_ids": the list of model IDs you recommend keeping for consensus clustering.
- "per_model_notes": a short justification (1–3 sentences) for KEEPING or EXCLUDING each model.

Do not include any text outside this JSON.

IMPORTANT: Keep each model note concise (under 50 words). Total response should be under 500 words."""
    
    messages = [{"role": "system", "content": system_content}]
    
    # Add H&E image if available
    if he_url:
        content_parts = [{"type": "text", "text": user_content_text}]
        content_parts.append({"type": "image_url", "image_url": {"url": he_url}})
    else:
        content_parts = [{"type": "text", "text": user_content_text}]
    
    # Add model images in order
    for idx, (real_id, url) in enumerate(sorted(top_model_urls.items()), start=1):
        content_parts.append({"type": "image_url", "image_url": {"url": url}})
    
    messages.append({"role": "user", "content": content_parts})
    
    return messages, generic_to_real


def run_second_round_reasoning(
    he_path: Optional[Path],
    top_model_images: Dict[str, Path],
    pairwise_files: List[Path],
    winning_rate_csv: Path,
    output_json: Path,
) -> Path:
    """
    Run second-round reasoning to identify poison pill models.
    
    Args:
        he_path: Path to H&E image (optional)
        top_model_images: Dict of top model_id -> image_path
        pairwise_files: List of pairwise JSONL files
        winning_rate_csv: Path to winning rate CSV
        output_json: Output JSON file path
    
    Returns:
        Path to output JSON
    """
    print_step("\n=== Second-Round Reasoning ===")
    print_step(f"Evaluating {len(top_model_images)} top models")
    
    # Read winning rates for top models
    top_model_stats = {}
    with winning_rate_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_id = row["model"]
            if model_id in top_model_images:
                top_model_stats[model_id] = float(row["win_rate"])
    
    print_step(f"Loaded winning rates for {len(top_model_stats)} top models")
    
    # Read pairwise comparison details for top models
    pairwise_comparisons = []
    for jsonl_path in pairwise_files:
        if not jsonl_path.exists():
            continue
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    m1 = obj.get("model1_label")
                    m2 = obj.get("model2_label")
                    if m1 in top_model_images and m2 in top_model_images:
                        pairwise_comparisons.append(obj)
                except json.JSONDecodeError:
                    continue
    
    print_step(f"Loaded {len(pairwise_comparisons)} pairwise comparisons for top models")
    
    # Build pairwise summary text
    pairwise_lines = []
    pairwise_lines.append("")
    for comp in pairwise_comparisons[:20]:
        m1 = comp.get("model1_label", "")
        m2 = comp.get("model2_label", "")
        choice = comp.get("gpt_choice", "unknown")
        rep = comp.get("repetition", "")
        winner = m1 if choice == "first" else (m2 if choice == "second" else "tie")
        pairwise_lines.append(f"  Rep {rep}: {m1} vs {m2} → Winner: {winner}")
    if len(pairwise_comparisons) > 20:
        pairwise_lines.append(f"  ... ({len(pairwise_comparisons) - 20} more comparisons)")
    pairwise_summary = "\n".join(pairwise_lines)
    
    # Encode images
    he_url = file_to_data_url(he_path) if he_path else None
    top_model_urls = {model_id: file_to_data_url(path) 
                      for model_id, path in top_model_images.items()}
    
    # Build messages
    messages, generic_to_real = build_second_round_messages(
        he_url, top_model_urls, top_model_stats, pairwise_summary
    )
    real_to_generic = {v: k for k, v in generic_to_real.items()}
    
    print_step("Calling LLM for second-round reasoning with HIGH reasoning effort...")
    print_step(f"Hyperparameters: temperature={SECOND_ROUND_TEMPERATURE}, reasoning_effort={SECOND_ROUND_REASONING_EFFORT}")
    print_step("Note: Output length controlled via prompt instructions (under 500 words)")
    
    try:
        # Convert messages to responses API input format (preserves images)
        input_messages = messages_to_responses_input(messages)
        
        # Build API call parameters for GPT-5 responses.create() with HIGH reasoning effort
        api_params = {
            "model": MODEL_NAME,
            "input": input_messages,
            "temperature": SECOND_ROUND_TEMPERATURE,
        }
        
        # Add reasoning parameter for GPT-5 (format: reasoning={"effort": "high"})
        if SECOND_ROUND_REASONING_EFFORT:
            api_params["reasoning"] = {"effort": SECOND_ROUND_REASONING_EFFORT}
            print_step(f"Using reasoning={{'effort': '{SECOND_ROUND_REASONING_EFFORT}'}} for careful poison pill analysis")
        
        resp = client.responses.create(**api_params)
        out_text = resp.output_text if hasattr(resp, 'output_text') else ""
        
        # Log reasoning tokens if available
        if hasattr(resp, 'usage') and hasattr(resp.usage, 'output_tokens_details'):
            reasoning_tokens = resp.usage.output_tokens_details.reasoning_tokens
            print_step(f"Received response (chars={len(out_text)}, reasoning_tokens={reasoning_tokens})")
        else:
            print_step(f"Received response (chars={len(out_text)})")
    
    except Exception as e:
        print_step(f"Error in second-round reasoning: {e}")
        raise
    
    # Parse JSON response
    try:
        # Remove markdown code blocks if present
        json_text = re.sub(r"```json\s*|\s*```", "", out_text).strip()
        
        result = json.loads(json_text)
        
        # Map generic IDs back to real IDs
        generic_keep_ids = result.get("final_keep_ids", [])
        real_keep_ids = [generic_to_real[gid] for gid in generic_keep_ids if gid in generic_to_real]
        
        # Map per_model_notes to real IDs
        per_model_notes_generic = result.get("per_model_notes", {})
        per_model_notes_real = {generic_to_real[gid]: note 
                                for gid, note in per_model_notes_generic.items() 
                                if gid in generic_to_real}
        
        # Generate summary
        excluded_models = [mid for mid in top_model_images.keys() if mid not in real_keep_ids]
        if excluded_models:
            summary = f"Excluded {len(excluded_models)} model(s) as potential poison pills: {', '.join(excluded_models)}. "
            summary += "These models showed substantial disagreement with the ensemble consensus and/or "
            summary += "lacked strong support from the H&E reference image, particularly regarding major tissue boundaries."
        else:
            summary = "All models were kept. The ensemble showed good consistency with each other and with the H&E reference."
        
        # Construct output
        output_data = {
            "n_final_models": len(real_keep_ids),
            "final_model_ids": real_keep_ids,
            "final_keep_ids_raw": generic_keep_ids,
            "per_model_notes": per_model_notes_real,
            "summary_poison_pill": summary,
            "raw_response": out_text,
        }
        
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print_step(f"Second-round reasoning saved to: {output_json}")
        print_step(f"\nFinal models kept: {len(real_keep_ids)}/{len(top_model_images)}")
        print_step(f"Kept models: {real_keep_ids}")
        if excluded_models:
            print_step(f"Excluded models: {excluded_models}")
        
        return output_json
        
    except Exception as e:
        print_step(f"Error parsing second-round reasoning response: {e}")
        print_step(f"Raw response: {out_text[:500]}...")
        raise


def run_full_pipeline(
    image_dir: str,
    reps: int = 5,
    top_k: int = 5,
    top_k_mode: str = "fixed",
    he_basename: str = "he.png",
    skip_pairwise: bool = False,
    simple_mode: bool = True,
):
    """
    Run the full second-round reasoning pipeline.
    
    Args:
        image_dir: Directory containing H&E and model images
        reps: Number of pairwise comparison repetitions
        top_k: Number of top models to select
        top_k_mode: "fixed" or "elbow"
        he_basename: Filename of H&E image
        skip_pairwise: Skip pairwise comparisons and reuse existing results
        simple_mode: If True, use simple prompts for pairwise comparisons
    """
    print_step("=" * 60)
    print_step("STARTING SECOND-ROUND REASONING PIPELINE")
    print_step("=" * 60)
    print_step(f"Model: {MODEL_NAME}")
    print_step(f"Skip pairwise: {skip_pairwise}")
    print_step(f"Simple mode: {simple_mode}")
    print_step(f"Repetitions: {reps}")
    print_step(f"Top-K: {top_k} (mode: {top_k_mode})")
    print_step("")
    print_step(f"Pairwise Comparison Prompts: {'SIMPLE' if simple_mode else 'COMPLEX (with bias warnings)'}")
    print_step("Pairwise Comparison Hyperparameters:")
    print_step(f"  - Temperature: {PAIRWISE_TEMPERATURE}")
    print_step(f"  - Reasoning Effort: {PAIRWISE_REASONING_EFFORT}")
    print_step(f"  - Output Length: <200 words (prompt-controlled)")
    print_step("")
    print_step("Second-Round Reasoning Hyperparameters (HIGHER EFFORT):")
    print_step(f"  - Temperature: {SECOND_ROUND_TEMPERATURE}")
    print_step(f"  - Reasoning Effort: {SECOND_ROUND_REASONING_EFFORT}")
    print_step(f"  - Output Length: <500 words (prompt-controlled)")
    print_step("=" * 60)
    
    image_path = Path(image_dir).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    # Create output paths parallel to image directory
    parent_dir = image_path.parent
    base_name = image_path.name
    
    pairwise_dir = parent_dir / f"{base_name}_pairwise_results"
    winning_rate_csv = parent_dir / f"{base_name}_winning_rates.csv"
    second_round_json = parent_dir / f"{base_name}_second_round_reasoning.json"
    
    print_step(f"Image directory: {image_path}")
    print_step(f"Output files will be created parallel to image directory:")
    print_step(f"  - Pairwise results: {pairwise_dir}")
    print_step(f"  - Winning rates CSV: {winning_rate_csv}")
    print_step(f"  - Second-round JSON: {second_round_json}")
    
    # Step 1: Discover models
    he_path, model_images = discover_models(str(image_path), he_basename)
    
    # Step 2 & 3: Run pairwise comparisons or skip
    if skip_pairwise:
        if not winning_rate_csv.exists():
            raise FileNotFoundError(
                f"Cannot skip pairwise: winning rate CSV not found at {winning_rate_csv}. "
                f"Please run with --skip-pairwise=False first to generate pairwise results."
            )
        print_step("Skipping pairwise comparisons, using existing winning rates")
        pairwise_files = []
        if pairwise_dir.exists():
            pairwise_files = sorted(pairwise_dir.glob("pairwise_results_rep*.jsonl"))
            print_step(f"Found {len(pairwise_files)} existing pairwise JSONL files")
    else:
        pairwise_files = run_pairwise_comparisons_for_reps(
            he_path, model_images, reps, pairwise_dir, simple_mode=simple_mode
        )
        compute_winning_rates_from_reps(pairwise_files, winning_rate_csv)
    
    # Step 4: Select top-K models
    top_model_ids = select_top_models(winning_rate_csv, top_k, top_k_mode)
    top_model_images = {mid: model_images[mid] for mid in top_model_ids}
    
    # Step 5: Second-round reasoning
    run_second_round_reasoning(
        he_path, top_model_images, pairwise_files, winning_rate_csv, second_round_json
    )
    
    print_step("\n" + "=" * 60)
    print_step("PIPELINE COMPLETED")
    print_step("=" * 60)
    print_step(f"Winning rates CSV: {winning_rate_csv}")
    print_step(f"Second-round reasoning JSON: {second_round_json}")
    print_step("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Second-round reasoning pipeline for spatial transcriptomics model evaluation"
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        help="Directory containing H&E and model output images"
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=5,
        help="Number of pairwise comparison repetitions (default: 5)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top models to select (default: 5)"
    )
    parser.add_argument(
        "--top-k-mode",
        choices=["fixed", "elbow"],
        default="fixed",
        help="Top-K selection mode: fixed or elbow (default: fixed)"
    )
    parser.add_argument(
        "--he-basename",
        default="he.png",
        help="Filename of H&E image (default: he.png)"
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (or set OPENAI_API_KEY environment variable)"
    )
    parser.add_argument(
        "--skip-pairwise",
        action="store_true",
        help="Skip pairwise comparisons and reuse existing winning-rate results"
    )
    parser.add_argument(
        "--complex-mode",
        action="store_true",
        help="Use complex prompts with bias warnings for pairwise comparisons (default: False, uses simple prompts)"
    )
    
    args = parser.parse_args()
    
    # Set up API key and client
    global API_KEY, client
    API_KEY = args.api_key or os.getenv("OPENAI_API_KEY")
    if not API_KEY:
        print("Error: OpenAI API key not provided. Use --api-key or set OPENAI_API_KEY environment variable.")
        sys.exit(1)
    
    client = OpenAI(api_key=API_KEY)
    
    # Run pipeline
    run_full_pipeline(
        image_dir=args.image_dir,
        reps=args.reps,
        top_k=args.top_k,
        top_k_mode=args.top_k_mode,
        he_basename=args.he_basename,
        skip_pairwise=args.skip_pairwise,
        simple_mode=(not args.complex_mode),
    )

if __name__ == "__main__":
    main()
