"""Second-round reasoning for poison pill model detection."""

import os
import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from lstar.pairwise import (
    discover_models,
    file_to_data_url,
    select_top_models,
    ask_llm_with_retries,
)
from lstar.config import DEFAULT_HE_BASENAME

logger = logging.getLogger(__name__)


def build_second_round_contents(
    he_url: Optional[str],
    top_model_urls: Dict[str, str],
    top_model_stats: Dict[str, float],
    pairwise_summary: str,
) -> Tuple[str, list, Dict[str, str]]:
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
    
    generic_to_real = {}
    model_list_text = []
    
    for idx, (real_id, url) in enumerate(sorted(top_model_urls.items()), start=1):
        generic_id = f"model_{idx}"
        generic_to_real[generic_id] = real_id
        model_list_text.append(f"   - id: \"{generic_id}\"\n"
                               f"     description: \"Top-K model {idx} clustering result ({real_id})\"\n"
                               f"     image: <MODEL_{idx}_IMAGE_HERE>")
    
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
    
    # Build contents list for OpenAI format (adapted from Gemini format)
    contents = []
    
    # Add User Text Context
    contents.append({"type": "text", "text": user_content_text})
    
    # Add H&E image if available
    if he_url:
        contents.append({"type": "image_url", "image_url": {"url": he_url}})
    
    # Add model images in order
    for idx, (real_id, url) in enumerate(sorted(top_model_urls.items()), start=1):
        contents.append({"type": "image_url", "image_url": {"url": url}})
    
    return system_content, contents, generic_to_real


def run_second_round_reasoning(
    image_dir: Path,
    pairwise_dir: Path,
    ranking_csv_path: Path,
    he_basename: str,
    top_k: int,
    model_name: str,
    second_round_temperature: float,
    second_round_reasoning_effort: str,
    api_key: Optional[str],
    api_base: Optional[str],
    dataset_name: str,
    output_dir: Path,
    disable_cache: bool = False,
) -> List[str]:
    """
    Run second-round 'poison pill' reasoning, using the same prompt and logic
    as the current external Two_Round_Reasoning script, and return a list of
    selected REAL model IDs. Also writes a JSON file with these models so the
    pipeline can read it later.
    
    Parameters
    ----------
    disable_cache : bool, default False
        If True, disable writing the second-round reasoning JSON file.
        The results will still be returned, but no JSON file will be created.
    """
    logger.info("\n=== Second-Round Reasoning ===")
    
    image_dir = Path(image_dir)
    pairwise_dir = Path(pairwise_dir)
    ranking_csv_path = Path(ranking_csv_path)
    output_dir = Path(output_dir)
    
    # 1. Read ranking and pick top-k models
    logger.info("Reading ranking CSV and selecting top-k models...")
    ranking_df = pd.read_csv(ranking_csv_path)
    top_model_ids = select_top_models(ranking_df, top_k, mode="fixed")
    logger.info(f"Selected top {len(top_model_ids)} models for second-round reasoning: {top_model_ids}")
    
    # 2. Build image URLs or data URLs
    logger.info("Loading and encoding images...")
    he_path, model_images = discover_models(image_dir, he_basename)
    he_url = file_to_data_url(he_path) if he_path else None
    
    # Get top model images and URLs
    top_model_urls: Dict[str, str] = {}
    with tqdm(total=len(top_model_ids), desc="Encoding images", unit="image") as pbar:
        for model_id in top_model_ids:
            if model_id in model_images:
                top_model_urls[model_id] = file_to_data_url(model_images[model_id])
                pbar.update(1)
            else:
                logger.warning(f"Model {model_id} not found in image directory, skipping")
                pbar.update(1)
    
    if len(top_model_urls) < 2:
        raise ValueError(
            f"Second-round reasoning requires at least 2 models with images, "
            f"but only {len(top_model_urls)} top models have images."
        )
    
    # Build top_model_stats from ranking
    top_model_stats: Dict[str, float] = {}
    if "win_rate" in ranking_df.columns:
        win_col = "win_rate"
    elif "winning_rate" in ranking_df.columns:
        win_col = "winning_rate"
    elif "WinningRate" in ranking_df.columns:
        win_col = "WinningRate"
    else:
        win_col = ranking_df.columns[-1]
    
    if "model" in ranking_df.columns:
        model_col = "model"
    elif "Model" in ranking_df.columns:
        model_col = "Model"
    else:
        model_col = ranking_df.columns[0]
    
    for _, row in ranking_df.iterrows():
        model_id = row[model_col]
        if model_id in top_model_ids:
            top_model_stats[model_id] = float(row[win_col])
    
    logger.info(f"Loaded winning rates for {len(top_model_stats)} top models")
    
    # 3. Build pairwise summary text
    logger.info("Loading pairwise comparison results...")
    pairwise_jsonl_files = sorted(pairwise_dir.glob("pairwise_results_rep*.jsonl"))
    pairwise_comparisons = []
    
    with tqdm(total=len(pairwise_jsonl_files), desc="Reading pairwise files", unit="file") as pbar:
        for jsonl_path in pairwise_jsonl_files:
            if not jsonl_path.exists():
                pbar.update(1)
                continue
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        m1 = obj.get("model1_label")
                        m2 = obj.get("model2_label")
                        if m1 in top_model_ids and m2 in top_model_ids:
                            pairwise_comparisons.append(obj)
                    except json.JSONDecodeError:
                        continue
            pbar.update(1)
    
    logger.info(f"Loaded {len(pairwise_comparisons)} pairwise comparisons for top models")
    
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
    
    # 4. Use the EXACT prompt via build_second_round_contents
    system_content, contents, generic_to_real = build_second_round_contents(
        he_url=he_url,
        top_model_urls=top_model_urls,
        top_model_stats=top_model_stats,
        pairwise_summary=pairwise_summary,
    )
    
    # 5. Call the OpenAI API
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": contents}
    ]
    
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not provided. Use api_key parameter or set OPENAI_API_KEY environment variable.")
    
    client = OpenAI(api_key=api_key, base_url=api_base) if api_base else OpenAI(api_key=api_key)
    
    logger.info("Calling LLM for second-round reasoning...")
    logger.info(f"Hyperparameters: temperature={second_round_temperature}, reasoning_effort={second_round_reasoning_effort}")
    
    # Progress indicator for LLM call
    with tqdm(total=1, desc="Second-round reasoning", unit="call") as pbar:
        try:
            out_text = ask_llm_with_retries(
                client, messages, model_name, second_round_temperature, second_round_reasoning_effort
            )
            pbar.update(1)
        except Exception as e:
            pbar.update(1)
            logger.error(f"Error in second-round reasoning: {e}")
            raise
    
    # 6. Parse the JSON output
    try:
        # Remove markdown code blocks if present
        json_text = re.sub(r"```json\s*|\s*```", "", out_text).strip()
        result = json.loads(json_text)
        
        final_keep_ids_generic = result.get("final_keep_ids", [])
        per_model_notes_generic = result.get("per_model_notes", {})
        
        # Map generic IDs to real model IDs
        final_model_ids = [
            generic_to_real[g_id]
            for g_id in final_keep_ids_generic
            if g_id in generic_to_real
        ]
        
        # Map per_model_notes to real IDs
        per_model_notes_real = {
            generic_to_real[g_id]: note
            for g_id, note in per_model_notes_generic.items()
            if g_id in generic_to_real
        }
        
        # 7. Write a JSON file compatible with read_second_round_results (only if not disabled)
        if not disable_cache:
            # Use dataset_name in filename if provided, otherwise use a generic name
            if dataset_name:
                # Sanitize dataset_name for filename
                safe_name = re.sub(r'[^\w\s-]', '', dataset_name).strip().replace(' ', '_')
                output_json = pairwise_dir / f"{safe_name}_second_round_reasoning.json"
            else:
                output_json = pairwise_dir / "second_round_reasoning.json"
            
            output_json.parent.mkdir(parents=True, exist_ok=True)
            
            output_data = {
                "final_keep_ids": final_keep_ids_generic,
                "final_model_ids": final_model_ids,
                "per_model_notes_generic": per_model_notes_generic,
                "per_model_notes": per_model_notes_real,
                "dataset_name": dataset_name,
                "raw_response": out_text,
            }
            
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Second-round reasoning saved to: {output_json}")
        else:
            logger.info("Second-round reasoning complete (JSON file disabled)")
        
        logger.info(f"\nFinal models kept: {len(final_model_ids)}/{len(top_model_ids)}")
        logger.info(f"Kept models: {final_model_ids}")
        excluded_models = [mid for mid in top_model_ids if mid not in final_model_ids]
        if excluded_models:
            logger.info(f"Excluded models: {excluded_models}")
        
        # 8. Return final_model_ids
        return final_model_ids
        
    except Exception as e:
        logger.error(f"Error parsing second-round reasoning response: {e}")
        logger.error(f"Raw response: {out_text[:500]}...")
        raise

