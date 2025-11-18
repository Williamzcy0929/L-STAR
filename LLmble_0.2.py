import os
import sys
import time
import json
import base64
import re
from itertools import combinations
from typing import Tuple, Dict, Optional, List


MODEL_NAME = "gpt-5"
DATASET_NAME = "<YOUR_DATASET_NAME_HERE>"

IMAGE_DIR = "PATH_TO_IMAGES"
OUTPUT_JSONL = "pairwise_results.jsonl"

REASONING_EFFORT = "minimal"
TEXT_VERBOSITY = "low"
TEMPERATURE = 0.0

ARI_MATRIX: List[List[float]] = [
    [ARI_1,   ARI_2,   ARI_3,   ARI_4],
    [ARI_5,   ARI_6,   ARI_7,   ARI_8],
    [ARI_9,   ARI_10,  ARI_11,  ARI_12],
    [ARI_13,  ARI_14,  ARI_15,  ARI_16],
]

MODEL_NAME_MATRIX: List[List[str]] = [
    ["MODEL_1",   "MODEL_2",    "MODEL_3",    "MODEL_4"],
    ["MODEL_5",   "MODEL_6",    "MODEL_7",    "MODEL_8"],
    ["MODEL_9",   "MODEL_10",   "MODEL_11",   "MODEL_12"],
    ["MODEL_13",  "MODEL_14",   "MODEL_15",   "MODEL_16"],
]

from openai import OpenAI
client = OpenAI()

def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def print_step(msg: str) -> None:
    print(f"[{now()}] {msg}")
    sys.stdout.flush()

def image_path(x: int, y: int) -> str:
    return os.path.join(IMAGE_DIR, f"row{x}_col{y}_p1.png")

def he_image_path() -> str:
    return os.path.join(IMAGE_DIR, "row1_col1_p1.png")

def file_to_data_url(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def ari_of(x: int, y: int) -> float:
    if not (2 <= x <= 5 and 1 <= y <= 4):
        raise ValueError(f"(x,y)=({x},{y}) out of range; expected x∈[2..5], y∈[1..4]")
    return ARI_MATRIX[x - 2][y - 1]

def label_of(x: int, y: int) -> str:
    if not (2 <= x <= 5 and 1 <= y <= 4):
        raise ValueError(f"(x,y)=({x},{y}) out of range; expected x∈[2..5], y∈[1..4]")
    return MODEL_NAME_MATRIX[x - 2][y - 1]

def parse_choice(text: str) -> Optional[str]:
    m = re.search(r"\b(first|second)\b", (text or "").strip(), flags=re.IGNORECASE)
    return m.group(1).lower() if m else None

def truth_better(ari1: float, ari2: float) -> Optional[str]:
    if ari1 > ari2:
        return "first"
    if ari2 > ari1:
        return "second"
    return None  # tie

def build_responses_input(he_url: str, img1_url: str, img2_url: str) -> list:
    return [
        {
            "role": "user",
            "content": [
                {"type": "input_text",
                 "text": (
                     f"The slices belong to the `{DATASET_NAME}`. "
                     "Based on the information, please compare the model performance of identifying "
                     "the layers of the slice provided in the next few messages."
                 )},
                {"type": "input_image", "image_url": he_url},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "input_text",
                 "text": (
                     "These two pictures are two identification results from two different models on this slice. "
                     "Please compare which model performed better. "
                     "Start your answer with EXACTLY ONE WORD — either 'first' or 'second' — "
                     "then give a 1–2 sentence justification."
                 )},
                {"type": "input_image", "image_url": img1_url},
                {"type": "input_image", "image_url": img2_url},
            ],
        },
    ]

def ask_gpt5_with_retries(he_url: str, img1_url: str, img2_url: str,
                          max_retries: int = 3, backoff_base: float = 1.5) -> str:
    """
    Call GPT-5 via Responses API, with simple exponential backoff.
    Returns output_text or raises last exception.
    """
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            print_step(f"Calling {MODEL_NAME} (attempt {attempt}) "
                       f"[effort={REASONING_EFFORT}, verbosity={TEXT_VERBOSITY}, temp={TEMPERATURE}] …")
            resp = client.responses.create(
                model=MODEL_NAME,
                input=build_responses_input(he_url, img1_url, img2_url),
                reasoning={"effort": REASONING_EFFORT},
                text={"verbosity": TEXT_VERBOSITY},
                temperature=TEMPERATURE,
            )
            out = resp.output_text
            print_step(f"Received response from {MODEL_NAME} (chars={len(out) if out else 0}).")
            return out
        except Exception as e:
            last_err = e
            wait_s = backoff_base ** attempt
            print_step(f"API error: {e}. Backing off {wait_s:.1f}s …")
            time.sleep(wait_s)
    print_step("Exhausted retries.")
    raise last_err

def validate_setup():
    print_step("Validating environment and inputs …")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Please set OPENAI_API_KEY in your environment.")
    if not os.path.exists(IMAGE_DIR):
        raise FileNotFoundError(f"IMAGE_DIR not found: {IMAGE_DIR}")
    if not os.path.exists(he_image_path()):
        raise FileNotFoundError(f"Missing H&E image: {he_image_path()}")
    if len(ARI_MATRIX) != 4 or any(len(row) != 4 for row in ARI_MATRIX):
        raise ValueError("ARI_MATRIX must be 4x4 for x=2..5, y=1..4.")
    if len(MODEL_NAME_MATRIX) != 4 or any(len(row) != 4 for row in MODEL_NAME_MATRIX):
        raise ValueError("MODEL_NAME_MATRIX must be 4x4 and match ARI_MATRIX shape.")
    for x in range(2, 6):
        for y in range(1, 5):
            p = image_path(x, y)
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing model image: {p}")
    print_step("Validation OK.")

def append_jsonl(path: str, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    print_step(f"Starting pipeline with model={MODEL_NAME}")
    print_step(f"Reasoning: {REASONING_EFFORT}, Verbosity: {TEXT_VERBOSITY}, Temperature: {TEMPERATURE}")
    validate_setup()

    if os.path.exists(OUTPUT_JSONL):
        print_step(f"Overwriting existing JSONL: {OUTPUT_JSONL}")
        os.remove(OUTPUT_JSONL)
    else:
        print_step(f"Creating JSONL: {OUTPUT_JSONL}")

    print_step("Encoding H&E image …")
    he_url = file_to_data_url(he_image_path())

    coords = [(x, y) for x in range(2, 6) for y in range(1, 5)]  # 16 cells
    print_step(f"Encoding {len(coords)} model result images …")
    data_urls: Dict[Tuple[int, int], str] = {(x, y): file_to_data_url(image_path(x, y))
                                             for (x, y) in coords}

    pairs = list(combinations(coords, 2))
    print_step(f"Total unordered pairs to compare: {len(pairs)}")

    correct = 0
    scored = 0

    for idx, ((x1, y1), (x2, y2)) in enumerate(pairs, start=1):
        m1, m2 = label_of(x1, y1), label_of(x2, y2)
        a1, a2 = ari_of(x1, y1), ari_of(x2, y2)
        print_step(f"[{idx}/{len(pairs)}] Pair: ({x1},{y1})[{m1}] vs ({x2},{y2})[{m2}] | ARIs: {a1:.3f} vs {a2:.3f}")
        print_step("Building request payload …")

        img1_url = data_urls[(x1, y1)]
        img2_url = data_urls[(x2, y2)]

        try:
            out_text = ask_gpt5_with_retries(he_url, img1_url, img2_url)
            choice = parse_choice(out_text)
            truth = truth_better(a1, a2)
            if truth is None:
                is_correct = None
                print_step("Truth is a TIE (equal ARIs) → pair excluded from accuracy.")
            else:
                is_correct = (choice == truth)
                scored += 1
                print_step(f"Model chose: {choice!r} | Truth: {truth!r} → "
                           f"{'CORRECT' if is_correct else 'WRONG'}")
                if is_correct:
                    correct += 1

            row = {
                "ts": now(),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "model1_label": m1, "model2_label": m2,
                "ari1": a1, "ari2": a2,
                "gpt_choice": choice,
                "truth_better": truth,
                "correct": is_correct,
                "gpt_output": out_text,
            }
            append_jsonl(OUTPUT_JSONL, row)
            print_step("Logged JSONL row.")
        except Exception as e:
            err_row = {
                "ts": now(),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "model1_label": m1, "model2_label": m2,
                "ari1": a1, "ari2": a2,
                "gpt_choice": None,
                "truth_better": truth_better(a1, a2),
                "correct": None,
                "gpt_output": f"[API ERROR] {e}",
            }
            append_jsonl(OUTPUT_JSONL, err_row)
            print_step(f"❗ Logged API error for this pair: {e}")

    print_step(f"Results written to: {OUTPUT_JSONL}")
    if scored:
        acc = correct / scored
        print_step(f"Final accuracy (excluding ties): {correct}/{scored} = {acc:.3f}")
    else:
        print_step("No non-tie pairs to score.")

if __name__ == "__main__":
    main()
