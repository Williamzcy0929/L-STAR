"""Minimal OpenAI transport used by structured domain annotation."""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional, Sequence


DEFAULT_ANNOTATION_MODEL = "gpt-5-2025-08-07"
DEFAULT_ANNOTATION_REASONING_EFFORT = "high"
SUPPORTED_REASONING_EFFORTS = frozenset({"minimal", "low", "medium", "high"})


def _json_type_for_const(value: Any) -> str:
    """Return the JSON Schema type for a JSON-compatible constant."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, Mapping):
        return "object"
    raise TypeError("JSON Schema const is not JSON-compatible: {!r}".format(value))


def _prepare_responses_api_schema(value: Any) -> Any:
    """Normalize a schema for the Responses API strict-schema subset."""
    if isinstance(value, Mapping):
        prepared = {
            key: _prepare_responses_api_schema(item)
            for key, item in value.items()
            if key != "uniqueItems"
        }
        if "const" in prepared and "type" not in prepared:
            prepared["type"] = _json_type_for_const(prepared["const"])
        return prepared
    if isinstance(value, list):
        return [_prepare_responses_api_schema(item) for item in value]
    return value


def _extract_output_text(response: Any) -> str:
    """Extract text from an OpenAI Responses result or compatible mock."""
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text:
        return output_text

    pieces = []
    refusals = []
    for output_item in getattr(response, "output", []) or []:
        for content_item in getattr(output_item, "content", []) or []:
            text_value = getattr(content_item, "text", None)
            if isinstance(text_value, str):
                pieces.append(text_value)
            if getattr(content_item, "type", None) == "refusal":
                refusal = getattr(content_item, "refusal", None)
                if isinstance(refusal, str):
                    refusals.append(refusal)
    if pieces:
        return "".join(pieces)
    if refusals:
        return "MODEL_REFUSAL: {}".format(" ".join(refusals))
    if getattr(response, "status", None) == "incomplete":
        details = getattr(response, "incomplete_details", None)
        reason = getattr(details, "reason", None) if details is not None else None
        return "MODEL_RESPONSE_INCOMPLETE: {}".format(reason or "unknown reason")
    return ""


def call_model(
    client: Any,
    *,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    schema: Mapping[str, Any],
    background_images: Sequence[Mapping[str, str]] = (),
    reasoning_effort: str = DEFAULT_ANNOTATION_REASONING_EFFORT,
) -> str:
    """Call the Responses API with a strict JSON-schema output request."""
    if reasoning_effort not in SUPPORTED_REASONING_EFFORTS:
        raise ValueError(
            "reasoning_effort must be one of: {}".format(
                ", ".join(sorted(SUPPORTED_REASONING_EFFORTS))
            )
        )

    user_content = [{"type": "input_text", "text": user_prompt}]
    for item in background_images:
        label = str(item.get("label", "Background image")).strip()
        image_url = str(item.get("image_url", "")).strip()
        if not image_url.startswith("data:image/"):
            raise ValueError("background image must be a base64 image data URL")
        user_content.extend(
            [
                {
                    "type": "input_text",
                    "text": "BACKGROUND_IMAGE: {}".format(label),
                },
                {"type": "input_image", "image_url": image_url},
            ]
        )

    response = client.responses.create(
        model=model_name,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "lstar_domain_annotation",
                "strict": True,
                "schema": _prepare_responses_api_schema(schema),
            }
        },
        reasoning={"effort": reasoning_effort},
        store=False,
    )
    return _extract_output_text(response)


def create_openai_client(api_key: Optional[str], api_base: Optional[str]) -> Any:
    """Create an OpenAI client lazily so deterministic utilities stay offline."""
    resolved_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_key:
        raise ValueError(
            "OpenAI API key not provided. Use api_key or set OPENAI_API_KEY."
        )
    try:
        from openai import OpenAI
    except ImportError as error:
        raise ImportError(
            "The openai package is required for domain annotation"
        ) from error
    client = (
        OpenAI(api_key=resolved_key, base_url=api_base)
        if api_base
        else OpenAI(api_key=resolved_key)
    )
    if not hasattr(client, "responses"):
        raise RuntimeError(
            "The installed openai package does not provide the Responses API"
        )
    return client


__all__ = [
    "DEFAULT_ANNOTATION_MODEL",
    "DEFAULT_ANNOTATION_REASONING_EFFORT",
    "SUPPORTED_REASONING_EFFORTS",
    "call_model",
    "create_openai_client",
]
