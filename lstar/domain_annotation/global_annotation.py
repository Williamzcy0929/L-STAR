"""Single-call, dataset-level LLM naming for L-STAR domains.

v3 replaces the earlier two-stage flow (per-domain candidate generation,
schema-bound to evidence-eligible genes, followed by a dataset-level
reconciliation call that selected among candidates and assigned an integrated
confidence label) with one dataset-level naming call: every evaluable domain
and its ranked positive markers are sent together, and the model returns a
name per domain. There is no separate candidate-generation step, no
candidate-selection step, and no annotation-confidence output.

This follows the "one simple prompt" pattern validated by GPTCelltype (Hou &
Ji, *Nature Methods*, 2024), which found that asking an LLM to name a cluster
directly from its top differentially expressed marker genes, in a single
call, performs on par with or better than heavier multi-step pipelines —
including on the Wilcoxon-ranked top genes used here (GPTCelltype reports
Wilcoxon as the best-performing marker-ranking method for this task). The
prompt in ``skills/domain_annotation/annotation.prompt.md`` adapts that
pattern to spatial domains: it still needs granularity guidance (a domain is
not a single cell type), a sampling-level branch (spot vs. cell), optional
dataset ``notes``, and an explicit abstention path, none of which a bare
marker list requires for GPTCelltype's per-cluster case — but it asks for
exactly one thing per domain: a name.

Evidence is positive-marker-only: there is no negative-marker or
contradiction-evidence pathway anywhere in this module. A domain with no
usable positive-marker evidence, or that failed a deterministic upstream
evidence gate, is named "Unknown" without spending an LLM call on it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

if __package__:
    from .llm_annotation import (
        DEFAULT_ANNOTATION_MODEL,
        DEFAULT_ANNOTATION_REASONING_EFFORT,
        call_model,
        create_openai_client,
    )
    from .output_utils import (
        domain_name_contains_annotation_confidence,
        domain_name_references_assignment_id,
    )
else:
    from llm_annotation import (
        DEFAULT_ANNOTATION_MODEL,
        DEFAULT_ANNOTATION_REASONING_EFFORT,
        call_model,
        create_openai_client,
    )
    from output_utils import (
        domain_name_contains_annotation_confidence,
        domain_name_references_assignment_id,
    )


GLOBAL_ANNOTATION_RULES_VERSION = "3.0.0"
RESOURCE_DIR = Path(__file__).resolve().parent / "skills" / "domain_annotation"
_MAX_DOMAIN_NAME_LENGTH = 120


@dataclass(frozen=True)
class GlobalAnnotationResources:
    annotation_prompt: str
    annotation_schema: Mapping[str, Any]
    version: str


@dataclass(frozen=True)
class DatasetAnnotationDecision:
    """Outcome of one dataset's naming run.

    ``domains`` is ordered exactly like the input evidence cards; each entry
    is ``{"domain_id", "domain_name", "named_by"}`` where ``named_by`` is one
    of ``"llm"``, ``"deterministic_unknown"`` (gated before any call was
    made), or ``"fallback_unknown"`` (the LLM call's response never passed
    validation, including repair attempts). ``payload`` is the exact user-
    prompt payload sent with the single annotation call (see
    :func:`_build_annotation_payload`), or ``None`` when every domain was
    gated deterministically and no call was made. ``response`` is the
    validated (or fallback) JSON payload returned by that call, covering only
    the domains that were actually sent; it is ``{"domains": []}`` when
    ``payload`` is ``None``. ``raw_responses`` holds every raw model response
    string across the original attempt and any repair attempts.
    ``deterministic_unknown_domain_ids`` lists the domains that never reached
    the model.
    """

    domains: Tuple[Mapping[str, Any], ...]
    payload: Optional[Mapping[str, Any]]
    response: Mapping[str, Any]
    raw_responses: Tuple[str, ...]
    validation_errors: Tuple[str, ...]
    deterministic_unknown_domain_ids: Tuple[str, ...]
    fallback_used: bool


def load_global_annotation_resources(
    resource_dir: Optional[Path] = None,
) -> GlobalAnnotationResources:
    root = Path(resource_dir) if resource_dir is not None else RESOURCE_DIR
    paths = {
        "annotation_prompt": root / "annotation.prompt.md",
        "annotation_schema": root / "annotation.schema.json",
        "version": root / "VERSION",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing global annotation resources: {}".format(", ".join(missing))
        )
    return GlobalAnnotationResources(
        annotation_prompt=paths["annotation_prompt"].read_text(
            encoding="utf-8"
        ).strip(),
        annotation_schema=json.loads(
            paths["annotation_schema"].read_text(encoding="utf-8")
        ),
        version=paths["version"].read_text(encoding="utf-8").strip(),
    )


def _schema_validate(payload: Any, schema: Mapping[str, Any]) -> None:
    try:
        import jsonschema  # type: ignore
    except ImportError:
        jsonschema = None
    if jsonschema is not None:
        try:
            jsonschema.validate(payload, dict(schema))
        except jsonschema.exceptions.ValidationError as error:
            raise ValueError("response schema validation failed: {}".format(error))
    if not isinstance(payload, dict):
        raise ValueError("response must be one JSON object")


def _call_with_repair(
    *,
    client: Any,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    schema: Mapping[str, Any],
    validator: Any,
    payload_normalizer: Optional[
        Callable[[Mapping[str, Any]], Mapping[str, Any]]
    ] = None,
    validation_failure_fallback: Optional[
        Callable[[Sequence[str]], Mapping[str, Any]]
    ] = None,
    background_images: Sequence[Mapping[str, str]] = (),
    reasoning_effort: str = DEFAULT_ANNOTATION_REASONING_EFFORT,
    max_validation_retries: int = 2,
) -> Tuple[Mapping[str, Any], List[str], List[str]]:
    raw_responses: List[str] = []
    validation_errors: List[str] = []
    prior_raw = ""
    for attempt in range(max_validation_retries + 1):
        prompt = user_prompt
        if attempt:
            prompt = (
                "{base}\n\nREPAIR_REQUEST\nThe previous response was invalid: "
                "{error}. Correct it without adding unsupported evidence and "
                "return JSON only. Previous response:\n{raw}"
            ).format(
                base=user_prompt,
                error=validation_errors[-1],
                raw=prior_raw,
            )
        try:
            raw = call_model(
                client,
                model_name=model_name,
                system_prompt=system_prompt,
                user_prompt=prompt,
                schema=schema,
                background_images=background_images,
                reasoning_effort=reasoning_effort,
            )
        except Exception as error:
            raise RuntimeError("Domain-annotation API call failed") from error
        prior_raw = raw
        raw_responses.append(raw)
        try:
            payload = json.loads(raw)
            if payload_normalizer is not None:
                payload = payload_normalizer(payload)
            _schema_validate(payload, schema)
            validator(payload)
            return payload, raw_responses, validation_errors
        except (
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            validation_errors.append(str(error))
    if validation_failure_fallback is not None:
        fallback_payload = validation_failure_fallback(validation_errors)
        _schema_validate(fallback_payload, schema)
        validator(fallback_payload)
        return fallback_payload, raw_responses, validation_errors
    raise RuntimeError(
        "LLM response remained invalid after {} attempts: {}".format(
            max_validation_retries + 1,
            validation_errors[-1] if validation_errors else "unknown error",
        )
    )


def _deterministic_unknown_reason(card: Mapping[str, Any]) -> Optional[str]:
    """Return why a domain must be Unknown without an LLM call, or None.

    A domain is gated deterministically, rather than sent to the model, when
    an upstream evidence gate already required abstention or when there is no
    positive-marker evidence at all to name it from. Both are structural
    properties of the evidence card, established before any LLM call.
    """
    gate = card.get("abstention_gate", {})
    if bool(gate.get("required", False)):
        reasons = [
            str(reason) for reason in gate.get("reasons", []) if str(reason).strip()
        ]
        return (
            "; ".join(reasons)
            if reasons
            else "A deterministic evidence gate required abstention."
        )
    if not card.get("positive_markers"):
        return "No positive marker evidence is available for this domain."
    return None


def _build_annotation_payload(
    sendable_cards: Sequence[Mapping[str, Any]],
    *,
    dataset_context: str,
) -> Dict[str, Any]:
    """Build the single dataset-level user-prompt payload.

    ``biological_context``, ``notes``, and ``sampling_level`` are dataset-wide
    and taken from the first card; every evidence card in one
    :func:`run_global_annotation` call shares the same dataset, so their
    values are identical across cards. Each domain contributes only its ID
    and its already Wilcoxon-ranked, deduplicated positive-marker gene list
    (:func:`annotation._deduplicate_biological_markers`) — no scores, tiers,
    or other per-marker metadata are sent, matching GPTCelltype's minimal
    marker-list input.
    """
    first_card = sendable_cards[0]
    return {
        "dataset_context": dataset_context,
        "biological_context": first_card.get("biological_context"),
        "notes": first_card.get("notes"),
        "sampling_level": first_card.get("sampling_level"),
        "domains": [
            {
                "domain_id": str(card["domain_id"]),
                "markers": [
                    str(marker["gene"]) for marker in card.get("positive_markers", [])
                ],
            }
            for card in sendable_cards
        ],
    }


def _validate_annotation_response(
    payload: Mapping[str, Any],
    expected_domain_ids: Sequence[str],
) -> None:
    domains = payload.get("domains")
    if not isinstance(domains, list):
        raise ValueError("annotation domains must be an array")
    observed = [str(domain.get("domain_id")) for domain in domains]
    if observed != list(expected_domain_ids):
        raise ValueError(
            "annotation domain order must exactly match the input order"
        )
    for domain in domains:
        domain_id = str(domain.get("domain_id"))
        domain_name = str(domain.get("domain_name", "")).strip()
        if not domain_name:
            raise ValueError("domain_name must be nonempty")
        if len(domain_name) > _MAX_DOMAIN_NAME_LENGTH:
            raise ValueError(
                "domain_name must be at most {} characters".format(
                    _MAX_DOMAIN_NAME_LENGTH
                )
            )
        if domain_name.lower() == "uncharacterized domain":
            raise ValueError(
                "domain_name must not be the generic placeholder "
                "'Uncharacterized domain'"
            )
        if domain_name_contains_annotation_confidence(domain_name):
            raise ValueError(
                "domain_name must not encode annotation confidence: {!r}".format(
                    domain_name
                )
            )
        if domain_name_references_assignment_id(domain_name, domain_id):
            raise ValueError(
                "domain_name must not explicitly reference its L-STAR "
                "assignment ID: {!r}".format(domain_name)
            )


def _deterministic_all_unknown_response(
    domain_ids: Sequence[str],
) -> Dict[str, Any]:
    """Build the all-Unknown fallback used once repair attempts are exhausted.

    Keeps the run from ever crashing on persistently invalid model output:
    every domain that was sent to the model is instead named "Unknown", and
    the accumulated validation errors remain available on the returned
    :class:`DatasetAnnotationDecision` for audit.
    """
    return {
        "domains": [
            {"domain_id": str(domain_id), "domain_name": "Unknown"}
            for domain_id in domain_ids
        ],
    }


def run_global_annotation(
    evidence_cards: Sequence[Mapping[str, Any]],
    *,
    dataset_context: str,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    model_name: Optional[str] = None,
    client: Any = None,
    background_images: Sequence[Mapping[str, str]] = (),
    reasoning_effort: str = DEFAULT_ANNOTATION_REASONING_EFFORT,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> DatasetAnnotationDecision:
    """Name every domain in one dataset with a single LLM call.

    Domains with ``abstention_gate.required`` or no positive-marker evidence
    are named "Unknown" deterministically and are never included in the
    payload; if every domain is gated this way, no LLM call is made at all.
    Otherwise the remaining domains are sent together in one call (see
    :func:`_build_annotation_payload`), validated (see
    :func:`_validate_annotation_response`), retried with a repair prompt up
    to twice on invalid output, and — only if every attempt still fails —
    named "Unknown" by a deterministic fallback so the run always completes.
    """
    if not evidence_cards:
        raise ValueError("evidence_cards must not be empty")

    def progress(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    ordered_domain_ids = [str(card["domain_id"]) for card in evidence_cards]
    gated_reasons: Dict[str, str] = {}
    sendable_cards: List[Mapping[str, Any]] = []
    for card in evidence_cards:
        domain_id = str(card["domain_id"])
        reason = _deterministic_unknown_reason(card)
        if reason is not None:
            gated_reasons[domain_id] = reason
        else:
            sendable_cards.append(card)

    if gated_reasons:
        progress(
            "{} of {} domain(s) are deterministically Unknown before any LLM "
            "call (no usable positive-marker evidence).".format(
                len(gated_reasons), len(evidence_cards)
            )
        )

    llm_named: Dict[str, str] = {}
    payload: Optional[Dict[str, Any]] = None
    response: Dict[str, Any] = {"domains": []}
    raw_responses: List[str] = []
    validation_errors: List[str] = []
    fallback_state = {"used": False}

    if sendable_cards:
        resources = load_global_annotation_resources()
        active_client = client or create_openai_client(api_key, api_base)
        active_model = model_name or DEFAULT_ANNOTATION_MODEL
        sendable_domain_ids = [str(card["domain_id"]) for card in sendable_cards]
        payload = _build_annotation_payload(
            sendable_cards, dataset_context=dataset_context
        )
        user_prompt = "DATASET_ANNOTATION_INPUT\n{}".format(
            json.dumps(payload, ensure_ascii=False, allow_nan=False)
        )

        def _fallback(errors: Sequence[str]) -> Mapping[str, Any]:
            fallback_state["used"] = True
            return _deterministic_all_unknown_response(sendable_domain_ids)

        progress(
            "Naming {} domain(s) in a single dataset-level call...".format(
                len(sendable_cards)
            )
        )
        response, raw_responses, validation_errors = _call_with_repair(
            client=active_client,
            model_name=active_model,
            system_prompt=resources.annotation_prompt,
            user_prompt=user_prompt,
            schema=resources.annotation_schema,
            validator=lambda candidate_response: _validate_annotation_response(
                candidate_response, sendable_domain_ids
            ),
            validation_failure_fallback=_fallback,
            background_images=background_images,
            reasoning_effort=reasoning_effort,
        )
        for domain in response["domains"]:
            llm_named[str(domain["domain_id"])] = str(domain["domain_name"])
        progress("Dataset annotation call complete.")
        progress("  - Validation retries: {}".format(len(validation_errors)))
        if fallback_state["used"]:
            progress(
                "  - Response remained invalid after every retry; affected "
                "domains were named Unknown deterministically."
            )

    fallback_used = fallback_state["used"]
    final_domains = []
    for index, domain_id in enumerate(ordered_domain_ids, start=1):
        if domain_id in gated_reasons:
            name = "Unknown"
            named_by = "deterministic_unknown"
        else:
            name = llm_named[domain_id]
            named_by = "fallback_unknown" if fallback_used else "llm"
        final_domains.append(
            {"domain_id": domain_id, "domain_name": name, "named_by": named_by}
        )
        progress(
            "[{}/{}] Domain {} -> {}".format(
                index, len(ordered_domain_ids), domain_id, name
            )
        )

    return DatasetAnnotationDecision(
        domains=tuple(final_domains),
        payload=payload,
        response=response,
        raw_responses=tuple(raw_responses),
        validation_errors=tuple(validation_errors),
        deterministic_unknown_domain_ids=tuple(gated_reasons),
        fallback_used=fallback_used,
    )


__all__ = [
    "DatasetAnnotationDecision",
    "GLOBAL_ANNOTATION_RULES_VERSION",
    "load_global_annotation_resources",
    "run_global_annotation",
]
