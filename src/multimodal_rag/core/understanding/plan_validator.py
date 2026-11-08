import math
from typing import Any, Dict, List, Optional

from jsonschema import Draft202012Validator, ValidationError

from .schemas import (
    EXECUTION_CONTEXT_SCHEMA,
    RETRIEVAL_MODES,
)


WORKSPACE_MODALITIES = {
    "All sources": {
        "document",
        "image",
        "video_transcript",
        "video_frames",
    },
    "PDFs and text only": {
        "document",
    },
    "Uploaded videos only": {
        "video_transcript",
        "video_frames",
    },
    "Uploaded images only": {
        "image",
    },
}


def validate_execution_plan(plan: Any) -> Dict[str, Any]:
    """
    Validate execution-plan structure and semantic consistency.

    This function never modifies the supplied plan.
    """

    errors: List[str] = []

    if not isinstance(plan, dict):
        return {
            "valid": False,
            "errors": ["Execution plan must be a dictionary."],
        }

    validator = Draft202012Validator(EXECUTION_CONTEXT_SCHEMA)

    try:
        validator.validate(plan)
    except ValidationError as exc:
        errors.append(
            f"Schema validation failed: {exc.message}"
        )

    if errors:
        return {
            "valid": False,
            "errors": errors,
        }

    evidence = plan["evidence_plan"]
    conversation = plan["conversation_context"]
    clarification = plan["clarification"]
    uncertainty = plan["uncertainty"]

    retrieval_mode = evidence["retrieval_mode"]
    retrieval_required = evidence["retrieval_required"]

    # Retrieval mode and retrieval flag must agree.
    expected_retrieval = retrieval_mode != "no_retrieval"

    if retrieval_required != expected_retrieval:
        errors.append(
            "retrieval_required must be False for no_retrieval "
            "and True for retrieve or hybrid."
        )

    # General knowledge requests must not request uploaded evidence.
    if retrieval_mode == "no_retrieval":
        if evidence["required_modalities"]:
            errors.append(
                "no_retrieval cannot request evidence modalities."
            )

    # Confidence must be a finite number.
    confidence = uncertainty["confidence"]

    if not math.isfinite(confidence):
        errors.append(
            "Uncertainty confidence must be a finite number."
        )

    # Required clarification must include a question.
    if clarification["needed"]:
        if not clarification["question"] or not clarification["question"].strip():
            errors.append(
                "A clarification question is required."
            )

    # Uncertainty must agree with clarification requirements.
    if (
        uncertainty["requires_clarification"]
        and not clarification["needed"]
    ):
        errors.append(
            "Uncertainty requires clarification, but clarification is not marked as needed."
        )

    # A clarification requirement must not be silently ignored.
    if (
        clarification["needed"]
        and not uncertainty["requires_clarification"]
    ):
        errors.append(
            "Clarification is needed, but uncertainty does not mark it as required."
        )

    # Conversation context cannot be required without a stated reason
    # in the plan's request type or task.
    if conversation["required"] and not conversation["resolved_references"]:
        relevant_context = conversation["relevant_context"]

        if not relevant_context:
            errors.append(
                "Conversation context is required, but no references "
                "or relevant context were identified."
            )

    return {
        "valid": not errors,
        "errors": errors,
    }


def _source_available(
    available_sources: Dict[str, Any],
    modality: str,
) -> bool:
    """Check availability of a specific evidence modality."""

    if modality in available_sources:
        return bool(available_sources[modality])

    # Backward compatibility with the existing combined video store.
    # This fallback applies only when modality-specific information
    # has not been supplied.
    if modality in ("video_transcript", "video_frames"):
        return bool(available_sources.get("video", False))

    return False


def check_execution_readiness(
    plan: Dict[str, Any],
    available_sources: Optional[Dict[str, Any]] = None,
    source_mode: str = "All sources",
    selected_document: Optional[str] = None,
    question_scope: Optional[str] = None,
    conversation_available: bool = False,
) -> Dict[str, Any]:
    """
    Check whether the application can execute a validated plan.

    Expected available_sources keys:
        document
        image
        video
        video_transcript (optional)
        video_frames (optional)

    Values should indicate whether usable evidence exists.
    This function does not retrieve or modify any evidence.
    """

    blockers: List[str] = []
    warnings: List[str] = []

    validation = validate_execution_plan(plan)

    if not validation["valid"]:
        blockers.extend(validation["errors"])

        return {
            "ready": False,
            "blockers": blockers,
            "warnings": warnings,
        }

    if source_mode not in WORKSPACE_MODALITIES:
        return {
            "ready": False,
            "blockers": [
                f"Unsupported workspace: {source_mode}"
            ],
            "warnings": warnings,
        }

    if available_sources is None:
        available_sources = {}

    if not isinstance(available_sources, dict):
        return {
            "ready": False,
            "blockers": [
                "available_sources must be a dictionary."
            ],
            "warnings": warnings,
        }

    evidence = plan["evidence_plan"]
    conversation = plan["conversation_context"]
    constraints = plan["constraints"]

    permitted_modalities = WORKSPACE_MODALITIES[source_mode]

    # The application controls source permissions.
    for modality in evidence["required_modalities"]:

        if modality not in permitted_modalities:
            blockers.append(
                f"Modality '{modality}' is not permitted "
                f"in workspace '{source_mode}'."
            )
            continue

        if not _source_available(available_sources, modality):
            blockers.append(
                f"Required evidence is unavailable: {modality}."
            )

    # Selected-document execution has its own application path.
    selected_document_scope = (
        question_scope == "Selected document"
    )

    if (
        selected_document_scope
        or constraints["must_use_selected_document"]
    ):
        if not selected_document:
            blockers.append(
                "A selected document is required but none is available."
            )

        if not _source_available(available_sources, "document"):
            blockers.append(
                "The selected-document evidence store is unavailable."
            )

    if (
        constraints["must_use_uploaded_sources"]
        and not evidence["retrieval_required"]
    ):
        blockers.append(
            "Uploaded sources are required, but retrieval is disabled."
        )

    if (
        conversation["required"]
        and not conversation_available
    ):
        blockers.append(
            "Conversation context is required but unavailable."
        )

    # The plan's source scope is informational. The application-provided
    # workspace remains authoritative.
    if evidence["source_scope"] != source_mode:
        warnings.append(
            "Plan source_scope differs from the application's active "
            "workspace. Application workspace permissions take priority."
        )

    return {
        "ready": not blockers,
        "blockers": blockers,
        "warnings": warnings,
    }
