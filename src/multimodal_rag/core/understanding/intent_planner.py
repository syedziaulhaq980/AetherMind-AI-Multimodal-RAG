import json
from typing import Any, Dict, Optional

from ..llm import generate_ollama_response
from .schemas import (
    REQUEST_TYPES,
    INTENTS,
    OPERATIONS,
    RETRIEVAL_MODES,
    EVIDENCE_MODALITIES,
    RESPONSE_FORMATS,
    DETAIL_LEVELS,
    UNCERTAINTY_LEVELS,
    EXECUTION_CONTEXT_SCHEMA,
)
from .plan_validator import validate_execution_plan


INTENT_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "request_type": {
            "type": "string",
            "enum": REQUEST_TYPES,
        },
        "task_plan": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": INTENTS,
                },
                "operation": {
                    "type": "string",
                    "enum": OPERATIONS,
                },
                "normalized_query": {
                    "type": "string",
                    "minLength": 1,
                },
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "intent",
                "operation",
                "normalized_query",
                "entities",
            ],
            "additionalProperties": False,
        },
        "evidence_plan": {
            "type": "object",
            "properties": {
                "retrieval_mode": {
                    "type": "string",
                    "enum": RETRIEVAL_MODES,
                },
                "retrieval_required": {
                    "type": "boolean",
                },
                "required_modalities": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": EVIDENCE_MODALITIES,
                    },
                },
                "evidence_requirements": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "general_knowledge_allowed": {
                    "type": "boolean",
                },
            },
            "required": [
                "retrieval_mode",
                "retrieval_required",
                "required_modalities",
                "evidence_requirements",
                "general_knowledge_allowed",
            ],
            "additionalProperties": False,
        },
        "response_plan": {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "enum": RESPONSE_FORMATS,
                },
                "requested_count": {
                    "type": ["integer", "null"],
                    "minimum": 1,
                },
                "detail_level": {
                    "type": "string",
                    "enum": DETAIL_LEVELS,
                },
                "language": {
                    "type": "string",
                    "minLength": 1,
                },
            },
            "required": [
                "format",
                "requested_count",
                "detail_level",
                "language",
            ],
            "additionalProperties": False,
        },
        "constraints": {
            "type": "object",
            "properties": {
                "must_use_selected_document": {
                    "type": "boolean",
                },
                "must_use_uploaded_sources": {
                    "type": "boolean",
                },
                "citation_required": {
                    "type": "boolean",
                },
            },
            "required": [
                "must_use_selected_document",
                "must_use_uploaded_sources",
                "citation_required",
            ],
            "additionalProperties": False,
        },
        "uncertainty": {
            "type": "object",
            "properties": {
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "level": {
                    "type": "string",
                    "enum": UNCERTAINTY_LEVELS,
                },
                "uncertain_fields": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "reason": {
                    "type": ["string", "null"],
                },
            },
            "required": [
                "confidence",
                "level",
                "uncertain_fields",
                "reason",
            ],
            "additionalProperties": False,
        },
    },
    "required": [
        "request_type",
        "task_plan",
        "evidence_plan",
        "response_plan",
        "constraints",
        "uncertainty",
    ],
    "additionalProperties": False,
}


def _serialize_context(value: Any) -> str:
    """Convert context data into readable JSON."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            default=str,
            indent=2,
        )
    except (TypeError, ValueError):
        return str(value)


def _uncertainty_level(confidence: float) -> str:
    """Map confidence to a consistent uncertainty level."""

    if confidence >= 0.80:
        return "low"

    if confidence >= 0.50:
        return "medium"

    return "high"


def _validate_or_raise(plan: Dict[str, Any]) -> None:
    """Reject invalid execution plans."""

    validation = validate_execution_plan(plan)

    if not validation["valid"]:
        errors = "\n".join(
            f"- {error}"
            for error in validation["errors"]
        )

        raise ValueError(
            f"Invalid execution plan:\n{errors}"
        )


def _build_clarification_context(
    question: str,
    context_interpretation: Dict[str, Any],
    source_scope: str,
) -> Dict[str, Any]:
    """Build a safe execution context when conversation is ambiguous."""

    confidence = float(
        context_interpretation.get("confidence", 0.5)
    )

    clarification_question = context_interpretation.get(
        "clarification_question"
    )

    reasoning = context_interpretation.get(
        "reasoning_summary",
        "The conversation reference could not be resolved confidently.",
    )

    plan = {
        "request_type": "conversational",
        "conversation_context": {
            "required": False,
            "resolved_references": [],
            "relevant_context": None,
        },
        "task_plan": {
            "intent": "other",
            "operation": "clarify",
            "normalized_query": (
                context_interpretation.get("normalized_query")
                or question
            ),
            "entities": [],
        },
        "evidence_plan": {
            "retrieval_mode": "no_retrieval",
            "retrieval_required": False,
            "required_modalities": [],
            "source_scope": source_scope,
            "evidence_requirements": [],
            "general_knowledge_allowed": False,
        },
        "response_plan": {
            "format": "paragraph",
            "requested_count": None,
            "detail_level": "appropriate",
            "language": "same_as_user",
        },
        "constraints": {
            "must_use_selected_document": False,
            "must_use_uploaded_sources": False,
            "citation_required": False,
        },
        "clarification": {
            "needed": True,
            "question": clarification_question,
        },
        "uncertainty": {
            "confidence": confidence,
            "level": _uncertainty_level(confidence),
            "uncertain_fields": ["conversation_reference"],
            "reason": reasoning,
            "requires_clarification": True,
        },
    }

    _validate_or_raise(plan)
    return plan


def build_execution_context(
    question: str,
    context_interpretation: Dict[str, Any],
    source_scope: str = "All sources",
    selected_document: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a complete execution context from the question
    and the conversation interpreter's result.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    if not isinstance(context_interpretation, dict):
        raise TypeError(
            "context_interpretation must be a dictionary."
        )

    required_context_fields = [
        "context_required",
        "normalized_query",
        "resolved_references",
        "relevant_context",
        "ambiguity_detected",
        "clarification_needed",
        "clarification_question",
        "confidence",
    ]

    missing_fields = [
        field
        for field in required_context_fields
        if field not in context_interpretation
    ]

    if missing_fields:
        raise ValueError(
            f"Missing context interpretation fields: {missing_fields}"
        )

    if context_interpretation["clarification_needed"]:
        return _build_clarification_context(
            question=question,
            context_interpretation=context_interpretation,
            source_scope=source_scope,
        )

    normalized_query = (
        context_interpretation["normalized_query"] or question
    )

    context_confidence = float(
        context_interpretation["confidence"]
    )

    if not 0.0 <= context_confidence <= 1.0:
        raise ValueError(
            "Context confidence must be between 0 and 1."
        )

    planning_input = {
        "original_question": question,
        "normalized_query": normalized_query,
        "conversation_context_required": (
            context_interpretation["context_required"]
        ),
        "resolved_references": (
            context_interpretation["resolved_references"]
        ),
        "relevant_context": (
            context_interpretation["relevant_context"]
        ),
        "source_scope": source_scope,
        "selected_document": selected_document,
    }

    prompt = f"""
You are the intent and execution planner for a multimodal RAG system.

Your task is to create an execution plan using the supplied
question and conversation interpretation.

Do not reinterpret or overwrite resolved conversation references.
Use the normalized query supplied to you.

PLANNING INPUT:
{_serialize_context(planning_input)}

PLANNING RULES:

1. Determine the request type:
   - general_knowledge
   - source_grounded
   - mixed_multimodal
   - conversational

2. Determine the intent and operation based on what the user
   actually requests.

3. Decide whether uploaded evidence is necessary.
   - no_retrieval: no uploaded evidence is needed.
   - retrieve: uploaded evidence is required.
   - hybrid: uploaded evidence and general knowledge are both useful.

4. Select only the evidence modalities genuinely needed:
   - document
   - image
   - video_transcript
   - video_frames

5. Respect the supplied source scope.
   Never request evidence outside the user's workspace scope.

6. Distinguish between:
   - describing an uploaded image,
   - answering a question about an image,
   - explaining information from a document,
   - answering questions about a video's transcript,
   - analyzing visual events or details in video frames.

7. Do not assume every question requires summarization.
   Follow the requested operation and response format.

8. Identify entities and evidence requirements when applicable.

9. Respect explicit user instructions:
   - requested number of items,
   - response format,
   - level of detail,
   - language.

10. Do not invent uploaded evidence or claim that a source exists.

11. Use selected-document constraints when the request requires
    information specifically from that selected document.

12. Set citation_required when retrieved uploaded evidence
    must support the answer.

13. Estimate planning confidence honestly.
    Record uncertain fields and explain uncertainty when relevant.

Return only a JSON object matching the supplied output schema.
"""

    raw_response = generate_ollama_response(
        prompt,
        output_format=INTENT_PLAN_SCHEMA,
    )

    try:
        planning_result = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Intent planner returned invalid JSON."
        ) from exc

    if not isinstance(planning_result, dict):
        raise ValueError(
            "Intent planner must return a JSON object."
        )

    planning_uncertainty = planning_result["uncertainty"]

    planning_confidence = float(
        planning_uncertainty["confidence"]
    )

    if not 0.0 <= planning_confidence <= 1.0:
        raise ValueError(
            "Planning confidence must be between 0 and 1."
        )

    final_confidence = min(
        context_confidence,
        planning_confidence,
    )

    uncertain_fields = list(
        planning_uncertainty["uncertain_fields"]
    )

    if context_interpretation["ambiguity_detected"]:
        uncertain_fields.append("conversation_reference")

    reason = planning_uncertainty["reason"]

    if not reason and context_interpretation["context_required"]:
        reason = context_interpretation.get("reasoning_summary")

    evidence_plan = planning_result["evidence_plan"]

    retrieval_required = evidence_plan["retrieval_required"]

    if evidence_plan["retrieval_mode"] == "no_retrieval":
        retrieval_required = False
        evidence_plan["required_modalities"] = []

    elif evidence_plan["retrieval_mode"] in ("retrieve", "hybrid"):
        retrieval_required = True

    evidence_plan["retrieval_required"] = retrieval_required
    evidence_plan["source_scope"] = source_scope

    constraints = planning_result["constraints"]

    if retrieval_required:
        constraints["must_use_uploaded_sources"] = True
        constraints["citation_required"] = True
    else:
        constraints["must_use_uploaded_sources"] = False
        constraints["citation_required"] = False

    plan = {
        "request_type": planning_result["request_type"],
        "conversation_context": {
            "required": context_interpretation["context_required"],
            "resolved_references": (
                context_interpretation["resolved_references"]
            ),
            "relevant_context": (
                context_interpretation["relevant_context"] or None
            ),
        },
        "task_plan": planning_result["task_plan"],
        "evidence_plan": evidence_plan,
        "response_plan": planning_result["response_plan"],
        "constraints": constraints,
        "clarification": {
            "needed": False,
            "question": None,
        },
        "uncertainty": {
            "confidence": final_confidence,
            "level": _uncertainty_level(final_confidence),
            "uncertain_fields": list(dict.fromkeys(uncertain_fields)),
            "reason": reason,
            "requires_clarification": False,
        },
    }

    _validate_or_raise(plan)

    return plan
