from typing import Any, Dict


REQUEST_TYPES = [
    "general_knowledge",
    "source_grounded",
    "mixed_multimodal",
    "conversational",
]

INTENTS = [
    "factual_question",
    "explanation",
    "summarization",
    "comparison",
    "extraction",
    "description",
    "analysis",
    "conversation",
    "other",
]

OPERATIONS = [
    "answer",
    "explain",
    "summarize",
    "compare",
    "extract",
    "describe",
    "analyze",
    "clarify",
]

RETRIEVAL_MODES = [
    "no_retrieval",
    "retrieve",
    "hybrid",
]

EVIDENCE_MODALITIES = [
    "document",
    "image",
    "video_transcript",
    "video_frames",
]

RESPONSE_FORMATS = [
    "paragraph",
    "numbered_list",
    "bullet_list",
    "table",
    "structured",
]

DETAIL_LEVELS = [
    "concise",
    "appropriate",
    "detailed",
    "comprehensive",
]

UNCERTAINTY_LEVELS = [
    "low",
    "medium",
    "high",
]


EXECUTION_CONTEXT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "request_type": {
            "type": "string",
            "enum": REQUEST_TYPES,
        },
        "conversation_context": {
            "type": "object",
            "properties": {
                "required": {"type": "boolean"},
                "resolved_references": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "relevant_context": {
                    "type": ["string", "null"],
                },
            },
            "required": [
                "required",
                "resolved_references",
                "relevant_context",
            ],
            "additionalProperties": False,
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
                "source_scope": {
                    "type": "string",
                    "minLength": 1,
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
                "source_scope",
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
        "clarification": {
            "type": "object",
            "properties": {
                "needed": {"type": "boolean"},
                "question": {
                    "type": ["string", "null"],
                },
            },
            "required": [
                "needed",
                "question",
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
                "requires_clarification": {
                    "type": "boolean",
                },
            },
            "required": [
                "confidence",
                "level",
                "uncertain_fields",
                "reason",
                "requires_clarification",
            ],
            "additionalProperties": False,
        },
    },
    "required": [
        "request_type",
        "conversation_context",
        "task_plan",
        "evidence_plan",
        "response_plan",
        "constraints",
        "clarification",
        "uncertainty",
    ],
    "additionalProperties": False,
}


def get_execution_context_schema() -> Dict[str, Any]:
    """Return the execution-context schema."""

    return EXECUTION_CONTEXT_SCHEMA
