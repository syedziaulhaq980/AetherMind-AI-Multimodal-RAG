import json
from typing import Any, Dict

from ..llm import generate_ollama_response


CONTEXT_INTERPRETATION_SCHEMA = {
    "type": "object",
    "properties": {
        "context_required": {
            "type": "boolean"
        },
        "normalized_query": {
            "type": "string"
        },
        "resolved_references": {
            "type": "array",
            "items": {"type": "string"}
        },
        "relevant_context": {
            "type": "string"
        },
        "ambiguity_detected": {
            "type": "boolean"
        },
        "clarification_needed": {
            "type": "boolean"
        },
        "clarification_question": {
            "type": "string"
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1
        },
        "reasoning_summary": {
            "type": "string"
        }
    },
    "required": [
        "context_required",
        "normalized_query",
        "resolved_references",
        "relevant_context",
        "ambiguity_detected",
        "clarification_needed",
        "clarification_question",
        "confidence",
        "reasoning_summary"
    ],
    "additionalProperties": False
}


def interpret_conversation_context(
    question: str,
    conversation_history: str = "",
    persistent_memory: str = "",
) -> Dict[str, Any]:
    """
    Interpret the current question using relevant conversation context.

    This function does not retrieve evidence, generate final answers,
    or modify conversation memory.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("The current question must be a non-empty string.")

    if not isinstance(conversation_history, str):
        raise TypeError("conversation_history must be a string.")

    if not isinstance(persistent_memory, str):
        raise TypeError("persistent_memory must be a string.")


    prompt = f"""
You are the conversation-context interpretation component
of a multimodal RAG system.

Your responsibility is to interpret the CURRENT QUESTION accurately
using conversation context only when it is genuinely necessary.

You receive three independent inputs:

1. CURRENT QUESTION
2. RECENT CONVERSATION HISTORY
3. PERSISTENT MEMORY

CORE DECISION RULES:

1. CONTEXT DEPENDENCY

Set context_required to true ONLY when understanding the current
question depends on information from previous conversation or
persistent memory.

A question is independent when it is understandable on its own.

A shared topic or subject does NOT automatically mean context
is required.

For independent questions:
- context_required = false
- resolved_references = []
- relevant_context = ""
- Do not incorporate unrelated conversation history.

2. REFERENCE RESOLUTION

When the current question contains references such as:
"it", "that", "this", "the previous one", or "explain it again",
look for their actual meaning in the supplied context.

When the reference has one clearly supported interpretation:
- context_required = true
- Resolve the reference.
- Rewrite normalized_query as a complete standalone question.
- Record the resolved reference.
- Include only the necessary supporting context.

Do not leave unresolved pronouns in normalized_query when
their meaning is clearly established.

3. AMBIGUITY DETECTION

When a reference has multiple plausible interpretations,
do NOT arbitrarily select one.

For example, if the conversation discusses machine learning,
deep learning, and neural networks, and the user asks:
"Explain that one again."

There are multiple possible referents.

In this situation:
- ambiguity_detected = true
- clarification_needed = true
- confidence should reflect the uncertainty.
- Ask one specific clarification question.
- Do not invent a normalized interpretation.

4. RELEVANT CONTEXT

Return only information necessary to interpret the current question.

Do not copy the entire conversation history.
Do not include unrelated topics.
Do not treat the existence of conversation history as evidence
that the current question depends on it.

5. PERSISTENT MEMORY

Use persistent memory only when it genuinely helps resolve
the current question or an explicit reference.

Do not force persistent memory into unrelated questions.

6. ORIGINAL QUESTION

Do not rewrite, alter, or answer the user's original question.
The application will preserve the exact original wording separately.

7. RESPONSE REQUIREMENTS

- normalized_query must preserve the user's actual intention.
- Never invent missing facts or references.
- Do not generate the final answer.
- Do not perform document, image, or video retrieval.
- reasoning_summary should briefly explain the interpretation.
- Do not expose hidden chain-of-thought.
- If clarification is needed, provide one concise question.



FEW-SHOT INTERPRETATION EXAMPLES:

EXAMPLE 1 — CLEAR REFERENCE

Current question:
"How is it different from deep learning?"

Recent conversation:
"USER: Explain machine learning.
ASSISTANT: Machine learning learns patterns from data."

Correct interpretation:
- context_required = true
- normalized_query = "How is machine learning different from deep learning?"
- resolved_references = ["it = machine learning"]
- relevant_context contains the machine-learning explanation.
- ambiguity_detected = false
- clarification_needed = false
- clarification_question = ""

Reason: "It" has one clearly supported referent.

EXAMPLE 2 — INDEPENDENT QUESTION

Current question:
"What is the capital of Japan?"

Recent conversation:
"USER: Explain machine learning.
ASSISTANT: Machine learning learns patterns from data."

Correct interpretation:
- context_required = false
- normalized_query = "What is the capital of Japan?"
- resolved_references = []
- relevant_context = ""
- ambiguity_detected = false
- clarification_needed = false
- clarification_question = ""

Reason: The question is fully understandable without previous conversation.

EXAMPLE 3 — AMBIGUOUS REFERENCE

Current question:
"Explain that one again."

Recent conversation:
"USER: Explain machine learning.
ASSISTANT: Machine learning learns patterns from data.
USER: What is deep learning?
ASSISTANT: Deep learning uses multilayer neural networks.
USER: What are neural networks?
ASSISTANT: Neural networks are computational models."

Correct interpretation:
- context_required = true
- normalized_query = "Explain that one again."
- resolved_references = []
- relevant_context contains the candidate topics.
- ambiguity_detected = true
- clarification_needed = true
- clarification_question = "Which topic would you like me to explain again?"

Reason: Several topics are plausible referents. Do not guess.


CURRENT QUESTION:
{question}

RECENT CONVERSATION HISTORY:
{conversation_history if conversation_history.strip() else "No conversation history provided."}

PERSISTENT MEMORY:
{persistent_memory if persistent_memory.strip() else "No persistent memory provided."}

Return only the structured result matching the supplied JSON schema.
"""


    raw_response = generate_ollama_response(
        prompt,
        output_format=CONTEXT_INTERPRETATION_SCHEMA,
    )

    try:
        result = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Context interpreter returned invalid JSON."
        ) from exc

    if not isinstance(result, dict):
        raise ValueError("Context interpreter must return a JSON object.")

    print(
        "[CONTEXT INTERPRETER RAW RESULT]",
        json.dumps(result, ensure_ascii=False, indent=2),
        flush=True,
    )

    normalized_query = result.get("normalized_query")

    if not isinstance(normalized_query, str) or not normalized_query.strip():
        raise ValueError("Context interpreter returned an empty normalized query.")

    confidence = result.get("confidence")

    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 <= confidence <= 1
    ):
        raise ValueError("Context interpreter returned an invalid confidence value.")


    # Validate interpretation flags.
    context_required = result.get("context_required")
    ambiguity_detected = result.get("ambiguity_detected")
    clarification_needed = result.get("clarification_needed")

    if not all(
        isinstance(value, bool)
        for value in (
            context_required,
            ambiguity_detected,
            clarification_needed,
        )
    ):
        raise ValueError("Context interpreter returned invalid decision flags.")

    clarification_question = result.get("clarification_question")

    if not isinstance(clarification_question, str):
        raise ValueError("Clarification question must be a string.")

    resolved_references = result.get("resolved_references")
    relevant_context = result.get("relevant_context")

    if not isinstance(resolved_references, list):
        raise ValueError("Resolved references must be a list.")

    if not isinstance(relevant_context, str):
        raise ValueError("Relevant context must be a string.")

    # An ambiguous interpretation must request clarification.
    if ambiguity_detected and not clarification_needed:
        raise ValueError(
            "Ambiguity was detected but clarification was not requested."
        )

    # Clarification requires conversation or memory context.
    if clarification_needed and not context_required:
        raise ValueError(
            "Clarification is required but context_required is false."
        )

    # A clarification request must contain an actual question.
    if clarification_needed and not clarification_question.strip():
        raise ValueError(
            "A clarification question is required when clarification is needed."
        )

    # Do not return clarification questions when none is required.
    if not clarification_needed and clarification_question.strip():
        raise ValueError(
            "Clarification question provided without clarification_needed."
        )

    # Independent questions must not carry unrelated context.
    if not context_required:
        if resolved_references or relevant_context.strip():
            raise ValueError(
                "Independent question contains unnecessary conversation context."
            )

        if ambiguity_detected or clarification_needed:
            raise ValueError(
                "Independent question has contradictory ambiguity flags."
            )


    result["original_question"] = question

    return result
