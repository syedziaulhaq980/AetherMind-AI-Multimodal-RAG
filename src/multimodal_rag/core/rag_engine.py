import json
import re
import traceback
import requests
from difflib import SequenceMatcher

from .config import (
    TOP_K,
    RELEVANCE_THRESHOLD,
    SHOW_EVIDENCE,
    MODEL_NAME,
)

from .llm import generate_ollama_response
from .sources import get_source_info
from .prompts import build_prompt
from .citations import validate_citations
from .retrieval import (
    load_vectorstore,
    retrieve_relevant_documents,
)

# Load the existing FAISS index.
vectorstore = load_vectorstore()

print("Ollama model:", MODEL_NAME)
# ---------- Ask a Question ----------

def ask_rag(question: str):

    # 1. Retrieve relevant document chunks.
    try:
        relevant = retrieve_relevant_documents(
            vectorstore,
            question
        )
    except Exception as exc:
        print(f"RAG retrieval failed: {exc}")
        traceback.print_exc()
        print("\nCould not retrieve relevant documents. Please try again.")
        return

    # Keep only the best TOP_K qualifying results.
    relevant = relevant[:TOP_K]

    print("Chunks passing threshold:", len(relevant))

    if not relevant:
        print("\nNo chunks passed the current relevance threshold.")
        print("Try rephrasing the question.")
        return

    # 3. Build context and assign source numbers
    context_parts = []
    source_records = []

    for i, (doc, distance) in enumerate(relevant, start=1):

        info = get_source_info(doc)

        context_parts.append(
            f"[Source {i}]\n"
            f"Document: {info['name']}\n"
            f"Page: {info['page_reference'] or 'Not available'}\n"
            f"Language: {info['language']}\n"
            f"Content:\n{doc.page_content}"
            )

        source_records.append({
            "id": i,
            "doc": doc,
            "distance": distance,
            "info": info,
        })

    context = "\n\n".join(context_parts)

    # The validator must use the same numbered source list
    # that is included in the prompt and printed below.
    source_count = len(source_records)

    # 4. Construct the prompt
    prompt = build_prompt(
        context=context,
        question=question
    )

    # 5. Generate the initial answer

    # 5. Generate a structured answer with source IDs.
    allowed_source_ids = [
        int(record["id"]) for record in source_records
    ]

    answer_schema = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string"
                    },
                    "source_id": {
                        "type": "integer",
                        "enum": allowed_source_ids
                    },
                    "evidence_quote": {
                        "type": "string"
                    }
                },
                "required": [
                    "text",
                    "source_id",
                    "evidence_quote"
                ],
                "additionalProperties": False
            },
            "minItems": 1,
            "maxItems": 3
        }
    },
    "required": ["claims"],
    "additionalProperties": False
}



    structured_prompt = f"""
{prompt}

STRUCTURED ANSWER INSTRUCTIONS:
Return only valid JSON matching the supplied schema.

ANSWER REQUIREMENTS:
1. Answer the original question directly.
2. Do not repeat or merely rephrase the question.
3. Make each claim a distinct, useful statement.
4. For "why" questions, explain different causes supported by the evidence.
5. For process questions, explain different steps in the correct order.
6. Do not produce multiple claims that express the same idea.
7. Do not fill the answer with generic statements when specific evidence
   is available.
8. Use only facts supported by the retrieved passages.

ADDITIONAL ANSWER-QUALITY RULES:
- For "why" questions, identify distinct, concrete causes mentioned in the evidence.
- Do not include a broad summary such as "poor retrieval quality"
  when more specific causes are available.
- Every claim must introduce a different cause, not paraphrase
  another claim.
- Prioritize specific evidence-supported causes, such as:
  incorrect document retrieval, chunking that removes context,
  missing or failed document ingestion, or retrieved context being
  accidentally omitted from the prompt.
- Distinguish retrieval failures from other pipeline failures.
- Do not invent a cause if the provided evidence does not support it.
- Return fewer claims rather than repeating the same explanation.

SOURCE SELECTION:
1. Assign exactly one source ID to each claim.
2. Choose the passage that most directly supports that specific claim.
3. Prefer specific evidence over a passage that only describes the
   general topic.
4. Do not reuse a general source for every claim when other sources
   provide more direct support.
5. Reusing a source is acceptable when that passage independently
   supports multiple distinct claims.
6. Never invent source IDs or unsupported details.

EVIDENCE QUOTE REQUIREMENTS:
1. Every claim must contain an evidence_quote.
2. The evidence_quote must be copied exactly from the Content
   section of the source assigned to that claim.
3. The quote must be a continuous passage, not reconstructed
   or paraphrased text.
4. The quote must directly support the entire claim.
5. If one passage cannot support the entire claim, split it
   into smaller claims.
6. Never invent or modify evidence quotes.
7. Do not use a quote merely because it contains similar
   keywords. It must provide actual supporting evidence.
8. If no retrieved passage supports a claim, omit that claim.
9. Return no more than 3 distinct claims.

SCOPE OF RETRIEVAL-FAILURE QUESTIONS:
- If the question specifically asks why retrieval failed, discuss
  causes that affect document extraction, indexing, chunking,
  or passage selection.
- Do not include failures that happen after retrieval, such as
  retrieved context being omitted from the generation prompt.
- If the original document contains the passage but the index
  does not, explain that as an ingestion or indexing problem.
- Distinguish indexing failures, retrieval failures, and
  generation failures instead of treating them as equivalent.
- Include only causes supported by the retrieved evidence.

CLAIM UNIQUENESS:
- Every claim must contribute a new fact or distinction.
- Do not restate a definition already given in another claim.
- Do not produce a broad summary alongside a claim expressing
  the same information more precisely.
- Prefer 2 or 3 distinct, useful claims over 4 claims with repetition.
- If fewer distinct claims are supported, return fewer claims.

QUESTION SCOPE AND CITATION ACCURACY:
- Answer the exact question asked. Do not add unrelated facts
  just because they appear in the retrieved evidence.
- For comparison questions, focus on the requested similarities,
  differences, advantages, or limitations.
- Each claim must add a distinct, relevant piece of information.
- For every claim, cite the source whose retrieved text directly
  supports that specific claim.
- Do not cite a source merely because it discusses the same
  general topic.
- If a source does not support a claim, do not cite it.
- If no retrieved source supports a useful claim, omit that claim
  rather than inventing support.
- Prefer three relevant, correctly cited claims over four claims
  containing unrelated information.

COMPLETENESS:
For questions asking why something fails, distinguish the supported
failure mechanisms instead of repeating "poor retrieval quality."
For questions asking how something works, explain the relevant stages
rather than merely naming broad stages.
If the evidence does not support enough distinct claims, return fewer
claims rather than inventing or repeating information.

Return only the JSON object. Do not include Markdown fences.
"""

    # Give causal questions more specific reasoning guidance.
    if re.search(r"\bwhy\b", question, flags=re.IGNORECASE):
        structured_prompt += (
            "\n\nCAUSAL QUESTION REQUIREMENTS:\n"
            "Identify concrete causes explicitly supported by the evidence.\n"
            "Explain how each cause leads to the stated problem.\n"
            "Use distinct causes rather than rewording the same cause.\n"
            "Do not use the question itself as an answer claim.\n"
            "Do not substitute generic statements about system quality "
            "or evaluation for actual causes.\n"
            "Do not invent causes that the retrieved passages do not support.\n"
            "Include two or more distinct causes only when the retrieved "
"evidence supports them. Otherwise, return fewer claims.\n"
        )
    print("\n========== PROMPT INSTRUCTIONS CHECK ==========")
    print(structured_prompt[-3000:])
    print("========== END PROMPT CHECK ==========\n")

    try:
        raw_answer = generate_ollama_response(
            structured_prompt,
            output_format=answer_schema,
        )
        print("\n========== RAW OLLAMA JSON ==========")
        print(raw_answer)
        print("========== END RAW JSON ==========\n")

        parsed_answer = json.loads(raw_answer)
        claims = parsed_answer.get("claims")

        if not isinstance(claims, list):
            raise ValueError("Ollama returned no valid claims list.")

        formatted_claims = []

        for claim in claims:
            if not isinstance(claim, dict):
                continue

            claim_text = claim.get("text")
            source_id = claim.get("source_id")
            evidence_quote = claim.get("evidence_quote")

            if (
                not isinstance(claim_text, str)
                or not claim_text.strip()
                or type(source_id) is not int
                or source_id not in allowed_source_ids
                or not isinstance(evidence_quote, str)
                or not evidence_quote.strip()
            ):
                continue

            source_record = next(
                record for record in source_records
                if record["id"] == source_id
            )

            source_text = " ".join(
                source_record["doc"].page_content.split()
            )

            normalized_quote = " ".join(evidence_quote.split())

            if normalized_quote not in source_text:
                print(
                    "Skipping a claim with an evidence quote not found in its source:",
                    source_id,
                )
                continue
            # Skip claims that merely restate the user's question.
            normalized_question = " ".join(
                re.findall(r"[a-z0-9]+", question.lower())
            )
            normalized_claim = " ".join(
                re.findall(r"[a-z0-9]+", claim_text.lower())
            )

            similarity = SequenceMatcher(
                None,
                normalized_claim,
                normalized_question,
            ).ratio()

            if similarity >= 0.80:
                print(
                    "Skipping a claim that repeats the question:",
                    claim_text,
                )
                continue

            formatted_claims.append(
                f"{claim_text.strip()} [Source {source_id}]"
            )

        if not formatted_claims:
            raise ValueError(
                "Ollama returned no usable claims with valid source IDs."
            )

        answer = "\n".join(formatted_claims)

    except requests.exceptions.ConnectionError:
        print("\nCannot connect to Ollama.")
        print("Check that Ollama is running on Windows.")
        return

    except requests.exceptions.RequestException as exc:
        print(f"\nOllama request failed: {exc}")
        return

    except (ValueError, KeyError, TypeError) as exc:
        print(f"\nCould not process Ollama's structured response: {exc}")
        traceback.print_exc()
        return

    # 6. Validate citations before displaying the answer
    citation_check = validate_citations(
        answer,
        source_count=source_count
    )

    if citation_check["valid"]:
        print("\nCitation check passed.")

    else:
        print(
            f"\nCitation warning: {citation_check['reason']}"
        )

        if citation_check["invalid"]:
            print(
                "Invalid source numbers:",
                citation_check["invalid"]
            )

        # 7. Attempt one correction if validation fails
        print("Attempting one citation correction...")

        retry_prompt = f"""
You are correcting an answer in a retrieval-augmented question-answering system.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

PREVIOUS ANSWER:
{answer}

YOUR TASK:
Write a corrected answer to the original question using only the retrieved context.

CITATION REQUIREMENTS:
1. Every factual claim must have a supporting inline citation.
2. Use the exact format [Source N], for example [Source 3].
3. Source numbers refer to the numbered passages in the retrieved context above.
4. Use only source numbers that exist in that context.
5. Choose sources based on their actual content, not just their titles.
6. Do not invent facts or citations.
7. If the context is insufficient, say so clearly.

Return only the corrected answer in 1–4 sentences.
Do not include explanations about the correction process.

CORRECTED ANSWER:
"""

        try:
            corrected_answer = generate_ollama_response(
                retry_prompt
            )

            retry_check = validate_citations(
                corrected_answer,
                source_count=source_count
            )

            if retry_check["valid"]:
                answer = corrected_answer
                citation_check = retry_check

                print(
                    "Citation check passed after correction."
                )

            else:
                print(
                    "Citation correction failed basic validation."
                )
                print(
                    "Keeping the original answer. "
                    "Review its citations before relying on it."
                )

        except requests.exceptions.RequestException as exc:
            print(
                f"Citation correction request failed: {exc}"
            )
            traceback.print_exc()
            print(
                "Keeping the original answer. "
                "Review its citations before relying on it."
            )

        except (ValueError, KeyError) as exc:
            print(
                f"Could not process the corrected response: {exc}"
            )
            traceback.print_exc()
            print(
                "Keeping the original answer. "
                "Review its citations before relying on it."
            )

    # 8. Display the answer
    print("\n========== ANSWER ==========\n")
    print(answer)

    # 9. Display source names and links
    print("\n========== SOURCES ==========")

    for record in source_records:
        info = record["info"]

        print(
            f"\n[Source {record['id']}] "
            f"(FAISS distance: {record['distance']:.4f})"
        )

        print("Name:", info["name"])

        if info["page_reference"]:
           print("Page:", info["page_reference"])

        print("Language:", info["language"])

        if info["timestamp_url"]:
            print("Video timestamp:", info["timestamp_url"])

        elif info["url"]:
            print("Link:", info["url"])

        else:
            print("Original source path:", info["raw_source"])

    # 10. Display retrieved evidence for manual grounding checks
    if SHOW_EVIDENCE:
        print("\n========== RETRIEVED EVIDENCE ==========")

        for record in source_records:
            doc = record["doc"]

            print(f"\n[Source {record['id']}]")
            print(doc.page_content[:700])

    # 11. Return results for automated evaluation
    return {
        "question": question,
        "answer": answer,
        "citation_check": citation_check,
        "source_count": source_count,
        "sources": [
            {
                "id": record["id"],
                "name": record["info"]["name"],
                "page_reference": record["info"]["page_reference"],
                "distance": record["distance"],
            }
            for record in source_records
        ],
    }
# ---------- Run the RAG Application ----------

if __name__ == "__main__":

    print(
        "\nAsk questions about your saved PDF, text and YouTube content."
    )
    print("Type 'exit' to stop.\n")

    while True:
        question = input("Your question: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("RAG session ended.")
            break

        if question:
            ask_rag(question)
