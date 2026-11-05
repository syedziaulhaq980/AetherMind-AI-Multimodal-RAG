from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path, PureWindowsPath
from statistics import mean

# --------------------------------------------------
# 1. Project paths and imports
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sentence_transformers import CrossEncoder

from src.multimodal_rag.core.config import (
    TOP_K,
    RELEVANCE_THRESHOLD,
)
from src.multimodal_rag.core.retrieval import load_vectorstore


TEST_FILE = PROJECT_ROOT / "evaluation" / "rag_test_cases.json"
REPORT_FILE = (
    PROJECT_ROOT / "evaluation" / "retrieval_evaluation_report.csv"
)
PASSAGE_REVIEW_FILE = (
    PROJECT_ROOT / "evaluation" / "passage_review.csv"
)

RERANKER_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

CANDIDATE_K = max(TOP_K * 3, 20)
EVALUATION_K = 5

ALLOWED_MEDIA_TYPES = (
    None,
    "",
    "pdf",
    "text",
    "document",
    "uploaded_document",
)


# --------------------------------------------------
# 2. Source filename normalization
# --------------------------------------------------

def source_filename(doc) -> str:
    metadata = getattr(doc, "metadata", {}) or {}

    raw_source = (
        metadata.get("uploaded_file_name")
        or metadata.get("source")
        or metadata.get("title")
        or "Unknown source"
    )

    return PureWindowsPath(str(raw_source)).name


def normalize_source(name: str) -> str:
    return PureWindowsPath(str(name)).name.strip().casefold()


def normalize_passage(text: str) -> str:
    """Normalize passage whitespace and casing."""

    return " ".join(str(text or "").split()).casefold()


def passage_fingerprint(source: str, text: str) -> str:
    """Generate a stable ID for a source passage."""

    identity = (
        normalize_source(source)
        + "\\n"
        + normalize_passage(text)
    )

    return hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()[:16]


# --------------------------------------------------
# 3. Source-level evaluation metrics
# --------------------------------------------------

def calculate_metrics(
    retrieved_sources: list[str],
    expected_sources: list[str],
) -> dict:

    expected = {
        normalize_source(source)
        for source in expected_sources
    }

    ranked = [
        normalize_source(source)
        for source in retrieved_sources[:EVALUATION_K]
    ]

    relevant_ranks = [
        rank
        for rank, source in enumerate(ranked, start=1)
        if source in expected
    ]

    precision = len(relevant_ranks) / EVALUATION_K

    recall = (
        len(relevant_ranks) / len(expected)
        if expected
        else 0.0
    )

    reciprocal_rank = (
        1.0 / relevant_ranks[0]
        if relevant_ranks
        else 0.0
    )

    return {
        "source_precision_at_5": precision,
        "source_recall_at_5": recall,
        "source_mrr_at_5": reciprocal_rank,
    }


# --------------------------------------------------
# 4. Main evaluation
# --------------------------------------------------

def main():

    print("=" * 65)
    print("ACTIVE PDF/TEXT RETRIEVAL EVALUATION")
    print("=" * 65)

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_FILE}"
        )

    with TEST_FILE.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        test_cases = json.load(file)

    print(f"Test cases loaded: {len(test_cases)}")
    print(f"TOP_K: {TOP_K}")
    print(f"Candidate pool: {CANDIDATE_K}")
    print(f"L2 threshold: {RELEVANCE_THRESHOLD}")
    print(f"Evaluation cutoff: {EVALUATION_K}")

    # Load the existing index. No index modification.
    print("\nLoading permanent FAISS index...")
    vectorstore = load_vectorstore()

    # Record which expected source files exist.
    index_sources = {
        normalize_source(source_filename(doc))
        for doc in vectorstore.docstore._dict.values()
        if (doc.metadata or {}).get("source")
    }

    print(f"Unique indexed sources: {len(index_sources)}")

    # Load the same Cross-Encoder as app.py.
    print("\nLoading Cross-Encoder on CPU...")
    reranker = CrossEncoder(
        RERANKER_NAME,
        device="cpu",
    )

    print("Cross-Encoder loaded successfully.")

    report_rows = []
    answerable_metrics = []
    passage_review_rows = []

    for case in test_cases:

        test_id = case["id"]
        question = case["question"]
        expected_sources = case.get("expected_sources", [])
        should_abstain = case.get("should_abstain", False)

        print("\n" + "-" * 65)
        print(f"Test: {test_id}")
        print(f"Question: {question}")

        # 1. Retrieve from permanent FAISS index.
        raw_results = vectorstore.similarity_search_with_score(
            question,
            k=CANDIDATE_K,
        )

        # 2. Keep PDF/text sources only.
        candidates = []

        for doc, distance in raw_results:

            metadata = doc.metadata or {}
            media_type = metadata.get("media_type")

            if media_type not in ALLOWED_MEDIA_TYPES:
                continue

            distance = float(distance)

            if distance <= RELEVANCE_THRESHOLD:
                candidates.append((doc, distance))

        print(f"Candidates passing filters: {len(candidates)}")

        # 3. Deduplicate using the same source + passage
        #    strategy as the active application.
        unique_candidates = []
        seen = set()

        for doc, distance in candidates:

            metadata = doc.metadata or {}

            original_source = (
                metadata.get("uploaded_file_name")
                or metadata.get("source")
                or metadata.get("title")
                or "Unknown source"
            )

            passage = " ".join(
                (doc.page_content or "").split()
            )

            if not passage:
                continue

            key = (str(original_source), passage)

            if key in seen:
                continue

            seen.add(key)
            unique_candidates.append((doc, distance))

        print(f"Unique passages: {len(unique_candidates)}")

        # 4. Cross-Encoder reranking.
        if unique_candidates:

            pairs = [
                (question, doc.page_content[:1800])
                for doc, _ in unique_candidates
            ]

            scores = reranker.predict(
                pairs,
                batch_size=8,
                show_progress_bar=False,
            )

            reranked = sorted(
                zip(unique_candidates, scores),
                key=lambda item: float(item[1]),
                reverse=True,
            )

        else:
            reranked = []

        # 5. Take the production TOP_K passages.
        final_results = [
            (doc, distance, float(score))
            for ((doc, distance), score) in reranked[:TOP_K]
        ]

        # Collect final passages for manual relevance annotation.
        for rank, (doc, distance, score) in enumerate(
            final_results,
            start=1,
        ):
            filename = source_filename(doc)
            passage_text = " ".join(
                (doc.page_content or "").split()
            )

            passage_review_rows.append({
                "test_id": test_id,
                "question": question,
                "rank": rank,
                "source_filename": filename,
                "passage_id": passage_fingerprint(
                    filename,
                    passage_text,
                ),
                "faiss_distance": distance,
                "cross_encoder_score": score,
                "is_relevant": "",
                "annotation_notes": "",
                "passage_text": passage_text,
            })

        # 6. Collapse passages to unique source ranking.
        retrieved_sources = []
        seen_sources = set()

        for doc, distance, score in final_results:

            filename = source_filename(doc)
            normalized = normalize_source(filename)

            if normalized in seen_sources:
                continue

            seen_sources.add(normalized)
            retrieved_sources.append(filename)

        # 7. Determine which expected sources exist in FAISS.
        missing_sources = [
            source
            for source in expected_sources
            if normalize_source(source) not in index_sources
        ]

        # 8. Calculate metrics for answerable questions.
        if not should_abstain:

            metrics = calculate_metrics(
                retrieved_sources,
                expected_sources,
            )

            answerable_metrics.append(metrics)

        else:

            metrics = {
                "source_precision_at_5": "",
                "source_recall_at_5": "",
                "source_mrr_at_5": "",
            }

        print("\nRetrieved source ranking:")

        if retrieved_sources:
            for rank, source in enumerate(
                retrieved_sources[:EVALUATION_K],
                start=1,
            ):
                print(f"{rank}. {source}")
        else:
            print("No sources retrieved.")

        print("Missing expected sources:", missing_sources)

        if not should_abstain:
            print(
                f"Source Precision@5: {metrics['source_precision_at_5']:.4f}"
            )
            print(
                f"Source Recall@5: {metrics['source_recall_at_5']:.4f}"
            )
            print(
                f"Source MRR@5: {metrics['source_mrr_at_5']:.4f}"
            )
        else:
            print(
                "Unsupported-question retrieval:",
                "No passages retrieved"
                if not final_results
                else f"{len(final_results)} passages retrieved",
            )

        # 9. Save detailed per-question results.
        report_rows.append({
            "test_id": test_id,
            "category": case.get("category", ""),
            "question": question,
            "should_abstain": should_abstain,
            "retrieved_passage_count": len(final_results),
            "retrieved_unique_sources_top5": " | ".join(
                retrieved_sources[:EVALUATION_K]
            ),
            "expected_sources": " | ".join(expected_sources),
            "missing_expected_sources": " | ".join(
                missing_sources
            ),
            "source_precision_at_5": metrics["source_precision_at_5"],
            "source_recall_at_5": metrics["source_recall_at_5"],
            "source_mrr_at_5": metrics["source_mrr_at_5"],
            "retrieved_any": bool(final_results),
        })

    # --------------------------------------------------
    # 10. Export CSV report
    # --------------------------------------------------

    fieldnames = list(report_rows[0].keys())

    with REPORT_FILE.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(report_rows)

    # --------------------------------------------------
    # 10A. Export passage review CSV
    # --------------------------------------------------

    if passage_review_rows:

        # Preserve existing manual relevance judgments.
        existing_annotations = {}

        if PASSAGE_REVIEW_FILE.exists():
            with PASSAGE_REVIEW_FILE.open(
                "r",
                newline="",
                encoding="utf-8-sig",
            ) as file:

                previous_rows = csv.DictReader(file)

                for previous in previous_rows:
                    key = (
                        previous.get("test_id", ""),
                        previous.get("passage_id", ""),
                    )

                    existing_annotations[key] = {
                        "is_relevant": previous.get("is_relevant", ""),
                        "annotation_notes": previous.get(
                            "annotation_notes", ""
                        ),
                    }

        # Restore annotations for passages that still exist.
        for row in passage_review_rows:
            key = (
                row["test_id"],
                row["passage_id"],
            )

            previous = existing_annotations.get(key)

            if previous:
                if previous["is_relevant"] != "":
                    row["is_relevant"] = previous["is_relevant"]

                if previous["annotation_notes"] != "":
                    row["annotation_notes"] = previous["annotation_notes"]

        passage_fieldnames = list(passage_review_rows[0].keys())

        with PASSAGE_REVIEW_FILE.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as file:

            passage_writer = csv.DictWriter(
                file,
                fieldnames=passage_fieldnames,
            )

            passage_writer.writeheader()
            passage_writer.writerows(passage_review_rows)

    print(
        f"Passage review file saved to:\\n{PASSAGE_REVIEW_FILE}"
    )

    # --------------------------------------------------
    # 11. Aggregate answerable-question metrics
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("FINAL RETRIEVAL EVALUATION SUMMARY")
    print("=" * 65)

    if answerable_metrics:

        print(
            "Mean Precision@5:",
            f"{mean(m['source_precision_at_5'] for m in answerable_metrics):.4f}",
        )

        print(
            "Mean Recall@5:",
            f"{mean(m['source_recall_at_5'] for m in answerable_metrics):.4f}",
        )

        print(
            "Mean MRR@5:",
            f"{mean(m['source_mrr_at_5'] for m in answerable_metrics):.4f}",
        )

    print(f"\nDetailed report saved to:\n{REPORT_FILE}")

    print(
        "\nNOTE: Unsupported-question retrieval is reported separately. "
        "This script does not evaluate LLM abstention."
    )


if __name__ == "__main__":
    main()
