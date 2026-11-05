"""Generate grounded explanations from extracted video evidence."""

from __future__ import annotations

import re
import requests
from typing import Any

from ..core.config import OLLAMA_URL, MODEL_NAME
from ..core.embeddings import MiniLMEmbeddings



def generate_video_explanation(video_result: Any) -> str:
    """Explain a video using extracted transcript and visual evidence."""

    video_docs = [
        doc
        for doc in video_result.documents
        if (doc.metadata or {}).get("media_type") in {
            "video_transcript",
            "video_frame_description",
        }
    ]

    if not video_docs:
        return "No transcript or visual evidence is available."

    source_records = []
    context_parts = []

    for source_id, doc in enumerate(video_docs, start=1):
        metadata = doc.metadata or {}
        media_type = metadata.get("media_type", "unknown")

        start = float(metadata.get("start_seconds", 0) or 0)
        timestamp = f"{int(start) // 60:02d}:{int(start) % 60:02d}"

        end_value = metadata.get("end_seconds")
        if end_value is not None:
            end = int(float(end_value))
            timestamp += f"–{end // 60:02d}:{end % 60:02d}"

        record = {
            "id": source_id,
            "doc": doc,
            "media_type": media_type,
            "timestamp": timestamp,
        }
        source_records.append(record)

        evidence_type = (
            "SPOKEN TRANSCRIPT"
            if media_type == "video_transcript"
            else "SAMPLED FRAME DESCRIPTION"
        )

        context_parts.append(
            f"[Source {source_id}]\n"
            f"Evidence type: {evidence_type}\n"
            f"Timestamp: {timestamp}\n"
            f"Content: {doc.page_content}"
        )

    evidence_context = "\n\n".join(context_parts)

    print("\n========== VIDEO EXPLANATION DEBUG ==========")
    print("Total video documents:", len(video_docs))
    print("Total evidence characters:", len(evidence_context))
    print(
        "Transcript documents:",
        sum(
            1 for doc in video_docs
            if (doc.metadata or {}).get("media_type") == "video_transcript"
        ),
    )
    print(
        "Frame descriptions:",
        sum(
            1 for doc in video_docs
            if (doc.metadata or {}).get("media_type") == "video_frame_description"
        ),
    )
    print("=============================================\n")

    prompt = f"""
You are a careful video analysis assistant.

Explain the uploaded video using ONLY the evidence below.

Write these sections:
## Video overview
## Explanation of the spoken content
## What is visible
## Key takeaways
## Evidence limitations

Rules:
- Describe only details supported by the supplied evidence.
- Keep spoken content separate from visual observations.
- Do not invent objects, people, actions, or speaker intentions.
- Sampled frames do not represent every moment of the video.
- Mention missing information only when it is genuinely missing.
- Write short paragraphs or bullet points.
- Do not create source labels yourself. The application will attach them.

VIDEO EVIDENCE:
{evidence_context}

Write a clear, beginner-friendly explanation.
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 700,
            },
        },
        timeout=(10, 240),
    )
    response.raise_for_status()

    answer = response.json().get("response", "").strip()

    if not answer:
        return "The model returned an empty video explanation."

    # Reuse the existing multilingual MiniLM embedding model.
    embedder = MiniLMEmbeddings()

    source_vectors = embedder.embed_documents(
        [record["doc"].page_content for record in source_records]
    )

    # Attach the closest evidence references to each answer paragraph.
    output_lines = []
    current_section = ""

    for line in answer.splitlines():
        stripped = line.strip()

        if stripped.startswith("## "):
            current_section = stripped.lower()
            output_lines.append(line)
            continue

        if not stripped:
            output_lines.append(line)
            continue

        claim = re.sub(
            r"\[Source\s+\d+\]",
            "",
            stripped,
            flags=re.IGNORECASE,
        ).strip()

        if not claim:
            output_lines.append(line)
            continue

        if "spoken content" in current_section:
            allowed_types = {"video_transcript"}

        elif "what is visible" in current_section:
            allowed_types = {"video_frame_description"}

        elif "key takeaways" in current_section:
            allowed_types = {
                "video_frame_description",
                "video_transcript",
            }

        else:
            allowed_types = {
                "video_transcript",
                "video_frame_description",
            }

        candidate_indices = [
            i
            for i, record in enumerate(source_records)
            if record["media_type"] in allowed_types
        ]

        if not candidate_indices:
            output_lines.append(line)
            continue

        claim_vector = embedder.embed_query(claim)

        scores = []

        for i in candidate_indices:
            score = sum(
                a * b
                for a, b in zip(claim_vector, source_vectors[i])
            )
            scores.append((score, i))

        scores.sort(reverse=True)

        best_matches = scores[:2]

        labels = " ".join(
            f"[Source {source_records[i]['id']}]"
            for score, i in best_matches
        )

        if stripped.startswith(("- ", "* ")):
            prefix = stripped[:2]
            claim = stripped[2:].strip()
            output_lines.append(f"{prefix}{claim} {labels}")
        else:
            output_lines.append(f"{claim} {labels}")

    # Build evidence limitations from actual extraction metadata.
    frame_records = [
        record
        for record in source_records
        if record["media_type"] == "video_frame_description"
    ]

    transcript_records = [
        record
        for record in source_records
        if record["media_type"] == "video_transcript"
    ]

    output_lines.extend(["", "## Evidence limitations"])

    if frame_records:
        timestamps = sorted({
            record["timestamp"]
            for record in frame_records
        })

        output_lines.append(
            f"- Visual analysis used {len(frame_records)} sampled "
            f"frame descriptions at: {', '.join(timestamps)}."
        )
        output_lines.append(
            "- Events occurring between sampled frames may not be captured."
        )
    else:
        output_lines.append(
            "- No usable visual frame descriptions were generated."
        )

    if transcript_records:
        output_lines.append(
            f"- Speech transcription produced {len(transcript_records)} "
            "timestamped transcript chunks. Automatic transcription "
            "may contain recognition errors."
        )
    else:
        output_lines.append(
            "- No usable speech transcript was generated."
        )

    output_lines.extend(["", "## Evidence index"])

    for record in source_records:
        doc = record["doc"]

        kind = (
            "Speech transcript"
            if record["media_type"] == "video_transcript"
            else "Video frame description"
        )

        excerpt = doc.page_content.strip()

        if len(excerpt) > 450:
            excerpt = excerpt[:450] + "..."

        output_lines.append(
            f"\n**[Source {record['id']}] {kind} — "
            f"{record['timestamp']}**\n\n{excerpt}"
        )

    output_lines.extend([
        "",
        "> **Citation note:** References are assigned using MiniLM "
        "semantic similarity. They identify the closest matching "
        "extracted evidence, but do not prove that every claim is "
        "fully supported. Review the evidence index.",
    ])

    return "\n".join(output_lines)