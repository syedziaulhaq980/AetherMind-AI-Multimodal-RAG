"""Extract searchable speech and visual descriptions from an uploaded video.

Requirements:
    pip install faster-whisper av pillow requests langchain-core

The speech and frame descriptions are returned as LangChain Documents. The caller
can add them to a temporary/session FAISS store without changing the saved index.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from time import perf_counter
import traceback
from typing import Any

import av
import requests
import os
from faster_whisper import WhisperModel
from langchain_core.documents import Document
from PIL import Image


OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://127.0.0.1:11434"
).rstrip("/")

OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"
VISION_MODEL_NAME =  "qwen2.5vl:3b"
WHISPER_MODEL_SIZE = "base"
MAX_FRAME_SAMPLES = 8
TARGET_FRAME_LONG_EDGE = 768
TRANSCRIPT_CHUNK_CHARS = 900
VISION_TIMEOUT = (10, 120)


@dataclass
class VideoIngestionResult:
    documents: list[Document] = field(default_factory=list)
    transcript_chunks: int = 0
    frame_descriptions: int = 0
    detected_language: str = "Unknown"
    duration_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)
    processing_seconds: float = 0.0


@lru_cache(maxsize=1)
def _load_whisper_model() -> WhisperModel:
    """Load a small multilingual transcription model once per app process."""
    return WhisperModel(
        WHISPER_MODEL_SIZE,
        device="cpu",
        compute_type="int8",
    )


def _format_timestamp(seconds: float | int) -> str:
    total = max(0, int(float(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _get_video_duration(container: av.container.InputContainer) -> float:
    if container.duration is not None:
        return max(0.0, float(container.duration / av.time_base))
    for stream in container.streams.video:
        if stream.duration is not None and stream.time_base is not None:
            return max(0.0, float(stream.duration * stream.time_base))
    return 0.0


def _sample_video_frames(video_path: Path) -> tuple[list[tuple[float, bytes]], float]:
    """Seek to a limited set of timestamps and return resized JPEG frames."""
    sampled: list[tuple[float, bytes]] = []
    with av.open(str(video_path)) as container:
        if not container.streams.video:
            return sampled, _get_video_duration(container)

        stream = container.streams.video[0]
        duration = _get_video_duration(container)
        if duration <= 0:
            target_times = [0.0]
        else:
            count = min(MAX_FRAME_SAMPLES, max(1, int(duration / 20.0) + 1))
            if count == 1:
                target_times = [0.0]
            else:
                target_times = [
                    i * max(0.0, duration - 0.25) / (count - 1)
                    for i in range(count)
                ]

        for target_seconds in target_times:
            try:
                offset = int(target_seconds / float(stream.time_base))
                container.seek(offset, stream=stream, backward=True)
                chosen_frame = None
                for frame in container.decode(video=0):
                    frame_seconds = float(frame.time or 0.0)
                    if frame_seconds + 0.05 >= target_seconds:
                        chosen_frame = frame
                        break
                if chosen_frame is None:
                    continue

                image = chosen_frame.to_image().convert("RGB")
                image.thumbnail((TARGET_FRAME_LONG_EDGE, TARGET_FRAME_LONG_EDGE))
                buffer = BytesIO()
                image.save(buffer, format="JPEG", quality=82, optimize=True)
                image_bytes = buffer.getvalue()

                print(
                        f"[FRAME DEBUG] "
                        f"Timestamp={float(chosen_frame.time or target_seconds):.2f}s | "
                        f"Dimensions={image.width}x{image.height} | "
                        f"JPEG size={len(image_bytes) / 1024:.1f} KB"
                    )

                sampled.append(
                        (float(chosen_frame.time or target_seconds), image_bytes)
                    )
            except Exception:
                # One unseekable timestamp should not prevent other frames being used.
                continue

    # Remove duplicate timestamps that can happen with very short videos.
    unique: list[tuple[float, bytes]] = []
    seen_times: set[int] = set()
    for seconds, image_bytes in sampled:
        key = int(seconds)
        if key not in seen_times:
            seen_times.add(key)
            unique.append((seconds, image_bytes))
    return unique, duration


def _describe_frame(image_bytes: bytes, timestamp: float) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    prompt = (
        f"Describe the visible content of this video frame at {_format_timestamp(timestamp)}. "
        "Mention clearly visible people, objects, actions, on-screen text, and scene changes. "
        "Do not guess names, motivations, or details that cannot be seen. "
        "Write a short factual description in 1-3 sentences."
    )
    payload: dict[str, Any] = {
        "model": VISION_MODEL_NAME,
        "prompt": prompt,
        "images": [encoded],
        "stream": False,
        "keep_alive": "10m",
        "options": {"temperature": 0, "num_predict": 110},
    }
    response = requests.post(
        OLLAMA_GENERATE_URL,
        json=payload,
        timeout=VISION_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()

    print(
            "[OLLAMA METRICS] "
            f"Total={data.get('total_duration', 0) / 1e9:.2f}s | "
            f"Load={data.get('load_duration', 0) / 1e9:.2f}s | "
            f"Prompt eval={data.get('prompt_eval_duration', 0) / 1e9:.2f}s | "
            f"Generation={data.get('eval_duration', 0) / 1e9:.2f}s | "
            f"Generated tokens={data.get('eval_count', 0)}"
        )

    caption = data.get("response", "")
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError("The vision model returned an empty frame description.")
    return caption.strip()


def _transcribe_video(video_path: Path) -> tuple[list[dict[str, Any]], str, list[str]]:
    warnings: list[str] = []
    transcript_segments: list[dict[str, Any]] = []
    language = "Unknown"
    try:
        model = _load_whisper_model()
        segments, info = model.transcribe(
            str(video_path),
            beam_size=1,
            vad_filter=True,
        )
        language = getattr(info, "language", None) or "Unknown"
        # The generator performs the actual transcription; keep it inside try/except.
        for segment in segments:
            text = (segment.text or "").strip()
            if text:
                transcript_segments.append({
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": text,
                })
    except Exception as exc:
        print(f"Speech transcription failed: {exc}")
        traceback.print_exc()
        warnings.append(
            "Speech transcription was unavailable. The video may have no usable audio, "
            "or transcription could not be completed."
        )
    return transcript_segments, language, warnings


def _segments_to_documents(
    segments: list[dict[str, Any]], source_name: str, language: str
) -> list[Document]:
    """Make timestamped, retrieval-friendly transcript chunks without losing timing."""
    if not segments:
        return []

    grouped: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0

    for segment in segments:
        segment_size = len(segment["text"])
        if current and current_chars + segment_size > TRANSCRIPT_CHUNK_CHARS:
            grouped.append(current)
            current = []
            current_chars = 0
        current.append(segment)
        current_chars += segment_size
    if current:
        grouped.append(current)

    docs: list[Document] = []
    for group in grouped:
        start = group[0]["start"]
        end = group[-1]["end"]
        transcript = " ".join(item["text"] for item in group)
        content = (
            f"Spoken transcript from {_format_timestamp(start)} to {_format_timestamp(end)}:\n"
            f"{transcript}"
        )
        docs.append(Document(
            page_content=content,
            metadata={
                "source": source_name,
                "title": source_name,
                "media_type": "video_transcript",
                "language": language,
                "start_seconds": round(start, 2),
                "end_seconds": round(end, 2),
            },
        ))
    return docs


def process_video_file(
    video_path: str | Path,
    source_name: str | None = None,
) -> VideoIngestionResult:
    """Create searchable transcript and visual-frame documents from a video file.

    This function does not save or modify the main FAISS index. It returns documents
    for the caller to add to a temporary/session FAISS store.
    """
    started = perf_counter()
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {path}")
    name = source_name or path.name
    result = VideoIngestionResult()

    # Audio and video are handled independently: a video without audio can still work visually.
    # Audio and video are handled independently.
    # Measure each processing stage separately.

    transcription_started = perf_counter()

    transcript_segments, language, transcript_warnings = _transcribe_video(path)

    transcription_seconds = perf_counter() - transcription_started
    result.warnings.extend(transcript_warnings)
    result.detected_language = language
    transcript_docs = _segments_to_documents(transcript_segments, name, language)
    result.documents.extend(transcript_docs)
    result.transcript_chunks = len(transcript_docs)
    frame_extraction_started = perf_counter()
    try:
        frames, duration = _sample_video_frames(path)
        result.duration_seconds = duration
    except Exception as exc:
        frames = []
        print(f"Video frame sampling failed: {exc}")
        traceback.print_exc()
        result.warnings.append(
            "Could not sample visual frames from this video. "
            "Please check that the video file is valid and try again."
        )
    frame_extraction_seconds = perf_counter() - frame_extraction_started

    if not frames:
        result.warnings.append("No video frames could be sampled from this file.")


    vision_started = perf_counter()
    vision_failed = False

    for frame_number, (timestamp, image_bytes) in enumerate(frames, start=1):
        frame_started = perf_counter()

        try:
            print(
                f"[VISION] Starting frame {frame_number}/{len(frames)} "
                f"at {_format_timestamp(timestamp)}"
            )

            caption = _describe_frame(image_bytes, timestamp)

            frame_seconds = perf_counter() - frame_started

            result.documents.append(Document(
                page_content=(
                    f"Visual description of the video at {_format_timestamp(timestamp)}:\n"
                    f"{caption}"
                ),
                metadata={
                    "source": name,
                    "title": name,
                    "media_type": "video_frame_description",
                    "language": "visual",
                    "start_seconds": round(timestamp, 2),
                },
            ))

            result.frame_descriptions += 1

            print(
                f"[VISION] Completed frame {frame_number}/{len(frames)} "
                f"in {frame_seconds:.2f} seconds"
            )

        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"

            print(
                f"[VISION] Failed frame {frame_number}/{len(frames)} "
                f"after {perf_counter() - frame_started:.2f} seconds"
            )

            if status == 404:
                result.warnings.append(
                    f"Ollama vision model '{VISION_MODEL_NAME}' is not installed. "
                    f"Run `ollama pull {VISION_MODEL_NAME}` and process the video again. "
                    "Speech transcript chunks, if available, are still usable."
                )
            else:
                result.warnings.append(
                    f"Visual description failed for a frame at {_format_timestamp(timestamp)} "
                    f"(HTTP {status})."
                )

            vision_failed = True
            break

        except requests.exceptions.RequestException as exc:
            print(
                f"[VISION] Failed frame {frame_number}/{len(frames)} "
                f"after {perf_counter() - frame_started:.2f} seconds"
            )

            result.warnings.append(
                "Could not call the local Ollama vision model. Check that Ollama is running "
                f"and the '{VISION_MODEL_NAME}' model is available ({type(exc).__name__})."
            )
            vision_failed = True
            break

        except (ValueError, KeyError) as exc:
            print(
                f"[VISION] Failed frame {frame_number}/{len(frames)} "
                f"after {perf_counter() - frame_started:.2f} seconds"
            )
            print(f"Video frame description failed: {exc}")
            traceback.print_exc()

            result.warnings.append(
                "Could not describe a video frame. "
                "Please check the frame content and local vision model."
            )
            vision_failed = True
            break

    vision_seconds = perf_counter() - vision_started

    if result.frame_descriptions == 0 and frames and not vision_failed:
        result.warnings.append("The video was opened, but no frame descriptions were generated.")

    if not result.documents:
        raise RuntimeError(
            "No searchable content could be extracted from this video. "
            "Check that it is a valid video file and that it contains audio or visible frames."
        )

    result.processing_seconds = perf_counter() - started
    print("\n========== VIDEO PROCESSING BREAKDOWN ==========")
    print(f"Whisper transcription: {transcription_seconds:.2f} seconds")
    print(f"Frame extraction:       {frame_extraction_seconds:.2f} seconds")
    print(f"Vision descriptions:    {vision_seconds:.2f} seconds")
    print(f"Total processing:       {result.processing_seconds:.2f} seconds")
    print(f"Frames described:       {result.frame_descriptions}/{len(frames)}")
    print("================================================\n")
    return result
