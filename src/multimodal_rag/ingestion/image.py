"""Extract searchable visual descriptions from uploaded images."""

from __future__ import annotations

import base64
import os
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any

import requests
from PIL import Image
from langchain_core.documents import Document


OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://127.0.0.1:11434"
).rstrip("/")

OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"

VISION_MODEL_NAME = "qwen2.5vl:3b"

VISION_TIMEOUT = (10, 120)

MAX_IMAGE_LONG_EDGE = 1536


def _prepare_image(image_path: Path) -> tuple[bytes, int, int]:
    """Validate, resize, and encode an image as JPEG."""

    with Image.open(image_path) as image:

        image = image.convert("RGB")

        original_width, original_height = image.size

        image.thumbnail(
            (MAX_IMAGE_LONG_EDGE, MAX_IMAGE_LONG_EDGE)
        )

        buffer = BytesIO()

        image.save(
            buffer,
            format="JPEG",
            quality=90,
            optimize=True,
        )

        return (
            buffer.getvalue(),
            original_width,
            original_height,
        )


def _describe_image(image_bytes: bytes) -> str:
    """Generate a factual image description using Ollama vision."""

    encoded = base64.b64encode(image_bytes).decode("ascii")

    prompt = (
        "Describe the visible content of this image accurately. "
        "Identify clearly visible objects, people, activities, "
        "background details, diagrams, charts, and readable text. "
        "For diagrams and charts, explain their visible structure "
        "and relationships. "
        "Do not guess names or details that cannot be seen. "
        "Write a detailed, factual description in clear paragraphs."
    )

    payload: dict[str, Any] = {
        "model": VISION_MODEL_NAME,
        "prompt": prompt,
        "images": [encoded],
        "stream": False,
        "keep_alive": "10m",
        "options": {
            "temperature": 0,
            "num_predict": 450,
        },
    }

    response = requests.post(
        OLLAMA_GENERATE_URL,
        json=payload,
        timeout=VISION_TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()

    description = data.get("response", "")

    if not isinstance(description, str) or not description.strip():
        raise ValueError(
            "The vision model returned an empty image description."
        )

    print(
        "[IMAGE OLLAMA METRICS] "
        f"Total={data.get('total_duration', 0) / 1e9:.2f}s | "
        f"Generated tokens={data.get('eval_count', 0)}"
    )

    return description.strip()


def process_image_file(
    image_path: str | Path,
    source_name: str | None = None,
) -> Document:
    """Create a searchable LangChain document from an image."""

    started = perf_counter()

    path = Path(image_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Image file does not exist: {path}"
        )

    name = source_name or path.name

    image_bytes, width, height = _prepare_image(path)

    description = _describe_image(image_bytes)

    content = (
        f"Image: {name}\n"
        f"Image dimensions: {width}x{height}\n\n"
        f"Visual description:\n{description}"
    )

    document = Document(
        page_content=content,
        metadata={
            "source": name,
            "title": name,
            "media_type": "image_description",
            "language": "visual",
            "width": width,
            "height": height,
        },
    )

    processing_seconds = perf_counter() - started

    print("\n========== IMAGE INGESTION RESULT ==========")
    print("Image:", name)
    print("Dimensions:", f"{width}x{height}")
    print("Description characters:", len(description))
    print("Processing seconds:", round(processing_seconds, 2))
    print("============================================\n")

    return document