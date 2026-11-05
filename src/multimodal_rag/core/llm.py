import requests

from .config import MODEL_NAME, OLLAMA_URL


def generate_ollama_response(
    prompt_text: str,
    output_format=None
) -> str:
    """Generate a response using the local Ollama model."""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt_text,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_ctx": 8192,
        }
    }

    if output_format is not None:
        payload["format"] = output_format

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=300
    )

    response.raise_for_status()

    result = response.json().get("response")

    if not isinstance(result, str):
        raise ValueError(
            "Ollama returned no valid text response."
        )

    return result.strip()
