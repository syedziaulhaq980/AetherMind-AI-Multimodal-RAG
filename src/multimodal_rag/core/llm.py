import requests
from time import perf_counter
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



    print("[OLLAMA REQUEST] Model:", MODEL_NAME)
    print("[OLLAMA REQUEST] Prompt characters:", len(prompt_text))
    print("[OLLAMA REQUEST] Output format:", output_format)    

    started = perf_counter()
    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=(10, 600)
    )

    response.raise_for_status()
    elapsed = perf_counter() - started

    print(
         f"[OLLAMA TIMING] "
         f"Request completed in {elapsed:.2f} seconds"
        )

    result = response.json().get("response")

    if not isinstance(result, str):
        raise ValueError(
            "Ollama returned no valid text response."
        )

    return result.strip()
