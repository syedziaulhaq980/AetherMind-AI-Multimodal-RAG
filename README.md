# AetherMind AI ? Multimodal Knowledge Assistant

A local Streamlit-based Multimodal Retrieval-Augmented Generation (RAG) application combining multilingual semantic search, FAISS retrieval, Cross-Encoder reranking, and local Ollama inference.

## Features

- Document, image, and video ingestion.
- Multilingual semantic embeddings.
- FAISS-based vector retrieval.
- Cross-Encoder reranking.
- Local LLM inference using Ollama and `qwen2.5vl:3b`.
- Source citations for generated answers.
- Session-based processing of uploaded content.
- Persistent FAISS knowledge base, unchanged by session uploads.

## Technology Stack

| Component | Technology |
|---|---|
| Interface | Streamlit |
| Language | Python |
| Embeddings | paraphrase-multilingual-MiniLM-L12-v2 |
| Vector search | FAISS |
| Reranking | Sentence Transformers Cross-Encoder |
| LLM | Ollama ? qwen2.5vl:3b |
| Document processing | LangChain, PyPDF |
| Video processing | Faster-Whisper, PyAV |
| Image processing | Pillow |

## Architecture

1. Documents, images, and videos enter their respective ingestion workflows.
2. Text is converted into multilingual vector representations.
3. FAISS retrieves relevant passages from the persistent knowledge base.
4. A Cross-Encoder reranks retrieved candidates.
5. Ollama generates an answer using the retrieved context.
6. Source information supports the generated response.

## Setup and Installation

1. Clone this repository and enter its folder.
2. Create a virtual environment: `python -m venv .venv`
3. Activate it: ` .\.venv\Scripts\Activate.ps1 ` (Windows PowerShell).
4. Install dependencies: `pip install -r requirements.txt`
5. Install Ollama from https://ollama.com/.
6. Download the model: `ollama pull qwen2.5vl:3b`
7. Provide a compatible FAISS index in `indexes/multimodal_rag_faiss/`.
8. Start the app: `streamlit run app.py`

The FAISS index and original knowledge-base data are not included in this repository. A compatible index must be provided or rebuilt before running the application.

## Retrieval Evaluation

The project includes a retrieval evaluation workflow with source-level Precision@5, Recall@5, and Mean Reciprocal Rank (MRR@5).

Preliminary manually judged passage results:

| Metric | Result |
|---|---:|
| Macro Passage Precision@5 | 0.85 |
| Macro Passage Precision@8 | 0.7812 |
| Macro Passage MRR@8 | 1.00 |

Source-level macro results:

| Metric | Result |
|---|---:|
| Source Precision@5 | 0.50 |
| Source Recall@5 | 0.8667 |
| Source MRR@5 | 1.00 |

These preliminary results use four answerable test cases and a local knowledge base containing 34 sources and 2,149 FAISS vectors. The benchmark is small and manually judged. Answer-generation quality has not yet been independently benchmarked.

## Privacy and Limitations

- Uploaded content is processed within the active application session.
- Session uploads do not modify the persistent FAISS index.
- Local conversation memory and knowledge-base files are excluded from this repository.
- LLM inference uses a locally running Ollama server.
- Retrieval evaluation is limited to a small manually judged test set.

## Author

**Syed Ziaul Haq**
AI & ML | Python | NLP | RAG | Generative AI
