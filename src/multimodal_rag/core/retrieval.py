from langchain_community.vectorstores import FAISS

from .config import INDEX_DIR
from .embeddings import MiniLMEmbeddings


from .config import INDEX_DIR, TOP_K, RELEVANCE_THRESHOLD


def load_vectorstore():
    """Load the existing persistent FAISS index."""

    if not INDEX_DIR.exists():
        raise FileNotFoundError(
            f"FAISS index directory not found: {INDEX_DIR}"
        )

    vectorstore = FAISS.load_local(
        str(INDEX_DIR),
        MiniLMEmbeddings(),
        allow_dangerous_deserialization=True
    )

    print("FAISS vectors:", vectorstore.index.ntotal)

    return vectorstore
def retrieve_relevant_documents(
    vectorstore,
    question: str
):
    """Retrieve and filter relevant document chunks."""

    candidate_k = max(TOP_K * 3, 20)

    results = vectorstore.similarity_search_with_score(
        question,
        k=candidate_k
    )

    print("\nTOP_K setting:", TOP_K)
    print("Candidate pool size:", len(results))

    relevant = [
        (doc, float(distance))
        for doc, distance in results
        if float(distance) <= RELEVANCE_THRESHOLD
    ]

    relevant = relevant[:TOP_K]

    print("Chunks passing threshold:", len(relevant))

    if not relevant:
        print("\nNo chunks passed the current relevance threshold.")
        print("Try rephrasing the question.")

    return relevant