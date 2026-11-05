from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings


# Load the existing multilingual embedding model once.
embedding_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    device="cpu"
)


class MiniLMEmbeddings(Embeddings):

    def embed_documents(self, texts):
        return embedding_model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).tolist()

    def embed_query(self, text):
        return embedding_model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0].tolist()
