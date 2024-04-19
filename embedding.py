import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingFunction:
    def __init__(self, model_name="all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def __call__(self, texts: list[str]) -> np.ndarray:
        """
        Generates embeddings for the given list of text inputs.

        Args:
            texts (list[str]): A list of text inputs.

        Returns:
            np.ndarray: A numpy array of embeddings, where each row represents the embedding for the corresponding text input.
        """
        embeddings = self.model.encode(texts)
        return embeddings