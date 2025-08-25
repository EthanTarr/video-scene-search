import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any
import pandas as pd

class EmbeddingStorage:
    def __init__(self, dimension: int = 512, storage_path: str = "data/embeddings"):
        self.dimension = dimension
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)

        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product for similarity
        self.metadata = []

    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        """Add embeddings to the index with associated metadata."""
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        self.index.add(embeddings.astype(np.float32))
        self.metadata.extend(metadata)
        print(f"Added {len(embeddings)} embeddings to index")

    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar embeddings."""
        # Normalize query
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['similarity_score'] = float(score)
                results.append(result)

        return results

    def save(self, filename: str = "scene_embeddings"):
        """Save index and metadata to disk."""
        index_path = os.path.join(self.storage_path, f"{filename}.index")
        metadata_path = os.path.join(self.storage_path, f"{filename}_metadata.pkl")

        faiss.write_index(self.index, index_path)
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        print(f"Saved embeddings to {index_path}")

    def load(self, filename: str = "scene_embeddings"):
        """Load index and metadata from disk."""
        index_path = os.path.join(self.storage_path, f"{filename}.index")
        metadata_path = os.path.join(self.storage_path, f"{filename}_metadata.pkl")

        if os.path.exists(index_path) and os.path.exists(metadata_path):
            self.index = faiss.read_index(index_path)
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            print(f"Loaded {len(self.metadata)} embeddings from {index_path}")
            return True
        return False

    def get_stats(self):
        """Get statistics about the stored embeddings."""
        return {
            'total_embeddings': len(self.metadata),
            'unique_videos': len(set(m['video_source'] for m in self.metadata)),
            'total_duration': sum(m['duration'] for m in self.metadata)
        }