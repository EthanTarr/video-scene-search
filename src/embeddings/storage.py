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
        try:
            # Ensure embeddings are float32 and contiguous
            embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
            
            # Validate embeddings shape and content
            if embeddings.size == 0:
                raise ValueError("Empty embeddings array")
            
            if embeddings.ndim != 2:
                raise ValueError(f"Expected 2D array, got {embeddings.ndim}D")
            
            if embeddings.shape[1] != self.dimension:
                raise ValueError(f"Expected dimension {self.dimension}, got {embeddings.shape[1]}")
            
            # Check for NaN or infinite values
            if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
                raise ValueError("Embeddings contain NaN or infinite values")
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)

            self.index.add(embeddings)
            self.metadata.extend(metadata)
            print(f"Added {len(embeddings)} embeddings to index")
            
        except Exception as e:
            print(f"Error adding embeddings: {e}")
            print(f"Embeddings shape: {embeddings.shape}")
            print(f"Embeddings dtype: {embeddings.dtype}")
            print(f"Embeddings min/max: {np.min(embeddings)}, {np.max(embeddings)}")
            raise

    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar embeddings."""
        # Ensure query is float32 and contiguous
        query_embedding = np.ascontiguousarray(query_embedding.reshape(1, -1), dtype=np.float32)
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
    
    def clear(self):
        """Clear all embeddings and metadata."""
        # Reset FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata.clear()
        
        # Remove saved files
        index_path = os.path.join(self.storage_path, "scene_embeddings.index")
        metadata_path = os.path.join(self.storage_path, "scene_embeddings_metadata.pkl")
        
        if os.path.exists(index_path):
            os.remove(index_path)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        
        print("Cleared all embeddings and metadata")