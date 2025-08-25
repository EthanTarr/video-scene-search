"""
GPT-4 Enhanced Video Search Engine
Provides advanced search capabilities using both CLIP and OpenAI embeddings.
"""

import numpy as np
import faiss
from typing import List, Dict, Any, Tuple, Optional
import os
import json
import logging
from pathlib import Path

from .gpt4_extractor import GPT4EnhancedEmbeddingExtractor
from .storage import EmbeddingStorage


class GPT4VideoSearchEngine:
    """
    Enhanced video search engine that combines CLIP visual similarity 
    with GPT-4 enhanced text understanding for superior search results.
    """
    
    def __init__(self, 
                 storage_path: str = "data/embeddings",
                 clip_model: str = "ViT-B/32",
                 openai_model: str = "text-embedding-3-large"):
        """
        Initialize the enhanced search engine.
        
        Args:
            storage_path: Path to embedding storage
            clip_model: CLIP model name
            openai_model: OpenAI embedding model name
        """
        self.storage_path = storage_path
        self.extractor = GPT4EnhancedEmbeddingExtractor(
            clip_model=clip_model,
            openai_model=openai_model
        )
        
        # Initialize storage for CLIP embeddings (existing system)
        self.clip_storage = EmbeddingStorage(storage_path=storage_path)
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Load existing embeddings
        self.load_embeddings()
    
    def load_embeddings(self):
        """Load existing CLIP embeddings from storage."""
        success = self.clip_storage.load()
        if success:
            self.logger.info(f"Loaded {len(self.clip_storage.metadata)} CLIP embeddings")
        else:
            self.logger.info("No existing embeddings found")
    
    def search_with_prompt(self, 
                          prompt: str, 
                          k: int = 10,
                          use_gpt4_enhancement: bool = True,
                          min_similarity: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for videos using a text prompt with GPT-4 enhancement.
        
        Args:
            prompt: User's search prompt
            k: Number of results to return
            use_gpt4_enhancement: Whether to enhance prompt with GPT-4
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of search results with metadata and scores
        """
        if not self.clip_storage.metadata:
            self.logger.warning("No video embeddings loaded. Process some videos first.")
            return []
        
        # Create search embeddings
        search_embeddings = self.extractor.create_hybrid_search_embedding(
            prompt, use_gpt4_enhancement
        )
        
        # Use CLIP embedding for search (compatible with existing video embeddings)
        clip_embedding = search_embeddings["clip_embedding"]
        
        # Perform similarity search
        results = self.clip_storage.search(clip_embedding, k)
        
        # Filter by minimum similarity
        filtered_results = [
            result for result in results 
            if result['similarity_score'] >= min_similarity
        ]
        
        # Add search metadata
        for result in filtered_results:
            result['search_prompt'] = prompt
            result['enhanced_search'] = use_gpt4_enhancement
        
        self.logger.info(f"Found {len(filtered_results)} results for prompt: '{prompt}'")
        return filtered_results
    
    def enhanced_search_with_reranking(self, 
                                     prompt: str, 
                                     k: int = 20,
                                     final_k: int = 10,
                                     use_gpt4_enhancement: bool = True) -> List[Dict[str, Any]]:
        """
        Perform enhanced search with GPT-4 reranking for better results.
        
        This method:
        1. Gets initial results using CLIP similarity
        2. Uses GPT-4 to rerank results based on semantic understanding
        3. Returns the top results
        
        Args:
            prompt: User's search prompt  
            k: Initial number of results to get
            final_k: Final number of results to return
            use_gpt4_enhancement: Whether to enhance the original prompt
            
        Returns:
            Reranked search results
        """
        # Get initial results with more candidates
        initial_results = self.search_with_prompt(
            prompt, k, use_gpt4_enhancement, min_similarity=0.0
        )
        
        if not initial_results:
            return []
        
        # Prepare context for GPT-4 reranking
        try:
            reranked_results = self._rerank_with_gpt4(prompt, initial_results, final_k)
            self.logger.info(f"Reranked {len(initial_results)} results to {len(reranked_results)}")
            return reranked_results
        except Exception as e:
            self.logger.error(f"GPT-4 reranking failed: {e}")
            # Fall back to original results
            return initial_results[:final_k]
    
    def _rerank_with_gpt4(self, 
                         prompt: str, 
                         results: List[Dict[str, Any]], 
                         final_k: int) -> List[Dict[str, Any]]:
        """
        Use GPT-4 to rerank search results based on semantic relevance.
        
        Args:
            prompt: Original search prompt
            results: Initial search results
            final_k: Number of final results to return
            
        Returns:
            Reranked results
        """
        # Prepare video descriptions for GPT-4
        video_descriptions = []
        for i, result in enumerate(results):
            description = f"Video {i+1}: {result.get('video_source', 'Unknown')} "
            description += f"(Scene {result.get('scene_id', 0)}, "
            description += f"{result.get('duration', 0):.1f}s, "
            description += f"similarity: {result.get('similarity_score', 0):.3f})"
            video_descriptions.append(description)
        
        system_message = f"""You are a video search expert. Given a search prompt and a list of video candidates, rank them by relevance to the search query.

Search prompt: "{prompt}"

Consider:
- Video content relevance to the prompt
- Scene duration (longer scenes might be more substantial)
- Current similarity scores as a baseline

Return only the video numbers (1-{len(results)}) in order of relevance, separated by commas. For example: "3,1,7,2,5"
"""
        
        videos_text = "\\n".join(video_descriptions)
        
        # Try different models for reranking
        models_to_try = ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o", "gpt-4"]
        
        response = None
        for model in models_to_try:
            try:
                response = self.extractor.openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": f"Rank these videos:\\n\\n{videos_text}"}
                    ],
                    max_tokens=100,
                    temperature=0.1
                )
                break
            except Exception as e:
                self.logger.warning(f"Reranking model {model} failed: {e}")
                continue
        
        if response is None:
            raise Exception("All reranking models failed")
        
        # Parse GPT-4 ranking
        ranking_text = response.choices[0].message.content.strip()
        try:
            # Extract numbers from response
            rankings = [int(x.strip()) - 1 for x in ranking_text.split(',') if x.strip().isdigit()]
            
            # Reorder results based on GPT-4 ranking
            reranked_results = []
            for rank_idx in rankings[:final_k]:
                if 0 <= rank_idx < len(results):
                    result = results[rank_idx].copy()
                    result['gpt4_rank'] = len(reranked_results) + 1
                    reranked_results.append(result)
            
            # Fill remaining spots with original ranking if needed
            while len(reranked_results) < final_k and len(reranked_results) < len(results):
                for result in results:
                    if result not in reranked_results:
                        result['gpt4_rank'] = len(reranked_results) + 1
                        reranked_results.append(result)
                        break
            
            return reranked_results
            
        except Exception as e:
            self.logger.error(f"Failed to parse GPT-4 ranking: {e}")
            return results[:final_k]
    
    def get_video_summary(self, video_source: str) -> Dict[str, Any]:
        """
        Get a summary of all scenes from a specific video.
        
        Args:
            video_source: Name of the video to summarize
            
        Returns:
            Dictionary with video summary information
        """
        # Find all scenes from the video
        video_scenes = [
            metadata for metadata in self.clip_storage.metadata
            if metadata.get('video_source') == video_source
        ]
        
        if not video_scenes:
            return {"error": f"No scenes found for video: {video_source}"}
        
        # Calculate summary statistics
        total_duration = sum(scene.get('duration', 0) for scene in video_scenes)
        scene_count = len(video_scenes)
        
        # Get scene time ranges
        scenes_info = [
            {
                "scene_id": scene.get('scene_id', i),
                "start_time": scene.get('start_time', 0),
                "end_time": scene.get('end_time', 0),
                "duration": scene.get('duration', 0),
                "scene_path": scene.get('scene_path', '')
            }
            for i, scene in enumerate(video_scenes)
        ]
        
        return {
            "video_source": video_source,
            "total_scenes": scene_count,
            "total_duration": total_duration,
            "scenes": scenes_info
        }
    
    def find_similar_scenes_in_video(self, 
                                   video_source: str, 
                                   scene_id: int, 
                                   k: int = 5) -> List[Dict[str, Any]]:
        """
        Find scenes similar to a specific scene within the same video or across videos.
        
        Args:
            video_source: Source video name
            scene_id: Scene ID to find similar scenes for
            k: Number of similar scenes to return
            
        Returns:
            List of similar scenes
        """
        # Find the reference scene
        reference_scene = None
        reference_idx = None
        
        for i, metadata in enumerate(self.clip_storage.metadata):
            if (metadata.get('video_source') == video_source and 
                metadata.get('scene_id') == scene_id):
                reference_scene = metadata
                reference_idx = i
                break
        
        if reference_scene is None:
            return []
        
        # Get the embedding for the reference scene
        if hasattr(self.clip_storage.index, 'reconstruct'):
            reference_embedding = self.clip_storage.index.reconstruct(reference_idx)
        else:
            # If we can't reconstruct, we need to search using the embedding itself
            # This is a limitation of some FAISS indices
            self.logger.warning("Cannot reconstruct embeddings from this index type")
            return []
        
        # Search for similar scenes
        results = self.clip_storage.search(reference_embedding, k + 1)  # +1 to exclude self
        
        # Remove the reference scene from results
        similar_scenes = [
            result for result in results
            if not (result.get('video_source') == video_source and 
                   result.get('scene_id') == scene_id)
        ]
        
        return similar_scenes[:k]
    
    def display_search_results(self, 
                             results: List[Dict[str, Any]], 
                             show_paths: bool = False):
        """
        Display search results in a formatted way.
        
        Args:
            results: Search results to display
            show_paths: Whether to show file paths
        """
        if not results:
            print("No results found.")
            return
        
        print("\\n" + "="*80)
        print("GPT-4 ENHANCED VIDEO SEARCH RESULTS")
        print("="*80)
        
        for i, result in enumerate(results, 1):
            print(f"\\n{i}. {result.get('video_source', 'Unknown Video')}")
            print(f"   Scene: {result.get('scene_id', 0)} | "
                  f"Duration: {result.get('duration', 0):.1f}s | "
                  f"Similarity: {result.get('similarity_score', 0):.4f}")
            
            if 'gpt4_rank' in result:
                print(f"   GPT-4 Rank: #{result['gpt4_rank']}")
            
            print(f"   Time: {result.get('start_time', 0):.1f}s - "
                  f"{result.get('end_time', 0):.1f}s")
            
            if show_paths and 'scene_path' in result:
                print(f"   Path: {result['scene_path']}")
            
            print("-" * 60)


def main():
    """
    Demo function for the enhanced video search engine.
    """
    print("GPT-4 Enhanced Video Search Engine Demo")
    print("="*50)
    
    try:
        # Initialize search engine
        search_engine = GPT4VideoSearchEngine()
        
        # Example searches
        test_queries = [
            "person walking outdoors",
            "car driving at night", 
            "people talking indoors",
            "dog running in park"
        ]
        
        for query in test_queries:
            print(f"\\n\\nSearching for: '{query}'")
            print("-" * 40)
            
            # Standard search
            print("\\nStandard search results:")
            results = search_engine.search_with_prompt(query, k=3)
            search_engine.display_search_results(results)
            
            # Enhanced search with reranking
            print("\\nGPT-4 enhanced search results:")
            enhanced_results = search_engine.enhanced_search_with_reranking(
                query, k=10, final_k=3
            )
            search_engine.display_search_results(enhanced_results)
            
    except Exception as e:
        print(f"Demo failed: {e}")
        print("Make sure you have:")
        print("1. OPENAI_API_KEY set in environment variables")
        print("2. Processed videos with embeddings in data/embeddings/")


if __name__ == "__main__":
    main()
