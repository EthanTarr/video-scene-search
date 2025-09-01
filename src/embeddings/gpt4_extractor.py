"""
GPT-4 Enhanced Embedding Extractor for Video Search
Combines CLIP visual embeddings with OpenAI's text embeddings for improved search capabilities.
"""

import torch
import open_clip
import cv2
import numpy as np
from PIL import Image
from typing import List, Optional, Dict, Any, Union, Tuple
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

class GPT4EnhancedEmbeddingExtractor:
    """
    Enhanced embedding extractor that combines CLIP visual embeddings 
    with OpenAI's GPT-4 text embeddings for superior search capabilities.
    """
    
    def __init__(self, 
                 clip_model: str = "ViT-B/32",
                 openai_model: str = "text-embedding-3-large",
                 device: str = "auto",
                 api_key: Optional[str] = None):
        """
        Initialize the enhanced extractor with both CLIP and OpenAI models.
        
        Args:
            clip_model: CLIP model name for visual embeddings
            openai_model: OpenAI embedding model name
            device: Device for CLIP model ('auto', 'cuda', 'cpu')
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
        """
        # Set up device for CLIP
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Initialize CLIP model
        print(f"Loading CLIP model {clip_model} on {self.device}")
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(clip_model, pretrained="openai", device=self.device)
        self.clip_model.eval()
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        self.openai_model = openai_model
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        print(f"Enhanced extractor initialized with CLIP ({clip_model}) and OpenAI ({openai_model})")
    
    def extract_video_embedding(self, video_path: str, sample_rate: int = 30) -> np.ndarray:
        """
        Extract CLIP embedding from video by sampling frames.
        This maintains compatibility with the existing system.
        
        Args:
            video_path: Path to video file
            sample_rate: Frame sampling rate (every Nth frame)
            
        Returns:
            Normalized CLIP video embedding
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
            
            frame_count += 1
        
        cap.release()
        
        if not frames:
            raise ValueError(f"No frames extracted from {video_path}")
        
        self.logger.info(f"Extracted {len(frames)} frames from {os.path.basename(video_path)}")
        
        # Process frames and get embeddings
        embeddings = []
        with torch.no_grad():
            for frame in frames:
                image_input = self.clip_preprocess(frame).unsqueeze(0).to(self.device)
                embedding = self.clip_model.encode_image(image_input)
                embeddings.append(embedding.cpu().numpy())
        
        # Average embeddings across frames
        scene_embedding = np.mean(embeddings, axis=0)
        return scene_embedding.flatten()
    
    def extract_clip_text_embedding(self, text: str) -> np.ndarray:
        """
        Extract CLIP text embedding for compatibility with existing video embeddings.
        
        Args:
            text: Input text
            
        Returns:
            CLIP text embedding
        """
        with torch.no_grad():
            text_tokens = open_clip.tokenize([text]).to(self.device)
            text_embedding = self.clip_model.encode_text(text_tokens)
            return text_embedding.cpu().numpy().flatten()
    
    def extract_openai_text_embedding(self, text: str) -> np.ndarray:
        """
        Extract OpenAI GPT-4 text embedding for enhanced text understanding.
        
        Args:
            text: Input text
            
        Returns:
            OpenAI text embedding
        """
        try:
            response = self.openai_client.embeddings.create(
                model=self.openai_model,
                input=text
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            self.logger.error(f"Error creating OpenAI embedding: {e}")
            return np.array([])
    
    def enhance_prompt_with_gpt4(self, prompt: str) -> Dict[str, str]:
        """
        Use GPT-4 or fallback models to enhance and expand the user prompt for better video matching.
        
        Args:
            prompt: Original user prompt
            
        Returns:
            Dictionary with enhanced prompt information
        """
        # Try different models in order of preference
        models_to_try = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "gpt-4"]
        
        system_message = """You are a video content analyst. Given a user's search prompt, help expand and enhance it to better match video content. 

Provide:
1. Visual elements that might be present
2. Potential actions or activities
3. Scene types or settings
4. Objects or people that might appear
5. Alternative descriptions or synonyms

Keep your response concise and focused on visual elements that would be detectable in video content."""
        
        for model in models_to_try:
            try:
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": f"Enhance this video search prompt: '{prompt}'"}
                    ],
                    max_tokens=300,
                    temperature=0.3
                )
                
                enhancement = response.choices[0].message.content
                
                return {
                    "original_prompt": prompt,
                    "enhanced_description": enhancement,
                    "combined_text": f"{prompt}. {enhancement}",
                    "model_used": model
                }
                
            except Exception as e:
                self.logger.warning(f"Model {model} failed: {e}")
                continue
        
        # If all models fail, return original prompt
        self.logger.error(f"All GPT models failed for prompt enhancement")
        return {
            "original_prompt": prompt,
            "enhanced_description": "(GPT enhancement unavailable)",
            "combined_text": prompt,
            "model_used": "none"
        }
    
    def create_hybrid_search_embedding(self, prompt: str, use_gpt4_enhancement: bool = True) -> Dict[str, np.ndarray]:
        """
        Create both CLIP and OpenAI embeddings for hybrid search capabilities.
        
        Args:
            prompt: User search prompt
            use_gpt4_enhancement: Whether to enhance prompt with GPT-4
            
        Returns:
            Dictionary with both embedding types
        """
        results = {}
        
        # Enhance prompt if requested
        if use_gpt4_enhancement:
            enhanced_prompt_info = self.enhance_prompt_with_gpt4(prompt)
            search_text = enhanced_prompt_info["combined_text"]
        else:
            search_text = prompt
        
        # Create CLIP embedding (for matching with video embeddings)
        clip_embedding = self.extract_clip_text_embedding(search_text)
        results["clip_embedding"] = clip_embedding
        
        # Create OpenAI embedding (for enhanced text understanding)
        openai_embedding = self.extract_openai_text_embedding(search_text)
        if openai_embedding.size > 0:
            results["openai_embedding"] = openai_embedding
        
        return results
    
    def extract_video_metadata_embedding(self, metadata: Dict[str, Any]) -> np.ndarray:
        """
        Create embedding from video metadata using OpenAI models.
        Useful for text-rich metadata like descriptions, titles, etc.
        
        Args:
            metadata: Video metadata dictionary
            
        Returns:
            OpenAI embedding of metadata text
        """
        # Combine relevant text fields from metadata
        text_fields = []
        
        if 'title' in metadata:
            text_fields.append(f"Title: {metadata['title']}")
        if 'description' in metadata:
            text_fields.append(f"Description: {metadata['description']}")
        if 'tags' in metadata:
            if isinstance(metadata['tags'], list):
                text_fields.append(f"Tags: {', '.join(metadata['tags'])}")
            else:
                text_fields.append(f"Tags: {metadata['tags']}")
        if 'transcript' in metadata:
            text_fields.append(f"Content: {metadata['transcript']}")
        
        combined_text = " ".join(text_fields)
        
        if combined_text.strip():
            return self.extract_openai_text_embedding(combined_text)
        else:
            return np.array([])


def main():
    """
    Demo function showing how to use the enhanced extractor.
    """
    print("GPT-4 Enhanced Video Search Demo")
    print("=" * 50)
    
    try:
        # Initialize extractor
        extractor = GPT4EnhancedEmbeddingExtractor()
        
        # Example prompt enhancement
        test_prompt = "person walking dog in park"
        print(f"\\nOriginal prompt: '{test_prompt}'")
        
        enhanced_info = extractor.enhance_prompt_with_gpt4(test_prompt)
        print(f"\\nEnhanced description:\\n{enhanced_info['enhanced_description']}")
        
        # Create hybrid embeddings
        embeddings = extractor.create_hybrid_search_embedding(test_prompt)
        print(f"\\nGenerated embeddings:")
        print(f"- CLIP embedding shape: {embeddings['clip_embedding'].shape}")
        if 'openai_embedding' in embeddings:
            print(f"- OpenAI embedding shape: {embeddings['openai_embedding'].shape}")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        print("Make sure you have OPENAI_API_KEY set in your environment variables.")


if __name__ == "__main__":
    main()
