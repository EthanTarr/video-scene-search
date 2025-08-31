import torch
import clip
import cv2
import numpy as np
from PIL import Image
from typing import List, Optional
import os

class SceneEmbeddingExtractor:
    def __init__(self, model_name: str = "ViT-B/32", device: str = "auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading CLIP model {model_name} on {self.device}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

    def extract_video_embedding(self, video_path: str,
                                sample_rate: int = 30) -> np.ndarray:
        """Extract embedding from video by sampling frames."""
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

        print(f"Extracted {len(frames)} frames from {os.path.basename(video_path)}")

        # Process frames and get embeddings
        embeddings = []
        with torch.no_grad():
            for frame in frames:
                image_input = self.preprocess(frame).unsqueeze(0).to(self.device)
                embedding = self.model.encode_image(image_input)
                embeddings.append(embedding.cpu().numpy())

        # Average embeddings across frames
        scene_embedding = np.mean(embeddings, axis=0)
        # Ensure float32 output for FAISS compatibility
        return scene_embedding.flatten().astype(np.float32)

    def extract_text_embedding(self, text: str) -> np.ndarray:
        """Extract embedding from text description."""
        with torch.no_grad():
            text_tokens = clip.tokenize([text]).to(self.device)
            text_embedding = self.model.encode_text(text_tokens)
            # Ensure float32 output for FAISS compatibility
            return text_embedding.cpu().numpy().flatten().astype(np.float32)