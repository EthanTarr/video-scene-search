#!/usr/bin/env python3
"""
Main script to process videos: detect scenes, chunk them, and generate embeddings.
"""

import os
import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from scene_detection.detector import SceneDetector
from scene_detection.chunker import VideoChunker
from embeddings.extractor import SceneEmbeddingExtractor
from embeddings.storage import EmbeddingStorage
import numpy as np

def process_video(video_path: str, output_base: str = "data"):
    """Process a single video through the complete pipeline."""
    print(f"\n{'='*60}")
    print(f"Processing video: {os.path.basename(video_path)}")
    print(f"{'='*60}")

    # Initialize components
    detector = SceneDetector(threshold=30.0, min_scene_len=1.0)
    chunker = VideoChunker(output_dir=f"{output_base}/scenes")
    extractor = SceneEmbeddingExtractor()
    storage = EmbeddingStorage(storage_path=f"{output_base}/embeddings")

    # Load existing index if available
    storage.load()

    # 1. Detect scenes
    print("\n1. Detecting scenes...")
    scenes = detector.detect_scenes(video_path)
    print(f"   Found {len(scenes)} scenes")

    if not scenes:
        print("   No scenes detected, skipping video")
        return 0

    # 2. Save scene metadata
    print("\n2. Saving scene metadata...")
    video_name = Path(video_path).stem
    metadata_path = detector.save_scene_metadata(
        video_path, scenes, f"{output_base}/metadata"
    )
    print(f"   Metadata saved to: {metadata_path}")

    # 3. Chunk video into scenes
    print("\n3. Chunking video into scenes...")
    scene_paths = chunker.chunk_video(video_path, scenes, video_name)
    print(f"   Created {len(scene_paths)} scene files")

    # 4. Extract embeddings for each scene
    print("\n4. Extracting embeddings...")
    embeddings = []
    metadata = []

    for i, scene_path in enumerate(scene_paths):
        if os.path.exists(scene_path):
            try:
                print(f"   Processing scene {i+1}/{len(scene_paths)}")
                embedding = extractor.extract_video_embedding(scene_path)
                embeddings.append(embedding)

                metadata.append({
                    'video_source': video_name,
                    'scene_id': i,
                    'scene_path': scene_path,
                    'start_time': scenes[i][0],
                    'end_time': scenes[i][1],
                    'duration': scenes[i][1] - scenes[i][0]
                })
            except Exception as e:
                print(f"   Error processing scene {scene_path}: {e}")

    # 5. Store embeddings
    if embeddings:
        print("\n5. Storing embeddings...")
        embeddings_array = np.vstack(embeddings)
        storage.add_embeddings(embeddings_array, metadata)
        storage.save()

        # Print statistics
        stats = storage.get_stats()
        print(f"   Total embeddings in database: {stats['total_embeddings']}")
        print(f"   Unique videos: {stats['unique_videos']}")
        print(f"   Total duration: {stats['total_duration']:.1f} seconds")

    return len(embeddings)

def main():
    parser = argparse.ArgumentParser(description="Process videos for scene detection and embedding")
    parser.add_argument("input_path", help="Path to video file or directory of videos")
    parser.add_argument("--output", default="data", help="Output directory base")
    parser.add_argument("--threshold", type=float, default=30.0, help="Scene detection threshold")
    parser.add_argument("--min-scene-len", type=float, default=1.0, help="Minimum scene length in seconds")

    args = parser.parse_args()

    input_path = Path(args.input_path)

    if input_path.is_file():
        # Process single video
        process_video(str(input_path), args.output)
    elif input_path.is_dir():
        # Process all videos in directory
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        video_files = [f for f in input_path.iterdir()
                       if f.suffix.lower() in video_extensions]

        if not video_files:
            print(f"No video files found in {input_path}")
            return

        print(f"Found {len(video_files)} video files to process")

        total_scenes = 0
        for i, video_file in enumerate(video_files, 1):
            print(f"\nProcessing video {i}/{len(video_files)}")
            scenes_count = process_video(str(video_file), args.output)
            total_scenes += scenes_count

        print(f"\n{'='*60}")
        print(f"PROCESSING COMPLETE")
        print(f"Processed {len(video_files)} videos, {total_scenes} total scenes")
        print(f"{'='*60}")
    else:
        print(f"Error: {input_path} is not a valid file or directory")

if __name__ == "__main__":
    main()