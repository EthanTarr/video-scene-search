#!/usr/bin/env python3
"""
Search video scenes using text prompts with GPT-4 enhanced capabilities.
"""

import os
import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from embeddings.extractor import SceneEmbeddingExtractor
from embeddings.storage import EmbeddingStorage
from embeddings.gpt4_search import GPT4VideoSearchEngine

def search_by_text(query: str, k: int = 10, data_dir: str = "data", use_gpt4: bool = False):
    """Search for scenes using a text query."""
    if use_gpt4:
        # Use GPT-4 enhanced search
        try:
            search_engine = GPT4VideoSearchEngine(storage_path=f"{data_dir}/embeddings")
            results = search_engine.search_with_prompt(query, k, use_gpt4_enhancement=True)
            search_engine.display_search_results(results, show_paths=True)
        except Exception as e:
            print(f"GPT-4 search failed: {e}")
            print("Falling back to CLIP search...")
            search_with_clip(query, k, data_dir)
    else:
        search_with_clip(query, k, data_dir)

def search_with_clip(query: str, k: int, data_dir: str):
    """Search using standard CLIP embeddings."""
    # Initialize components
    extractor = SceneEmbeddingExtractor()
    storage = EmbeddingStorage(storage_path=f"{data_dir}/embeddings")
    
    # Load existing embeddings
    if not storage.load():
        print(f"No embeddings found in {data_dir}/embeddings")
        print("Please run process_videos.py first to create embeddings.")
        return
    
    print(f"Loaded {len(storage.metadata)} video scene embeddings")
    
    # Create text embedding for query
    query_embedding = extractor.extract_text_embedding(query)
    
    # Search for similar scenes
    results = storage.search(query_embedding, k)
    
    # Display results
    display_results(results, query)

def search_by_video(video_path: str, k: int = 10, data_dir: str = "data"):
    """Search for scenes similar to a given video clip."""
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return
    
    # Initialize components
    extractor = SceneEmbeddingExtractor()
    storage = EmbeddingStorage(storage_path=f"{data_dir}/embeddings")
    
    # Load existing embeddings
    if not storage.load():
        print(f"No embeddings found in {data_dir}/embeddings")
        print("Please run process_videos.py first to create embeddings.")
        return
    
    print(f"Loaded {len(storage.metadata)} video scene embeddings")
    print(f"\nExtracting embedding from: {video_path}")
    
    try:
        # Extract embedding from the query video
        query_embedding = extractor.extract_video_embedding(video_path)
        
        # Search for similar scenes
        results = storage.search(query_embedding, k)
        
        # Display results
        display_results(results, f"video: {os.path.basename(video_path)}")
        
    except Exception as e:
        print(f"Error processing video: {e}")

def display_results(results, query):
    """Display search results in a formatted way."""
    if not results:
        print("No results found.")
        return
    
    print(f"\n{'='*80}")
    print(f"SEARCH RESULTS FOR: {query}")
    print(f"{'='*80}")
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.get('video_source', 'Unknown Video')}")
        print(f"   Scene: {result.get('scene_id', 0)} | Duration: {result.get('duration', 0):.1f}s | Similarity: {result.get('similarity_score', 0):.4f}")
        print(f"   Time: {result.get('start_time', 0):.1f}s - {result.get('end_time', 0):.1f}s")
        
        if 'scene_path' in result:
            print(f"   Path: {result['scene_path']}")
        
        print("-" * 60)

def show_stats(data_dir: str = "data"):
    """Show database statistics."""
    storage = EmbeddingStorage(storage_path=f"{data_dir}/embeddings")
    
    if not storage.load():
        print(f"No embeddings found in {data_dir}/embeddings")
        return
    
    stats = storage.get_stats()
    
    print(f"\n{'='*50}")
    print("DATABASE STATISTICS")
    print(f"{'='*50}")
    print(f"Total embeddings: {stats['total_embeddings']}")
    print(f"Unique videos: {stats['unique_videos']}")
    print(f"Total duration: {stats['total_duration']:.1f} seconds ({stats['total_duration']/60:.1f} minutes)")
    
    # Show videos in database
    video_sources = set(m['video_source'] for m in storage.metadata)
    print(f"\nVideos in database:")
    for video in sorted(video_sources):
        scene_count = sum(1 for m in storage.metadata if m['video_source'] == video)
        video_duration = sum(m['duration'] for m in storage.metadata if m['video_source'] == video)
        print(f"  - {video}: {scene_count} scenes, {video_duration:.1f}s")

def interactive_search(data_dir: str = "data", use_gpt4: bool = False):
    """Interactive search mode."""
    print("\n" + "="*60)
    print("INTERACTIVE VIDEO SEARCH")
    if use_gpt4:
        print("GPT-4 Enhanced Mode")
    else:
        print("Standard CLIP Mode")
    print("="*60)
    print("Commands:")
    print("  - Type your search query to find matching scenes")
    print("  - 'stats' to show database statistics")
    print("  - 'gpt4 on/off' to toggle GPT-4 enhancement")
    print("  - 'quit' or 'exit' to quit")
    print("-" * 60)
    
    while True:
        try:
            query = input("\nEnter search query: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            elif query.lower() == 'stats':
                show_stats(data_dir)
            elif query.lower().startswith('gpt4 '):
                if 'on' in query.lower():
                    use_gpt4 = True
                    print("GPT-4 enhancement enabled")
                elif 'off' in query.lower():
                    use_gpt4 = False
                    print("GPT-4 enhancement disabled")
            else:
                search_by_text(query, k=5, data_dir=data_dir, use_gpt4=use_gpt4)
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Search video scenes using text or video queries")
    parser.add_argument("--query", "-q", help="Text query to search for")
    parser.add_argument("--video", "-v", help="Video file to find similar scenes")
    parser.add_argument("--top-k", "-k", type=int, default=10, help="Number of results to return")
    parser.add_argument("--data", default="data", help="Data directory path")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive search mode")
    parser.add_argument("--gpt4", action="store_true", help="Use GPT-4 enhanced search")
    
    args = parser.parse_args()
    
    if args.stats:
        show_stats(args.data)
    elif args.interactive:
        interactive_search(args.data, args.gpt4)
    elif args.query:
        search_by_text(args.query, args.top_k, args.data, args.gpt4)
    elif args.video:
        search_by_video(args.video, args.top_k, args.data)
    else:
        # Default to interactive mode
        interactive_search(args.data, args.gpt4)

if __name__ == "__main__":
    main()
