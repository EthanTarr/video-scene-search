"""
Tests for scene detection and video chunking functionality.
"""
import pytest
import cv2
import os
import tempfile
import shutil
import numpy as np
from unittest.mock import patch
from pathlib import Path

# Import the real modules we're testing
from scene_detection.detector import SceneDetector
from scene_detection.chunker import VideoChunker


class TestSceneDetector:
    """Test SceneDetector class functionality."""
    
    def test_scene_detector_initialization(self):
        """Test SceneDetector initialization with default parameters."""
        detector = SceneDetector()
        
        assert detector.threshold == 30.0
        assert detector.min_scene_len == 1.0
        assert detector.method == "auto"  # Will try PySceneDetect first, then fallback
    
    def test_scene_detector_custom_parameters(self):
        """Test SceneDetector initialization with custom parameters."""
        detector = SceneDetector(threshold=25.0, min_scene_len=2.0)
        
        assert detector.threshold == 25.0
        assert detector.min_scene_len == 2.0
        assert detector.method == "auto"  # This is set internally
    
    def test_detect_scenes_with_real_video(self, test_video_path):
        """Test scene detection with actual test video."""
        detector = SceneDetector()
        
        # Test with the actual test video
        scenes = detector.detect_scenes(test_video_path)
        
        # Verify that scenes are returned correctly
        assert isinstance(scenes, list)
        assert len(scenes) > 0
        
        # Verify scene structure
        for start_time, end_time in scenes:
            assert isinstance(start_time, (int, float))
            assert isinstance(end_time, (int, float))
            assert start_time >= 0
            assert end_time > start_time
        
        # Verify scene durations meet minimum requirements
        for start_time, end_time in scenes:
            duration = end_time - start_time
            assert duration >= detector.min_scene_len
    
    def test_detect_scenes_video_not_found(self):
        """Test scene detection with non-existent video."""
        detector = SceneDetector()
        
        # Should handle missing video gracefully
        scenes = detector.detect_scenes("nonexistent_video.mp4")
        assert isinstance(scenes, list)
        assert len(scenes) > 0  # Should fall back to single scene
    
    def test_detect_scenes_with_different_thresholds(self, test_video_path):
        """Test scene detection with different threshold values."""
        # Test with lower threshold (more sensitive)
        detector_low = SceneDetector(threshold=15.0)
        scenes_low = detector_low.detect_scenes(test_video_path)
        
        # Test with higher threshold (less sensitive)
        detector_high = SceneDetector(threshold=45.0)
        scenes_high = detector_high.detect_scenes(test_video_path)
        
        # Lower threshold should generally detect more scenes
        # (though this depends on video content)
        assert isinstance(scenes_low, list)
        assert isinstance(scenes_high, list)
        assert len(scenes_low) > 0
        assert len(scenes_high) > 0
    
    def test_save_scene_metadata(self, test_video_path, temp_dir):
        """Test saving scene metadata to CSV."""
        detector = SceneDetector()
        scenes = detector.detect_scenes(test_video_path)
        
        # Save metadata
        metadata_path = detector.save_scene_metadata(test_video_path, scenes, temp_dir)
        
        # Verify file was created
        assert os.path.exists(metadata_path)
        assert metadata_path.endswith('.csv')
        
        # Verify file content
        import pandas as pd
        df = pd.read_csv(metadata_path)
        
        assert len(df) == len(scenes)
        assert 'start_time' in df.columns
        assert 'end_time' in df.columns
        assert 'duration' in df.columns
        assert 'video_source' in df.columns
        assert 'scene_id' in df.columns
        
        # Verify data integrity with approximate comparison for floating point precision
        for i, (start_time, end_time) in enumerate(scenes):
            assert np.isclose(df.iloc[i]['start_time'], start_time, rtol=1e-5)
            assert np.isclose(df.iloc[i]['end_time'], end_time, rtol=1e-5)
            assert np.isclose(df.iloc[i]['duration'], end_time - start_time, rtol=1e-5)
            assert df.iloc[i]['scene_id'] == i


class TestVideoChunker:
    """Test VideoChunker class functionality."""
    
    def test_video_chunker_initialization(self, temp_dir):
        """Test VideoChunker initialization."""
        output_dir = str(temp_dir / "scenes")
        chunker = VideoChunker(output_dir)
        
        assert chunker.output_dir == Path(output_dir)
        assert chunker.output_dir.exists()
    
    def test_video_chunker_default_output_dir(self):
        """Test VideoChunker with default output directory."""
        chunker = VideoChunker()
        
        # Should create default directory
        assert chunker.output_dir == Path("data/scenes")
    
    def test_extract_scenes_with_real_video(self, test_video_path, temp_dir):
        """Test scene extraction with actual test video."""
        # First detect scenes
        detector = SceneDetector()
        scenes = detector.detect_scenes(test_video_path)
        
        # Then extract scenes
        output_dir = str(temp_dir / "scenes")
        chunker = VideoChunker(output_dir)
        
        video_name = Path(test_video_path).stem
        output_paths = chunker.chunk_video(test_video_path, scenes, video_name)
        
        assert isinstance(output_paths, list)
        assert len(output_paths) == len(scenes)
        assert all(isinstance(path, str) for path in output_paths)
        
        # Verify that scene files were created
        for scene_path in output_paths:
            assert os.path.exists(scene_path)
            assert scene_path.endswith('.mp4')
    
    def test_extract_scenes_empty_scenes_list(self, test_video_path, temp_dir):
        """Test scene extraction with empty scenes list."""
        output_dir = str(temp_dir / "scenes")
        chunker = VideoChunker(output_dir)
        
        # Empty scenes should return empty list
        output_paths = chunker.chunk_video(test_video_path, [], "test_video")
        assert output_paths == []
    
    def test_extract_scenes_invalid_output_dir(self):
        """Test scene extraction with invalid output directory."""
        # The VideoChunker should handle invalid paths gracefully
        chunker = VideoChunker("/invalid/path")
        
        # Should still be able to create the chunker
        assert chunker.output_dir == Path("/invalid/path")
    
    def test_full_scene_detection_workflow(self, test_video_path, temp_dir):
        """Test complete scene detection and extraction workflow."""
        # Initialize components
        detector = SceneDetector()
        output_dir = str(temp_dir / "scenes")
        chunker = VideoChunker(output_dir)
        
        # Detect scenes
        scenes = detector.detect_scenes(test_video_path)
        assert isinstance(scenes, list)
        assert len(scenes) > 0
        
        # Extract scenes
        video_name = Path(test_video_path).stem
        output_paths = chunker.chunk_video(test_video_path, scenes, video_name)
        
        assert isinstance(output_paths, list)
        assert len(output_paths) == len(scenes)
        
        # Verify all scene files exist and have reasonable sizes
        for scene_path in output_paths:
            assert os.path.exists(scene_path)
            file_size = os.path.getsize(scene_path)
            assert file_size > 0  # Should not be empty
    
    def test_scene_quality_verification(self, test_video_path, temp_dir):
        """Test that extracted scenes maintain video quality."""
        detector = SceneDetector()
        scenes = detector.detect_scenes(test_video_path)
        
        output_dir = str(temp_dir / "scenes")
        chunker = VideoChunker(output_dir)
        
        video_name = Path(test_video_path).stem
        output_paths = chunker.chunk_video(test_video_path, scenes, video_name)
        
        # Verify each scene can be opened and has valid properties
        for scene_path in output_paths:
            cap = cv2.VideoCapture(scene_path)
            assert cap.isOpened()
            
            # Check basic video properties
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            assert frame_count > 0
            assert fps > 0
            assert width > 0
            assert height > 0
            
            cap.release()


class TestSceneDetectionIntegration:
    """Integration tests for scene detection workflow."""
    
    def test_end_to_end_workflow(self, test_video_path, temp_dir):
        """Test complete workflow from video input to scene files."""
        # 1. Detect scenes
        detector = SceneDetector()
        scenes = detector.detect_scenes(test_video_path)
        
        # 2. Save metadata
        metadata_path = detector.save_scene_metadata(test_video_path, scenes, temp_dir)
        
        # 3. Extract scenes
        chunker = VideoChunker(str(temp_dir / "scenes"))
        video_name = Path(test_video_path).stem
        scene_paths = chunker.chunk_video(test_video_path, scenes, video_name)
        
        # 4. Verify results
        assert len(scene_paths) == len(scenes)
        assert os.path.exists(metadata_path)
        
        # 5. Verify metadata matches scenes
        import pandas as pd
        df = pd.read_csv(metadata_path)
        assert len(df) == len(scenes)
        
        # 6. Verify scene files exist and are valid
        for scene_path in scene_paths:
            assert os.path.exists(scene_path)
            assert os.path.getsize(scene_path) > 0
    
    def test_multiple_video_processing(self, test_video_path, temp_dir):
        """Test processing multiple videos in sequence."""
        # Simulate processing the same video multiple times
        detector = SceneDetector()
        chunker = VideoChunker(str(temp_dir / "scenes"))
        
        results = []
        for i in range(3):
            # Process video with different parameters
            detector_i = SceneDetector(threshold=30.0 + i*5)
            scenes = detector_i.detect_scenes(test_video_path)
            
            video_name = f"{Path(test_video_path).stem}_run_{i}"
            scene_paths = chunker.chunk_video(test_video_path, scenes, video_name)
            
            results.append({
                'run': i,
                'scenes': scenes,
                'paths': scene_paths
            })
        
        # Verify all runs produced results
        assert len(results) == 3
        for result in results:
            assert len(result['scenes']) > 0
            assert len(result['paths']) == len(result['scenes'])

