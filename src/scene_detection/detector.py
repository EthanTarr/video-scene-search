import cv2
import os
import numpy as np
from typing import List, Tuple, Optional
import pandas as pd

class SceneDetector:
    def __init__(self, threshold: float = 30.0, min_scene_len: float = 1.0):
        self.threshold = threshold
        self.min_scene_len = min_scene_len
        self.method = "auto"  # Will try PySceneDetect first, then fallback

    def detect_scenes(self, video_path: str) -> List[Tuple[float, float]]:
        """Detect scenes in video and return list of (start_time, end_time) tuples."""
        print(f"Detecting scenes in: {os.path.basename(video_path)}")

        # Try different methods in order of preference
        methods = [
            ("pyscenedetect_simple", self._detect_with_pyscenedetect_simple),
            ("frame_difference", self._detect_with_frame_difference),
            ("fixed_segments", self._detect_with_fixed_segments)
        ]

        for method_name, method_func in methods:
            try:
                print(f"Trying {method_name} method...")
                scenes = method_func(video_path)
                if scenes and len(scenes) > 0:
                    print(f"✅ Success with {method_name}: {len(scenes)} scenes detected")
                    return scenes
            except Exception as e:
                print(f"❌ {method_name} failed: {e}")
                continue

        # Ultimate fallback
        print("⚠️  All methods failed, using single scene fallback")
        return self._get_single_scene_fallback(video_path)

    def _detect_with_pyscenedetect_simple(self, video_path: str) -> List[Tuple[float, float]]:
        """Try the simplest PySceneDetect approach."""
        try:
            import scenedetect
            from scenedetect import detect, ContentDetector

            # Use minimal parameters to avoid compatibility issues
            scene_list = detect(video_path, ContentDetector())

            if not scene_list:
                raise ValueError("No scenes detected")

            scenes = []
            for scene in scene_list:
                start_time = scene[0].get_seconds()
                end_time = scene[1].get_seconds()
                duration = end_time - start_time

                if duration >= self.min_scene_len:
                    scenes.append((start_time, end_time))

            return scenes if scenes else [(0.0, self._get_video_duration(video_path))]

        except Exception as e:
            raise Exception(f"PySceneDetect simple method failed: {e}")

    def _detect_with_frame_difference(self, video_path: str) -> List[Tuple[float, float]]:
        """Detect scenes using frame difference analysis."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or frame_count <= 0:
            cap.release()
            raise ValueError("Invalid video properties")

        # Sample frames for analysis (every 30th frame to speed up)
        sample_rate = max(1, int(fps // 2))  # Sample every 0.5 seconds
        prev_frame = None
        scene_changes = [0]  # Start with frame 0

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                # Convert to grayscale and resize for faster processing
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (320, 240))

                if prev_frame is not None:
                    # Calculate frame difference
                    diff = cv2.absdiff(prev_frame, gray)
                    mean_diff = np.mean(diff)

                    # If difference is above threshold, it's a scene change
                    if mean_diff > self.threshold:
                        scene_changes.append(frame_idx)

                prev_frame = gray

            frame_idx += 1

        cap.release()

        # Add final frame
        scene_changes.append(frame_count - 1)

        # Convert frame numbers to time tuples
        scenes = []
        for i in range(len(scene_changes) - 1):
            start_frame = scene_changes[i]
            end_frame = scene_changes[i + 1]

            start_time = start_frame / fps
            end_time = end_frame / fps

            if end_time - start_time >= self.min_scene_len:
                scenes.append((start_time, end_time))

        # Merge very short scenes with neighbors
        scenes = self._merge_short_scenes(scenes)

        return scenes if scenes else [(0.0, frame_count / fps)]

    def _detect_with_fixed_segments(self, video_path: str) -> List[Tuple[float, float]]:
        """Create fixed-length segments as scenes."""
        duration = self._get_video_duration(video_path)
        if duration <= 0:
            raise ValueError("Cannot determine video duration")

        # Create segments of 30 seconds each (or custom length)
        segment_length = max(30.0, self.min_scene_len * 2)
        scenes = []

        current_time = 0.0
        while current_time < duration:
            end_time = min(current_time + segment_length, duration)
            if end_time - current_time >= self.min_scene_len:
                scenes.append((current_time, end_time))
            current_time = end_time

        return scenes

    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.0

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0.0

        cap.release()
        return duration

    def _get_single_scene_fallback(self, video_path: str) -> List[Tuple[float, float]]:
        """Fallback: treat entire video as single scene."""
        duration = self._get_video_duration(video_path)
        if duration > 0:
            return [(0.0, duration)]
        else:
            return [(0.0, 60.0)]  # Default 60 seconds

    def _merge_short_scenes(self, scenes: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Merge scenes that are too short with neighboring scenes."""
        if not scenes:
            return scenes

        merged = []
        current_start, current_end = scenes[0]

        for i in range(1, len(scenes)):
            next_start, next_end = scenes[i]

            # If current scene is too short, extend it
            if current_end - current_start < self.min_scene_len:
                current_end = next_end  # Merge with next scene
            else:
                merged.append((current_start, current_end))
                current_start, current_end = next_start, next_end

        # Add the last scene
        merged.append((current_start, current_end))

        return merged

    def save_scene_metadata(self, video_path: str, scenes: List[Tuple[float, float]],
                            output_dir: str) -> str:
        """Save scene detection metadata to CSV."""
        os.makedirs(output_dir, exist_ok=True)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        metadata_path = os.path.join(output_dir, f"{video_name}_scenes.csv")

        df = pd.DataFrame(scenes, columns=['start_time', 'end_time'])
        df['duration'] = df['end_time'] - df['start_time']
        df['video_source'] = video_name
        df['scene_id'] = df.index

        df.to_csv(metadata_path, index=False)
        print(f"Scene metadata saved to: {metadata_path}")
        return metadata_path