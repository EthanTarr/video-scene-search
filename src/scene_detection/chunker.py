import os
import subprocess
import shutil
from typing import List, Tuple
from pathlib import Path
import cv2

class VideoChunker:
    def __init__(self, output_dir: str = "data/scenes"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ffmpeg_available = self._check_ffmpeg()

    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available."""
        try:
            # Try ffmpeg command
            subprocess.run(['ffmpeg', '-version'],
                           capture_output=True, check=True)
            print("✅ FFmpeg found")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                # Try ffmpeg-python
                import ffmpeg
                ffmpeg.probe('test')  # This will fail but imports work
                return True
            except:
                print("⚠️  FFmpeg not found. Will use OpenCV fallback.")
                return False

    def chunk_video(self, video_path: str, scenes: List[Tuple[float, float]],
                    video_name: str = None) -> List[str]:
        """Split video into scene chunks and return list of output paths."""
        if video_name is None:
            video_name = Path(video_path).stem

        output_paths = []

        if self.ffmpeg_available:
            output_paths = self._chunk_with_ffmpeg(video_path, scenes, video_name)
        else:
            output_paths = self._chunk_with_opencv(video_path, scenes, video_name)

        return output_paths

    def _chunk_with_ffmpeg(self, video_path: str, scenes: List[Tuple[float, float]],
                           video_name: str) -> List[str]:
        """Chunk video using FFmpeg."""
        import ffmpeg
        output_paths = []

        for i, (start_time, end_time) in enumerate(scenes):
            output_filename = f"{video_name}_scene_{i:03d}.mp4"
            output_path = self.output_dir / output_filename

            try:
                (
                    ffmpeg
                    .input(video_path, ss=start_time, t=end_time - start_time)
                    .output(str(output_path), vcodec='libx264', acodec='aac')
                    .overwrite_output()
                    .run(quiet=True)
                )
                output_paths.append(str(output_path))
                print(f"Created scene {i}: {output_filename} ({end_time - start_time:.1f}s)")
            except Exception as e:
                print(f"Error processing scene {i} with FFmpeg: {e}")
                # Try OpenCV fallback for this scene
                fallback_path = self._chunk_single_scene_opencv(
                    video_path, start_time, end_time, output_path
                )
                if fallback_path:
                    output_paths.append(fallback_path)

        return output_paths

    def _chunk_with_opencv(self, video_path: str, scenes: List[Tuple[float, float]],
                           video_name: str) -> List[str]:
        """Chunk video using OpenCV (slower but doesn't require FFmpeg)."""
        print("Using OpenCV for video chunking (this may be slower)...")
        output_paths = []

        for i, (start_time, end_time) in enumerate(scenes):
            output_filename = f"{video_name}_scene_{i:03d}.mp4"
            output_path = self.output_dir / output_filename

            scene_path = self._chunk_single_scene_opencv(
                video_path, start_time, end_time, output_path
            )
            if scene_path:
                output_paths.append(scene_path)
                print(f"Created scene {i}: {output_filename} ({end_time - start_time:.1f}s)")

        return output_paths

    def _chunk_single_scene_opencv(self, video_path: str, start_time: float,
                                   end_time: float, output_path: Path) -> str:
        """Extract a single scene using OpenCV."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Cannot open video: {video_path}")
                return None

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Calculate frame numbers
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)

            # Set up video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

            # Seek to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            frame_count = 0
            target_frames = end_frame - start_frame

            while frame_count < target_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                out.write(frame)
                frame_count += 1

            cap.release()
            out.release()

            return str(output_path)

        except Exception as e:
            print(f"Error creating scene with OpenCV: {e}")
            return None