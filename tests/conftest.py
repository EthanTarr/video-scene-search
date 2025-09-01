"""
Pytest configuration and fixtures for video-scene-search tests.
"""
import pytest
import tempfile
import os
import shutil
from pathlib import Path
import numpy as np
from unittest.mock import Mock, patch

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(str(temp_dir))

@pytest.fixture
def test_data_dir():
    """Get the path to the test data directory."""
    return Path(__file__).parent / "data"

@pytest.fixture
def test_video_path(test_data_dir):
    """Get the path to the test video file."""
    video_path = test_data_dir / "raw_videos" / "ForBiggerBlazes.mp4"
    if not video_path.exists():
        pytest.skip(f"Test video not found at {video_path}")
    return str(video_path)

@pytest.fixture
def test_scenes_dir(test_data_dir):
    """Get the path to the test scenes directory."""
    return test_data_dir / "scenes"

@pytest.fixture
def test_embeddings_dir(test_data_dir):
    """Get the path to the test embeddings directory."""
    return test_data_dir / "embeddings"

@pytest.fixture
def test_metadata_dir(test_data_dir):
    """Get the path to the test metadata directory."""
    return test_data_dir / "metadata"

@pytest.fixture
def sample_video_path(temp_dir):
    """Create a mock video file path for testing."""
    video_path = os.path.join(temp_dir, "sample_video.mp4")
    # Create an empty file to simulate video
    with open(video_path, 'w') as f:
        f.write("mock video content")
    return video_path

@pytest.fixture
def sample_scene_path(temp_dir):
    """Create a mock scene file path for testing."""
    scene_path = os.path.join(temp_dir, "sample_scene.mp4")
    with open(scene_path, 'w') as f:
        f.write("mock scene content")
    return scene_path

@pytest.fixture
def mock_embeddings():
    """Create mock embeddings for testing."""
    return np.random.rand(10, 512).astype(np.float32)

@pytest.fixture
def mock_metadata():
    """Create mock metadata for testing."""
    return {
        'video_name': 'test_video.mp4',
        'scene_number': 0,
        'start_time': 0.0,
        'end_time': 5.0,
        'duration': 5.0,
        'file_path': '/path/to/scene.mp4'
    }

@pytest.fixture
def mock_openai_response():
    """Create mock OpenAI API response for testing."""
    return {
        'choices': [{
            'message': {
                'content': 'Enhanced search query with visual details'
            }
        }],
        'usage': {
            'total_tokens': 100,
            'prompt_tokens': 50,
            'completion_tokens': 50
        }
    }

@pytest.fixture
def mock_clip_model():
    """Create mock CLIP model for testing."""
    mock_model = Mock()
    mock_model.encode_image.return_value = np.random.rand(1, 512).astype(np.float32)
    mock_model.encode_text.return_value = np.random.rand(1, 512).astype(np.float32)
    return mock_model

@pytest.fixture
def mock_faiss_index():
    """Create mock FAISS index for testing."""
    mock_index = Mock()
    mock_index.search.return_value = (
        np.array([[0.8, 0.7, 0.6]]),  # Distances
        np.array([[0, 1, 2]])          # Indices
    )
    return mock_index

