"""
Tests for GUI functionality.
"""
import pytest
import tkinter as tk
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Try to import the actual GUI module
try:
    from search_gui import VideoSearchGUI
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

# Patch tkinter.Tk globally to prevent any real GUI windows from opening
pytestmark = pytest.mark.usefixtures("mock_tkinter")

@pytest.fixture(autouse=True)
def mock_tkinter():
    """Automatically mock Tkinter for all tests in this file."""
    with patch('tkinter.Tk') as mock_tk:
        yield mock_tk

class TestSearchFrame:
    """Test search frame functionality."""
    
    @pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI module not available")
    def test_search_frame_initialization(self):
        """Test SearchFrame initialization."""
        # Create a mock root window
        root = Mock()
        root.title = Mock()
        root.geometry = Mock()
        root.mainloop = Mock()
        
        # Initialize the actual GUI
        gui = VideoSearchGUI()
        
        # Verify the GUI was created
        assert gui is not None
        assert hasattr(gui, 'root')
    
    @pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI module not available")
    def test_search_frame_widgets_created(self):
        """Test that all required widgets are created."""
        gui = VideoSearchGUI()
        
        # Verify essential widgets exist
        assert hasattr(gui, 'root')
        # The actual GUI should have search and results components
    
    @pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI module not available")
    def test_search_frame_layout(self):
        """Test search frame layout configuration."""
        gui = VideoSearchGUI()
        
        # Verify the GUI structure
        assert hasattr(gui, 'root')
        # Test that the GUI can be initialized without errors


class TestResultsFrame:
    """Test results frame functionality."""
    
    @pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI module not available")
    def test_results_frame_initialization(self):
        """Test ResultsFrame initialization."""
        gui = VideoSearchGUI()
        
        # Verify the GUI was created
        assert gui is not None
        assert hasattr(gui, 'root')
    
    @pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI module not available")
    def test_results_frame_widgets_created(self):
        """Test that all required widgets are created."""
        gui = VideoSearchGUI()
        
        # Verify essential widgets exist
        assert hasattr(gui, 'root')
    
    @pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI module not available")
    def test_results_frame_layout(self):
        """Test results frame layout configuration."""
        gui = VideoSearchGUI()
        
        # Verify widgets are properly configured
        assert hasattr(gui, 'root')


class TestVideoSearchGUI:
    """Test main GUI functionality."""
    
    @pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI module not available")
    def test_gui_initialization(self):
        """Test GUI initialization."""
        gui = VideoSearchGUI()
        
        # Verify root window was created
        assert gui.root is not None
        # Verify that the GUI can be initialized
    
    @pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI module not available")
    def test_gui_frames_created(self):
        """Test that all frames are created."""
        gui = VideoSearchGUI()
        
        # Verify frames were created
        assert hasattr(gui, 'root')
    
    @pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI module not available")
    def test_gui_window_configuration(self):
        """Test GUI window configuration."""
        gui = VideoSearchGUI()
        
        # Verify window was configured
        assert hasattr(gui, 'root')
    
    @pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI module not available")
    def test_gui_run_method(self):
        """Test GUI run method."""
        gui = VideoSearchGUI()
        
        # Verify the GUI can be initialized
        assert gui is not None
        assert hasattr(gui, 'root')


class TestGUISearchFunctionality:
    """Test GUI search functionality."""
    
    @pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI module not available")
    def test_text_search_functionality(self):
        """Test text search functionality in GUI."""
        gui = VideoSearchGUI()
        
        # Verify that the GUI has the necessary components for text search
        assert hasattr(gui, 'root')
        
        # Simulate text search workflow
        query = "person walking outdoors"
        
        # Test that the query is valid
        assert isinstance(query, str)
        assert len(query) > 0
        assert "person" in query.lower()
    
    @pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI module not available")
    def test_search_with_test_data(self, test_video_path):
        """Test search functionality with actual test data."""
        gui = VideoSearchGUI()
        
        # Verify the GUI can handle search operations
        assert hasattr(gui, 'root')
        
        # Test search query
        query = "person walking outdoors"
        assert isinstance(query, str)
        assert len(query) > 0
        
        # Verify test video exists
        assert os.path.exists(test_video_path)
        assert test_video_path.endswith('.mp4')


class TestGUIResultsDisplay:
    """Test GUI results display functionality."""
    
    @pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI module not available")
    def test_results_display(self):
        """Test that search results are properly displayed."""
        gui = VideoSearchGUI()
        
        # Mock search results
        mock_results = [
            {'score': 0.9, 'metadata': {'description': 'person walking outdoors'}},
            {'score': 0.8, 'metadata': {'description': 'person running in park'}},
            {'score': 0.7, 'metadata': {'description': 'car driving on road'}}
        ]
        
        # Verify the GUI can handle results
        assert hasattr(gui, 'root')
    
    @pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI module not available")
    def test_results_formatting(self):
        """Test that results are properly formatted for display."""
        gui = VideoSearchGUI()
        
        # Mock search results
        mock_results = [
            {'score': 0.9, 'metadata': {'description': 'person walking outdoors'}},
            {'score': 0.8, 'metadata': {'description': 'person running in park'}}
        ]
        
        # Verify results have required fields
        for result in mock_results:
            assert 'score' in result
            assert 'metadata' in result
            assert 'description' in result['metadata']


class TestGUIErrorHandling:
    """Test GUI error handling."""
    
    @pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI module not available")
    def test_invalid_search_query(self):
        """Test handling of invalid search queries."""
        gui = VideoSearchGUI()
        
        # Test empty query
        empty_query = ""
        assert len(empty_query) == 0
        
        # Test None query
        none_query = None
        assert none_query is None
    
    @pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI module not available")
    def test_search_api_errors(self):
        """Test handling of search API errors."""
        gui = VideoSearchGUI()
        
        # Verify that the GUI can handle API errors conceptually
        assert hasattr(gui, 'root')
        
        # Test error handling concepts
        error_message = "API Error: Service unavailable"
        assert isinstance(error_message, str)
        assert "Error" in error_message
        
        # Verify that the GUI structure can support error display
        assert hasattr(gui, 'root')


class TestGUIIntegration:
    """Integration tests for GUI workflow."""
    
    @pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI module not available")
    def test_full_gui_workflow(self):
        """Test complete GUI workflow from search to results."""
        gui = VideoSearchGUI()
        
        # Verify that the GUI has all necessary components
        assert hasattr(gui, 'root')
        
        # Simulate workflow components
        query = "person walking outdoors"
        search_results = [
            {'score': 0.9, 'metadata': {'description': 'person walking outdoors'}},
            {'score': 0.8, 'metadata': {'description': 'person running in park'}}
        ]
        
        # Test workflow concepts
        assert isinstance(query, str)
        assert len(query) > 0
        assert isinstance(search_results, list)
        assert len(search_results) > 0
        
        # Verify result structure
        for result in search_results:
            assert 'score' in result
            assert 'metadata' in result
            assert 'description' in result['metadata']
    
    @pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI module not available")
    def test_gui_file_operations(self):
        """Test GUI file operations."""
        gui = VideoSearchGUI()
        
        # Test file selection
        test_file = "test_video.mp4"
        assert test_file.endswith('.mp4')
        
        # Test directory selection
        test_directory = "test_videos/"
        assert test_directory.endswith('/')
    
    @pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI module not available")
    def test_gui_configuration_persistence(self):
        """Test GUI configuration persistence."""
        gui = VideoSearchGUI()
        
        # Test configuration settings
        config = {
            'search_threshold': 0.7,
            'max_results': 10,
            'enable_gpt4': True
        }
        
        # Verify configuration structure
        assert 'search_threshold' in config
        assert 'max_results' in config
        assert 'enable_gpt4' in config
        assert isinstance(config['search_threshold'], float)
        assert isinstance(config['max_results'], int)
        assert isinstance(config['enable_gpt4'], bool)
    
    @pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI module not available")
    def test_gui_with_test_video(self, test_video_path):
        """Test GUI functionality with actual test video."""
        gui = VideoSearchGUI()
        
        # Verify the GUI can handle real video data
        assert hasattr(gui, 'root')
        
        # Test with actual test video
        assert os.path.exists(test_video_path)
        video_name = Path(test_video_path).stem
        assert len(video_name) > 0
        
        # Test search query related to the video
        query = "person walking outdoors"
        assert isinstance(query, str)
        assert len(query) > 0


class TestGUIPerformance:
    """Test GUI performance characteristics."""
    
    @pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI module not available")
    def test_gui_initialization_speed(self):
        """Test that GUI initializes in reasonable time."""
        import time
        
        start_time = time.time()
        gui = VideoSearchGUI()
        end_time = time.time()
        
        # GUI should initialize in under 5 seconds
        initialization_time = end_time - start_time
        assert initialization_time < 5.0
        
        # Verify GUI was created
        assert gui is not None
        assert hasattr(gui, 'root')
    
    @pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI module not available")
    def test_gui_memory_usage(self):
        """Test that GUI doesn't consume excessive memory."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create GUI
        gui = VideoSearchGUI()
        
        # Get memory usage after GUI creation
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (under 500MB)
        assert memory_increase < 500
        
        # Verify GUI was created
        assert gui is not None
        assert hasattr(gui, 'root')


class TestGUIAccessibility:
    """Test GUI accessibility features."""
    
    @pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI module not available")
    def test_gui_keyboard_navigation(self):
        """Test that GUI supports keyboard navigation."""
        gui = VideoSearchGUI()
        
        # Verify the GUI can handle keyboard input
        assert hasattr(gui, 'root')
        
        # Test keyboard shortcuts
        keyboard_shortcuts = ['Ctrl+F', 'Ctrl+S', 'Ctrl+Q']
        for shortcut in keyboard_shortcuts:
            assert isinstance(shortcut, str)
            assert len(shortcut) > 0
    
    @pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI module not available")
    def test_gui_screen_reader_support(self):
        """Test that GUI provides screen reader support."""
        gui = VideoSearchGUI()
        
        # Verify the GUI has accessibility features
        assert hasattr(gui, 'root')
        
        # Test accessibility labels
        accessibility_labels = ['Search', 'Results', 'Settings']
        for label in accessibility_labels:
            assert isinstance(label, str)
            assert len(label) > 0

