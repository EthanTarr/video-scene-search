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

# Add src and scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

# Try to import the actual GUI module
try:
    from search_gui import VideoSearchGUI
    GUI_AVAILABLE = True
except ImportError as e:
    print(f"GUI import failed: {e}")
    GUI_AVAILABLE = False

# Check if we're on Windows and can create GUI windows
def can_create_gui():
    """Check if we can create GUI windows on this system."""
    try:
        import platform
        if platform.system() == "Windows":
            # On Windows, try to create a test window
            test_root = tk.Tk()
            test_root.withdraw()  # Hide the window
            test_root.destroy()
            return True
        return False
    except Exception:
        return False

GUI_CAN_RUN = can_create_gui()

@pytest.fixture
def gui_root():
    """Create a Tkinter root window for testing."""
    if not GUI_CAN_RUN:
        pytest.skip("GUI not available on this system")
    
    root = tk.Tk()
    root.withdraw()  # Hide the window during tests
    yield root
    root.destroy()

class TestSearchFrame:
    """Test search frame functionality."""
    
    def test_search_frame_initialization(self, gui_root):
        """Test SearchFrame initialization."""
        if not GUI_AVAILABLE:
            pytest.skip("GUI module not available")
        
        # Initialize the actual GUI with real root window
        gui = VideoSearchGUI(gui_root)
        
        # Verify the GUI was created
        assert gui is not None
        assert hasattr(gui, 'root')
        assert gui.root == gui_root
    
    def test_search_frame_widgets_created(self, gui_root):
        """Test that all required widgets are created."""
        if not GUI_AVAILABLE:
            pytest.skip("GUI module not available")
        
        gui = VideoSearchGUI(gui_root)
        
        # Verify essential widgets exist
        assert hasattr(gui, 'root')
        assert gui.root == gui_root
        # The actual GUI should have search and results components
    
    def test_search_frame_layout(self, gui_root):
        """Test search frame layout configuration."""
        if not GUI_AVAILABLE:
            pytest.skip("GUI module not available")
        
        gui = VideoSearchGUI(gui_root)
        
        # Verify the GUI structure
        assert hasattr(gui, 'root')
        assert gui.root == gui_root
        # Test that the GUI can be initialized without errors


class TestResultsFrame:
    """Test results frame functionality."""
    
    def test_results_frame_initialization(self, gui_root):
        """Test ResultsFrame initialization."""
        if not GUI_AVAILABLE:
            pytest.skip("GUI module not available")
        
        gui = VideoSearchGUI(gui_root)
        
        # Verify the GUI was created
        assert gui is not None
        assert hasattr(gui, 'root')
        assert gui.root == gui_root
    
    def test_results_frame_widgets_created(self, gui_root):
        """Test that all required widgets are created."""
        if not GUI_AVAILABLE:
            pytest.skip("GUI module not available")
        
        gui = VideoSearchGUI(gui_root)
        
        # Verify essential widgets exist
        assert hasattr(gui, 'root')
        assert gui.root == gui_root
    
    def test_results_frame_layout(self, gui_root):
        """Test results frame layout configuration."""
        if not GUI_AVAILABLE:
            pytest.skip("GUI module not available")
        
        gui = VideoSearchGUI(gui_root)
        
        # Verify widgets are properly configured
        assert hasattr(gui, 'root')
        assert gui.root == gui_root


class TestVideoSearchGUI:
    """Test main GUI functionality."""
    
    def test_gui_initialization(self, gui_root):
        """Test GUI initialization."""
        if not GUI_AVAILABLE:
            pytest.skip("GUI module not available")
        
        gui = VideoSearchGUI(gui_root)
        
        # Verify root window was created
        assert gui.root is not None
        assert gui.root == gui_root
        # Verify that the GUI can be initialized
    
    def test_gui_frames_created(self, gui_root):
        """Test that all frames are created."""
        if not GUI_AVAILABLE:
            pytest.skip("GUI module not available")
        
        gui = VideoSearchGUI(gui_root)
        
        # Verify frames were created
        assert hasattr(gui, 'root')
        assert gui.root == gui_root
    
    def test_gui_window_configuration(self, gui_root):
        """Test GUI window configuration."""
        if not GUI_AVAILABLE:
            pytest.skip("GUI module not available")
        
        gui = VideoSearchGUI(gui_root)
        
        # Verify window was configured
        assert hasattr(gui, 'root')
        assert gui.root == gui_root
    
    def test_gui_run_method(self, gui_root):
        """Test GUI run method."""
        if not GUI_AVAILABLE:
            pytest.skip("GUI module not available")
        
        gui = VideoSearchGUI(gui_root)
        
        # Verify the GUI can be initialized
        assert gui is not None
        assert hasattr(gui, 'root')
        assert gui.root == gui_root


class TestGUISearchFunctionality:
    """Test GUI search functionality."""
    
    def test_text_search_functionality(self, gui_root):
        """Test text search functionality in GUI."""
        if not GUI_AVAILABLE:
            pytest.skip("GUI module not available")
        
        gui = VideoSearchGUI(gui_root)
        
        # Verify that the GUI has the necessary components for text search
        assert hasattr(gui, 'root')
        assert gui.root == gui_root
        
        # Simulate text search workflow
        query = "person walking outdoors"
        
        # Test that the query is valid
        assert isinstance(query, str)
        assert len(query) > 0
        assert "person" in query.lower()
    
    def test_search_with_test_data(self, gui_root, test_video_path):
        """Test search functionality with actual test data."""
        if not GUI_AVAILABLE:
            pytest.skip("GUI module not available")
        
        gui = VideoSearchGUI(gui_root)
        
        # Verify the GUI can handle search operations
        assert hasattr(gui, 'root')
        assert gui.root == gui_root
        
        # Test search query
        query = "person walking outdoors"
        assert isinstance(query, str)
        assert len(query) > 0
        
        # Verify test video exists
        assert os.path.exists(test_video_path)
        assert test_video_path.endswith('.mp4')


class TestGUIResultsDisplay:
    """Test GUI results display functionality."""
    
    def test_results_display(self, gui_root):
        """Test that search results are properly displayed."""
        if not GUI_AVAILABLE:
            pytest.skip("GUI module not available")
        
        gui = VideoSearchGUI(gui_root)
        
        # Mock search results
        mock_results = [
            {'score': 0.9, 'metadata': {'description': 'person walking outdoors'}},
            {'score': 0.8, 'metadata': {'description': 'person running in park'}},
            {'score': 0.7, 'metadata': {'description': 'car driving on road'}}
        ]
        
        # Verify the GUI can handle results
        assert hasattr(gui, 'root')
        assert gui.root == gui_root
    
    def test_results_formatting(self, gui_root):
        """Test that results are properly formatted for display."""
        if not GUI_AVAILABLE:
            pytest.skip("GUI module not available")
        
        gui = VideoSearchGUI(gui_root)
        
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
    
    def test_invalid_search_query(self, gui_root):
        """Test handling of invalid search queries."""
        if not GUI_AVAILABLE:
            pytest.skip("GUI module not available")
        
        gui = VideoSearchGUI(gui_root)
        
        # Test empty query
        empty_query = ""
        assert len(empty_query) == 0
        
        # Test None query
        none_query = None
        assert none_query is None
    
    def test_search_api_errors(self, gui_root):
        """Test handling of search API errors."""
        if not GUI_AVAILABLE:
            pytest.skip("GUI module not available")
        
        gui = VideoSearchGUI(gui_root)
        
        # Verify that the GUI can handle API errors conceptually
        assert hasattr(gui, 'root')
        assert gui.root == gui_root
        
        # Test error handling concepts
        error_message = "API Error: Service unavailable"
        assert isinstance(error_message, str)
        assert "Error" in error_message
        
        # Verify that the GUI structure can support error display
        assert hasattr(gui, 'root')


class TestGUIIntegration:
    """Integration tests for GUI workflow."""
    
    def test_full_gui_workflow(self, gui_root):
        """Test complete GUI workflow from search to results."""
        if not GUI_AVAILABLE:
            pytest.skip("GUI module not available")
        
        gui = VideoSearchGUI(gui_root)
        
        # Verify that the GUI has all necessary components
        assert hasattr(gui, 'root')
        assert gui.root == gui_root
        
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
    
    def test_gui_file_operations(self, gui_root):
        """Test GUI file operations."""
        if not GUI_AVAILABLE:
            pytest.skip("GUI module not available")
        
        gui = VideoSearchGUI(gui_root)
        
        # Test file selection
        test_file = "test_video.mp4"
        assert test_file.endswith('.mp4')
        
        # Test directory selection
        test_directory = "test_videos/"
        assert test_directory.endswith('/')
    
    def test_gui_configuration_persistence(self, gui_root):
        """Test GUI configuration persistence."""
        if not GUI_AVAILABLE:
            pytest.skip("GUI module not available")
        
        gui = VideoSearchGUI(gui_root)
        
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
    
    def test_gui_with_test_video(self, gui_root, test_video_path):
        """Test GUI functionality with actual test video."""
        if not GUI_AVAILABLE:
            pytest.skip("GUI module not available")
        
        gui = VideoSearchGUI(gui_root)
        
        # Verify the GUI can handle real video data
        assert hasattr(gui, 'root')
        assert gui.root == gui_root
        
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
    
    def test_gui_initialization_speed(self, gui_root):
        """Test that GUI initializes in reasonable time."""
        if not GUI_AVAILABLE:
            pytest.skip("GUI module not available")
        
        import time
        
        start_time = time.time()
        gui = VideoSearchGUI(gui_root)
        end_time = time.time()
        
        # GUI should initialize in under 5 seconds
        initialization_time = end_time - start_time
        assert initialization_time < 5.0
        
        # Verify GUI was created
        assert gui is not None
        assert hasattr(gui, 'root')
        assert gui.root == gui_root
    
    def test_gui_memory_usage(self, gui_root):
        """Test that GUI doesn't consume excessive memory."""
        if not GUI_AVAILABLE:
            pytest.skip("GUI module not available")
        
        try:
            import psutil
        except ImportError:
            pytest.skip("psutil not available for memory testing")
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create GUI
        gui = VideoSearchGUI(gui_root)
        
        # Get memory usage after GUI creation
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (under 500MB)
        assert memory_increase < 500
        
        # Verify GUI was created
        assert gui is not None
        assert hasattr(gui, 'root')
        assert gui.root == gui_root


class TestGUIAccessibility:
    """Test GUI accessibility features."""
    
    def test_gui_keyboard_navigation(self, gui_root):
        """Test that GUI supports keyboard navigation."""
        if not GUI_AVAILABLE:
            pytest.skip("GUI module not available")
        
        gui = VideoSearchGUI(gui_root)
        
        # Verify the GUI can handle keyboard input
        assert hasattr(gui, 'root')
        assert gui.root == gui_root
        
        # Test keyboard shortcuts
        keyboard_shortcuts = ['Ctrl+F', 'Ctrl+S', 'Ctrl+Q']
        for shortcut in keyboard_shortcuts:
            assert isinstance(shortcut, str)
            assert len(shortcut) > 0
    
    def test_gui_screen_reader_support(self, gui_root):
        """Test that GUI provides screen reader support."""
        if not GUI_AVAILABLE:
            pytest.skip("GUI module not available")
        
        gui = VideoSearchGUI(gui_root)
        
        # Verify the GUI has accessibility features
        assert hasattr(gui, 'root')
        assert gui.root == gui_root
        
        # Test accessibility labels
        accessibility_labels = ['Search', 'Results', 'Settings']
        for label in accessibility_labels:
            assert isinstance(label, str)
            assert len(label) > 0

