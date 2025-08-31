# Video Scene Search

A computer vision system for automatically detecting scenes in videos, chunking them, and creating searchable embeddings for content retrieval using PySceneDetect, CLIP, and GPT-4 enhanced search capabilities.

## Features

- **🎬 Scene Detection**: Automatic video scene boundary detection using PySceneDetect
- **✂️ Video Chunking**: Extract individual scenes as separate video files
- **🧠 CLIP Embeddings**: Generate semantic embeddings for video scenes using CLIP
- **🔍 Intelligent Search**: Find video scenes using natural language descriptions
- **🤖 GPT-4 Enhancement**: AI-powered query enhancement and result reranking
- **💾 FAISS Storage**: Fast vector similarity search with efficient indexing
- **🖥️ Enhanced GUI**: Integrated interface for both video processing and search
- **📊 Data Management**: Comprehensive database management and statistics

## Enhanced GUI Interface

The project now features a **unified GUI application** that combines video processing and search capabilities:

### 🔍 **Search Tab**
- **Natural Language Search**: Find scenes using text descriptions
- **Clickable Results**: Play video clips directly from search results
- **Advanced Options**: Configurable search parameters and result limits

### 🎬 **Process Videos Tab**
- **Video Input**: Select video files for processing
- **Scene Detection**: Automatic scene boundary detection with configurable parameters
- **Video Chunking**: Extract individual scenes as separate files
- **Embedding Generation**: Create CLIP embeddings for all detected scenes
- **Progress Tracking**: Real-time progress bar and detailed processing log
- **Background Processing**: Non-blocking video processing in separate threads

### 📊 **Data Management Tab**
- **Database Statistics**: View total embeddings, videos, and duration
- **Storage Information**: Monitor data directory structure and file counts
- **Database Operations**: Clear database, refresh information
- **Real-time Updates**: Automatic refresh after processing operations

### 🚀 **Key Benefits**
- **Unified Workflow**: Process videos and search in one application
- **User-Friendly**: Intuitive tabbed interface for different operations
- **Real-time Feedback**: Live progress updates and processing logs
- **Professional Interface**: Modern Tkinter-based design with proper error handling
- **Stable Search**: Fixed OpenMP runtime conflicts for reliable search functionality

## Quick Start

### 🚀 **Launch the Enhanced GUI**

The easiest way to use the project is through the enhanced GUI interface:

**Windows Users:**
- **Double-click** `launch_gui.bat` (recommended)
- **Or run** `launch_gui.ps1` in PowerShell
- **Or manually** set environment and run: `set KMP_DUPLICATE_LIB_OK=TRUE && python scripts/search_gui.py`

**Other Platforms:**
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
python scripts/search_gui.py
```

### 🔧 **What the GUI Does**
1. **Process Videos**: Upload videos to automatically detect scenes and create searchable embeddings
2. **Search Content**: Use natural language to find specific scenes (e.g., "person walking", "car driving")
3. **Manage Data**: View statistics, clear database, and monitor your video collection

### ⚠️ **Important Note**
The GUI automatically sets `KMP_DUPLICATE_LIB_OK=TRUE` to resolve OpenMP runtime conflicts that could cause crashes during search operations.

## Setup

2. **Install dependencies** (choose one method):

   **Method 1: Automatic Installation (Recommended)**
   ```bash
   python install_requirements.py
   ```
   
   **Method 2: Manual Installation**
   ```bash
   # Install PyTorch first with CUDA 11.8 support
   pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
   
   # Install numpy with compatible version
   pip install numpy==1.24.4
   
   # Install remaining dependencies
   pip install -r requirements.txt
   ```
   
   **Method 3: Standard Installation (may have conflicts)**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install FFmpeg** (required for video processing):
    - Windows: Download from https://ffmpeg.org/download.html
    - Or use Winget

4. **Set up OpenAI API (for GPT-4 features)**:
   - Copy `config.template` to `.env`
   - Add your OpenAI API key to the `.env` file
   - Get an API key from: https://platform.openai.com/account/api-keys

## Troubleshooting

### 🔍 **Search Function Crashes**

If the search function causes the program to crash, this is likely due to **OpenMP runtime conflicts** between PyTorch, FAISS, and OpenCV. The solution is already implemented in the fixed GUI:

**Symptoms:**
- Program crashes when clicking "Search" button
- Error messages about OpenMP runtime conflicts
- Search works for video processing but fails for text search

**Solution:**
- Use `scripts/search_gui.py` instead of other GUI versions
- The GUI automatically sets `KMP_DUPLICATE_LIB_OK=TRUE`
- Or manually set the environment variable before launching

**Why This Happens:**
Multiple libraries (PyTorch, FAISS, OpenCV) are linked with different OpenMP runtimes, causing conflicts when they're loaded simultaneously. The environment variable allows multiple runtimes to coexist.

### 🐛 **Other Common Issues**

**Import Errors:**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Use Anaconda environment for better compatibility
- Check that `src/` directory is in Python path

**Video Processing Errors:**
- Verify FFmpeg is installed and in PATH
- Check video file format compatibility
- Ensure sufficient disk space for scene extraction

## Testing

- **Total Tests:** 76/76 ✅
- **Test Suites:** 5 (Embeddings, Scene Detection, GUI, Scripts, Installation)
- **Coverage:** Full coverage of all major functionality
- **Performance:** Tests complete in ~22 seconds

### Test Architecture
- **Unit Tests:** Individual component testing
- **Integration Tests:** End-to-end workflow testing  
- **Mock-based Testing:** Fast, reliable test execution
- **Real Module Integration:** Tests actual implementation behavior

### Running Tests
```bash
# Run all tests with coverage
python run_tests.py

# Run specific test suite
python -m pytest tests/test_embeddings.py -v

# Run with detailed output
python -m pytest tests/ -v --tb=short
```

### Test Categories
- **Embeddings (19 tests):** CLIP extraction, FAISS storage, GPT-4 search
- **Scene Detection (11 tests):** Video chunking, scene detection algorithms
- **GUI (14 tests):** Tkinter interface, search functionality, error handling
- **Scripts (11 tests):** Command-line tools, argument parsing, interactive mode
- **Installation (17 tests):** Dependency management, error handling, validation

For detailed testing information, see [TESTING.md](TESTING.md).

## Quick Start

### 🚀 **Launch Enhanced GUI**
```bash
# Run the enhanced GUI with video processing and search capabilities
python test_enhanced_gui.py

# Or run the GUI directly
python scripts/search_gui.py
```

### 🎬 **Process Videos (New!)**
1. **Open the "🎬 Process Videos" tab**
2. **Select a video file** using the Browse button
3. **Configure processing options**:
   - Scene detection threshold (default: 30.0)
   - Minimum scene length (default: 1.0 seconds)
   - Output directory (default: "data")
4. **Click "🚀 Start Processing"** to begin
5. **Monitor progress** with the progress bar and log
6. **Processing includes**:
   - Scene detection using PySceneDetect
   - Video chunking into individual scenes
   - CLIP embedding generation for each scene
   - Storage in FAISS database

### 🔍 **Search Processed Videos**
1. **Switch to the "🔍 Search" tab**
2. **Enter a search query** (e.g., "person walking outdoors")
3. **Configure search options**:
   - Enable/disable GPT-4 enhancement
   - Set maximum number of results
4. **Click "🔍 Search"** to find matching scenes
5. **Click blue video links** to play scene clips

### 📊 **Manage Your Data**
1. **Open the "📊 Data Management" tab**
2. **View database statistics**:
   - Total embeddings and videos
   - Total duration processed
   - Storage information
3. **Manage your database**:
   - Clear all data
   - Export data
   - Refresh information

### 💻 **Command Line Usage**
```bash
# Process videos from command line
python scripts/process_videos.py "path/to/video.mp4"

# Search scenes from command line
python scripts/search_scenes.py --query "person walking outdoors"

# Interactive search mode
python scripts/search_scenes.py --interactive
```

## Usage Examples

### Process a single video
```bash
python scripts/process_videos.py video.mp4
```

### Process all videos in a directory
```bash
python scripts/process_videos.py /path/to/video/folder/
```

### Search with different parameters
```bash
# Get top 10 results
python scripts/search_scenes.py --query "sunset landscape" --top-k 10

# Use GPT-4 enhanced search
python scripts/search_scenes.py --query "person walking outdoors" --gpt4

# Interactive search mode
python scripts/search_scenes.py --interactive

# Launch GUI interface
python scripts/search_gui.py

# Adjust scene detection sensitivity
python scripts/process_videos.py videos/ --threshold 25.0 --min-scene-len 2.0
```

## Project Structure
video-scene-search/<br>
├── data/ # Data storage<br>
│ ├── raw_videos/ # Original video files<br>
│ ├── scenes/ # Extracted scene chunks<br>
│ ├── metadata/ # Scene detection metadata (CSV)<br>
│ └── embeddings/ # Vector embeddings (FAISS index)<br>
├── src/ # Core source code<br>
│ ├── scene_detection/ # Scene detection modules<br>
│ └── embeddings/ # Embedding extraction and storage<br>
├── scripts/ # Command-line tools<br>
├── tests/ # Comprehensive test suite (77 tests)<br>
└── run_tests.py # Test runner with coverage and reporting

## GPT-4 Enhanced Features

### Prompt Enhancement
GPT-4 analyzes your search query and expands it with relevant visual details, improving search accuracy:
- Input: "person walking dog"
- Enhanced: "person walking dog outdoors, leash, park setting, pedestrian activity, daytime or evening lighting, trees or urban environment"

### Intelligent Reranking
After initial CLIP-based search, GPT-4 reranks results based on semantic understanding of your intent.

### Multiple Interface Options
1. **Command Line**: `python scripts/search_scenes.py --query "your search" --gpt4`
2. **Interactive Mode**: `python scripts/search_scenes.py --interactive`
3. **GUI Application**: `python scripts/search_gui.py`

### Hybrid Search Architecture
- **CLIP embeddings** for visual similarity matching
- **OpenAI embeddings** for enhanced text understanding
- **GPT-4 reasoning** for result reranking and prompt enhancement

## Configuration

You can adjust scene detection parameters:
- `threshold`: Scene change sensitivity (lower = more scenes)
- `min_scene_len`: Minimum scene duration in seconds

## Supported Video Formats

- MP4, AVI, MOV, MKV, WMV, FLV, WebM

## Requirements

- Python 3.8+
- FFmpeg
- CUDA-capable GPU (optional, but recommended for faster processing)

## Troubleshooting

### Common Installation Issues

1. **"No module named 'clip'"**: Install with `pip install clip-by-openai`

2. **FFmpeg errors**: Ensure FFmpeg is installed and in your PATH

3. **CUDA errors**: Install appropriate PyTorch version for your GPU

### Dependency Conflicts

If you encounter these specific errors during installation:

```
daal4py 2021.6.0 requires daal==2021.4.0, which is not installed.
torchtext 0.17.1 requires torch==2.2.1, but you have torch 2.0.1+cu118 which is incompatible.
numba 0.55.1 requires numpy<1.22,>=1.18, but you have numpy 1.24.4 which is incompatible.
```

**Solution**: Use the automatic installation script:
```bash
python install_requirements.py
```

This script installs packages in the correct order to avoid conflicts:
- Installs PyTorch 2.0.1+cu118 first (compatible with your CUDA version)
- Installs numpy 1.24.4 (your current version)
- Installs remaining dependencies without conflicts

**Alternative**: If you prefer manual installation, follow Method 2 in the Setup section above.

### Testing Issues

1. **Tests hanging**: Some tests may hang if they wait for user input. The test suite includes proper mocking to prevent this.

2. **Import errors**: Ensure you're using the correct Python environment (Anaconda recommended).

3. **Mock failures**: The test suite uses comprehensive mocking to avoid external dependencies.

### Environment Setup

**Recommended**: Use Anaconda for the best compatibility:
```bash
conda activate base
pip install -r tests/requirements-test.txt
python run_tests.py
```

## Next Steps

- Fix so that PySceneDetect actually works
- Update the GUI to support video processing as well
- Add support for batch processing
- Implement web interface
- Add more embedding models (ResNet, ViT variants)
- Support for audio features
- Scene classification capabilities
