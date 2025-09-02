#!/usr/bin/env python3
"""
Fixed Enhanced GUI interface with video processing and search.
This version handles OpenMP runtime conflicts and provides better error handling.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from pathlib import Path
import threading
import tempfile
from PIL import Image, ImageTk

# Fix OpenMP runtime conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add src to path FIRST, before any other imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent / "src"
sys.path.insert(0, str(src_dir))

# Import modules with detailed error handling
import numpy as np

try:
    import faiss
except ImportError as e:
    faiss = None

try:
    import torch
except ImportError as e:
    torch = None

try:
    import open_clip
except ImportError as e:
    open_clip = None

try:
    import cv2
except ImportError as e:
    cv2 = None

try:
    import pandas as pd
except ImportError as e:
    pd = None

# Now try to import our source modules
try:
    from embeddings.gpt4_search import GPT4VideoSearchEngine
except ImportError as e:
    GPT4VideoSearchEngine = None

try:
    from embeddings.extractor import SceneEmbeddingExtractor
except ImportError as e:
    SceneEmbeddingExtractor = None

try:
    from embeddings.storage import EmbeddingStorage
except ImportError as e:
    EmbeddingStorage = None

try:
    from scene_detection.detector import SceneDetector
except ImportError as e:
    SceneDetector = None

try:
    from scene_detection.chunker import VideoChunker
except ImportError as e:
    VideoChunker = None

# Check if we have the minimum required modules
required_modules = {
    'FAISS': faiss is not None,
    'PyTorch': torch is not None,
    'CLIP': open_clip is not None,
    'OpenCV': cv2 is not None,
    'Pandas': pd is not None,
    'SceneEmbeddingExtractor': SceneEmbeddingExtractor is not None,
    'EmbeddingStorage': EmbeddingStorage is not None,
    'SceneDetector': SceneDetector is not None,
    'VideoChunker': VideoChunker is not None,
}



# Check if we have the core functionality
core_modules_available = all([
    faiss is not None,
    torch is not None,
    open_clip is not None,
    cv2 is not None,
    pd is not None
])

# Check if we have our source modules
source_modules_available = all([
    SceneEmbeddingExtractor is not None,
    EmbeddingStorage is not None,
    SceneDetector is not None,
    VideoChunker is not None
])


class ClickableText(tk.Text):
    """Custom Text widget that properly handles clickable links."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(state='disabled')
        self.bind('<Button-1>', self._on_click)
        self.bind('<Motion>', self._on_motion)
        
    def _on_click(self, event):
        """Handle click events on text."""
        self.configure(state='normal')
        index = self.index(f"@{event.x},{event.y}")
        self.configure(state='disabled')
        
        for tag in self.tag_names(index):
            if tag.startswith('link_'):
                if hasattr(self, '_link_callbacks') and tag in self._link_callbacks:
                    self._link_callbacks[tag]()
                break
    
    def _on_motion(self, event):
        """Handle mouse motion for cursor changes."""
        self.configure(state='normal')
        index = self.index(f"@{event.x},{event.y}")
        self.configure(state='disabled')
        
        is_over_link = False
        for tag in self.tag_names(index):
            if tag.startswith('link_'):
                is_over_link = True
                break
        
        if is_over_link:
            self.configure(cursor='hand2')
        else:
            self.configure(cursor='')
    
    def insert_with_tags(self, text, tags=None):
        """Insert text with tags while maintaining disabled state."""
        self.configure(state='normal')
        if tags:
            self.insert(tk.END, text, tags)
        else:
            self.insert(tk.END, text)
        self.configure(state='disabled')
    
    def clear_text(self):
        """Clear all text content."""
        self.configure(state='normal')
        self.delete(1.0, tk.END)
        self.configure(state='disabled')
    
    def add_link(self, tag_name, text, callback):
        """Add a clickable link with proper callback handling."""
        if not hasattr(self, '_link_callbacks'):
            self._link_callbacks = {}
        
        self._link_callbacks[tag_name] = callback
        
        self.tag_config(tag_name,
                       foreground='blue',
                       underline=True,
                       font=('Arial', 10, 'underline bold'))


class ImageResultsFrame(tk.Frame):
    """Frame for displaying search results as clickable images."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.images = []
        self.image_labels = []
        self.current_row = 0
        self.current_col = 0
        self.max_cols = 3
        
    def clear(self):
        """Clear all images from the frame."""
        for label in self.image_labels:
            label.destroy()
        self.image_labels.clear()
        self.images.clear()
        self.current_row = 0
        self.current_col = 0
    
    def add_result(self, thumbnail_image, video_path, metadata, callback):
        """Add a search result with thumbnail image."""
        try:
            if thumbnail_image is None:
                # Create placeholder image if thumbnail extraction failed
                placeholder = Image.new('RGB', (200, 150), color='lightgray')
                # Add text to placeholder
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(placeholder)
                try:
                    font = ImageFont.truetype("arial.ttf", 16)
                except:
                    font = ImageFont.load_default()
                text = "No Preview"
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                x = (200 - text_width) // 2
                y = (150 - text_height) // 2
                draw.text((x, y), text, fill='black', font=font)
                thumbnail_image = placeholder
            
            # Convert PIL image to PhotoImage
            photo = ImageTk.PhotoImage(thumbnail_image)
            self.images.append(photo)  # Keep reference to prevent garbage collection
            
            # Create frame for this result
            result_frame = tk.Frame(self, relief='raised', bd=2, bg='white')
            result_frame.grid(row=self.current_row, column=self.current_col, padx=5, pady=5, sticky='nsew')
            
            # Create clickable image label
            image_label = tk.Label(result_frame, image=photo, cursor='hand2', bg='white')
            image_label.pack(pady=5)
            image_label.bind('<Button-1>', lambda e, path=video_path: callback(path))
            self.image_labels.append(image_label)
            
            # Add metadata text
            video_name = metadata.get('video_source', 'Unknown')
            if len(video_name) > 20:
                video_name = video_name[:17] + "..."
                
            info_text = f"{video_name}\n"
            info_text += f"Scene: {metadata.get('scene_id', 'N/A')}\n"
            info_text += f"Time: {metadata.get('start_time', 0):.1f}s - {metadata.get('end_time', 0):.1f}s\n"
            info_text += f"Similarity: {metadata.get('similarity', 0):.3f}"
            
            info_label = tk.Label(result_frame, text=info_text, font=('Arial', 8), 
                                 bg='white', justify='center', wraplength=180)
            info_label.pack(pady=2)
            
            # Update grid position
            self.current_col += 1
            if self.current_col >= self.max_cols:
                self.current_col = 0
                self.current_row += 1
            
            # Configure grid weights
            self.grid_columnconfigure(self.current_col, weight=1)
            self.grid_rowconfigure(self.current_row, weight=1)
            
        except Exception as e:
            print(f"Error adding result: {e}")
            # Create a simple text-based result as fallback
            result_frame = tk.Frame(self, relief='raised', bd=2, bg='lightgray')
            result_frame.grid(row=self.current_row, column=self.current_col, padx=5, pady=5, sticky='nsew')
            
            error_label = tk.Label(result_frame, text=f"Error loading\n{metadata.get('video_source', 'Unknown')}", 
                                  font=('Arial', 10), bg='lightgray', justify='center')
            error_label.pack(pady=20)
            error_label.bind('<Button-1>', lambda e, path=video_path: callback(path))
            
            self.current_col += 1
            if self.current_col >= self.max_cols:
                self.current_col = 0
                self.current_row += 1


class VideoSearchGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Scene Search")
        self.root.geometry("1000x800")
        
        # Initialize components
        self.clip_extractor = None
        self.clip_storage = None
        self.data_dir = "data"
        self.current_results = []
        
        # Create GUI
        self.create_widgets()
        
        # Load embeddings on startup if possible
        if source_modules_available:
            self.load_embeddings()
        
        # Load initial stats and info
        self.refresh_stats()
        self.refresh_info()
    

    
    def create_widgets(self):
        """Create the GUI widgets with tabbed interface."""
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🎬 Video Scene Search", 
                               font=('Arial', 18, 'bold'))
        title_label.grid(row=0, column=0, pady=(0, 10))
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create tabs
        self.create_search_tab()
        self.create_processing_tab()
        self.create_management_tab()
    
    def create_search_tab(self):
        """Create the search functionality tab."""
        search_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(search_frame, text="🔍 Search")
        
        # Configure grid weights
        search_frame.columnconfigure(1, weight=1)
        search_frame.rowconfigure(4, weight=1)
        
        # Search query
        ttk.Label(search_frame, text="Search Query:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.query_var = tk.StringVar()
        query_entry = ttk.Entry(search_frame, textvariable=self.query_var, width=50, font=('Arial', 11))
        query_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=(5, 0))
        query_entry.bind('<Return>', lambda e: self.search())
        
        # Search options
        options_frame = ttk.Frame(search_frame)
        options_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(options_frame, text="Max Results:").grid(row=0, column=0, padx=(0, 5))
        self.max_results_var = tk.StringVar(value="10")
        results_spinbox = ttk.Spinbox(options_frame, from_=1, to=50, width=5,
                                     textvariable=self.max_results_var)
        results_spinbox.grid(row=0, column=1)
        
        # Buttons
        button_frame = ttk.Frame(search_frame)
        button_frame.grid(row=2, column=0, columnspan=3, pady=10)
        
        search_btn = ttk.Button(button_frame, text="🔍 Search", command=self.search)
        search_btn.grid(row=0, column=0, padx=5)
        
        clear_btn = ttk.Button(button_frame, text="Clear", command=self.clear_results)
        clear_btn.grid(row=0, column=1, padx=5)
        
        stats_btn = ttk.Button(button_frame, text="Show Stats", command=self.show_stats)
        stats_btn.grid(row=0, column=2, padx=5)
        
        # Results area
        results_label = ttk.Label(search_frame, text="Search Results:", font=('Arial', 12, 'bold'))
        results_label.grid(row=3, column=0, sticky=(tk.W, tk.N), pady=(10, 5))
        
        # Create frame for image results only
        results_frame = ttk.Frame(search_frame)
        results_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(25, 0))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Create scrollable frame for images
        self.results_canvas = tk.Canvas(results_frame, bg='white')
        self.results_scrollbar = ttk.Scrollbar(results_frame, orient='vertical', command=self.results_canvas.yview)
        self.scrollable_frame = ttk.Frame(self.results_canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))
        )
        
        self.results_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.results_canvas.configure(yscrollcommand=self.results_scrollbar.set)
        
        # Image results frame
        self.image_results = ImageResultsFrame(self.scrollable_frame)
        self.image_results.pack(fill='both', expand=True)
        
        self.results_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.results_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Add mousewheel scrolling
        def _on_mousewheel(event):
            self.results_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.results_canvas.bind_all("<MouseWheel>", _on_mousewheel)
    
    def create_processing_tab(self):
        """Create the video processing tab."""
        processing_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(processing_frame, text="🎬 Process Videos")
        
        # Configure grid weights
        processing_frame.columnconfigure(1, weight=1)
        processing_frame.rowconfigure(5, weight=1)
        
        # Video selection
        ttk.Label(processing_frame, text="Video File:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.video_path_var = tk.StringVar()
        video_entry = ttk.Entry(processing_frame, textvariable=self.video_path_var, width=60)
        video_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=(5, 0))
        
        browse_btn = ttk.Button(processing_frame, text="Browse", command=self.browse_video)
        browse_btn.grid(row=0, column=2, padx=(5, 0))
        
        # Processing options
        options_frame = ttk.LabelFrame(processing_frame, text="Processing Options", padding="5")
        options_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(options_frame, text="Scene Detection Threshold:").grid(row=0, column=0, sticky=tk.W)
        self.threshold_var = tk.StringVar(value="30.0")
        threshold_entry = ttk.Entry(options_frame, textvariable=self.threshold_var, width=10)
        threshold_entry.grid(row=0, column=1, padx=(5, 0))
        
        ttk.Label(options_frame, text="Min Scene Length (s):").grid(row=0, column=2, padx=(20, 0))
        self.min_scene_len_var = tk.StringVar(value="1.0")
        min_len_entry = ttk.Entry(options_frame, textvariable=self.min_scene_len_var, width=10)
        min_len_entry.grid(row=0, column=2, padx=(5, 0))
        
        # Output directory
        ttk.Label(processing_frame, text="Output Directory:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.output_dir_var = tk.StringVar(value="data")
        output_entry = ttk.Entry(processing_frame, textvariable=self.output_dir_var, width=60)
        output_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5, padx=(5, 0))
        
        output_browse_btn = ttk.Button(processing_frame, text="Browse", command=self.browse_output_dir)
        output_browse_btn.grid(row=2, column=2, padx=(5, 0))
        
        # Processing controls
        control_frame = ttk.Frame(processing_frame)
        control_frame.grid(row=3, column=0, columnspan=3, pady=10)
        
        self.process_btn = ttk.Button(control_frame, text="🚀 Start Processing", command=self.start_processing)
        self.process_btn.grid(row=0, column=0, padx=5)
        
        self.stop_btn = ttk.Button(control_frame, text="⏹️ Stop", command=self.stop_processing, state='disabled')
        self.stop_btn.grid(row=0, column=1, padx=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(processing_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # Status and log
        status_frame = ttk.LabelFrame(processing_frame, text="Processing Status", padding="5")
        status_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(1, weight=1)
        
        self.status_var = tk.StringVar(value="Ready to process videos")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, font=('Arial', 10, 'bold'))
        status_label.grid(row=0, column=0, sticky=tk.W)
        
        # Processing log
        self.log_text = scrolledtext.ScrolledText(status_frame, height=15, width=80, 
                                                 font=('Consolas', 9), bg='black', fg='white')
        self.log_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(5, 0))
    
    def create_management_tab(self):
        """Create the data management tab."""
        management_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(management_frame, text="📊 Data Management")
        
        # Configure grid weights
        management_frame.columnconfigure(1, weight=1)
        management_frame.rowconfigure(3, weight=1)
        
        # Statistics display
        stats_frame = ttk.LabelFrame(management_frame, text="Database Statistics", padding="5")
        stats_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.stats_text = scrolledtext.ScrolledText(stats_frame, height=8, width=80, 
                                                   font=('Consolas', 9), bg='white')
        self.stats_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Management buttons
        button_frame = ttk.Frame(management_frame)
        button_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        refresh_btn = ttk.Button(button_frame, text="🔄 Refresh Stats", command=self.refresh_stats)
        refresh_btn.grid(row=0, column=0, padx=5)
        
        clear_btn = ttk.Button(button_frame, text="🗑️ Clear Database", command=self.clear_database)
        clear_btn.grid(row=0, column=1, padx=5)
        
        # Data directory info
        info_frame = ttk.LabelFrame(management_frame, text="Data Directory Information", padding="5")
        info_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.info_text = scrolledtext.ScrolledText(info_frame, height=6, width=80, 
                                                  font=('Consolas', 9), bg='white')
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Refresh info
        refresh_info_btn = ttk.Button(info_frame, text="🔄 Refresh Info", command=self.refresh_info)
        refresh_info_btn.grid(row=1, column=0, pady=(5, 0))
    
    def browse_video(self):
        """Browse for video file."""
        path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All files", "*.*")]
        )
        if path:
            self.video_path_var.set(path)
    
    def browse_output_dir(self):
        """Browse for output directory."""
        path = filedialog.askdirectory(title="Select Output Directory")
        if path:
            self.output_dir_var.set(path)
    
    def start_processing(self):
        """Start video processing in a separate thread."""
        if not source_modules_available:
            messagebox.showerror("Error", "Required modules not available. Please check your installation.")
            return
            
        video_path = self.video_path_var.get().strip()
        if not video_path:
            messagebox.showerror("Error", "Please select a video file")
            return
        
        if not os.path.exists(video_path):
            messagebox.showerror("Error", "Selected path does not exist")
            return
        
        # Update UI state
        self.processing = True
        self.process_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.progress_var.set(0)
        self.status_var.set("Processing started...")
        
        # Clear log
        self.log_text.delete(1.0, tk.END)
        
        # Start processing in background thread
        self.processing_thread = threading.Thread(target=self.process_video_worker, args=(video_path,))
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def stop_processing(self):
        """Stop video processing."""
        self.processing = False
        self.status_var.set("Processing stopped by user")
        self.process_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.log_message("Processing stopped by user")
    
    def process_video_worker(self, video_path):
        """Worker thread for video processing."""
        try:
            self.log_message(f"Starting processing of: {video_path}")
            
            # Initialize components
            threshold = float(self.threshold_var.get())
            min_scene_len = float(self.min_scene_len_var.get())
            output_base = self.output_dir_var.get()
            
            self.log_message(f"Scene detection threshold: {threshold}")
            self.log_message(f"Minimum scene length: {min_scene_len} seconds")
            self.log_message(f"Output directory: {output_base}")
            
            # Initialize processing components
            detector = SceneDetector(threshold=threshold, min_scene_len=min_scene_len)
            chunker = VideoChunker(output_dir=f"{output_base}/scenes")
            extractor = SceneEmbeddingExtractor()
            storage = EmbeddingStorage(storage_path=f"{output_base}/embeddings")
            
            # Load existing index if available
            try:
                storage.load()
                self.log_message("Loaded existing embeddings database")
            except Exception as e:
                self.log_message(f"Created new embeddings database: {e}")
            
            # 1. Detect scenes
            self.log_message("\n1. Detecting scenes...")
            self.update_status("Detecting scenes...")
            scenes = detector.detect_scenes(video_path)
            self.log_message(f"   Found {len(scenes)} scenes")
            
            if not scenes:
                self.log_message("   No scenes detected, skipping video")
                return
            
            # Update progress
            self.progress_var.set(20)
            
            # 2. Save scene metadata
            self.log_message("\n2. Saving scene metadata...")
            self.update_status("Saving scene metadata...")
            video_name = Path(video_path).stem
            metadata_path = detector.save_scene_metadata(
                video_path, scenes, f"{output_base}/metadata"
            )
            self.log_message(f"   Metadata saved to: {metadata_path}")
            
            # Update progress
            self.progress_var.set(40)
            
            # 3. Chunk video into scenes
            self.log_message("\n3. Chunking video into scenes...")
            self.update_status("Chunking video into scenes...")
            scene_paths = chunker.chunk_video(video_path, scenes, video_name)
            self.log_message(f"   Created {len(scene_paths)} scene files")
            
            # Update progress
            self.progress_var.set(60)
            
            # 4. Extract embeddings for each scene
            self.log_message("\n4. Extracting embeddings...")
            self.update_status("Extracting embeddings...")
            embeddings = []
            metadata = []
            
            for i, scene_path in enumerate(scene_paths):
                if not self.processing:  # Check if stopped
                    return
                
                if os.path.exists(scene_path):
                    try:
                        self.log_message(f"   Processing scene {i+1}/{len(scene_paths)}")
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
                        
                        # Update progress for embedding extraction
                        progress = 60 + (i + 1) / len(scene_paths) * 30
                        self.progress_var.set(progress)
                        
                    except Exception as e:
                        self.log_message(f"   Error processing scene {scene_path}: {e}")
            
            # 5. Store embeddings
            if embeddings and self.processing:
                self.log_message("\n5. Storing embeddings...")
                self.update_status("Storing embeddings...")
                
                # Debug information
                self.log_message(f"   Embeddings count: {len(embeddings)}")
                self.log_message(f"   First embedding shape: {embeddings[0].shape}")
                self.log_message(f"   First embedding dtype: {embeddings[0].dtype}")
                
                embeddings_array = np.vstack(embeddings)
                self.log_message(f"   Stacked array shape: {embeddings_array.shape}")
                self.log_message(f"   Stacked array dtype: {embeddings_array.dtype}")
                
                storage.add_embeddings(embeddings_array, metadata)
                storage.save()
                
                # Print statistics
                stats = storage.get_stats()
                self.log_message(f"   Total embeddings in database: {stats['total_embeddings']}")
                self.log_message(f"   Unique videos: {stats['unique_videos']}")
                self.log_message(f"   Total duration: {stats['total_duration']:.1f} seconds")
                
                # Update progress
                self.progress_var.set(100)
                self.update_status("Processing completed successfully!")
                self.log_message("\n✅ Processing completed successfully!")
                
                # Reload embeddings for search
                self.load_embeddings()
                
            else:
                self.log_message("\n❌ Processing stopped or no embeddings generated")
                self.update_status("Processing stopped or failed")
                
        except Exception as e:
            self.log_message(f"\n❌ Error during processing: {e}")
            self.update_status(f"Processing failed: {e}")
        finally:
            # Reset UI state
            self.processing = False
            self.root.after(0, self.reset_processing_ui)
    
    def reset_processing_ui(self):
        """Reset the processing UI state."""
        self.process_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
    
    def log_message(self, message):
        """Add message to processing log."""
        self.root.after(0, lambda: self.log_text.insert(tk.END, f"{message}\n"))
        self.root.after(0, lambda: self.log_text.see(tk.END))
    
    def update_status(self, status):
        """Update the status display."""
        self.root.after(0, lambda: self.status_var.set(status))
    
    def load_embeddings(self):
        """Load embeddings in a separate thread."""
        if not source_modules_available:
            self.status_var.set("Required modules not available")
            return
            
        def load_in_thread():
            self.status_var.set("Loading embeddings...")
            self.root.update()
            
            try:
                # Initialize standard CLIP components
                self.clip_extractor = SceneEmbeddingExtractor()
                self.clip_storage = EmbeddingStorage(storage_path=f"{self.data_dir}/embeddings")
                self.clip_storage.load()
                
                embedding_count = len(self.clip_storage.metadata)
                self.status_var.set(f"Ready - {embedding_count} embeddings loaded from {self.data_dir}")
                
                # Show initial stats
                self.show_initial_info()
                
            except Exception as e:
                self.status_var.set(f"Error loading embeddings: {str(e)}")
                messagebox.showerror("Loading Error", f"Failed to load embeddings: {str(e)}")
        
        # Run in thread to avoid blocking GUI
        thread = threading.Thread(target=load_in_thread)
        thread.daemon = True
        thread.start()
    
    def show_initial_info(self):
        """Show initial information about the database."""
        if not self.clip_storage or not self.clip_storage.metadata:
            # Show no data message in the image area
            self.image_results.clear()
            no_data_label = tk.Label(self.scrollable_frame, 
                                   text="No video embeddings found.\nPlease process videos first using the Process Videos tab.", 
                                   font=('Arial', 14), 
                                   bg='white', 
                                   fg='gray',
                                   justify='center')
            no_data_label.pack(pady=50)
            return
        
        stats = self.clip_storage.get_stats()
        
        # Show welcome message in the image area
        self.image_results.clear()
        welcome_label = tk.Label(self.scrollable_frame, 
                               text=f"🎬 Video Scene Search - Ready!\n\nDatabase Statistics:\n• Total embeddings: {stats['total_embeddings']}\n• Unique videos: {stats['unique_videos']}\n• Total duration: {stats['total_duration']:.1f} seconds\n\n✨ Enter a search query above and click 'Search' to find matching video scenes.\n📹 Examples: 'person walking', 'car driving', 'sunset', 'people talking'\n🖱️ Click on thumbnails to play video clips!", 
                               font=('Arial', 12), 
                               bg='white', 
                               fg='black',
                               justify='center')
        welcome_label.pack(pady=50)
    
    def search(self):
        """Perform search with improved error handling."""
        query = self.query_var.get().strip()
        if not query:
            messagebox.showwarning("Empty Query", "Please enter a search query.")
            return
        
        if not self.clip_storage or not self.clip_storage.metadata:
            messagebox.showerror("No Data", "No embeddings loaded. Please process videos first.")
            return
        
        # Run search in separate thread
        def search_in_thread():
            self.status_var.set(f"Searching for: {query}...")
            self.root.update()
            
            try:
                max_results = int(self.max_results_var.get())
                
                # Clear previous results
                self.image_results.clear()
                
                # Use standard CLIP search with error handling
                try:
                    query_embedding = self.clip_extractor.extract_text_embedding(query)
                    print(f"Query embedding shape: {query_embedding.shape}")
                    
                    results = self.clip_storage.search(query_embedding, max_results)
                    print(f"Search returned {len(results)} results")
                    
                    # Display results in main thread
                    self.root.after(0, lambda: self.display_results(results, query))
                    
                    self.root.after(0, lambda: self.status_var.set(f"Found {len(results)} results for: {query}"))
                    
                except Exception as search_error:
                    print(f"Search error: {search_error}")
                    error_msg = f"Search failed: {str(search_error)}"
                    self.root.after(0, lambda: self.status_var.set(error_msg))
                    self.root.after(0, lambda: messagebox.showerror("Search Error", error_msg))
                    
                    # Show error in results area
                    def show_error():
                        self.image_results.clear()
                        error_label = tk.Label(self.scrollable_frame, 
                                             text=f"❌ Search Error: {error_msg}\n\nThis may be due to:\n• OpenMP runtime conflicts\n• Memory issues\n• Corrupted embeddings\n\nTry restarting the application or reprocessing the videos.", 
                                             font=('Arial', 12), 
                                             bg='white', 
                                             fg='red',
                                             justify='center')
                        error_label.pack(pady=50)
                    self.root.after(0, show_error)
                
            except Exception as e:
                print(f"General search error: {e}")
                self.root.after(0, lambda: self.status_var.set(f"Search error: {str(e)}"))
                self.root.after(0, lambda: messagebox.showerror("Search Error", f"Search failed: {str(e)}"))
        
        thread = threading.Thread(target=search_in_thread)
        thread.daemon = True
        thread.start()
    
    def display_results(self, results, query):
        """Display search results as clickable image thumbnails."""
        if not results:
            # Show no results message in the image area
            self.image_results.clear()
            # Create a simple text label for no results
            no_results_label = tk.Label(self.scrollable_frame, 
                                      text=f"No results found for: '{query}'", 
                                      font=('Arial', 14), 
                                      bg='white', 
                                      fg='gray')
            no_results_label.pack(pady=50)
            self.current_results = []
            return
        
        # Store current results for context menu operations
        self.current_results = results
        
        # Clear image results
        self.image_results.clear()
        
        # Update status
        self.status_var.set(f"Extracting thumbnails for {len(results)} results...")
        self.root.update()
        
        for i, result in enumerate(results, 1):
            # Video info
            video_name = result.get('video_source', 'Unknown Video')
            scene_id = result.get('scene_id', 0)
            duration = result.get('duration', 0)
            similarity = result.get('similarity_score', 0)
            start_time = result.get('start_time', 0)
            end_time = result.get('end_time', 0)
            scene_path = result.get('scene_path', '')
            
            # Update status for each result
            self.status_var.set(f"Processing result {i}/{len(results)}: {video_name}")
            self.root.update()
            
            # Extract thumbnail and add to image results
            print(f"Processing result {i}: scene_path={scene_path}, exists={os.path.exists(scene_path) if scene_path else False}")
            if scene_path and os.path.exists(scene_path):
                try:
                    # Extract thumbnail from the beginning of the scene
                    thumbnail_timestamp = 0.1  # Use 0.1 seconds from the start
                    print(f"Extracting thumbnail at timestamp: {thumbnail_timestamp}")
                    thumbnail = self.extract_video_thumbnail(scene_path, thumbnail_timestamp)
                    
                    metadata = {
                        'video_source': video_name,
                        'scene_id': scene_id,
                        'start_time': start_time,
                        'end_time': end_time,
                        'similarity': similarity,
                        'duration': duration
                    }
                    self.image_results.add_result(thumbnail, scene_path, metadata, self.open_video)
                    
                except Exception as e:
                    print(f"Error processing result {i}: {e}")
                    # Add placeholder for failed extraction
                    metadata = {
                        'video_source': video_name,
                        'scene_id': scene_id,
                        'start_time': start_time,
                        'end_time': end_time,
                        'similarity': similarity,
                        'duration': duration
                    }
                    self.image_results.add_result(None, scene_path, metadata, self.open_video)
            else:
                print(f"Video file not found: {scene_path}")
        
        # Update status
        self.status_var.set(f"Found {len(results)} results - Click thumbnails to play videos")
        
        # Update canvas scroll region
        self.results_canvas.update_idletasks()
        self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))
    
    def clear_results(self):
        """Clear search results."""
        self.image_results.clear()
        self.query_var.set("")
        self.current_results = []
        self.status_var.set("Ready")
        
        # Clear any existing widgets in scrollable frame
        for widget in self.scrollable_frame.winfo_children():
            if isinstance(widget, tk.Label) and widget.cget('text').startswith('No results found'):
                widget.destroy()
    
    def show_stats(self):
        """Show database statistics."""
        if not self.clip_storage or not self.clip_storage.metadata:
            messagebox.showinfo("No Data", "No embeddings loaded.")
            return
        
        stats = self.clip_storage.get_stats()
        
        # Clear image results and show stats
        self.image_results.clear()
        
        # Create stats text
        stats_text = "📊 DATABASE STATISTICS\n" + "=" * 50 + "\n\n"
        stats_text += f"Total embeddings: {stats['total_embeddings']}\n"
        stats_text += f"Unique videos: {stats['unique_videos']}\n"
        stats_text += f"Total duration: {stats['total_duration']:.1f} seconds ({stats['total_duration']/60:.1f} minutes)\n\n"
        
        # Show videos in database
        video_sources = set(m['video_source'] for m in self.clip_storage.metadata)
        stats_text += "📹 Videos in database:\n"
        for video in sorted(video_sources):
            scene_count = sum(1 for m in self.clip_storage.metadata if m['video_source'] == video)
            video_duration = sum(m['duration'] for m in self.clip_storage.metadata if m['video_source'] == video)
            stats_text += f"  - {video}: {scene_count} scenes, {video_duration:.1f}s\n"
        
        # Display stats in the image area
        stats_label = tk.Label(self.scrollable_frame, 
                             text=stats_text, 
                             font=('Consolas', 10), 
                             bg='white', 
                             fg='black',
                             justify='left')
        stats_label.pack(pady=20)
    
    def extract_video_thumbnail(self, video_path, timestamp=0.1):
        """Extract a thumbnail frame from video at specified timestamp."""
        if not cv2:
            print("OpenCV not available for thumbnail extraction")
            return None
            
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Failed to open video: {video_path}")
                return None
            
            # Get video duration to ensure we don't exceed it
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            video_duration = frame_count / fps if fps > 0 else 0
            
            # Use a safe timestamp within the video duration (prefer beginning)
            safe_timestamp = min(timestamp, video_duration * 0.5)  # Use max 50% of duration
            if safe_timestamp < 0.05:  # If video is very short, use 5% of duration
                safe_timestamp = max(0.05, video_duration * 0.1)
            
            print(f"Extracting thumbnail from: {video_path} at {safe_timestamp:.2f}s (video duration: {video_duration:.2f}s)")
            
            # Set to specific timestamp (in seconds)
            cap.set(cv2.CAP_PROP_POS_MSEC, safe_timestamp * 1000)
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                print(f"Failed to read frame from: {video_path}")
                return None
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize to thumbnail size
            height, width = frame_rgb.shape[:2]
            max_size = 200
            if width > height:
                new_width = max_size
                new_height = int(height * max_size / width)
            else:
                new_height = max_size
                new_width = int(width * max_size / height)
            
            resized_frame = cv2.resize(frame_rgb, (new_width, new_height))
            
            # Convert to PIL Image
            pil_image = Image.fromarray(resized_frame)
            print(f"Successfully extracted thumbnail: {new_width}x{new_height}")
            
            return pil_image
            
        except Exception as e:
            print(f"Error extracting thumbnail from {video_path}: {e}")
            return None

    def open_video(self, video_path):
        """Open/play a video file using the default system video player."""
        if not os.path.exists(video_path):
            messagebox.showerror("File Not Found", f"Video file not found:\n{video_path}")
            return
        
        try:
            # Update status
            self.status_var.set(f"Opening: {os.path.basename(video_path)}")
            self.root.update()
            
            # Open with default application based on OS
            if sys.platform == "win32":
                os.startfile(video_path)
            elif sys.platform == "darwin":
                import subprocess
                subprocess.run(["open", video_path], check=True)
            else:
                import subprocess
                subprocess.run(["xdg-open", video_path], check=True)
            
            # Reset status after a short delay
            self.root.after(2000, lambda: self.status_var.set("Ready"))
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while opening the video:\n{str(e)}")
            self.status_var.set("Ready")
    
    def refresh_stats(self):
        """Refresh the database statistics display."""
        try:
            if self.clip_storage:
                stats = self.clip_storage.get_stats()
                stats_text = f"""Database Statistics:
                
Total Embeddings: {stats.get('total_embeddings', 0)}
Unique Videos: {stats.get('unique_videos', 0)}
Total Duration: {stats.get('total_duration', 0):.1f} seconds
                """
            else:
                stats_text = "No embeddings database loaded"
            
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(1.0, stats_text)
            
        except Exception as e:
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(1.0, f"Error loading stats: {e}")
    
    def clear_database(self):
        """Clear the embeddings database."""
        if messagebox.askyesno("Confirm", "Are you sure you want to clear the entire database? This cannot be undone."):
            try:
                if self.clip_storage:
                    # Clear the storage
                    self.clip_storage.clear()
                    self.log_message("Database cleared successfully")
                    self.refresh_stats()
                    self.refresh_info()
                else:
                    messagebox.showwarning("Warning", "No database loaded")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to clear database: {e}")
    
    def refresh_info(self):
        """Refresh the data directory information."""
        try:
            output_dir = self.output_dir_var.get()
            info_text = f"""Data Directory Information:
            
Output Directory: {output_dir}
Scenes Directory: {output_dir}/scenes
Metadata Directory: {output_dir}/metadata
Embeddings Directory: {output_dir}/embeddings

Directory Status:
"""
            
            # Check directory status
            for subdir in ['scenes', 'metadata', 'embeddings']:
                full_path = os.path.join(output_dir, subdir)
                if os.path.exists(full_path):
                    file_count = len([f for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f))])
                    info_text += f"{subdir.capitalize()}: {file_count} files\n"
                else:
                    info_text += f"{subdir.capitalize()}: Directory not found\n"
            
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(1.0, info_text)
            
        except Exception as e:
            self.info_text.insert(1.0, f"Error loading info: {e}")


def main():
    """Main function."""
    root = tk.Tk()
    app = VideoSearchGUI(root)
    
    # Handle window closing
    def on_closing():
        root.quit()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
