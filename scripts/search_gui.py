#!/usr/bin/env python3
"""
Enhanced GUI interface with working clickable video links for GPT-4 enhanced video search.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from pathlib import Path
import threading
import webbrowser
import subprocess
from urllib.parse import quote

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from embeddings.gpt4_search import GPT4VideoSearchEngine
    from embeddings.extractor import SceneEmbeddingExtractor
    from embeddings.storage import EmbeddingStorage
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all dependencies are installed.")
    sys.exit(1)


class ClickableText(tk.Text):
    """Custom Text widget that properly handles clickable links."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(state='disabled')  # Make read-only but still allow tag clicks
        self.bind('<Button-1>', self._on_click)
        self.bind('<Motion>', self._on_motion)
        
    def _on_click(self, event):
        """Handle click events on text."""
        # Temporarily enable to get current position
        self.configure(state='normal')
        index = self.index(f"@{event.x},{event.y}")
        self.configure(state='disabled')
        
        # Check if click is on a link tag
        for tag in self.tag_names(index):
            if tag.startswith('link_'):
                # Get the callback from tag
                if hasattr(self, '_link_callbacks') and tag in self._link_callbacks:
                    self._link_callbacks[tag]()
                break
    
    def _on_motion(self, event):
        """Handle mouse motion for cursor changes."""
        # Temporarily enable to get current position
        self.configure(state='normal')
        index = self.index(f"@{event.x},{event.y}")
        self.configure(state='disabled')
        
        # Check if mouse is over a link
        is_over_link = False
        for tag in self.tag_names(index):
            if tag.startswith('link_'):
                is_over_link = True
                break
        
        # Change cursor
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
        
        # Configure tag appearance
        self.tag_config(tag_name,
                       foreground='blue',
                       underline=True,
                       font=('Arial', 10, 'underline bold'))


class VideoSearchGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Scene Search - GPT-4 Enhanced")
        self.root.geometry("900x700")
        
        # Initialize search engines
        self.gpt4_engine = None
        self.clip_extractor = None
        self.clip_storage = None
        self.data_dir = "data"
        self.current_results = []  # Store current search results
        
        # Create GUI
        self.create_widgets()
        
        # Load embeddings on startup
        self.load_embeddings()
    
    def create_widgets(self):
        """Create the GUI widgets."""
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🎬 Video Scene Search", 
                               font=('Arial', 18, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        subtitle_label = ttk.Label(main_frame, text="Click blue video links to play clips!", 
                                  font=('Arial', 10, 'italic'))
        subtitle_label.grid(row=0, column=0, columnspan=3, pady=(30, 10))
        
        # Search query
        ttk.Label(main_frame, text="Search Query:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.query_var = tk.StringVar()
        query_entry = ttk.Entry(main_frame, textvariable=self.query_var, width=50, font=('Arial', 11))
        query_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5, padx=(5, 0))
        query_entry.bind('<Return>', lambda e: self.search())
        
        # Search options frame
        options_frame = ttk.Frame(main_frame)
        options_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Search options
        self.use_gpt4_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Use GPT-4 Enhancement", 
                       variable=self.use_gpt4_var).grid(row=0, column=0, sticky=tk.W)
        
        ttk.Label(options_frame, text="Max Results:").grid(row=0, column=1, padx=(20, 5))
        self.max_results_var = tk.StringVar(value="10")
        results_spinbox = ttk.Spinbox(options_frame, from_=1, to=50, width=5,
                                     textvariable=self.max_results_var)
        results_spinbox.grid(row=0, column=2)
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=3, pady=10)
        
        # Buttons
        search_btn = ttk.Button(button_frame, text="🔍 Search", command=self.search)
        search_btn.grid(row=0, column=0, padx=5)
        
        clear_btn = ttk.Button(button_frame, text="Clear", command=self.clear_results)
        clear_btn.grid(row=0, column=1, padx=5)
        
        stats_btn = ttk.Button(button_frame, text="Show Stats", command=self.show_stats)
        stats_btn.grid(row=0, column=2, padx=5)
        
        load_btn = ttk.Button(button_frame, text="Load Data Dir", command=self.load_data_dir)
        load_btn.grid(row=0, column=3, padx=5)
        
        # Results area
        results_label = ttk.Label(main_frame, text="Search Results:", font=('Arial', 12, 'bold'))
        results_label.grid(row=4, column=0, sticky=(tk.W, tk.N), pady=(10, 5))
        
        # Create frame for results text with scrollbar
        results_frame = ttk.Frame(main_frame)
        results_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(25, 0))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Custom clickable text widget
        self.results_text = ClickableText(results_frame, width=90, height=25, wrap=tk.WORD,
                                         font=('Consolas', 10), bg='white', relief='sunken', bd=2)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(results_frame, orient='vertical', command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Add context menu to results
        self.create_context_menu()
    
    def create_context_menu(self):
        """Create context menu for results."""
        self.context_menu = tk.Menu(self.root, tearoff=0)
        self.context_menu.add_command(label="Copy Selection", command=self.copy_text)
        self.context_menu.add_command(label="Select All", command=self.select_all)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Open Video Folder", command=self.open_video_folder)
        self.context_menu.add_command(label="Copy Video Paths", command=self.copy_video_paths)
        
        self.results_text.bind("<Button-3>", self.show_context_menu)
    
    def show_context_menu(self, event):
        """Show context menu."""
        try:
            self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.context_menu.grab_release()
    
    def copy_text(self):
        """Copy selected text."""
        try:
            selected_text = self.results_text.selection_get()
            self.root.clipboard_clear()
            self.root.clipboard_append(selected_text)
        except tk.TclError:
            pass
    
    def select_all(self):
        """Select all text."""
        self.results_text.tag_add(tk.SEL, "1.0", tk.END)
    
    def open_video_folder(self):
        """Open the video data folder."""
        video_folder = os.path.join(self.data_dir, "scenes")
        if os.path.exists(video_folder):
            if sys.platform == "win32":
                os.startfile(video_folder)
            elif sys.platform == "darwin":
                os.system(f"open '{video_folder}'")
            else:
                os.system(f"xdg-open '{video_folder}'")
        else:
            messagebox.showwarning("Folder not found", f"Video folder not found: {video_folder}")
    
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
                subprocess.run(["open", video_path], check=True)
            else:
                subprocess.run(["xdg-open", video_path], check=True)
            
            # Reset status after a short delay
            self.root.after(2000, lambda: self.status_var.set("Ready"))
            
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Failed to Open Video", 
                               f"Failed to open video with default player:\n{video_path}\n\nError: {e}")
            self.status_var.set("Ready")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while opening the video:\n{str(e)}")
            self.status_var.set("Ready")
    
    def copy_video_paths(self):
        """Copy all video file paths from current search results to clipboard."""
        if not self.current_results:
            messagebox.showinfo("No Results", "No search results to copy paths from.")
            return
        
        video_paths = []
        for result in self.current_results:
            scene_path = result.get('scene_path', '')
            if scene_path and os.path.exists(scene_path):
                video_paths.append(scene_path)
        
        if video_paths:
            paths_text = '\n'.join(video_paths)
            self.root.clipboard_clear()
            self.root.clipboard_append(paths_text)
            messagebox.showinfo("Copied", f"Copied {len(video_paths)} video file paths to clipboard.")
        else:
            messagebox.showinfo("No Paths", "No valid video file paths found in current results.")
    
    def load_data_dir(self):
        """Load a different data directory."""
        new_dir = filedialog.askdirectory(title="Select Data Directory", initialdir=self.data_dir)
        if new_dir:
            self.data_dir = new_dir
            self.load_embeddings()
    
    def load_embeddings(self):
        """Load embeddings in a separate thread."""
        def load_in_thread():
            self.status_var.set("Loading embeddings...")
            self.root.update()
            
            try:
                # Initialize GPT-4 search engine
                self.gpt4_engine = GPT4VideoSearchEngine(storage_path=f"{self.data_dir}/embeddings")
                
                # Initialize standard CLIP components
                self.clip_extractor = SceneEmbeddingExtractor()
                self.clip_storage = EmbeddingStorage(storage_path=f"{self.data_dir}/embeddings")
                self.clip_storage.load()
                
                embedding_count = len(self.gpt4_engine.clip_storage.metadata)
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
        if not self.gpt4_engine or not self.gpt4_engine.clip_storage.metadata:
            self.results_text.insert_with_tags("No video embeddings found.\n")
            self.results_text.insert_with_tags("Please run process_videos.py first to create embeddings.\n\n")
            self.results_text.insert_with_tags("Example usage:\n")
            self.results_text.insert_with_tags("python scripts/process_videos.py data/raw_videos/\n\n")
            return
        
        stats = self.gpt4_engine.clip_storage.get_stats()
        
        self.results_text.insert_with_tags("=" * 60 + "\n")
        self.results_text.insert_with_tags("🎬 GPT-4 Enhanced Video Search - Ready!\n")
        self.results_text.insert_with_tags("=" * 60 + "\n\n")
        
        self.results_text.insert_with_tags(f"Database Statistics:\n")
        self.results_text.insert_with_tags(f"  • Total embeddings: {stats['total_embeddings']}\n")
        self.results_text.insert_with_tags(f"  • Unique videos: {stats['unique_videos']}\n")
        self.results_text.insert_with_tags(f"  • Total duration: {stats['total_duration']:.1f} seconds\n\n")
        
        self.results_text.insert_with_tags("✨ Enter a search query above and click 'Search' to find matching video scenes.\n")
        self.results_text.insert_with_tags("📹 Examples: 'person walking', 'car driving', 'sunset', 'people talking'\n")
        self.results_text.insert_with_tags("🖱️  Click on blue video links to play clips instantly!\n\n")
    
    def search(self):
        """Perform search."""
        query = self.query_var.get().strip()
        if not query:
            messagebox.showwarning("Empty Query", "Please enter a search query.")
            return
        
        if not self.gpt4_engine or not self.gpt4_engine.clip_storage.metadata:
            messagebox.showerror("No Data", "No embeddings loaded. Please load a data directory first.")
            return
        
        # Run search in separate thread
        def search_in_thread():
            self.status_var.set(f"Searching for: {query}...")
            self.root.update()
            
            try:
                max_results = int(self.max_results_var.get())
                use_gpt4 = self.use_gpt4_var.get()
                
                # Clear previous results
                self.results_text.clear_text()
                
                if use_gpt4:
                    # Use GPT-4 enhanced search
                    results = self.gpt4_engine.search_with_prompt(
                        query, k=max_results, use_gpt4_enhancement=True
                    )
                else:
                    # Use standard CLIP search
                    query_embedding = self.clip_extractor.extract_text_embedding(query)
                    results = self.clip_storage.search(query_embedding, max_results)
                
                # Display results
                self.display_results(results, query, use_gpt4)
                
                self.status_var.set(f"Found {len(results)} results for: {query}")
                
            except Exception as e:
                self.status_var.set(f"Search error: {str(e)}")
                messagebox.showerror("Search Error", f"Search failed: {str(e)}")
        
        thread = threading.Thread(target=search_in_thread)
        thread.daemon = True
        thread.start()
    
    def display_results(self, results, query, use_gpt4):
        """Display search results with clickable video links."""
        if not results:
            self.results_text.insert_with_tags(f"No results found for: '{query}'\n")
            self.current_results = []  # Clear stored results
            return
        
        # Store current results for context menu operations
        self.current_results = results
        
        # Header
        mode = "GPT-4 Enhanced" if use_gpt4 else "Standard CLIP"
        self.results_text.insert_with_tags("=" * 80 + "\n")
        self.results_text.insert_with_tags(f"🎯 {mode} SEARCH RESULTS FOR: {query}\n")
        self.results_text.insert_with_tags("=" * 80 + "\n\n")
        
        for i, result in enumerate(results, 1):
            # Video info
            video_name = result.get('video_source', 'Unknown Video')
            scene_id = result.get('scene_id', 0)
            duration = result.get('duration', 0)
            similarity = result.get('similarity_score', 0)
            start_time = result.get('start_time', 0)
            end_time = result.get('end_time', 0)
            scene_path = result.get('scene_path', '')
            
            # Result header
            self.results_text.insert_with_tags(f"{i}. 📹 {video_name}\n")
            self.results_text.insert_with_tags(f"   Scene: {scene_id} | Duration: {duration:.1f}s | Similarity: {similarity:.4f}\n")
            
            if 'gpt4_rank' in result:
                self.results_text.insert_with_tags(f"   🎯 GPT-4 Rank: #{result['gpt4_rank']}\n")
            
            self.results_text.insert_with_tags(f"   ⏱️  Time: {start_time:.1f}s - {end_time:.1f}s\n")
            
            # Add clickable video link if path exists
            if scene_path and os.path.exists(scene_path):
                # Create clickable link
                link_text = f"   🎬 ▶️ CLICK TO PLAY: {os.path.basename(scene_path)}"
                tag_name = f"link_{i}_{scene_id}"
                
                self.results_text.add_link(tag_name, link_text, lambda path=scene_path: self.open_video(path))
                self.results_text.insert_with_tags(link_text + "\n", [tag_name])
                
                # Also add path for reference
                self.results_text.insert_with_tags(f"   📁 Path: {scene_path}\n")
            else:
                self.results_text.insert_with_tags(f"   ❌ Video file not found: {scene_path}\n")
            
            self.results_text.insert_with_tags("-" * 60 + "\n\n")
        
        # Add instruction text
        self.results_text.insert_with_tags("\n💡 TIP: Click on the blue ▶️ CLICK TO PLAY links above to open video clips!\n")
        self.results_text.insert_with_tags("🖱️  Right-click for more options (copy paths, open folder, etc.)\n")
        
        # Scroll to top
        self.results_text.see(1.0)
    
    def clear_results(self):
        """Clear search results."""
        self.results_text.clear_text()
        self.query_var.set("")
        self.current_results = []
        self.status_var.set("Ready")
    
    def show_stats(self):
        """Show database statistics."""
        if not self.gpt4_engine or not self.gpt4_engine.clip_storage.metadata:
            messagebox.showinfo("No Data", "No embeddings loaded.")
            return
        
        stats = self.gpt4_engine.clip_storage.get_stats()
        
        self.results_text.clear_text()
        self.results_text.insert_with_tags("=" * 50 + "\n")
        self.results_text.insert_with_tags("📊 DATABASE STATISTICS\n")
        self.results_text.insert_with_tags("=" * 50 + "\n\n")
        
        self.results_text.insert_with_tags(f"Total embeddings: {stats['total_embeddings']}\n")
        self.results_text.insert_with_tags(f"Unique videos: {stats['unique_videos']}\n")
        self.results_text.insert_with_tags(f"Total duration: {stats['total_duration']:.1f} seconds ({stats['total_duration']/60:.1f} minutes)\n\n")
        
        # Show videos in database
        video_sources = set(m['video_source'] for m in self.gpt4_engine.clip_storage.metadata)
        self.results_text.insert_with_tags("📹 Videos in database:\n")
        for video in sorted(video_sources):
            scene_count = sum(1 for m in self.gpt4_engine.clip_storage.metadata if m['video_source'] == video)
            video_duration = sum(m['duration'] for m in self.gpt4_engine.clip_storage.metadata if m['video_source'] == video)
            self.results_text.insert_with_tags(f"  - {video}: {scene_count} scenes, {video_duration:.1f}s\n")


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
