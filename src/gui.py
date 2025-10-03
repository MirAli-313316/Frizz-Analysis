"""
Modern GUI for Hair Frizz Analysis Tool using customtkinter.

Provides a user-friendly interface for selecting images, processing them,
and viewing results with visualization and Excel export capabilities.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from pathlib import Path
import threading
from typing import List, Optional
import logging
import os
import sys
from PIL import Image, ImageTk
import pandas as pd
import traceback

from .batch_processor import BatchProcessor
from .time_parser import TimePointParser
from .analysis import ImageAnalysis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set appearance mode and color theme
ctk.set_appearance_mode("System")  # Modes: "System", "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue", "green", "dark-blue"


class FrizzAnalysisGUI:
    """Main GUI application for Frizz Analysis."""
    
    def __init__(self):
        """Initialize the GUI application."""
        self.root = ctk.CTk()
        self.root.title("Hair Frizz Analysis Tool")
        self.root.geometry("1200x800")
        
        # Data storage
        self.selected_files: List[str] = []
        self.time_points: List = []
        self.results: List[ImageAnalysis] = []
        self.excel_path: Optional[Path] = None
        self.summary_df: Optional[pd.DataFrame] = None
        
        # Processing state
        self.processing = False
        
        # Create UI components
        self._create_title()
        self._create_main_container()
        self._create_left_panel()
        self._create_right_panel()
        self._create_bottom_panel()
        
        logger.info("GUI initialized successfully")
    
    def _create_title(self):
        """Create title bar."""
        title_frame = ctk.CTkFrame(self.root, height=60, corner_radius=0)
        title_frame.pack(fill="x", padx=0, pady=0)
        title_frame.pack_propagate(False)
        
        title_label = ctk.CTkLabel(
            title_frame,
            text="🔬 Hair Frizz Analysis Tool",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=15)
    
    def _create_main_container(self):
        """Create main container for left and right panels."""
        self.main_container = ctk.CTkFrame(self.root, corner_radius=0)
        self.main_container.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Configure grid weights
        self.main_container.grid_columnconfigure(0, weight=1, minsize=400)
        self.main_container.grid_columnconfigure(1, weight=2)
        self.main_container.grid_rowconfigure(0, weight=1)
    
    def _create_left_panel(self):
        """Create left panel for file management."""
        left_frame = ctk.CTkFrame(self.main_container)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=0)
        
        # Section title
        title_label = ctk.CTkLabel(
            left_frame,
            text="📁 File Management",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title_label.pack(pady=(15, 10))
        
        # Button frame
        button_frame = ctk.CTkFrame(left_frame, fg_color="transparent")
        button_frame.pack(fill="x", padx=15, pady=5)
        
        # Select Images button
        self.select_btn = ctk.CTkButton(
            button_frame,
            text="Select Images",
            command=self._select_images,
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40,
            fg_color="#2B7A78",
            hover_color="#17252A"
        )
        self.select_btn.pack(fill="x", pady=5)
        
        # Clear button
        self.clear_btn = ctk.CTkButton(
            button_frame,
            text="Clear Selection",
            command=self._clear_selection,
            font=ctk.CTkFont(size=14),
            height=35,
            fg_color="#D32F2F",
            hover_color="#B71C1C"
        )
        self.clear_btn.pack(fill="x", pady=5)
        
        # File count label
        self.file_count_label = ctk.CTkLabel(
            left_frame,
            text="No images selected",
            font=ctk.CTkFont(size=12)
        )
        self.file_count_label.pack(pady=(10, 5))
        
        # Listbox frame with scrollbar
        list_frame = ctk.CTkFrame(left_frame)
        list_frame.pack(fill="both", expand=True, padx=15, pady=10)
        
        # Create scrollable frame for file list
        self.file_list_frame = ctk.CTkScrollableFrame(
            list_frame,
            label_text="Selected Files",
            label_font=ctk.CTkFont(size=14, weight="bold")
        )
        self.file_list_frame.pack(fill="both", expand=True)
        
        # Instructions
        instructions = ctk.CTkLabel(
            left_frame,
            text="Select images to process.\nTime points will be auto-detected.",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        instructions.pack(pady=(0, 15))
    
    def _create_right_panel(self):
        """Create right panel with tabbed interface."""
        right_frame = ctk.CTkFrame(self.main_container)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=0)
        
        # Create tabview
        self.tabview = ctk.CTkTabview(right_frame)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create tabs
        self.tabview.add("Analysis")
        self.tabview.add("Visualization")
        self.tabview.add("Excel Report")
        
        # --- Analysis Tab ---
        self._create_analysis_tab()
        
        # --- Visualization Tab ---
        self._create_visualization_tab()
        
        # --- Excel Report Tab ---
        self._create_excel_tab()
    
    def _create_analysis_tab(self):
        """Create analysis results tab."""
        tab = self.tabview.tab("Analysis")
        
        # Title
        title = ctk.CTkLabel(
            tab,
            text="📊 Analysis Results",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title.pack(pady=(10, 5))
        
        # Results frame with scrollbar
        self.analysis_frame = ctk.CTkScrollableFrame(tab)
        self.analysis_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Placeholder text
        self.analysis_placeholder = ctk.CTkLabel(
            self.analysis_frame,
            text="No analysis results yet.\nSelect images and click 'Process Images' to begin.",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        self.analysis_placeholder.pack(pady=50)
    
    def _create_visualization_tab(self):
        """Create visualization tab."""
        tab = self.tabview.tab("Visualization")
        
        # Title
        title = ctk.CTkLabel(
            tab,
            text="🖼️ Processed Images",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title.pack(pady=(10, 5))
        
        # Image selector frame
        selector_frame = ctk.CTkFrame(tab, fg_color="transparent")
        selector_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(
            selector_frame,
            text="Select Image:",
            font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=5)
        
        # Image dropdown
        self.image_selector = ctk.CTkComboBox(
            selector_frame,
            values=["No images processed"],
            command=self._display_selected_image,
            state="readonly",
            width=300
        )
        self.image_selector.pack(side="left", padx=5)
        self.image_selector.set("No images processed")
        
        # Scrollable frame for image display
        self.viz_frame = ctk.CTkScrollableFrame(tab)
        self.viz_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Placeholder
        self.viz_placeholder = ctk.CTkLabel(
            self.viz_frame,
            text="No visualizations yet.\nProcessed images will appear here.",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        self.viz_placeholder.pack(pady=50)
    
    def _create_excel_tab(self):
        """Create Excel report tab."""
        tab = self.tabview.tab("Excel Report")
        
        # Title
        title = ctk.CTkLabel(
            tab,
            text="📈 Excel Report Summary",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title.pack(pady=(10, 5))
        
        # Report frame
        self.excel_frame = ctk.CTkScrollableFrame(tab)
        self.excel_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Placeholder
        self.excel_placeholder = ctk.CTkLabel(
            self.excel_frame,
            text="No Excel report generated yet.\nReport will be created after processing.",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        self.excel_placeholder.pack(pady=50)
    
    def _create_bottom_panel(self):
        """Create bottom panel with process button and progress."""
        bottom_frame = ctk.CTkFrame(self.root, corner_radius=0, height=120)
        bottom_frame.pack(fill="x", padx=10, pady=(0, 10))
        bottom_frame.pack_propagate(False)
        
        # Process button
        self.process_btn = ctk.CTkButton(
            bottom_frame,
            text="🚀 Process Images",
            command=self._process_images,
            font=ctk.CTkFont(size=16, weight="bold"),
            height=50,
            state="disabled",
            fg_color="#1B5E20",
            hover_color="#2E7D32"
        )
        self.process_btn.pack(pady=(15, 10), padx=20, fill="x")
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(bottom_frame)
        self.progress_bar.pack(fill="x", padx=20, pady=5)
        self.progress_bar.set(0)
        
        # Status text
        self.status_label = ctk.CTkLabel(
            bottom_frame,
            text="Ready. Select images to begin.",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.pack(pady=5)
    
    # --- Event Handlers ---
    
    def _select_images(self):
        """Open file dialog to select images."""
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.JPG *.JPEG *.png *.PNG"),
            ("All files", "*.*")
        ]
        
        files = filedialog.askopenfilenames(
            title="Select Hair Tress Images",
            filetypes=filetypes
        )
        
        if files:
            self.selected_files = list(files)
            self._update_file_list()
            self.process_btn.configure(state="normal")
            self._update_status(f"Selected {len(files)} images")
            logger.info(f"Selected {len(files)} images")
    
    def _clear_selection(self):
        """Clear selected files."""
        self.selected_files = []
        self.time_points = []
        self._update_file_list()
        self.process_btn.configure(state="disabled")
        self._update_status("Selection cleared. Ready to select new images.")
        logger.info("Selection cleared")
    
    def _update_file_list(self):
        """Update the file list display."""
        # Clear existing items
        for widget in self.file_list_frame.winfo_children():
            widget.destroy()
        
        if not self.selected_files:
            self.file_count_label.configure(text="No images selected")
            placeholder = ctk.CTkLabel(
                self.file_list_frame,
                text="No files selected",
                text_color="gray"
            )
            placeholder.pack(pady=20)
            return
        
        # Update count
        self.file_count_label.configure(text=f"{len(self.selected_files)} images selected")
        
        # Parse time points
        parser = TimePointParser()
        self.time_points = parser.parse_batch(self.selected_files)
        
        # Sort by time
        sorted_items = sorted(
            zip(self.time_points, self.selected_files),
            key=lambda x: x[0].hours
        )
        
        # Display files with time points
        for tp, filepath in sorted_items:
            filename = Path(filepath).name
            
            # Create frame for each file
            file_frame = ctk.CTkFrame(self.file_list_frame, fg_color="transparent")
            file_frame.pack(fill="x", pady=2)
            
            # Time point label
            time_label = ctk.CTkLabel(
                file_frame,
                text=f"{tp.label:>10}",
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color="#2B7A78",
                width=80
            )
            time_label.pack(side="left", padx=5)
            
            # Filename label
            name_label = ctk.CTkLabel(
                file_frame,
                text=filename,
                font=ctk.CTkFont(size=11),
                anchor="w"
            )
            name_label.pack(side="left", fill="x", expand=True, padx=5)
    
    def _process_images(self):
        """Start image processing in background thread."""
        if not self.selected_files:
            messagebox.showwarning("No Files", "Please select images first.")
            return
        
        if self.processing:
            messagebox.showinfo("Processing", "Processing already in progress.")
            return
        
        # Disable buttons during processing
        self.processing = True
        self.process_btn.configure(state="disabled")
        self.select_btn.configure(state="disabled")
        self.clear_btn.configure(state="disabled")
        
        # Reset progress
        self.progress_bar.set(0)
        self._update_status("Starting processing...")
        
        # Start processing thread
        thread = threading.Thread(target=self._process_thread, daemon=True)
        thread.start()
    
    def _process_thread(self):
        """Background thread for processing images."""
        try:
            # Create batch processor
            processor = BatchProcessor(output_dir="outputs")
            
            total_images = len(self.selected_files)
            
            # Process each image with progress updates
            self._update_status(f"Processing {total_images} images...")
            
            results = []
            for i, filepath in enumerate(self.selected_files):
                # Update progress
                progress = (i + 1) / total_images
                self.root.after(0, self._update_progress, progress, 
                               f"Processing image {i+1}/{total_images}...")
                
                try:
                    from .analysis import analyze_image
                    result = analyze_image(
                        filepath,
                        visualize=True,
                        output_dir="outputs",
                        max_processing_dim=1024
                    )
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to process {filepath}: {e}")
                    logger.error(traceback.format_exc())
                    # Continue with next image
                    continue
            
            if not results:
                raise RuntimeError("No images were successfully processed")
            
            self.results = results
            
            # Parse time points again for sorted results
            time_points = TimePointParser().parse_batch(self.selected_files)
            
            # Sort by time
            sorted_items = sorted(
                zip(time_points, results),
                key=lambda x: x[0].hours
            )
            sorted_time_points, sorted_results = zip(*sorted_items)
            sorted_time_points = list(sorted_time_points)
            sorted_results = list(sorted_results)
            
            # Generate Excel report
            self._update_status("Generating Excel report...")
            excel_path = processor.generate_excel_report(
                sorted_results,
                sorted_time_points,
                output_filename="test_results.xlsx"
            )
            self.excel_path = excel_path
            
            # Create summary dataframe
            self.summary_df = processor._create_summary_dataframe(
                sorted_results,
                sorted_time_points
            )
            
            # Update UI with results
            self.root.after(0, self._display_results)
            
            # Success
            self.root.after(0, self._update_status, 
                           f"✓ Processing complete! {len(results)} images analyzed.")
            self.root.after(0, self._processing_complete)
            
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.root.after(0, messagebox.showerror, "Processing Error", error_msg)
            self.root.after(0, self._update_status, f"❌ Error: {str(e)}")
            self.root.after(0, self._processing_complete)
    
    def _processing_complete(self):
        """Re-enable UI after processing."""
        self.processing = False
        self.process_btn.configure(state="normal")
        self.select_btn.configure(state="normal")
        self.clear_btn.configure(state="normal")
        
        # Open Excel file if generated
        if self.excel_path and self.excel_path.exists():
            try:
                os.startfile(str(self.excel_path))  # Windows
            except AttributeError:
                try:
                    import subprocess
                    subprocess.call(['open', str(self.excel_path)])  # macOS
                except:
                    try:
                        subprocess.call(['xdg-open', str(self.excel_path)])  # Linux
                    except:
                        logger.warning("Could not auto-open Excel file")
    
    def _update_progress(self, value: float, status: str):
        """Update progress bar and status."""
        self.progress_bar.set(value)
        self._update_status(status)
    
    def _update_status(self, text: str):
        """Update status label."""
        self.status_label.configure(text=text)
    
    def _display_results(self):
        """Display analysis results in the UI."""
        # Update Analysis tab
        self._populate_analysis_tab()
        
        # Update Visualization tab
        self._populate_visualization_tab()
        
        # Update Excel Report tab
        self._populate_excel_tab()
        
        # Switch to Analysis tab
        self.tabview.set("Analysis")
    
    def _populate_analysis_tab(self):
        """Populate the analysis tab with results."""
        # Clear existing content
        for widget in self.analysis_frame.winfo_children():
            widget.destroy()
        
        if not self.results:
            return
        
        # Create summary cards
        for result in self.results:
            card = ctk.CTkFrame(self.analysis_frame)
            card.pack(fill="x", pady=5, padx=5)
            
            # Header
            header = ctk.CTkLabel(
                card,
                text=f"📷 {result.image_name}",
                font=ctk.CTkFont(size=14, weight="bold"),
                anchor="w"
            )
            header.pack(fill="x", padx=10, pady=(10, 5))
            
            # Info frame
            info_frame = ctk.CTkFrame(card, fg_color="transparent")
            info_frame.pack(fill="x", padx=10, pady=5)
            
            # Stats
            stats_text = (
                f"Tresses: {len(result.tresses)} | "
                f"Total Area: {result.get_total_area():.2f} cm² | "
                f"Processing: {result.processing_time:.2f}s"
            )
            stats_label = ctk.CTkLabel(
                info_frame,
                text=stats_text,
                font=ctk.CTkFont(size=11)
            )
            stats_label.pack(anchor="w")
            
            # Individual tresses
            for tress in result.tresses:
                tress_text = f"  • Tress {tress.tress_id}: {tress.area_cm2:.2f} cm²"
                tress_label = ctk.CTkLabel(
                    card,
                    text=tress_text,
                    font=ctk.CTkFont(size=10),
                    text_color="gray",
                    anchor="w"
                )
                tress_label.pack(fill="x", padx=20, pady=2)
    
    def _populate_visualization_tab(self):
        """Populate the visualization tab."""
        # Clear existing
        for widget in self.viz_frame.winfo_children():
            widget.destroy()
        
        if not self.results:
            return
        
        # Update dropdown
        output_dir = Path("outputs")
        image_names = []
        
        for result in self.results:
            analysis_image = output_dir / f"analysis_{result.image_name}.jpg"
            if analysis_image.exists():
                image_names.append(result.image_name)
        
        if image_names:
            self.image_selector.configure(values=image_names)
            self.image_selector.set(image_names[0])
            self._display_selected_image(image_names[0])
    
    def _display_selected_image(self, choice):
        """Display the selected visualization image."""
        # Clear existing
        for widget in self.viz_frame.winfo_children():
            widget.destroy()
        
        if choice == "No images processed":
            return
        
        # Find and display image
        output_dir = Path("outputs")
        image_path = output_dir / f"analysis_{choice}.jpg"
        
        if not image_path.exists():
            error_label = ctk.CTkLabel(
                self.viz_frame,
                text=f"Image not found: {image_path}",
                text_color="red"
            )
            error_label.pack(pady=20)
            return
        
        try:
            # Load and display image
            img = Image.open(image_path)
            
            # Resize to fit (max 750 width)
            max_width = 750
            if img.width > max_width:
                ratio = max_width / img.width
                new_height = int(img.height * ratio)
                img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)
            
            # Display in label
            img_label = tk.Label(self.viz_frame, image=photo)
            img_label.image = photo  # Keep reference
            img_label.pack(pady=10)
            
        except Exception as e:
            error_label = ctk.CTkLabel(
                self.viz_frame,
                text=f"Error loading image: {str(e)}",
                text_color="red"
            )
            error_label.pack(pady=20)
    
    def _populate_excel_tab(self):
        """Populate the Excel report tab."""
        # Clear existing
        for widget in self.excel_frame.winfo_children():
            widget.destroy()
        
        if not self.excel_path or not self.excel_path.exists():
            return
        
        # Display Excel info
        info_card = ctk.CTkFrame(self.excel_frame)
        info_card.pack(fill="x", pady=10, padx=10)
        
        title = ctk.CTkLabel(
            info_card,
            text="📊 Excel Report Generated",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title.pack(pady=(15, 5))
        
        path_label = ctk.CTkLabel(
            info_card,
            text=f"Location: {self.excel_path}",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        path_label.pack(pady=5)
        
        # Open button
        open_btn = ctk.CTkButton(
            info_card,
            text="📂 Open Excel File",
            command=lambda: os.startfile(str(self.excel_path)),
            height=35
        )
        open_btn.pack(pady=10)
        
        # Display summary dataframe
        if self.summary_df is not None and not self.summary_df.empty:
            summary_title = ctk.CTkLabel(
                self.excel_frame,
                text="Summary Preview",
                font=ctk.CTkFont(size=14, weight="bold")
            )
            summary_title.pack(pady=(20, 10))
            
            # Create text widget to display dataframe
            text_frame = ctk.CTkFrame(self.excel_frame)
            text_frame.pack(fill="both", expand=True, padx=10, pady=10)
            
            text_widget = ctk.CTkTextbox(text_frame, wrap="none", height=300)
            text_widget.pack(fill="both", expand=True)
            
            # Format dataframe as string
            df_str = self.summary_df.to_string(index=False)
            text_widget.insert("1.0", df_str)
            text_widget.configure(state="disabled")
    
    def run(self):
        """Start the GUI application."""
        logger.info("Starting Frizz Analysis GUI...")
        self.root.mainloop()


def main():
    """Entry point for GUI application."""
    try:
        app = FrizzAnalysisGUI()
        app.run()
    except Exception as e:
        logger.error(f"GUI error: {e}")
        logger.error(traceback.format_exc())
        messagebox.showerror("Application Error", f"Failed to start GUI: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()


