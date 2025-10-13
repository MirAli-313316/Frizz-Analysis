import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import cv2
import numpy as np
from pathlib import Path
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class QuarterSegmenterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Quarter Segmenter - SAM2")
        self.root.geometry("500x300")
        self.root.resizable(False, False)
        
        self.image_path = None
        self.export_folder = None
        self.predictor = None
        
        # Create GUI first
        self.create_widgets()
        
        # Initialize SAM2 model after GUI is created
        self.init_sam2()
    
    def init_sam2(self):
        """Initialize SAM2 model"""
        try:
            # Use tiny model for 4GB VRAM
            model_cfg = "sam2_hiera_t.yaml"
            checkpoint = "checkpoints/sam2_hiera_tiny.pt"
            
            # Build model
            sam2_model = build_sam2(model_cfg, checkpoint, device="cuda")
            self.predictor = SAM2ImagePredictor(sam2_model)
            
            self.status_label.config(text="✓ SAM2 Model Loaded", fg="green")
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load SAM2 model:\n{str(e)}\n\nMake sure sam2_hiera_tiny.pt is in the checkpoints folder.")
    
    def create_widgets(self):
        # Title
        title_label = tk.Label(self.root, text="Quarter Segmenter", font=("Arial", 16, "bold"))
        title_label.pack(pady=20)
        
        # Status label
        self.status_label = tk.Label(self.root, text="Loading SAM2 model...", font=("Arial", 10))
        self.status_label.pack(pady=5)
        
        # Image selection frame
        image_frame = tk.Frame(self.root)
        image_frame.pack(pady=15, padx=20, fill="x")
        
        tk.Label(image_frame, text="Image:", font=("Arial", 10)).grid(row=0, column=0, sticky="w", pady=5)
        self.image_label = tk.Label(image_frame, text="No image selected", fg="gray", anchor="w")
        self.image_label.grid(row=0, column=1, sticky="w", padx=10)
        
        select_image_btn = ttk.Button(image_frame, text="Select Image", command=self.select_image)
        select_image_btn.grid(row=0, column=2, padx=5)
        
        # Export folder selection frame
        export_frame = tk.Frame(self.root)
        export_frame.pack(pady=15, padx=20, fill="x")
        
        tk.Label(export_frame, text="Export:", font=("Arial", 10)).grid(row=0, column=0, sticky="w", pady=5)
        self.export_label = tk.Label(export_frame, text="No folder selected", fg="gray", anchor="w")
        self.export_label.grid(row=0, column=1, sticky="w", padx=10)
        
        select_export_btn = ttk.Button(export_frame, text="Select Folder", command=self.select_export_folder)
        select_export_btn.grid(row=0, column=2, padx=5)
        
        # Segment button
        self.segment_btn = ttk.Button(self.root, text="Segment Quarter", command=self.segment_quarter, state="disabled")
        self.segment_btn.pack(pady=30)
        
        # Progress label
        self.progress_label = tk.Label(self.root, text="", font=("Arial", 9), fg="blue")
        self.progress_label.pack(pady=5)
    
    def select_image(self):
        """Select input image"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("JPEG files", "*.jpg *.jpeg"), ("All files", "*.*")]
        )
        if file_path:
            self.image_path = file_path
            self.image_label.config(text=Path(file_path).name, fg="black")
            self.check_ready()
    
    def select_export_folder(self):
        """Select export folder"""
        folder_path = filedialog.askdirectory(title="Select Export Folder")
        if folder_path:
            self.export_folder = folder_path
            self.export_label.config(text=Path(folder_path).name, fg="black")
            self.check_ready()
    
    def check_ready(self):
        """Enable segment button if both paths are selected"""
        if self.image_path and self.export_folder and self.predictor:
            self.segment_btn.config(state="normal")
    
    def find_largest_circular_region(self, image):
        """Find the center point of the largest circular object (quarter)"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Detect circles using Hough Circle Transform
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=100,
            param1=50,
            param2=30,
            minRadius=50,
            maxRadius=400
        )
        
        if circles is not None:
            # Get the largest circle
            circles = np.round(circles[0, :]).astype("int")
            largest_circle = max(circles, key=lambda c: c[2])  # Sort by radius
            x, y, r = largest_circle
            return np.array([[x, y]]), np.array([1])  # Return point and label
        else:
            # Fallback: use center of image
            h, w = image.shape[:2]
            return np.array([[w // 2, h // 2]]), np.array([1])
    
    def segment_quarter(self):
        """Perform segmentation using SAM2"""
        try:
            self.progress_label.config(text="Loading image...")
            self.root.update()
            
            # Load image
            image = cv2.imread(self.image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            self.progress_label.config(text="Finding quarter...")
            self.root.update()
            
            # Find the quarter (largest circular object)
            input_point, input_label = self.find_largest_circular_region(image_rgb)
            
            self.progress_label.config(text="Segmenting with SAM2...")
            self.root.update()
            
            # Set image in predictor
            self.predictor.set_image(image_rgb)
            
            # Predict mask
            masks, scores, logits = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True
            )
            
            # Get the best mask (highest score)
            best_mask_idx = np.argmax(scores)
            mask = masks[best_mask_idx]
            
            self.progress_label.config(text="Creating overlay...")
            self.root.update()
            
            # Create colored overlay
            overlay = image_rgb.copy().astype(np.float32)
            color = np.array([0, 255, 0], dtype=np.float32)  # Green mask
            
            # Apply mask overlay
            mask_bool = mask.astype(bool)
            overlay[mask_bool] = overlay[mask_bool] * 0.5 + color * 0.5
            
            # Save result
            output_filename = Path(self.image_path).stem + "_segmented.jpg"
            output_path = Path(self.export_folder) / output_filename
            
            # Convert back to BGR for saving
            overlay_bgr = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), overlay_bgr)
            
            self.progress_label.config(text=f"✓ Saved to: {output_filename}", fg="green")
            messagebox.showinfo("Success", f"Segmentation complete!\n\nSaved to:\n{output_path}")
            
        except Exception as e:
            self.progress_label.config(text="Error occurred", fg="red")
            messagebox.showerror("Error", f"Segmentation failed:\n{str(e)}")

def main():
    root = tk.Tk()
    app = QuarterSegmenterApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
