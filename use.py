import os
import tkinter as tk
from tkinter import messagebox
import numpy as np
from PIL import Image, ImageDraw
import torch
from utils import preprocess_image, DiagonNet

# Configuration
CANVAS_SIZE = 400
GRID_SIZE = 100
BRUSH_SIZE = 16
MODEL_PATH = "model.pth"
HIDDEN_LAYERS = [256, 128, 64]
DIGITS = [1, 2, 3, 4, 5, 6, 7, 8, 9]

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Digit Recognizer (PyTorch GPU)")
        self.root.resizable(False, False)
        
        # Dark Theme Palette
        self.bg_color = "#121212"
        self.card_color = "#1E1E1E"
        self.text_color = "#FFFFFF"
        self.accent_color = "#8A2BE2"  # Purple
        self.button_color = "#2D2D2D"
        self.button_hover = "#3D3D3D"
        self.success_color = "#00FF7F"
        
        self.root.configure(bg=self.bg_color)
        
        # State variables
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.classes = DIGITS
        self.load_model()
        
        # Drawing state
        self.last_x = None
        self.last_y = None
        
        # Initialize PIL image (grayscale)
        self.pil_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), "black")
        self.draw = ImageDraw.Draw(self.pil_image)
        
        self.create_ui()
        self.bind_events()
        
    def load_model(self):
        if not os.path.exists(MODEL_PATH):
            root.withdraw() # Hide main window
            messagebox.showerror(
                "Model Not Found", 
                f"Could not find PyTorch model weights '{MODEL_PATH}'.\n\nPlease run 'train.py' to train on your GPU first!"
            )
            os._exit(0)
            
        try:
            # Recreate model architecture and load state_dict
            self.model = DiagonNet(
                input_size=GRID_SIZE, 
                hidden_dims=HIDDEN_LAYERS, 
                num_classes=10
            )
            # Load weights
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device, weights_only=True))
            self.model.to(self.device)
            self.model.eval()
            print(f"Loaded PyTorch model weights successfully on device: {self.device}")
        except Exception as e:
            root.withdraw()
            messagebox.showerror("Error", f"Failed to load the PyTorch model: {e}")
            os._exit(0)

    def create_ui(self):
        # Header Frame
        header_frame = tk.Frame(self.root, bg=self.bg_color, pady=15)
        header_frame.pack(fill=tk.X)
        
        self.title_label = tk.Label(
            header_frame,
            text=f"Real-Time Digit Recognition ({self.device.type.upper()})",
            font=("Helvetica", 16, "bold"),
            bg=self.bg_color,
            fg=self.accent_color
        )
        self.title_label.pack()
        
        self.inst_label = tk.Label(
            header_frame,
            text="Draw a digit inside the canvas. It will be preprocessed and classified in real time.",
            font=("Helvetica", 10),
            bg=self.bg_color,
            fg="#AAAAAA"
        )
        self.inst_label.pack(pady=2)

        # Main Workspace: 2-column layout (Drawing on Left, Stats on Right)
        workspace = tk.Frame(self.root, bg=self.bg_color)
        workspace.pack(padx=20, pady=5)

        # LEFT COLUMN: Canvas Card
        left_card = tk.Frame(workspace, bg=self.card_color, padx=15, pady=15)
        left_card.pack(side=tk.LEFT, padx=(0, 10))

        canvas_border = tk.Frame(left_card, bg=self.accent_color, bd=2)
        canvas_border.pack()

        self.canvas = tk.Canvas(
            canvas_border,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="black",
            highlightthickness=0,
            cursor="pencil"
        )
        self.canvas.pack()

        self.canvas_text = self.canvas.create_text(
            CANVAS_SIZE // 2,
            CANVAS_SIZE // 2,
            text="Draw digit here",
            fill="#444444",
            font=("Helvetica", 14, "italic")
        )

        # Clear button under canvas
        self.btn_clear = tk.Button(
            left_card,
            text="Clear Canvas (C)",
            command=self.clear_canvas,
            font=("Helvetica", 11, "bold"),
            bg=self.button_color,
            fg=self.text_color,
            activebackground=self.button_hover,
            activeforeground=self.text_color,
            bd=0,
            pady=8
        )
        self.btn_clear.pack(fill=tk.X, pady=(15, 0))
        self.btn_clear.bind("<Enter>", lambda e: self.btn_clear.configure(bg=self.button_hover))
        self.btn_clear.bind("<Leave>", lambda e: self.btn_clear.configure(bg=self.button_color))

        # RIGHT COLUMN: Predictions Display
        right_card = tk.Frame(workspace, bg=self.card_color, padx=20, pady=15, width=280)
        right_card.pack(side=tk.LEFT, fill=tk.Y)
        right_card.pack_propagate(False)

        # Big prediction output
        pred_label_title = tk.Label(
            right_card,
            text="Prediction:",
            font=("Helvetica", 12),
            bg=self.card_color,
            fg="#AAAAAA"
        )
        pred_label_title.pack(anchor=tk.W)

        self.pred_val_label = tk.Label(
            right_card,
            text="?",
            font=("Helvetica", 64, "bold"),
            bg=self.card_color,
            fg=self.success_color
        )
        self.pred_val_label.pack(pady=5)

        # Confidences bar title
        conf_label_title = tk.Label(
            right_card,
            text="Confidence Scores:",
            font=("Helvetica", 11, "bold"),
            bg=self.card_color,
            fg=self.text_color
        )
        conf_label_title.pack(anchor=tk.W, pady=(10, 5))

        # We will dynamically create a bar row for each class
        self.bar_rows = {}
        for digit in self.classes:
            row_frame = tk.Frame(right_card, bg=self.card_color)
            row_frame.pack(fill=tk.X, pady=3)

            lbl_digit = tk.Label(
                row_frame,
                text=f"{digit}:",
                font=("Helvetica", 10, "bold"),
                bg=self.card_color,
                fg="#CCCCCC",
                width=3,
                anchor=tk.W
            )
            lbl_digit.pack(side=tk.LEFT)

            # Container for the progress bar
            bar_bg = tk.Frame(row_frame, bg="#222222", height=12, width=150)
            bar_bg.pack(side=tk.LEFT, padx=5)
            bar_bg.pack_propagate(False)

            # The actual colored bar
            bar_fill = tk.Frame(bar_bg, bg=self.accent_color, height=12, width=0)
            bar_fill.pack(side=tk.LEFT)

            # Percentage label
            lbl_pct = tk.Label(
                row_frame,
                text="0%",
                font=("Helvetica", 9),
                bg=self.card_color,
                fg="#888888",
                width=5,
                anchor=tk.E
            )
            lbl_pct.pack(side=tk.LEFT)

            # Store references to update later
            self.bar_rows[digit] = {
                "fill": bar_fill,
                "pct": lbl_pct
            }

        # Footer
        footer_label = tk.Label(
            self.root,
            text=f"Hardware Acceleration: {self.device.type.upper()}",
            font=("Helvetica", 9),
            bg=self.bg_color,
            fg="#555555",
            pady=10
        )
        footer_label.pack()

    def bind_events(self):
        # Draw on canvas
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        # Keyboard binds
        self.root.bind("c", lambda event: self.clear_canvas())
        self.root.bind("C", lambda event: self.clear_canvas())
        self.root.bind("<Escape>", lambda event: self.clear_canvas())

    def draw_line(self, event):
        if self.canvas_text:
            self.canvas.delete(self.canvas_text)
            self.canvas_text = None
            
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            self.canvas.create_line(
                self.last_x, self.last_y, x, y,
                width=BRUSH_SIZE, fill="white", capstyle=tk.ROUND, joinstyle=tk.ROUND
            )
            self.draw.line(
                [self.last_x, self.last_y, x, y],
                fill="white", width=BRUSH_SIZE
            )
            
        self.last_x = x
        self.last_y = y
        
        # Run prediction on active drawing
        self.predict_digit()

    def on_release(self, event):
        self.last_x = None
        self.last_y = None
        self.predict_digit()

    def clear_canvas(self):
        self.canvas.delete("all")
        self.canvas_text = self.canvas.create_text(
            CANVAS_SIZE // 2,
            CANVAS_SIZE // 2,
            text="Draw digit here",
            fill="#444444",
            font=("Helvetica", 14, "italic")
        )
        self.pil_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), "black")
        self.draw = ImageDraw.Draw(self.pil_image)
        
        # Reset prediction display
        self.pred_val_label.configure(text="?")
        for digit in self.classes:
            self.bar_rows[digit]["fill"].configure(width=0)
            self.bar_rows[digit]["pct"].configure(text="0%", fg="#888888")

    def predict_digit(self):
        extrema = self.pil_image.getextrema()
        if extrema == (0, 0) or extrema is None:
            return
            
        # Double-resize to match the training data pipeline
        resized_100 = self.pil_image.resize((GRID_SIZE, GRID_SIZE), Image.Resampling.LANCZOS)
        preprocessed = preprocess_image(resized_100, target_size=GRID_SIZE)
        
        # Normalize and flatten (100x100 -> 10000)
        img_array = (np.array(preprocessed, dtype=np.float32) / 255.0).flatten()
        
        try:
            # Convert to PyTorch tensor and move to GPU
            tensor_img = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(tensor_img)
                # Compute softmax probability distributions
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                
                # Retrieve best prediction class strictly among classes 1-9
                # Output logits are size 10 (indices 0..9). 
                # argmax over indices 1..9 ensures index 0 (unused) is never selected
                prediction = int(torch.argmax(outputs[0, 1:]).item() + 1)
                
            self.pred_val_label.configure(text=str(prediction))
            
            for digit in self.classes:
                prob = probabilities[digit]
                percentage = int(prob * 100)
                
                # Update progress bar width (max width is 150px)
                pixel_width = int(prob * 150)
                
                # Color highlighting for high confidence
                if digit == prediction:
                    bar_color = self.success_color
                    text_color = self.success_color
                else:
                    bar_color = self.accent_color
                    text_color = "#888888"
                    
                self.bar_rows[digit]["fill"].configure(width=pixel_width, bg=bar_color)
                self.bar_rows[digit]["pct"].configure(text=f"{percentage}%", fg=text_color)
                
        except Exception as e:
            print(f"Error predicting digit: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
