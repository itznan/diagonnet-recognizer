import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw

# Configuration
CANVAS_SIZE = 400
GRID_SIZE = 100
BRUSH_SIZE = 16
NUM_ATTEMPTS = 30
DIGITS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
DATA_DIR = "data"

class DataCollectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Data Collector (100x100 Grid)")
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
        self.current_digit_index = 0
        self.current_attempt = 1
        
        # Drawing state
        self.last_x = None
        self.last_y = None
        
        # Initialize PIL image for drawing (grayscale)
        self.pil_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), "black")
        self.draw = ImageDraw.Draw(self.pil_image)
        
        self.create_ui()
        self.bind_events()
        self.update_status()

    def create_ui(self):
        # Header Frame
        header_frame = tk.Frame(self.root, bg=self.bg_color, pady=15)
        header_frame.pack(fill=tk.X)
        
        self.title_label = tk.Label(
            header_frame,
            text="Digit Drawing Dataset Collector",
            font=("Helvetica", 16, "bold"),
            bg=self.bg_color,
            fg=self.accent_color
        )
        self.title_label.pack()
        
        self.inst_label = tk.Label(
            header_frame,
            text="Draw the digit shown below inside the canvas.",
            font=("Helvetica", 10),
            bg=self.bg_color,
            fg="#AAAAAA"
        )
        self.inst_label.pack(pady=2)

        # Main Workspace (Card Style)
        card_frame = tk.Frame(self.root, bg=self.card_color, bd=0, padx=20, pady=20)
        card_frame.pack(padx=20, pady=10)

        # Target digit prompt
        self.prompt_frame = tk.Frame(card_frame, bg=self.card_color)
        self.prompt_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.digit_prompt_label = tk.Label(
            self.prompt_frame,
            text="Draw Digit:",
            font=("Helvetica", 14),
            bg=self.card_color,
            fg="#CCCCCC"
        )
        self.digit_prompt_label.pack(side=tk.LEFT)
        
        self.digit_val_label = tk.Label(
            self.prompt_frame,
            text="1",
            font=("Helvetica", 28, "bold"),
            bg=self.card_color,
            fg=self.success_color
        )
        self.digit_val_label.pack(side=tk.LEFT, padx=10)
        
        self.attempt_label = tk.Label(
            self.prompt_frame,
            text="Attempt: 1 / 30",
            font=("Helvetica", 12),
            bg=self.card_color,
            fg="#999999"
        )
        self.attempt_label.pack(side=tk.RIGHT, pady=10)

        # Drawing Canvas Frame
        canvas_border = tk.Frame(card_frame, bg=self.accent_color, bd=2)
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

        # Canvas overlay instructions (fade out on draw)
        self.canvas_text = self.canvas.create_text(
            CANVAS_SIZE // 2,
            CANVAS_SIZE // 2,
            text="Draw here\n(Hold Left Mouse Button)",
            fill="#444444",
            font=("Helvetica", 14, "italic"),
            justify=tk.CENTER
        )

        # Progress bar
        self.progress_canvas = tk.Canvas(
            card_frame,
            width=CANVAS_SIZE,
            height=6,
            bg="#222222",
            highlightthickness=0
        )
        self.progress_canvas.pack(pady=(15, 0))
        self.progress_bar = self.progress_canvas.create_rectangle(
            0, 0, 0, 6, fill=self.accent_color, width=0
        )

        # Control Panel
        control_frame = tk.Frame(self.root, bg=self.bg_color, pady=15)
        control_frame.pack(fill=tk.X, padx=20)
        
        # Styled Buttons
        self.btn_clear = tk.Button(
            control_frame,
            text="Clear (C)",
            command=self.clear_canvas,
            font=("Helvetica", 11, "bold"),
            bg=self.button_color,
            fg=self.text_color,
            activebackground=self.button_hover,
            activeforeground=self.text_color,
            bd=0,
            padx=15,
            pady=8,
            width=10
        )
        self.btn_clear.pack(side=tk.LEFT, padx=5)

        self.btn_back = tk.Button(
            control_frame,
            text="Undo / Back",
            command=self.go_back,
            font=("Helvetica", 11, "bold"),
            bg=self.button_color,
            fg="#FF5555",
            activebackground=self.button_hover,
            activeforeground="#FF5555",
            bd=0,
            padx=15,
            pady=8,
            width=10
        )
        self.btn_back.pack(side=tk.LEFT, padx=5)

        self.btn_save = tk.Button(
            control_frame,
            text="Save & Next (Space)",
            command=self.save_and_next,
            font=("Helvetica", 11, "bold"),
            bg=self.accent_color,
            fg=self.text_color,
            activebackground="#A020F0",
            activeforeground=self.text_color,
            bd=0,
            padx=20,
            pady=8
        )
        self.btn_save.pack(side=tk.RIGHT, padx=5)

        # Bind hovers
        self.btn_clear.bind("<Enter>", lambda e: self.btn_clear.configure(bg=self.button_hover))
        self.btn_clear.bind("<Leave>", lambda e: self.btn_clear.configure(bg=self.button_color))
        self.btn_back.bind("<Enter>", lambda e: self.btn_back.configure(bg=self.button_hover))
        self.btn_back.bind("<Leave>", lambda e: self.btn_back.configure(bg=self.button_color))
        self.btn_save.bind("<Enter>", lambda e: self.btn_save.configure(bg="#9B30FF"))
        self.btn_save.bind("<Leave>", lambda e: self.btn_save.configure(bg=self.accent_color))

        # Footer Status Label
        self.status_label = tk.Label(
            self.root,
            text="Draw a clear digit in the center.",
            font=("Helvetica", 9),
            bg=self.bg_color,
            fg="#777777",
            pady=10
        )
        self.status_label.pack()

    def bind_events(self):
        # Canvas drawing binds
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.reset_coordinates)
        
        # Keyboard shortcuts
        self.root.bind("<space>", lambda event: self.save_and_next())
        self.root.bind("<Return>", lambda event: self.save_and_next())
        self.root.bind("c", lambda event: self.clear_canvas())
        self.root.bind("C", lambda event: self.clear_canvas())
        self.root.bind("<Escape>", lambda event: self.clear_canvas())

    def draw_line(self, event):
        # Hide instructions on first draw
        if self.canvas_text:
            self.canvas.delete(self.canvas_text)
            self.canvas_text = None
            
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            # Draw on Tkinter canvas
            self.canvas.create_line(
                self.last_x, self.last_y, x, y,
                width=BRUSH_SIZE, fill="white", capstyle=tk.ROUND, joinstyle=tk.ROUND
            )
            # Draw on PIL image for saving
            self.draw.line(
                [self.last_x, self.last_y, x, y],
                fill="white", width=BRUSH_SIZE
            )
            
        self.last_x = x
        self.last_y = y

    def reset_coordinates(self, event):
        self.last_x = None
        self.last_y = None

    def clear_canvas(self):
        self.canvas.delete("all")
        # Re-draw the canvas instruction text if it was deleted
        self.canvas_text = self.canvas.create_text(
            CANVAS_SIZE // 2,
            CANVAS_SIZE // 2,
            text="Draw here\n(Hold Left Mouse Button)",
            fill="#444444",
            font=("Helvetica", 14, "italic"),
            justify=tk.CENTER
        )
        # Re-initialize PIL image
        self.pil_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), "black")
        self.draw = ImageDraw.Draw(self.pil_image)

    def get_current_digit(self):
        if self.current_digit_index < len(DIGITS):
            return DIGITS[self.current_digit_index]
        return None

    def update_status(self):
        digit = self.get_current_digit()
        if digit is not None:
            self.digit_val_label.configure(text=str(digit))
            self.attempt_label.configure(text=f"Attempt: {self.current_attempt} / {NUM_ATTEMPTS}")
            
            # Progress bar update
            total_drawings = len(DIGITS) * NUM_ATTEMPTS
            completed = self.current_digit_index * NUM_ATTEMPTS + (self.current_attempt - 1)
            progress_width = (completed / total_drawings) * CANVAS_SIZE
            self.progress_canvas.coords(self.progress_bar, 0, 0, progress_width, 6)
            
            self.status_label.configure(text=f"Total Progress: {completed} / {total_drawings} saved.")
        else:
            # Done!
            self.digit_val_label.configure(text="Done!", fg=self.success_color)
            self.attempt_label.configure(text="")
            self.canvas.delete("all")
            self.canvas.create_text(
                CANVAS_SIZE // 2,
                CANVAS_SIZE // 2,
                text="Dataset Collection Complete!\n\nYou can now run train.py to train the model.",
                fill=self.success_color,
                font=("Helvetica", 12, "bold"),
                justify=tk.CENTER
            )
            self.canvas.unbind("<B1-Motion>")
            self.canvas.unbind("<ButtonRelease-1>")
            self.btn_save.configure(state=tk.DISABLED)
            self.btn_clear.configure(state=tk.DISABLED)
            self.status_label.configure(text="All data saved! Run python train.py next.")

    def save_and_next(self):
        digit = self.get_current_digit()
        if digit is None:
            return
            
        # Check if the canvas is empty (basic check: if we haven't deleted the instruction text or the image is all black)
        # Let's check if the image has any non-zero pixels
        extrema = self.pil_image.getextrema()
        if extrema == (0, 0) or extrema is None:
            messagebox.showwarning("Empty Canvas", "Please draw something before saving!")
            return

        # Ensure directory exists
        digit_dir = os.path.join(DATA_DIR, str(digit))
        os.makedirs(digit_dir, exist_ok=True)
        
        # Resize to 100x100 and save
        resized_img = self.pil_image.resize((GRID_SIZE, GRID_SIZE), Image.Resampling.LANCZOS)
        file_path = os.path.join(digit_dir, f"{digit}_{self.current_attempt:02d}.png")
        resized_img.save(file_path)
        
        # Advance state
        self.current_attempt += 1
        if self.current_attempt > NUM_ATTEMPTS:
            self.current_attempt = 1
            self.current_digit_index += 1
            
        self.clear_canvas()
        self.update_status()

    def go_back(self):
        # Go back one step
        if self.current_digit_index == 0 and self.current_attempt == 1:
            messagebox.showinfo("First Step", "Already at the first attempt!")
            return
            
        self.current_attempt -= 1
        if self.current_attempt < 1:
            self.current_digit_index -= 1
            self.current_attempt = NUM_ATTEMPTS
            
        digit = self.get_current_digit()
        file_path = os.path.join(DATA_DIR, str(digit), f"{digit}_{self.current_attempt:02d}.png")
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error removing file {file_path}: {e}")
                
        self.clear_canvas()
        # If we had completed and disabled things, we should re-enable them
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.reset_coordinates)
        self.btn_save.configure(state=tk.NORMAL)
        self.btn_clear.configure(state=tk.NORMAL)
        self.update_status()

if __name__ == "__main__":
    root = tk.Tk()
    app = DataCollectorApp(root)
    root.mainloop()
