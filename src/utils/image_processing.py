import numpy as np
from PIL import Image

def preprocess_image(img, target_size=100, padding=12):
    """
    Crops the drawn digit to its bounding box, pads it to a square aspect ratio,
    and resizes it to target_size x target_size.
    This ensures the model is invariant to where the user draws and how large they draw.
    """
    img = img.convert("L")
    bbox = img.getbbox()
    if bbox is None:
        return Image.new("L", (target_size, target_size), "black")
        
    cropped = img.crop(bbox)
    w, h = cropped.size
    max_dim = max(w, h)
    
    square_size = max_dim + padding * 2
    square_img = Image.new("L", (square_size, square_size), "black")
    
    paste_x = padding + (max_dim - w) // 2
    paste_y = padding + (max_dim - h) // 2
    square_img.paste(cropped, (paste_x, paste_y))
    
    return square_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
