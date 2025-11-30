import cv2
import numpy as np
from PIL import Image


def load_image(path):
    """
    Loads an image from disk and returns a NumPy array (RGB).
    """
    img = Image.open(path).convert("RGB")
    return np.array(img)


def resize_background(bg_img, target_height, target_width):
    """
    Resize the background image to match the foreground dimensions.
    """
    bg_resized = cv2.resize(bg_img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    return bg_resized


def alpha_blend(fg, bg, mask):
    """
    Performs alpha blending between a foreground (with removed background) and a new background.

    fg: foreground image (H×W×3)
    bg: background image (H×W×3)
    mask: binary or soft mask (H×W), where 1 = foreground, 0 = background
    """
    if mask.ndim == 2:
        mask = np.expand_dims(mask, axis=-1)

    mask = mask.astype(np.float32)
    fg = fg.astype(np.float32)
    bg = bg.astype(np.float32)

    blended = fg * mask + bg * (1 - mask)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return blended


def apply_background_replacement(foreground_img, background_img, foreground_mask):
    """
    High‑level wrapper to:
    1. Resize background image to match foreground
    2. Blend using the existing mask from the pipeline
    """
    h, w, _ = foreground_img.shape
    bg_resized = resize_background(background_img, h, w)

    result = alpha_blend(foreground_img, bg_resized, foreground_mask)
    return result