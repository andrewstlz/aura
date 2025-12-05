"""
Background Blur / Portrait Mode
Creates professional depth-of-field effect by blurring background while keeping subject sharp
Uses existing background removal mask for segmentation
"""

import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")


@dataclass
class BlurConfig:
    blur_strength: int = 21  # Kernel size (odd number, 5-51)
    blur_type: str = "gaussian"  # "gaussian", "motion", "lens"
    edge_softness: int = 15  # Edge transition softness (1-50)
    foreground_enhance: bool = True  # Sharpen foreground slightly


class BackgroundBlur:
    """Professional background blur effect (portrait mode / bokeh)"""
    
    def __init__(self):
        print("Background blur processor initialized")
    
    def create_depth_mask(self, alpha_mask: np.ndarray, edge_softness: int = 15) -> np.ndarray:
        """
        Create smooth depth mask from binary alpha mask
        
        Args:
            alpha_mask: Binary mask (0-255) where 255 = foreground
            edge_softness: How soft the transition is (higher = softer)
        
        Returns:
            Smooth depth mask (0.0 to 1.0) where 1.0 = foreground
        """
        # Ensure mask is single channel
        if len(alpha_mask.shape) == 3:
            alpha_mask = cv2.cvtColor(alpha_mask, cv2.COLOR_BGR2GRAY)
        
        # Normalize to 0-1
        mask_float = alpha_mask.astype(np.float32) / 255.0
        
        # Create distance transform for smooth falloff
        # This creates a gradient from edges
        mask_binary = (alpha_mask > 128).astype(np.uint8) * 255
        
        # Dilate slightly to expand foreground
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_binary = cv2.dilate(mask_binary, kernel, iterations=1)
        
        # Apply heavy Gaussian blur for smooth transition
        blur_size = edge_softness * 2 + 1  # Ensure odd
        depth_mask = cv2.GaussianBlur(mask_binary.astype(np.float32), 
                                      (blur_size, blur_size), 
                                      edge_softness / 2.0)
        
        # Normalize to 0-1
        depth_mask = depth_mask / 255.0
        
        # Apply gamma curve for more realistic falloff
        depth_mask = np.power(depth_mask, 0.8)
        
        return depth_mask
    
    def apply_gaussian_blur(self, image: np.ndarray, strength: int) -> np.ndarray:
        """Apply Gaussian blur (circular bokeh)"""
        # Ensure strength is odd
        if strength % 2 == 0:
            strength += 1
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (strength, strength), strength / 3.0)
        
        return blurred
    
    def apply_lens_blur(self, image: np.ndarray, strength: int) -> np.ndarray:
        """Apply lens blur (more realistic bokeh with hexagonal shape)"""
        # Ensure strength is odd
        if strength % 2 == 0:
            strength += 1
        
        # First pass: Gaussian for base blur
        blurred = cv2.GaussianBlur(image, (strength, strength), strength / 3.0)
        
        # Second pass: Median filter for lens-like effect
        median_size = max(5, strength // 3)
        if median_size % 2 == 0:
            median_size += 1
        blurred = cv2.medianBlur(blurred, median_size)
        
        # Third pass: Light Gaussian to smooth
        light_blur = max(5, strength // 4)
        if light_blur % 2 == 0:
            light_blur += 1
        blurred = cv2.GaussianBlur(blurred, (light_blur, light_blur), light_blur / 3.0)
        
        return blurred
    
    def apply_motion_blur(self, image: np.ndarray, strength: int, angle: float = 0) -> np.ndarray:
        """Apply directional motion blur"""
        # Create motion blur kernel
        kernel_size = strength
        kernel = np.zeros((kernel_size, kernel_size))
        
        # Create line in the middle
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        
        # Rotate kernel if angle specified
        if angle != 0:
            center = (kernel_size // 2, kernel_size // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            kernel = cv2.warpAffine(kernel, rotation_matrix, (kernel_size, kernel_size))
        
        # Apply kernel
        blurred = cv2.filter2D(image, -1, kernel)
        
        return blurred
    
    def apply_bokeh_blur(self, image: np.ndarray, strength: int) -> np.ndarray:
        """
        Apply bokeh-style blur with multiple passes for realistic depth-of-field
        This creates the most realistic DSLR-like background blur
        """
        # Ensure strength is odd
        if strength % 2 == 0:
            strength += 1
        
        # Multiple blur passes with different kernels simulate lens bokeh
        result = image.copy()
        
        # Pass 1: Box blur (simulates aperture shape)
        box_size = max(5, strength // 2)
        if box_size % 2 == 0:
            box_size += 1
        result = cv2.blur(result, (box_size, box_size))
        
        # Pass 2: Gaussian blur (smooth bokeh)
        gauss_size = max(7, strength // 2 + strength // 4)
        if gauss_size % 2 == 0:
            gauss_size += 1
        result = cv2.GaussianBlur(result, (gauss_size, gauss_size), gauss_size / 3.0)
        
        # Pass 3: Another Gaussian for smoothness
        result = cv2.GaussianBlur(result, (strength, strength), strength / 3.0)
        
        return result
    
    def sharpen_foreground(self, image: np.ndarray, amount: float = 0.3) -> np.ndarray:
        """Slightly sharpen foreground to enhance separation"""
        # Unsharp mask
        blurred = cv2.GaussianBlur(image, (0, 0), 2.0)
        sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
        
        return sharpened
    
    def apply_background_blur(self, image: np.ndarray, alpha_mask: np.ndarray,
                              config: Optional[BlurConfig] = None) -> np.ndarray:
        """
        Main function: Apply background blur with smooth transitions
        
        Args:
            image: Input image (BGR)
            alpha_mask: Alpha mask from background removal (0-255, 255=foreground)
            config: Blur configuration
        
        Returns:
            Image with blurred background
        """
        if config is None:
            config = BlurConfig()
        
        print("  Creating depth mask...")
        depth_mask = self.create_depth_mask(alpha_mask, config.edge_softness)
        
        # Expand depth mask to 3 channels
        depth_mask_3ch = np.stack([depth_mask] * 3, axis=2)
        
        print(f"  Applying {config.blur_type} blur (strength={config.blur_strength})...")
        
        # Apply selected blur type to entire image
        if config.blur_type == "lens":
            blurred_bg = self.apply_lens_blur(image, config.blur_strength)
        elif config.blur_type == "bokeh":
            blurred_bg = self.apply_bokeh_blur(image, config.blur_strength)
        elif config.blur_type == "motion":
            blurred_bg = self.apply_motion_blur(image, config.blur_strength)
        else:  # gaussian (default)
            blurred_bg = self.apply_gaussian_blur(image, config.blur_strength)
        
        # Optionally sharpen foreground
        if config.foreground_enhance:
            print("  Enhancing foreground sharpness...")
            foreground_sharp = self.sharpen_foreground(image, amount=0.3)
        else:
            foreground_sharp = image
        
        print("  Blending with depth mask...")
        # Blend: foreground (sharp) where mask=1, background (blur) where mask=0
        result = (depth_mask_3ch * foreground_sharp.astype(np.float32) + 
                 (1 - depth_mask_3ch) * blurred_bg.astype(np.float32))
        
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def process_image(self, image_path: str, alpha_mask_path: str, output_path: str,
                     config: Optional[BlurConfig] = None) -> bool:
        """
        Process image with background blur
        
        Args:
            image_path: Path to input image
            alpha_mask_path: Path to alpha mask (from background removal)
            output_path: Path to save output
            config: Blur configuration
        
        Returns:
            Success status
        """
        if config is None:
            config = BlurConfig()
        
        try:
            print("\n" + "="*60)
            print("BACKGROUND BLUR / PORTRAIT MODE")
            print("="*60)
            
            print("Loading image...")
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot load image: {image_path}")
            
            print(f"  Image shape: {image.shape}")
            
            print("Loading alpha mask...")
            alpha_mask = cv2.imread(alpha_mask_path, cv2.IMREAD_GRAYSCALE)
            if alpha_mask is None:
                raise ValueError(f"Cannot load mask: {alpha_mask_path}")
            
            print(f"  Mask shape: {alpha_mask.shape}")
            
            # Apply blur
            result = self.apply_background_blur(image, alpha_mask, config)
            
            print("Saving result...")
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            Image.fromarray(result_rgb).save(output_path, quality=95)
            
            print(f"SUCCESS! Saved to: {output_path}")
            print("="*60 + "\n")
            return True
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return False


# Preset blur configurations
PRESET_BLUR = {
    'subtle': BlurConfig(
        blur_strength=15,
        blur_type='gaussian',
        edge_softness=20,
        foreground_enhance=True
    ),
    'moderate': BlurConfig(
        blur_strength=25,
        blur_type='lens',
        edge_softness=15,
        foreground_enhance=True
    ),
    'strong': BlurConfig(
        blur_strength=35,
        blur_type='bokeh',
        edge_softness=12,
        foreground_enhance=True
    ),
    'portrait': BlurConfig(
        blur_strength=31,
        blur_type='lens',
        edge_softness=18,
        foreground_enhance=True
    ),
    'dramatic': BlurConfig(
        blur_strength=45,
        blur_type='bokeh',
        edge_softness=10,
        foreground_enhance=True
    )
}


if __name__ == "__main__":
    print("Background blur module - creates professional depth-of-field effect")