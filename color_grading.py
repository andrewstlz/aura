"""
Color Grading Filters
Professional film-style color grading with 8 preset filters
"""

import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")


@dataclass
class ColorGradingConfig:
    filter_name: str = "none"
    intensity: float = 1.0  # 0.0 = no effect, 1.0 = full effect


class ColorGradingFilters:
    """Professional color grading filters for photo enhancement"""
    
    def __init__(self):
        print("Color grading filters initialized")
    
    def apply_warm_sunset(self, image: np.ndarray, intensity: float = 1.0) -> np.ndarray:
        """Golden hour warm tones - orange/yellow enhancement"""
        # Convert to LAB for better color control
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        l, a, b = cv2.split(lab)
        
        # Shift toward warm (increase b channel = more yellow)
        a = a + 10 * intensity  # Slight red shift
        b = b + 20 * intensity  # Strong yellow shift
        
        # Slight exposure boost
        l = l * (1 + 0.1 * intensity)
        
        # Merge and convert back
        lab = cv2.merge([np.clip(l, 0, 255), np.clip(a, 0, 255), np.clip(b, 0, 255)])
        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        
        # Add orange tint in highlights
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        
        # Boost saturation in mid-tones
        s = s * (1 + 0.2 * intensity)
        
        hsv = cv2.merge([h, np.clip(s, 0, 255), v])
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return result
    
    def apply_cool_blue(self, image: np.ndarray, intensity: float = 1.0) -> np.ndarray:
        """Cinematic cool blue tones - cyan/blue enhancement"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        l, a, b = cv2.split(lab)
        
        # Shift toward cool (decrease b = more blue)
        a = a - 5 * intensity   # Slight cyan shift
        b = b - 15 * intensity  # Strong blue shift
        
        # Slight contrast boost
        l = (l - 128) * (1 + 0.15 * intensity) + 128
        
        lab = cv2.merge([np.clip(l, 0, 255), np.clip(a, 0, 255), np.clip(b, 0, 255)])
        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        
        return result
    
    def apply_vintage_film(self, image: np.ndarray, intensity: float = 1.0) -> np.ndarray:
        """Vintage film look - faded colors, lifted blacks, grain"""
        # Reduce contrast (lift blacks, lower highlights)
        img_float = image.astype(np.float32) / 255.0
        
        # Lift shadows (add to darks)
        img_float = img_float * (1 - 0.2 * intensity) + 0.2 * intensity
        
        # Reduce saturation
        hsv = cv2.cvtColor((img_float * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        s = s * (1 - 0.3 * intensity)
        
        hsv = cv2.merge([h, s, v])
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # Add slight yellow/sepia tint
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
        l, a, b = cv2.split(lab)
        b = b + 10 * intensity  # Yellow tint
        lab = cv2.merge([l, a, np.clip(b, 0, 255)])
        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        
        # Add film grain
        if intensity > 0.5:
            noise = np.random.normal(0, 3 * intensity, result.shape).astype(np.int16)
            result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return result
    
    def apply_high_contrast(self, image: np.ndarray, intensity: float = 1.0) -> np.ndarray:
        """Dramatic high contrast - deep blacks, bright highlights"""
        # S-curve contrast
        img_float = image.astype(np.float32) / 255.0
        
        # Apply contrast curve
        contrast_factor = 1.0 + 0.5 * intensity
        img_float = (img_float - 0.5) * contrast_factor + 0.5
        img_float = np.clip(img_float, 0, 1)
        
        result = (img_float * 255).astype(np.uint8)
        
        # Boost saturation slightly
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        s = s * (1 + 0.3 * intensity)
        
        hsv = cv2.merge([h, np.clip(s, 0, 255), v])
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return result
    
    def apply_soft_pastel(self, image: np.ndarray, intensity: float = 1.0) -> np.ndarray:
        """Soft pastel tones - light, dreamy colors"""
        # Lift overall exposure
        img_float = image.astype(np.float32) / 255.0
        img_float = img_float * (1 - 0.15 * intensity) + 0.15 * intensity
        
        result = (np.clip(img_float, 0, 1) * 255).astype(np.uint8)
        
        # Reduce saturation for pastel effect
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        s = s * (1 - 0.4 * intensity)
        
        hsv = cv2.merge([h, s, v])
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # Slight pink/warm tint
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
        l, a, b = cv2.split(lab)
        a = a + 5 * intensity
        b = b + 5 * intensity
        lab = cv2.merge([l, np.clip(a, 0, 255), np.clip(b, 0, 255)])
        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        
        return result
    
    def apply_cinematic(self, image: np.ndarray, intensity: float = 1.0) -> np.ndarray:
        """Cinematic movie look - teal shadows, orange highlights"""
        # Split toning: teal in shadows, orange in highlights
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        l, a, b = cv2.split(lab)
        
        # Create luminance mask (0 = dark, 1 = bright)
        lum_mask = l / 255.0
        
        # Shadows: push toward teal (cyan + slight green)
        shadow_strength = (1 - lum_mask) * intensity
        a = a - 10 * shadow_strength  # Cyan
        b = b - 5 * shadow_strength   # Slight blue
        
        # Highlights: push toward orange
        highlight_strength = lum_mask * intensity
        a = a + 8 * highlight_strength   # Red
        b = b + 12 * highlight_strength  # Yellow
        
        # Increase contrast slightly
        l = (l - 128) * (1 + 0.2 * intensity) + 128
        
        lab = cv2.merge([np.clip(l, 0, 255), np.clip(a, 0, 255), np.clip(b, 0, 255)])
        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        
        return result
    
    def apply_moody_dark(self, image: np.ndarray, intensity: float = 1.0) -> np.ndarray:
        """Moody dark tones - crushed blacks, muted colors"""
        # Reduce overall exposure
        img_float = image.astype(np.float32) / 255.0
        img_float = img_float * (1 - 0.25 * intensity)
        
        result = (np.clip(img_float, 0, 1) * 255).astype(np.uint8)
        
        # Crush blacks (make darks pure black)
        result = np.where(result < 30 * (1 + intensity), 
                         (result * 0.3).astype(np.uint8), 
                         result)
        
        # Desaturate
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        s = s * (1 - 0.3 * intensity)
        
        hsv = cv2.merge([h, s, v])
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # Add cool tint
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
        l, a, b = cv2.split(lab)
        b = b - 8 * intensity  # Blue tint
        lab = cv2.merge([l, a, np.clip(b, 0, 255)])
        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        
        return result
    
    def apply_bright_airy(self, image: np.ndarray, intensity: float = 1.0) -> np.ndarray:
        """Bright airy look - lifted exposure, reduced contrast"""
        # Lift overall exposure significantly
        img_float = image.astype(np.float32) / 255.0
        img_float = img_float * (1 - 0.1 * intensity) + 0.25 * intensity
        
        result = (np.clip(img_float, 0, 1) * 255).astype(np.uint8)
        
        # Reduce contrast (flatten curve)
        result_float = result.astype(np.float32) / 255.0
        result_float = (result_float - 0.5) * (1 - 0.3 * intensity) + 0.5
        result = (np.clip(result_float, 0, 1) * 255).astype(np.uint8)
        
        # Slight desaturation
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        s = s * (1 - 0.2 * intensity)
        
        hsv = cv2.merge([h, s, v])
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # Add slight warm tint
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
        l, a, b = cv2.split(lab)
        b = b + 8 * intensity  # Warm
        lab = cv2.merge([l, a, np.clip(b, 0, 255)])
        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        
        return result
    
    def apply_filter(self, image: np.ndarray, filter_name: str, intensity: float = 1.0) -> np.ndarray:
        """Apply specified filter with given intensity"""
        if filter_name == "warm_sunset" or filter_name == "warm":
            return self.apply_warm_sunset(image, intensity)
        elif filter_name == "cool_blue" or filter_name == "cool":
            return self.apply_cool_blue(image, intensity)
        elif filter_name == "vintage_film" or filter_name == "vintage":
            return self.apply_vintage_film(image, intensity)
        elif filter_name == "high_contrast" or filter_name == "contrast":
            return self.apply_high_contrast(image, intensity)
        elif filter_name == "soft_pastel" or filter_name == "pastel":
            return self.apply_soft_pastel(image, intensity)
        elif filter_name == "cinematic" or filter_name == "movie":
            return self.apply_cinematic(image, intensity)
        elif filter_name == "moody_dark" or filter_name == "moody":
            return self.apply_moody_dark(image, intensity)
        elif filter_name == "bright_airy" or filter_name == "airy":
            return self.apply_bright_airy(image, intensity)
        else:
            print(f"Unknown filter: {filter_name}, returning original")
            return image
    
    def apply_color_grading(self, image_path: str, output_path: str, 
                           config: Optional[ColorGradingConfig] = None) -> bool:
        """Apply color grading filter to image"""
        if config is None:
            config = ColorGradingConfig()
        
        try:
            print("\n" + "="*60)
            print("COLOR GRADING")
            print("="*60)
            
            print("Loading image...")
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot load image: {image_path}")
            
            print(f"  Image shape: {image.shape}")
            print(f"  Filter: {config.filter_name}")
            print(f"  Intensity: {config.intensity}")
            
            print("Applying color grade...")
            result = self.apply_filter(image, config.filter_name, config.intensity)
            
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


# Preset filter configurations
PRESET_FILTERS = {
    'warm_sunset': ColorGradingConfig(
        filter_name='warm_sunset',
        intensity=0.8
    ),
    'cool_blue': ColorGradingConfig(
        filter_name='cool_blue',
        intensity=0.7
    ),
    'vintage_film': ColorGradingConfig(
        filter_name='vintage_film',
        intensity=0.8
    ),
    'high_contrast': ColorGradingConfig(
        filter_name='high_contrast',
        intensity=0.7
    ),
    'soft_pastel': ColorGradingConfig(
        filter_name='soft_pastel',
        intensity=0.8
    ),
    'cinematic': ColorGradingConfig(
        filter_name='cinematic',
        intensity=0.8
    ),
    'moody_dark': ColorGradingConfig(
        filter_name='moody_dark',
        intensity=0.7
    ),
    'bright_airy': ColorGradingConfig(
        filter_name='bright_airy',
        intensity=0.8
    )
}


if __name__ == "__main__":
    print("Color grading filters module - 8 professional film-style filters")