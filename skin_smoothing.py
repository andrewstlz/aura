import cv2
import numpy as np
from PIL import Image
import logging
from typing import Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

try:
    import face_alignment
    FACE_ALIGNMENT_AVAILABLE = True
except ImportError:
    FACE_ALIGNMENT_AVAILABLE = False
    print("Warning: face_alignment not installed. Install with: pip install face-alignment")


@dataclass
class BeautyConfig:
    smooth_strength: float = 0.6
    brightness_boost: float = 1.05
    contrast_adjustment: float = 1.02
    sharpen_amount: float = 0.3
    skin_tone_balance: float = 0.15
    blemish_removal: bool = True


class SkinSmoothingFilter:
    
    def __init__(self, device: str = 'cpu'):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        if FACE_ALIGNMENT_AVAILABLE:
            print(f"Loading FAN landmark detector on {device}...")
            self.fa = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D,
                device=device,
                flip_input=False
            )
            print("FAN loaded successfully")
        else:
            print("FAN not available, will use full-image processing")
            self.fa = None
    
    def get_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        # Extract facial landmarks if available
        if self.fa is None:
            return None
        
        try:
            if len(image.shape) == 2:
                rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3:
                if image.shape[2] == 4:
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                elif image.shape[2] == 3:
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    return None
            else:
                return None
            
            landmarks_list = self.fa.get_landmarks(rgb)
            
            if landmarks_list is None or len(landmarks_list) == 0:
                return None
            
            return landmarks_list[0][:, :2]
            
        except Exception as e:
            self.logger.error(f"Error extracting landmarks: {e}")
            return None
    
    def create_skin_mask(self, image: np.ndarray, landmarks: Optional[np.ndarray] = None) -> np.ndarray:
        # Generate mask for skin regions using color-based segmentation and optional landmarks
        h, w = image.shape[:2]
        
        if landmarks is not None:
            mask = np.zeros((h, w), dtype=np.uint8)
            
            face_contour = []
            face_contour.extend(landmarks[list(range(0, 17))])
            
            left_brow = landmarks[list(range(17, 22))]
            right_brow = landmarks[list(range(22, 27))]
            
            forehead_offset = int(h * 0.15)
            forehead_points = [
                left_brow[0] + np.array([0, -forehead_offset]),
                left_brow[2] + np.array([0, -forehead_offset * 1.2]),
                right_brow[2] + np.array([0, -forehead_offset * 1.2]),
                right_brow[4] + np.array([0, -forehead_offset]),
            ]
            face_contour.extend(forehead_points)
            
            hull = cv2.convexHull(np.array(face_contour, dtype=np.int32))
            cv2.fillConvexPoly(mask, hull, 255)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
            mask = cv2.dilate(mask, kernel, iterations=1)
            
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            
            skin_mask = cv2.inRange(ycrcb, np.array([0, 130, 70]), np.array([255, 180, 135]))
            
            mask = cv2.bitwise_and(mask, skin_mask)
        else:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            skin_mask = cv2.inRange(ycrcb, np.array([0, 130, 70]), np.array([255, 180, 135]))
            mask = skin_mask
        
        closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel, iterations=2)
        
        mask = cv2.GaussianBlur(mask, (31, 31), 15)
        
        return mask
    
    def edge_preserving_smooth(self, image: np.ndarray, strength: float = 0.6) -> np.ndarray:
        # Apply bilateral filter for smoothing while preserving edges
        d = int(9 * strength)
        if d % 2 == 0:
            d += 1
        d = max(5, min(d, 15))
        
        sigma_color = 75 * strength
        sigma_space = 75 * strength
        
        smoothed = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        
        return smoothed
    
    def guided_filter(self, image: np.ndarray, guide: np.ndarray, radius: int = 8, eps: float = 0.01) -> np.ndarray:
        # Apply guided filter for edge-preserving smoothing
        guide_gray = cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY) if len(guide.shape) == 3 else guide
        guide_gray = guide_gray.astype(np.float32)
        
        if len(image.shape) == 3:
            output = np.zeros_like(image, dtype=np.float32)
            for c in range(3):
                channel = image[:, :, c].astype(np.float32)
                
                mean_I = cv2.boxFilter(guide_gray, cv2.CV_32F, (radius, radius))
                mean_p = cv2.boxFilter(channel, cv2.CV_32F, (radius, radius))
                mean_Ip = cv2.boxFilter(guide_gray * channel, cv2.CV_32F, (radius, radius))
                
                cov_Ip = mean_Ip - mean_I * mean_p
                
                mean_II = cv2.boxFilter(guide_gray * guide_gray, cv2.CV_32F, (radius, radius))
                var_I = mean_II - mean_I * mean_I
                
                a = cov_Ip / (var_I + eps)
                b = mean_p - a * mean_I
                
                mean_a = cv2.boxFilter(a, cv2.CV_32F, (radius, radius))
                mean_b = cv2.boxFilter(b, cv2.CV_32F, (radius, radius))
                
                output[:, :, c] = mean_a * guide_gray + mean_b
            
            return np.clip(output, 0, 255).astype(np.uint8)
        else:
            channel = image.astype(np.float32)
            
            mean_I = cv2.boxFilter(guide_gray, cv2.CV_32F, (radius, radius))
            mean_p = cv2.boxFilter(channel, cv2.CV_32F, (radius, radius))
            mean_Ip = cv2.boxFilter(guide_gray * channel, cv2.CV_32F, (radius, radius))
            
            cov_Ip = mean_Ip - mean_I * mean_p
            
            mean_II = cv2.boxFilter(guide_gray * guide_gray, cv2.CV_32F, (radius, radius))
            var_I = mean_II - mean_I * mean_I
            
            a = cov_Ip / (var_I + eps)
            b = mean_p - a * mean_I
            
            mean_a = cv2.boxFilter(a, cv2.CV_32F, (radius, radius))
            mean_b = cv2.boxFilter(b, cv2.CV_32F, (radius, radius))
            
            output = mean_a * guide_gray + mean_b
            
            return np.clip(output, 0, 255).astype(np.uint8)
    
    def remove_blemishes(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # Detect and reduce small blemishes using morphological operations
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        blemish_candidates = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(blemish_candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        result = image.copy()
        mask_binary = (mask > 128).astype(np.uint8)
        
        total_blemish_mask = np.zeros(gray.shape, dtype=np.uint8)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if 3 < area < 200:
                
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if the blemish is mostly inside the skin mask
                if mask_binary[y:y+h, x:x+w].mean() > 0.5:
                    # Add this contour to our total mask
                    cv2.drawContours(total_blemish_mask, [contour], -1, 255, -1)

        # Now, inpaint everything at once
        if np.any(total_blemish_mask):
            result = cv2.inpaint(result, total_blemish_mask, 3, cv2.INPAINT_TELEA)
        
        return result
    
    def adjust_brightness_contrast(self, image: np.ndarray, brightness: float = 1.0, 
                                     contrast: float = 1.0) -> np.ndarray:
        # Adjust image brightness and contrast in LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        l = l.astype(np.float32)
        l = l * contrast
        l = l + (brightness - 1.0) * 50
        l = np.clip(l, 0, 255).astype(np.uint8)
        
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return result
    
    def balance_skin_tone(self, image: np.ndarray, mask: np.ndarray, strength: float = 0.15) -> np.ndarray:
        # Normalize skin tone color cast while preserving natural variation
        mask_norm = mask.astype(np.float32) / 255.0
        
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        skin_pixels_a = a[mask > 128]
        skin_pixels_b = b[mask > 128]
        
        if len(skin_pixels_a) > 0:
            mean_a = np.median(skin_pixels_a)
            mean_b = np.median(skin_pixels_b)
            
            target_a = 128 + (mean_a - 128) * (1 - strength * 0.5)
            target_b = 128 + (mean_b - 128) * (1 - strength * 0.5)
            
            a_adjusted = a.astype(np.float32)
            b_adjusted = b.astype(np.float32)
            
            a_adjusted = a_adjusted + (target_a - mean_a) * mask_norm * strength
            b_adjusted = b_adjusted + (target_b - mean_b) * mask_norm * strength
            
            a_adjusted = np.clip(a_adjusted, 0, 255).astype(np.uint8)
            b_adjusted = np.clip(b_adjusted, 0, 255).astype(np.uint8)
            
            lab = cv2.merge([l, a_adjusted, b_adjusted])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            return result
        
        return image
    
    def sharpen_details(self, image: np.ndarray, amount: float = 0.3, 
                        mask: Optional[np.ndarray] = None) -> np.ndarray:
        # Apply selective sharpening to non-skin areas like eyes and hair
        if mask is not None:
            inverse_mask = 255 - mask
            inverse_mask = cv2.GaussianBlur(inverse_mask, (15, 15), 8)
            inverse_mask_norm = inverse_mask.astype(np.float32) / 255.0
        else:
            inverse_mask_norm = np.ones((image.shape[0], image.shape[1]), dtype=np.float32)
        
        blurred = cv2.GaussianBlur(image, (0, 0), 2.0)
        sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
        
        result = image.copy().astype(np.float32)
        for c in range(3):
            result[:, :, c] = (inverse_mask_norm * sharpened[:, :, c].astype(np.float32) + 
                             (1 - inverse_mask_norm) * image[:, :, c].astype(np.float32))
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def apply_beauty_filter(self, image_path: str, output_path: str,
                            config: Optional[BeautyConfig] = None) -> bool:
        # Main pipeline: detect face, create skin mask, smooth, enhance, and save
        if config is None:
            config = BeautyConfig()
        
        try:
            print("\n" + "="*60)
            print("SKIN SMOOTHING & BEAUTY FILTER")
            print("="*60)
            
            print("Loading image...")
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                raise ValueError(f"Cannot load image: {image_path}")
            
            print(f"    Image shape: {image_bgr.shape}")
            
            print("Detecting face landmarks...")
            landmarks = self.get_landmarks(image_bgr)
            if landmarks is not None:
                print(f"    Detected {len(landmarks)} landmarks")
            else:
                print("    No landmarks detected, processing full image")
            
            print("Creating skin mask...")
            skin_mask = self.create_skin_mask(image_bgr, landmarks)
            
            result = image_bgr.copy()
            
            if config.blemish_removal:
                print("Removing blemishes...")
                result = self.remove_blemishes(result, skin_mask)
            
            print("Applying edge-preserving smoothing...")
            smoothed = self.edge_preserving_smooth(result, config.smooth_strength)
            
            print("Applying guided filter...")
            strong_radius = 12
            strong_eps = 0.02
            
            for _ in range(3): # Increased from 2 to 3 iterations
                smoothed = self.guided_filter(smoothed, result, 
                                              radius=strong_radius, eps=strong_eps)
            
            print("Blending smoothed skin...")
            mask_norm = skin_mask.astype(np.float32) / 255.0
            for c in range(3):
                result[:, :, c] = (mask_norm * smoothed[:, :, c].astype(np.float32) + 
                                  (1 - mask_norm) * result[:, :, c].astype(np.float32))
            result = result.astype(np.uint8)
            
            print("Balancing skin tone...")
            result = self.balance_skin_tone(result, skin_mask, config.skin_tone_balance)
            
            print("Adjusting brightness and contrast...")
            result = self.adjust_brightness_contrast(result, config.brightness_boost, config.contrast_adjustment)
            
            print("Sharpening details...")
            result = self.sharpen_details(result, config.sharpen_amount, skin_mask)
            
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            
            print("Saving result...")
            Image.fromarray(result_rgb).save(output_path, quality=95)
            
            print(f"SUCCESS! Beauty filter applied: {output_path}")
            print("="*60 + "\n")
            return True
            
        except Exception as e:
            self.logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return False


PRESET_CONFIGS = {
    'subtle': BeautyConfig(
        smooth_strength=0.4,
        brightness_boost=1.03,
        contrast_adjustment=1.01,
        sharpen_amount=0.2,
        skin_tone_balance=0.10,
        blemish_removal=True
    ),
    'natural': BeautyConfig(
        smooth_strength=0.6,
        brightness_boost=1.05,
        contrast_adjustment=1.02,
        sharpen_amount=0.3,
        skin_tone_balance=0.15,
        blemish_removal=True
    ),
    'strong': BeautyConfig(
        smooth_strength=0.8,
        brightness_boost=1.08,
        contrast_adjustment=1.03,
        sharpen_amount=0.4,
        skin_tone_balance=0.20,
        blemish_removal=True
    ),
    'intense': BeautyConfig(
        smooth_strength=2.5,
        brightness_boost=1.05,
        contrast_adjustment=1.02,
        sharpen_amount=0.2,
        skin_tone_balance=0.25,
        blemish_removal=True
    )
}


if __name__ == "__main__":
    INPUT_IMAGE = "/Users/zheng/Desktop/CIS5810Project/trysmoothing.webp"
    OUTPUT_IMAGE = "/Users/zheng/Desktop/CIS5810Project/testsmooth.png"
    
    print("\nInitializing Beauty Filter...")
    beauty_filter = SkinSmoothingFilter(device='cpu') 
    
    config = PRESET_CONFIGS['intense']

    success = beauty_filter.apply_beauty_filter(INPUT_IMAGE, OUTPUT_IMAGE, config)
    
    if success:
        print("\nBeauty filter applied successfully!")
    else:
        print("\nBeauty filter failed.")