"""
Virtual Makeup Application
Uses facial landmarks and advanced blending techniques for realistic makeup effects
"""

import cv2
import numpy as np
from PIL import Image
import logging
from typing import Optional, Tuple, List
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
class MakeupConfig:
    lipstick_color: Tuple[int, int, int] = (180, 38, 45)  # RGB
    lipstick_intensity: float = 0.7
    blush_color: Tuple[int, int, int] = (255, 150, 150)
    blush_intensity: float = 0.3
    eyeshadow_color: Tuple[int, int, int] = (150, 100, 80)
    eyeshadow_intensity: float = 0.4
    eyeliner_color: Tuple[int, int, int] = (20, 20, 20)
    eyeliner_thickness: int = 2
    eyebrow_color: Tuple[int, int, int] = (60, 40, 30)
    eyebrow_intensity: float = 0.3


class VirtualMakeup:
    
    def __init__(self, device: str = 'cpu'):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        if not FACE_ALIGNMENT_AVAILABLE:
            raise ImportError("face_alignment required: pip install face-alignment")
        
        print(f"Loading FAN landmark detector on {device}...")
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            device=device,
            flip_input=False
        )
        print("FAN loaded successfully")
    
    def get_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        # Extract 68 facial landmarks
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
    
    def create_lip_mask(self, landmarks: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        # Generate smooth mask for lip region from landmarks 48-67
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        outer_lip = landmarks[48:60].astype(np.int32)
        inner_lip = landmarks[60:68].astype(np.int32)
        
        cv2.fillPoly(mask, [outer_lip], 255)
        cv2.fillPoly(mask, [inner_lip], 255)
        
        mask = cv2.GaussianBlur(mask, (7, 7), 3)
        
        return mask
    
    def create_blush_mask(self, landmarks: np.ndarray, image_shape: Tuple[int, int], 
                         side: str = 'left') -> np.ndarray:
        # Generate elliptical mask for blush on cheeks
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if side == 'left':
            cheekbone = landmarks[1]
            eye_corner = landmarks[36]
            nose_point = landmarks[31]
        else:
            cheekbone = landmarks[15]
            eye_corner = landmarks[45]
            nose_point = landmarks[35]
        
        center_x = int((cheekbone[0] + nose_point[0]) / 2)
        center_y = int((cheekbone[1] + eye_corner[1]) / 2)
        
        width = int(abs(cheekbone[0] - nose_point[0]) * 0.8)
        height = int(abs(cheekbone[1] - eye_corner[1]) * 1.2)
        
        cv2.ellipse(mask, (center_x, center_y), (width, height), 0, 0, 360, 255, -1)
        
        mask = cv2.GaussianBlur(mask, (51, 51), 25)
        
        return mask
    
    def create_eyeshadow_mask(self, landmarks: np.ndarray, image_shape: Tuple[int, int], 
                             side: str = 'left') -> np.ndarray:
        # Generate mask for eyeshadow region above eyes
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if side == 'left':
            eye_points = landmarks[36:42]
            brow_points = landmarks[17:22]
        else:
            eye_points = landmarks[42:48]
            brow_points = landmarks[22:27]
        
        eye_top = eye_points[[1, 2]].mean(axis=0)
        brow_bottom = brow_points.mean(axis=0)
        
        shadow_region = np.vstack([
            eye_points[[0, 1, 2, 3]],
            brow_points[::-1]
        ]).astype(np.int32)
        
        cv2.fillPoly(mask, [shadow_region], 255)
        
        mask = cv2.GaussianBlur(mask, (21, 21), 10)
        
        return mask
    
    def create_eyeliner_mask(self, landmarks: np.ndarray, image_shape: Tuple[int, int], 
                            thickness: int = 2, side: str = 'left') -> np.ndarray:
        # Generate mask for eyeliner along upper lash line
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if side == 'left':
            eye_points = landmarks[36:42]
            upper_lid = eye_points[[0, 1, 2, 3]]
        else:
            eye_points = landmarks[42:48]
            upper_lid = eye_points[[0, 1, 2, 3]]
        
        pts = upper_lid.astype(np.int32)
        
        for i in range(len(pts) - 1):
            cv2.line(mask, tuple(pts[i]), tuple(pts[i+1]), 255, thickness)
        
        outer_corner = pts[-1]
        extended_point = outer_corner + np.array([int(thickness * 3), -int(thickness * 2)])
        cv2.line(mask, tuple(outer_corner), tuple(extended_point.astype(np.int32)), 255, thickness)
        
        mask = cv2.GaussianBlur(mask, (5, 5), 1)
        
        return mask
    
    def create_eyebrow_mask(self, landmarks: np.ndarray, image_shape: Tuple[int, int], 
                           side: str = 'left') -> np.ndarray:
        # Generate mask for eyebrow enhancement
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if side == 'left':
            brow_points = landmarks[17:22]
        else:
            brow_points = landmarks[22:27]
        
        pts = brow_points.astype(np.int32)
        
        for i in range(len(pts) - 1):
            cv2.line(mask, tuple(pts[i]), tuple(pts[i+1]), 255, 3)
        
        mask = cv2.GaussianBlur(mask, (7, 7), 2)
        
        return mask
    
    def blend_multiply(self, base: np.ndarray, blend: np.ndarray, mask: np.ndarray, 
                      intensity: float) -> np.ndarray:
        # Multiply blend mode for darker colors (lipstick, eyeliner)
        mask_norm = (mask.astype(np.float32) / 255.0) * intensity
        mask_3ch = np.stack([mask_norm] * 3, axis=2)
        
        base_norm = base.astype(np.float32) / 255.0
        blend_norm = blend.astype(np.float32) / 255.0
        
        result = base_norm * (1 - mask_3ch) + (base_norm * blend_norm) * mask_3ch
        
        return (result * 255).astype(np.uint8)
    
    def blend_screen(self, base: np.ndarray, blend: np.ndarray, mask: np.ndarray, 
                    intensity: float) -> np.ndarray:
        # Screen blend mode for lighter colors (highlights)
        mask_norm = (mask.astype(np.float32) / 255.0) * intensity
        mask_3ch = np.stack([mask_norm] * 3, axis=2)
        
        base_norm = base.astype(np.float32) / 255.0
        blend_norm = blend.astype(np.float32) / 255.0
        
        result = base_norm * (1 - mask_3ch) + (1 - (1 - base_norm) * (1 - blend_norm)) * mask_3ch
        
        return (result * 255).astype(np.uint8)
    
    def blend_overlay(self, base: np.ndarray, blend: np.ndarray, mask: np.ndarray, 
                     intensity: float) -> np.ndarray:
        # Overlay blend mode for natural color mixing (blush, eyeshadow)
        mask_norm = (mask.astype(np.float32) / 255.0) * intensity
        mask_3ch = np.stack([mask_norm] * 3, axis=2)
        
        base_norm = base.astype(np.float32) / 255.0
        blend_norm = blend.astype(np.float32) / 255.0
        
        overlay = np.where(
            base_norm < 0.5,
            2 * base_norm * blend_norm,
            1 - 2 * (1 - base_norm) * (1 - blend_norm)
        )
        
        result = base_norm * (1 - mask_3ch) + overlay * mask_3ch
        
        return (result * 255).astype(np.uint8)
    
    def blend_soft_light(self, base: np.ndarray, blend: np.ndarray, mask: np.ndarray, 
                        intensity: float) -> np.ndarray:
        # Soft light blend for subtle color application
        mask_norm = (mask.astype(np.float32) / 255.0) * intensity
        mask_3ch = np.stack([mask_norm] * 3, axis=2)
        
        base_norm = base.astype(np.float32) / 255.0
        blend_norm = blend.astype(np.float32) / 255.0
        
        soft_light = np.where(
            blend_norm < 0.5,
            2 * base_norm * blend_norm + base_norm**2 * (1 - 2 * blend_norm),
            2 * base_norm * (1 - blend_norm) + np.sqrt(base_norm) * (2 * blend_norm - 1)
        )
        
        result = base_norm * (1 - mask_3ch) + soft_light * mask_3ch
        
        return (result * 255).astype(np.uint8)
    
    def apply_lipstick(self, image: np.ndarray, landmarks: np.ndarray, 
                      color: Tuple[int, int, int], intensity: float) -> np.ndarray:
        # Apply lipstick with multiply blend for rich color
        mask = self.create_lip_mask(landmarks, image.shape)
        
        color_layer = np.zeros_like(image)
        color_layer[:, :] = color[::-1]
        
        result = self.blend_multiply(image, color_layer, mask, intensity)
        
        return result
    
    def apply_blush(self, image: np.ndarray, landmarks: np.ndarray, 
                   color: Tuple[int, int, int], intensity: float) -> np.ndarray:
        # Apply blush with soft light blend for natural glow
        mask_left = self.create_blush_mask(landmarks, image.shape, 'left')
        mask_right = self.create_blush_mask(landmarks, image.shape, 'right')
        mask_combined = cv2.bitwise_or(mask_left, mask_right)
        
        color_layer = np.zeros_like(image)
        color_layer[:, :] = color[::-1]
        
        result = self.blend_soft_light(image, color_layer, mask_combined, intensity)
        
        return result
    
    def apply_eyeshadow(self, image: np.ndarray, landmarks: np.ndarray, 
                       color: Tuple[int, int, int], intensity: float) -> np.ndarray:
        # Apply eyeshadow with overlay blend for dimensional color
        mask_left = self.create_eyeshadow_mask(landmarks, image.shape, 'left')
        mask_right = self.create_eyeshadow_mask(landmarks, image.shape, 'right')
        mask_combined = cv2.bitwise_or(mask_left, mask_right)
        
        color_layer = np.zeros_like(image)
        color_layer[:, :] = color[::-1]
        
        result = self.blend_overlay(image, color_layer, mask_combined, intensity)
        
        return result
    
    def apply_eyeliner(self, image: np.ndarray, landmarks: np.ndarray, 
                      color: Tuple[int, int, int], thickness: int) -> np.ndarray:
        # Apply eyeliner with multiply blend for defined lines
        mask_left = self.create_eyeliner_mask(landmarks, image.shape, thickness, 'left')
        mask_right = self.create_eyeliner_mask(landmarks, image.shape, thickness, 'right')
        mask_combined = cv2.bitwise_or(mask_left, mask_right)
        
        color_layer = np.zeros_like(image)
        color_layer[:, :] = color[::-1]
        
        result = self.blend_multiply(image, color_layer, mask_combined, 0.9)
        
        return result
    
    def apply_eyebrow_enhancement(self, image: np.ndarray, landmarks: np.ndarray, 
                                 color: Tuple[int, int, int], intensity: float) -> np.ndarray:
        # Enhance eyebrows with multiply blend for fuller appearance
        mask_left = self.create_eyebrow_mask(landmarks, image.shape, 'left')
        mask_right = self.create_eyebrow_mask(landmarks, image.shape, 'right')
        mask_combined = cv2.bitwise_or(mask_left, mask_right)
        
        color_layer = np.zeros_like(image)
        color_layer[:, :] = color[::-1]
        
        result = self.blend_multiply(image, color_layer, mask_combined, intensity)
        
        return result
    
    def apply_makeup(self, image_path: str, output_path: str,
                    config: Optional[MakeupConfig] = None) -> bool:
        # Main pipeline: detect landmarks and apply all makeup effects
        if config is None:
            config = MakeupConfig()
        
        try:
            print("\n" + "="*60)
            print("VIRTUAL MAKEUP APPLICATION")
            print("="*60)
            
            print("Loading image...")
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                raise ValueError(f"Cannot load image: {image_path}")
            
            print(f"   Image shape: {image_bgr.shape}")
            
            print("Detecting face landmarks...")
            landmarks = self.get_landmarks(image_bgr)
            if landmarks is None:
                self.logger.error("No face detected")
                return False
            
            print(f"   Detected {len(landmarks)} landmarks")
            
            result = image_bgr.copy()
            
            if config.eyeshadow_intensity > 0:
                print("Applying eyeshadow...")
                result = self.apply_eyeshadow(result, landmarks, config.eyeshadow_color, 
                                             config.eyeshadow_intensity)
            
            if config.eyeliner_thickness > 0:
                print("Applying eyeliner...")
                result = self.apply_eyeliner(result, landmarks, config.eyeliner_color, 
                                            config.eyeliner_thickness)
            
            if config.eyebrow_intensity > 0:
                print("Enhancing eyebrows...")
                result = self.apply_eyebrow_enhancement(result, landmarks, config.eyebrow_color, 
                                                       config.eyebrow_intensity)
            
            if config.blush_intensity > 0:
                print("Applying blush...")
                result = self.apply_blush(result, landmarks, config.blush_color, 
                                        config.blush_intensity)
            
            if config.lipstick_intensity > 0:
                print("Applying lipstick...")
                result = self.apply_lipstick(result, landmarks, config.lipstick_color, 
                                           config.lipstick_intensity)
            
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            
            print("Saving result...")
            Image.fromarray(result_rgb).save(output_path, quality=95)
            
            print(f"SUCCESS! Makeup applied: {output_path}")
            print("="*60 + "\n")
            return True
            
        except Exception as e:
            self.logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return False


PRESET_MAKEUPS = {
    'natural': MakeupConfig(
        lipstick_color=(200, 80, 90),
        lipstick_intensity=0.5,
        blush_color=(255, 180, 180),
        blush_intensity=0.25,
        eyeshadow_color=(180, 150, 130),
        eyeshadow_intensity=0.3,
        eyeliner_color=(40, 30, 30),
        eyeliner_thickness=2,
        eyebrow_color=(70, 50, 40),
        eyebrow_intensity=0.2
    ),
    'glamorous': MakeupConfig(
        lipstick_color=(180, 38, 45),
        lipstick_intensity=0.8,
        blush_color=(255, 140, 140),
        blush_intensity=0.4,
        eyeshadow_color=(120, 80, 60),
        eyeshadow_intensity=0.6,
        eyeliner_color=(20, 20, 20),
        eyeliner_thickness=3,
        eyebrow_color=(50, 35, 25),
        eyebrow_intensity=0.4
    ),
    'soft_pink': MakeupConfig(
        lipstick_color=(255, 150, 170),
        lipstick_intensity=0.6,
        blush_color=(255, 190, 200),
        blush_intensity=0.3,
        eyeshadow_color=(220, 180, 190),
        eyeshadow_intensity=0.4,
        eyeliner_color=(80, 60, 60),
        eyeliner_thickness=2,
        eyebrow_color=(90, 70, 60),
        eyebrow_intensity=0.25
    ),
    'bold_red': MakeupConfig(
        lipstick_color=(160, 20, 30),
        lipstick_intensity=0.85,
        blush_color=(255, 130, 130),
        blush_intensity=0.35,
        eyeshadow_color=(100, 60, 50),
        eyeshadow_intensity=0.5,
        eyeliner_color=(10, 10, 10),
        eyeliner_thickness=3,
        eyebrow_color=(40, 25, 20),
        eyebrow_intensity=0.45
    ),
    'subtle': MakeupConfig(
        lipstick_color=(220, 120, 130),
        lipstick_intensity=0.4,
        blush_color=(255, 200, 200),
        blush_intensity=0.2,
        eyeshadow_color=(200, 180, 170),
        eyeshadow_intensity=0.25,
        eyeliner_color=(60, 50, 50),
        eyeliner_thickness=1,
        eyebrow_color=(80, 60, 50),
        eyebrow_intensity=0.15
    )
}


if __name__ == "__main__":
    INPUT_IMAGE = "/Users/zheng/Downloads/Shi-2631-1.jpg"
    OUTPUT_IMAGE = "/Users/zheng/Desktop/CIS5810Project/Shi-makeup.png"
    
    print("\nInitializing Virtual Makeup...")
    makeup = VirtualMakeup(device='cpu')
    
    config = PRESET_MAKEUPS['natural']
    success = makeup.apply_makeup(INPUT_IMAGE, OUTPUT_IMAGE, config)
    
    if success:
        print("\nMakeup applied successfully!")
    else:
        print("\nMakeup application failed.")