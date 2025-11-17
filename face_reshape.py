"""
Face Reshaping using FAN Landmarks + Optical Flow Warping
"""

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
class ReshapeConfig:
    slim_factor: float = 0.12
    eye_factor: float = 0.25
    chin_factor: float = 0.15
    nose_factor: float = 0.1


class OpticalFlowReshaper:
    
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
        # Extract 68 facial landmarks using FAN
        try:
            if len(image.shape) == 2:
                rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3:
                if image.shape[2] == 4:
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                elif image.shape[2] == 3:
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    raise ValueError("Unsupported format")
            else:
                raise ValueError("Invalid dimensions")
            
            landmarks_list = self.fa.get_landmarks(rgb)
            
            if landmarks_list is None or len(landmarks_list) == 0:
                return None
            
            return landmarks_list[0][:, :2]
            
        except Exception as e:
            self.logger.error(f"Error extracting landmarks: {e}")
            return None
    
    def compute_target_landmarks(self, landmarks: np.ndarray, config: ReshapeConfig) -> np.ndarray:
        # Calculate desired landmark positions based on reshape parameters
        target = landmarks.copy()
        face_center = landmarks[33]
        
        jaw_indices = list(range(0, 17))
        for idx in jaw_indices:
            if idx in [3, 4, 5, 6, 10, 11, 12, 13]:
                slim_amount = config.slim_factor
            elif idx in [1, 2, 14, 15]:
                slim_amount = config.slim_factor * 0.8
            else:
                slim_amount = config.slim_factor * 0.5
            
            vec = landmarks[idx] - face_center
            target[idx] = face_center + vec * (1 - slim_amount)
        
        chin_idx = 8
        chin_vec = landmarks[33] - landmarks[chin_idx]
        target[chin_idx] = landmarks[chin_idx] + chin_vec * config.chin_factor
        
        nose_indices = list(range(31, 36))
        for idx in nose_indices:
            vec = landmarks[idx] - face_center
            target[idx] = face_center + vec * (1 - config.nose_factor * 0.3)
        
        for eye_indices in [list(range(36, 42)), list(range(42, 48))]:
            eye_center = landmarks[eye_indices].mean(axis=0)
            for idx in eye_indices:
                vec = landmarks[idx] - eye_center
                target[idx] = eye_center + vec * (1 + config.eye_factor * 0.2)
        
        return target
    
    def compute_optical_flow(self, source_lm: np.ndarray, target_lm: np.ndarray, 
                            image_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        # Generate dense flow field from sparse landmark correspondences using thin-plate spline
        from scipy.interpolate import Rbf
        
        h, w = image_shape[:2]
        
        boundary_points_src = []
        boundary_points_tgt = []
        
        corners = [
            [0, 0], [w-1, 0], [0, h-1], [w-1, h-1],
            [w//2, 0], [0, h//2], [w-1, h//2], [w//2, h-1]
        ]
        for corner in corners:
            boundary_points_src.append(corner)
            boundary_points_tgt.append(corner)
        
        num_edge_points = 10
        for i in range(num_edge_points):
            t = i / (num_edge_points - 1)
            boundary_points_src.append([int(t * w), 0])
            boundary_points_tgt.append([int(t * w), 0])
            boundary_points_src.append([int(t * w), h-1])
            boundary_points_tgt.append([int(t * w), h-1])
            boundary_points_src.append([0, int(t * h)])
            boundary_points_tgt.append([0, int(t * h)])
            boundary_points_src.append([w-1, int(t * h)])
            boundary_points_tgt.append([w-1, int(t * h)])
        
        all_source = np.vstack([source_lm, np.array(boundary_points_src)])
        all_target = np.vstack([target_lm, np.array(boundary_points_tgt)])
        
        print(f"   Creating RBF with {len(all_source)} control points...")
        rbf_x = Rbf(all_target[:, 0], all_target[:, 1], all_source[:, 0],
                   function='thin_plate', smooth=0.1)
        rbf_y = Rbf(all_target[:, 0], all_target[:, 1], all_source[:, 1],
                   function='thin_plate', smooth=0.1)
        
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        
        print(f"   Interpolating dense flow field...")
        map_x = rbf_x(grid_x.ravel(), grid_y.ravel()).reshape(h, w).astype(np.float32)
        map_y = rbf_y(grid_x.ravel(), grid_y.ravel()).reshape(h, w).astype(np.float32)
        
        map_x = np.clip(map_x, 0, w - 1)
        map_y = np.clip(map_y, 0, h - 1)
        
        return map_x, map_y
    
    def apply_optical_flow(self, image: np.ndarray, map_x: np.ndarray, 
                          map_y: np.ndarray) -> np.ndarray:
        # Warp image using computed flow field
        warped = cv2.remap(image, map_x, map_y, 
                          interpolation=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REFLECT_101)
        return warped
    
    def create_blend_mask(self, landmarks: np.ndarray, 
                         image_shape: Tuple[int, int]) -> np.ndarray:
        # Generate smooth mask for blending reshaped region with original
        h, w = image_shape[:2]
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
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100, 100))
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (71, 71), 40)
        
        return mask
    
    def reshape_face(self, image_path: str, output_path: str,
                    config: Optional[ReshapeConfig] = None) -> bool:
        # Main pipeline: load image, detect landmarks, warp, blend, and save
        if config is None:
            config = ReshapeConfig()
        
        try:
            print("\n" + "="*60)
            print("OPTICAL FLOW FACE RESHAPING")
            print("="*60)
            
            print("Loading image...")
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                raise ValueError(f"Cannot load image: {image_path}")
            
            print(f"   Image shape: {image_bgr.shape}")
            
            print("Detecting landmarks...")
            landmarks = self.get_landmarks(image_bgr)
            if landmarks is None:
                self.logger.error("No face detected")
                return False
            
            print(f"   Detected {len(landmarks)} landmarks")
            
            print("Computing target face shape...")
            target_landmarks = self.compute_target_landmarks(landmarks, config)
            
            print("Computing dense optical flow...")
            map_x, map_y = self.compute_optical_flow(landmarks, target_landmarks, image_bgr.shape)
            
            print("Warping image...")
            warped = self.apply_optical_flow(image_bgr, map_x, map_y)
            
            print("Creating blend mask...")
            mask = self.create_blend_mask(landmarks, image_bgr.shape)
            mask_norm = mask.astype(np.float32) / 255.0
            
            print("Blending with original...")
            blended = image_bgr.copy().astype(np.float32)
            
            for c in range(3):
                blended[:, :, c] = (mask_norm * warped[:, :, c].astype(np.float32) + 
                                  (1 - mask_norm) * image_bgr[:, :, c].astype(np.float32))
            
            result_bgr = blended.astype(np.uint8)
            result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
            
            print("Saving result...")
            Image.fromarray(result_rgb).save(output_path, quality=95)
            
            print(f"SUCCESS! Saved to: {output_path}")
            print("="*60 + "\n")
            return True
            
        except Exception as e:
            self.logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return False


PRESET_CONFIGS = {
    'subtle': ReshapeConfig(
        slim_factor=0.08,
        eye_factor=0.15,
        chin_factor=0.10,
        nose_factor=0.05
    ),
    'natural': ReshapeConfig(
        slim_factor=0.10,
        eye_factor=0.20,
        chin_factor=0.12,
        nose_factor=0.08
    ),
    'moderate': ReshapeConfig(
        slim_factor=0.12,
        eye_factor=0.25,
        chin_factor=0.15,
        nose_factor=0.10
    )
}


if __name__ == "__main__":
    INPUT_IMAGE = "/Users/zheng/Downloads/Shi-2631-1.jpg"
    OUTPUT_IMAGE = "/Users/zheng/Desktop/CIS5810Project/Shi-New-reshaped.png"
    
    print("\nInitializing Face Reshaper...")
    reshaper = OpticalFlowReshaper(device='cpu')
    
    config = PRESET_CONFIGS['natural']
    success = reshaper.reshape_face(INPUT_IMAGE, OUTPUT_IMAGE, config)
    
    if success:
        print("\nFace reshaping complete!")
    else:
        print("\nReshaping failed.")