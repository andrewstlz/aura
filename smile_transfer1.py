"""
Smile Transfer Module
Transfers smile from reference image to target image
Compatible with FAN landmark detector (already in pipeline)
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import os


@dataclass
class SmileConfig:
    intensity: float = 1.0  # 0.0 = no smile, 1.0 = full smile
    blend_method: str = "seamless"  # "seamless" or "alpha"
    feather_amount: int = 20  # Edge softness
    reference_image: Optional[str] = None  # Path to smile reference


class SmileTransfer:
    """Transfer smile from reference image to target image using facial landmarks"""
    
    def __init__(self):
        # Mouth landmark indices (68-point model)
        self.MOUTH_POINTS = list(range(48, 68))
        self.OUTER_MOUTH = list(range(48, 60))
        self.INNER_MOUTH = list(range(60, 68))
        
        # Default reference image path (you should set this)
        self.default_reference = None
        
        print("Smile transfer processor initialized")
    
    def get_mouth_mask(self, image_shape: Tuple[int, int], landmarks: np.ndarray, 
                      expansion_factor: float = 1.4) -> np.ndarray:
        """
        Create mask for mouth region with expansion
        
        Args:
            image_shape: (height, width) of image
            landmarks: 68 facial landmarks
            expansion_factor: How much to expand mask beyond mouth
        
        Returns:
            Binary mask where mouth region is 255
        """
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        mouth_pts = landmarks[self.MOUTH_POINTS]
        
        # Expand mouth region for better blending
        center = mouth_pts.mean(axis=0)
        expanded_pts = center + (mouth_pts - center) * expansion_factor
        expanded_pts = expanded_pts.astype(np.int32)
        
        # Create convex hull and fill
        hull = cv2.convexHull(expanded_pts)
        cv2.fillConvexPoly(mask, hull, 255)
        
        return mask
    
    def align_mouths(self, src_landmarks: np.ndarray, 
                    dst_landmarks: np.ndarray) -> np.ndarray:
        """
        Compute affine transformation to align source mouth to target mouth
        
        Args:
            src_landmarks: Source image 68 landmarks
            dst_landmarks: Target image 68 landmarks
        
        Returns:
            2x3 affine transformation matrix
        """
        # Use key mouth corner points for alignment
        key_indices = [48, 54, 51, 57]  # Left corner, right corner, top center, bottom center
        src_pts = src_landmarks[key_indices].astype(np.float32)
        dst_pts = dst_landmarks[key_indices].astype(np.float32)
        
        # Estimate affine transform (rotation + scale + translation)
        matrix = cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]
        
        return matrix
    
    def warp_mouth(self, src_img: np.ndarray, tgt_img: np.ndarray,
                   src_landmarks: np.ndarray, dst_landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Warp source image to align mouth with target
        
        Args:
            src_img: Source image with smile
            tgt_img: Target image
            src_landmarks: Source 68 landmarks
            dst_landmarks: Target 68 landmarks
        
        Returns:
            Warped source image, transformation matrix
        """
        matrix = self.align_mouths(src_landmarks, dst_landmarks)
        warped = cv2.warpAffine(src_img, matrix, (tgt_img.shape[1], tgt_img.shape[0]))
        
        return warped, matrix
    
    def alpha_blend(self, src: np.ndarray, dst: np.ndarray, 
                   mask: np.ndarray, feather: int = 20) -> np.ndarray:
        """
        Blend source and destination with soft edges
        
        Args:
            src: Source image (warped)
            dst: Destination image
            mask: Binary mask
            feather: Gaussian blur kernel size for soft edges
        
        Returns:
            Blended image
        """
        # Normalize mask to 0-1
        mask_f = mask.astype(float) / 255.0
        
        # Apply Gaussian blur for soft edges
        mask_f = cv2.GaussianBlur(mask_f, (feather * 2 + 1, feather * 2 + 1), 0)
        
        # Expand to 3 channels
        mask_f = np.stack([mask_f] * 3, axis=2)
        
        # Blend
        result = (src * mask_f + dst * (1 - mask_f)).astype(np.uint8)
        
        return result
    
    def seamless_clone(self, src: np.ndarray, dst: np.ndarray,
                      mask: np.ndarray, center: Tuple[int, int]) -> np.ndarray:
        """
        Use OpenCV's seamless cloning for natural blending
        
        Args:
            src: Source image (warped)
            dst: Destination image
            mask: Binary mask
            center: Center point of region to clone
        
        Returns:
            Seamlessly blended image
        """
        try:
            result = cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)
            return result
        except Exception as e:
            print(f"  Seamless clone failed, using alpha blend: {e}")
            return self.alpha_blend(src, dst, mask)
    
    def transfer_smile(self, source_img: np.ndarray, source_landmarks: np.ndarray,
                      target_img: np.ndarray, target_landmarks: np.ndarray,
                      config: SmileConfig) -> np.ndarray:
        """
        Main transfer function: transfer smile from source to target
        
        Args:
            source_img: Image with smile (BGR)
            source_landmarks: 68 landmarks of source
            target_img: Image to add smile to (BGR)
            target_landmarks: 68 landmarks of target
            config: Smile transfer configuration
        
        Returns:
            Target image with transferred smile
        """
        print("  Warping source mouth to target...")
        warped_src, matrix = self.warp_mouth(
            source_img, target_img, source_landmarks, target_landmarks
        )
        
        print("  Creating mouth mask...")
        mask = self.get_mouth_mask(target_img.shape, target_landmarks, expansion_factor=1.4)
        
        print("  Blending...")
        if config.blend_method == "alpha":
            result = self.alpha_blend(warped_src, target_img, mask, feather=config.feather_amount)
        else:  # seamless
            mouth_center = tuple(np.mean(target_landmarks[self.MOUTH_POINTS], axis=0).astype(int))
            result = self.seamless_clone(warped_src, target_img, mask, mouth_center)
        
        # Apply intensity
        if config.intensity < 1.0:
            print(f"  Applying intensity: {config.intensity}")
            result = cv2.addWeighted(target_img, 1.0 - config.intensity, result, config.intensity, 0)
        
        return result
    
    def apply_smile(self, target_img: np.ndarray, target_landmarks: np.ndarray,
                   reference_path: Optional[str] = None,
                   reference_landmarks_getter = None,
                   config: Optional[SmileConfig] = None) -> np.ndarray:
        """
        Apply smile to target image using reference
        
        Args:
            target_img: Image to add smile to (BGR)
            target_landmarks: 68 landmarks of target
            reference_path: Path to reference smile image
            reference_landmarks_getter: Function to get landmarks from reference
            config: Smile configuration
        
        Returns:
            Target image with smile applied
        """
        if config is None:
            config = SmileConfig()
        
        if config.intensity == 0:
            return target_img
        
        # Load reference image
        ref_path = reference_path or config.reference_image or self.default_reference
        
        if ref_path is None:
            raise ValueError("No reference image provided. Set reference_path or config.reference_image")
        
        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"Reference image not found: {ref_path}")
        
        print(f"  Loading reference smile: {ref_path}")
        reference_img = cv2.imread(ref_path)
        
        if reference_img is None:
            raise ValueError(f"Could not load reference image: {ref_path}")
        
        # Get landmarks from reference
        if reference_landmarks_getter is None:
            raise ValueError("reference_landmarks_getter function is required")
        
        print("  Detecting landmarks in reference image...")
        reference_landmarks = reference_landmarks_getter(reference_img)
        
        if reference_landmarks is None:
            raise ValueError("Could not detect face in reference image")
        
        # Transfer smile
        result = self.transfer_smile(
            source_img=reference_img,
            source_landmarks=reference_landmarks,
            target_img=target_img,
            target_landmarks=target_landmarks,
            config=config
        )
        
        return result


# Preset configurations
PRESET_SMILES = {
    'subtle': SmileConfig(
        intensity=0.5,
        blend_method='seamless',
        feather_amount=25
    ),
    'natural': SmileConfig(
        intensity=0.7,
        blend_method='seamless',
        feather_amount=20
    ),
    'strong': SmileConfig(
        intensity=1.0,
        blend_method='seamless',
        feather_amount=20
    ),
    'soft': SmileConfig(
        intensity=0.6,
        blend_method='alpha',
        feather_amount=30
    )
}


if __name__ == "__main__":
    print("Smile transfer module - transfers smile expression from reference to target")