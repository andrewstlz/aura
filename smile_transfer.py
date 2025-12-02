import cv2
import numpy as np
import dlib
from dataclasses import dataclass
from PIL import Image

@dataclass
class SmileConfig:
    intensity: float = 1.0
    color_match: bool = True
    blend_method: str = "seamless"
    source_image_path: str = None


class SmileTransfer:
    def __init__(self, predictor_path="shape_predictor_68_face_landmarks.dat"):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

        self.MOUTH_POINTS = list(range(48, 68))
        self.OUTER_MOUTH = list(range(48, 60))
        self.INNER_MOUTH = list(range(60, 68))

    def get_landmarks(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        if len(faces) == 0:
            raise ValueError("No face detected.")

        shape = self.predictor(gray, faces[0])
        return np.array([[p.x, p.y] for p in shape.parts()])
    
    def get_mouth_mask(self, image_shape, landmarks, expansion_factor=1.4):
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        mouth_pts = landmarks[self.MOUTH_POINTS]

        center = mouth_pts.mean(axis=0)
        expanded_pts = center + (mouth_pts - center) * expansion_factor
        expanded_pts = expanded_pts.astype(np.int32)

        hull = cv2.convexHull(expanded_pts)
        cv2.fillConvexPoly(mask, hull, 255)

        return mask
    
    def align_mouths(self, src_lm, dst_lm):
        key_idx = [48, 54, 51, 57]
        src_pts = src_lm[key_idx].astype(np.float32)
        dst_pts = dst_lm[key_idx].astype(np.float32)

        matrix = cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]
        return matrix
    
    def warp_mouth(self, src_img, tgt_img, src_lm, dst_lm):
        matrix = self.align_mouths(src_lm, dst_lm)
        warped = cv2.warpAffine(src_img, matrix, (tgt_img.shape[1], tgt_img.shape[0]))
        return warped, matrix
    
    def color_correct(self, src_region, dst_region, mask):
        mask_idx = mask > 0
        corrected = src_region.copy().astype(np.float32)

        for c in range(3):
            src_vals = src_region[..., c][mask_idx]
            dst_vals = dst_region[..., c][mask_idx]

            if len(src_vals) == 0 or len(dst_vals) == 0:
                continue

            s_m, s_s = src_vals.mean(), src_vals.std()
            d_m, d_s = dst_vals.mean(), dst_vals.std()

            if s_s > 0:
                corrected[..., c] = (corrected[..., c] - s_m) * (d_s / s_s) + d_m

        return np.clip(corrected, 0, 255).astype(np.uint8)

    def alpha_blend(self, src, dst, mask, feather=20):
        mask_f = mask.astype(float) / 255.0
        mask_f = cv2.GaussianBlur(mask_f, (feather*2+1, feather*2+1), 0)
        mask_f = np.stack([mask_f]*3, axis=2)

        return (src * mask_f + dst * (1 - mask_f)).astype(np.uint8)
    
    def seamless_clone(self, src, dst, mask, center):
        try:
            return cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)
        except:
            return self.alpha_blend(src, dst, mask)

    def transfer(self, source_img, target_img, blend_method="seamless", color_match=True):
        src_lm = self.get_landmarks(source_img)
        dst_lm = self.get_landmarks(target_img)

        warped_src, matrix = self.warp_mouth(source_img, target_img, src_lm, dst_lm)

        mask = self.get_mouth_mask(target_img.shape, dst_lm)

        if color_match:
            warped_src = self.color_correct(warped_src, target_img, mask)

        mouth_center = tuple(np.mean(dst_lm[self.MOUTH_POINTS], axis=0).astype(int))

        if blend_method == "alpha":
            result = self.alpha_blend(warped_src, target_img, mask)
        else:
            result = self.seamless_clone(warped_src, target_img, mask, mouth_center)

        return result
    
    def apply_smile(self, target_image_path, output_path, config: SmileConfig):
        if config.source_image_path is None:
            raise ValueError("SmileConfig.source_image_path must be set.")

        src = cv2.imread(config.source_image_path)
        tgt = cv2.imread(target_image_path)

        if src is None:
            raise FileNotFoundError("Source smile image not found.")
        if tgt is None:
            raise FileNotFoundError("Target image not found.")

        result = self.transfer(
            source_img=src,
            target_img=tgt,
            blend_method=config.blend_method,
            color_match=config.color_match
        )

        cv2.imwrite(output_path, result)
        return True