import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class SmileConfig:
    intensity: float = 1.0
    blend_method: str = "seamless"



class SmileTransfer:

    def __init__(self):
        # OpenCV DNN face detector
        modelFile = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_detector = cv2.CascadeClassifier(modelFile)

        # OpenCV LBF landmark model (comes with opencv-contrib-python)
        landmark_model = cv2.face.createFacemarkLBF()
        landmark_model.loadModel(cv2.data.face + "lbfmodel.yaml")
        self.facemark = landmark_model

        # Mouth landmark indices (same as dlib’s 68-point model)
        self.MOUTH_POINTS = list(range(48, 68))
        self.OUTER_MOUTH = list(range(48, 60))
        self.INNER_MOUTH = list(range(60, 68))

    def get_landmarks(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            raise ValueError("No face detected.")

        # Convert to required bounding box format
        face_rects = []
        for (x, y, w, h) in faces:
            face_rects.append((x, y, x + w, y + h))

        success, landmarks = self.facemark.fit(img, face_rects)
        if not success or len(landmarks) == 0:
            raise ValueError("Landmarks could not be detected.")

        # Return shape (68, 2)
        return landmarks[0][0]

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

    def alpha_blend(self, src, dst, mask, feather=20):
        mask_f = mask.astype(float) / 255.0
        mask_f = cv2.GaussianBlur(mask_f, (feather * 2 + 1, feather * 2 + 1), 0)
        mask_f = np.stack([mask_f] * 3, axis=2)
        return (src * mask_f + dst * (1 - mask_f)).astype(np.uint8)

    def seamless_clone(self, src, dst, mask, center):
        try:
            return cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)
        except:
            return self.alpha_blend(src, dst, mask)

    def transfer(self, source_img, target_img, blend_method="seamless"):

        src_lm = self.get_landmarks(source_img)
        dst_lm = self.get_landmarks(target_img)

        warped_src, matrix = self.warp_mouth(
            source_img, target_img, src_lm, dst_lm
        )

        mask = self.get_mouth_mask(target_img.shape, dst_lm)
        mouth_center = tuple(np.mean(dst_lm[self.MOUTH_POINTS], axis=0).astype(int))

        if blend_method == "alpha":
            return self.alpha_blend(warped_src, target_img, mask)

        return self.seamless_clone(warped_src, target_img, mask, mouth_center)

    def apply_smile(self, target_image_path, output_path, config: SmileConfig):
        if config.intensity == 0:
            return False  # nothing to change

        src = cv2.imread("smile_ref.webp")  # your reference smile
        tgt = cv2.imread(target_image_path)

        if src is None or tgt is None:
            raise FileNotFoundError("Image not found.")

        result = self.transfer(
            source_img=src,
            target_img=tgt,
            blend_method=config.blend_method,
        )

        cv2.imwrite(output_path, result)
        return True
