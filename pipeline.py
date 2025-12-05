# pipeline.py
import os
from pathlib import Path

from skin_smoothing import SkinSmoothingFilter
from face_reshape import OpticalFlowReshaper, ReshapeConfig
from virtual_makeup import VirtualMakeup, MakeupConfig
from background_removal import load_model as load_bg_model, process_image as remove_bg
from blend_background import apply_background_replacement
from blur_background import BackgroundBlur, BlurConfig, PRESET_BLUR
from color_grading import ColorGradingFilters, ColorGradingConfig, PRESET_FILTERS
from smile_transfer import SmileTransfer, SmileConfig, PRESET_SMILES
import cv2
import numpy as np


def run_aura_pipeline(input_path: str, output_path: str, params: dict):
    """
    Runs ALL Aura modules in sequence.
    Compatible with NLP_interface.py output.
    
    Features:
    1. Background Removal
    2. Skin Smoothing
    3. Face Reshape
    4. Virtual Makeup
    5. Smile Transfer
    6. Background Blur (Portrait Mode)
    7. Color Grading
    8. Background Replacement
    """

    features = params.get("features", {})
    do_enhance = features.get("enhancement", False)
    do_makeup = features.get("makeup", False)
    do_reshape = features.get("reshape", False)
    do_smoothing = features.get("smoothing", False)
    do_smile = features.get("smile_transfer", False)
    do_bg_blur = features.get("background_blur", False)
    do_color_grade = features.get("color_grading", False)
    do_bg_replace = features.get("background_replacement", False)

    working = input_path
    alpha_mask_path = None  # Store alpha mask for later use

    # ------------------------------------
    # 1. BACKGROUND REMOVAL
    # ------------------------------------
    print("STEP 1 – Background Removal")
    try:
        model = load_bg_model("u2netp.pth")  # adjust if needed
        tmp = str(Path(working).with_name("fg.png"))
        remove_bg(model, working, tmp)
        
        # Store alpha mask path for background blur
        alpha_mask_path = tmp
        working = tmp
    except Exception as e:
        print("Skipping background removal:", e)

    # ------------------------------------
    # 2. SKIN SMOOTHING
    # ------------------------------------
    if do_smoothing:
        print("STEP 2 – Skin Smoothing")
        tmp = str(Path(working).with_name("smooth.png"))
        
        smooth = SkinSmoothingFilter(device="cpu")
        
        # Get beauty config from params or use defaults
        beauty_cfg = params.get("beauty_config", {})
        from skin_smoothing import BeautyConfig, PRESET_CONFIGS
        
        if beauty_cfg.get("preset"):
            config = PRESET_CONFIGS.get(beauty_cfg["preset"])
        else:
            config = BeautyConfig(
                smooth_strength=beauty_cfg.get("smooth_strength", 0.6),
                brightness_boost=beauty_cfg.get("brightness_boost", 1.05),
                blemish_removal=beauty_cfg.get("blemish_removal", True)
            )
        
        smooth.apply_beauty_filter(working, tmp, config=config)
        working = tmp

    # ------------------------------------
    # 3. FACE RESHAPE
    # ------------------------------------
    if do_reshape:
        print("STEP 3 – Face Reshape")
        
        rcfg = params.get("reshape_config", {})
        
        # Use preset if specified
        from face_reshape import PRESET_CONFIGS
        if rcfg.get("preset"):
            cfg = PRESET_CONFIGS.get(rcfg["preset"], ReshapeConfig())
        else:
            cfg = ReshapeConfig(
                slim_factor=rcfg.get("slim_factor", 0.1),
                eye_factor=rcfg.get("eye_factor", 0.1),
                chin_factor=rcfg.get("chin_factor", 0.1),
                nose_factor=rcfg.get("nose_factor", 0.1),
            )

        reshaper = OpticalFlowReshaper(device="cpu")
        tmp = str(Path(working).with_name("reshaped.png"))
        reshaper.reshape_face(working, tmp, cfg)
        working = tmp

    # ------------------------------------
    # 4. MAKEUP
    # ------------------------------------
    if do_makeup:
        print("STEP 4 – Virtual Makeup")
        
        mc = params.get("makeup_config", {})
        
        # Use preset if specified
        from virtual_makeup import PRESET_MAKEUPS
        if mc.get("preset"):
            mk = PRESET_MAKEUPS.get(mc["preset"], MakeupConfig())
        else:
            mk = MakeupConfig(
                lipstick_color=tuple(mc.get("lipstick_color", (180, 38, 45))),
                lipstick_intensity=mc.get("lipstick_intensity", 0.0),
                blush_color=tuple(mc.get("blush_color", (255, 150, 150))),
                blush_intensity=mc.get("blush_intensity", 0.0),
                eyeshadow_color=tuple(mc.get("eyeshadow_color", (150, 100, 80))),
                eyeshadow_intensity=mc.get("eyeshadow_intensity", 0.0),
                eyeliner_color=tuple(mc.get("eyeliner_color", (20, 20, 20))),
                eyeliner_thickness=mc.get("eyeliner_thickness", 0),
                eyebrow_color=tuple(mc.get("eyebrow_color", (60, 40, 30))),
                eyebrow_intensity=mc.get("eyebrow_intensity", 0.0),
            )

        vm = VirtualMakeup(device="cpu")
        tmp = str(Path(working).with_name("makeup.png"))
        vm.apply_makeup(working, tmp, mk)
        working = tmp

    # ------------------------------------
    # 5. SMILE TRANSFER
    # ------------------------------------
    if do_smile:
        print("STEP 5 – Smile Transfer")
        try:
            smile_cfg = params.get("smile_config", {})
            
            # Get smile reference path
            smile_source = smile_cfg.get("reference_image") or "assets/smile_ref.webp"

            if not os.path.exists(smile_source):
                raise FileNotFoundError(f"Smile source image not found at: {smile_source}")

            # Load current working image
            target_img = cv2.imread(working)
            
            # Initialize smile transfer
            smiler = SmileTransfer()
            
            # Get landmarks for target image
            from skin_smoothing import SkinSmoothingFilter
            landmark_detector = SkinSmoothingFilter(device="cpu")
            target_landmarks = landmark_detector.get_landmarks(target_img)
            
            if target_landmarks is None:
                raise ValueError("Could not detect face in target image")
            
            # Use preset if specified
            if smile_cfg.get("preset"):
                config = PRESET_SMILES.get(smile_cfg["preset"], SmileConfig())
            else:
                config = SmileConfig(
                    intensity=smile_cfg.get("intensity", 1.0),
                    blend_method=smile_cfg.get("blend_method", "seamless"),
                    feather_amount=smile_cfg.get("feather_amount", 20),
                    reference_image=smile_source
                )
            
            # Apply smile transfer
            result_img = smiler.apply_smile(
                target_img=target_img,
                target_landmarks=target_landmarks,
                reference_path=smile_source,
                reference_landmarks_getter=lambda img: landmark_detector.get_landmarks(img),
                config=config
            )
            
            # Save result
            tmp = str(Path(working).with_name("smile.png"))
            cv2.imwrite(tmp, result_img)
            working = tmp

        except Exception as e:
            print("Skipping smile transfer:", e)
            import traceback
            traceback.print_exc()

    # ------------------------------------
    # 6. BACKGROUND BLUR (Portrait Mode)
    # ------------------------------------
    if do_bg_blur and alpha_mask_path:
        print("STEP 6 – Background Blur (Portrait Mode)")
        try:
            blur_cfg = params.get("blur_config", {})
            
            # Use preset if specified
            if blur_cfg.get("preset"):
                config = PRESET_BLUR.get(blur_cfg["preset"], BlurConfig())
            else:
                config = BlurConfig(
                    blur_strength=blur_cfg.get("blur_strength", 21),
                    blur_type=blur_cfg.get("blur_type", "gaussian"),
                    edge_softness=blur_cfg.get("edge_softness", 15),
                    foreground_enhance=blur_cfg.get("foreground_enhance", True)
                )
            
            # Load current working image and alpha mask
            current_img = cv2.imread(working)
            alpha_mask = cv2.imread(alpha_mask_path, cv2.IMREAD_UNCHANGED)
            
            # Extract alpha channel if RGBA
            if len(alpha_mask.shape) == 3 and alpha_mask.shape[2] == 4:
                alpha_mask = alpha_mask[:, :, 3]
            elif len(alpha_mask.shape) == 3:
                alpha_mask = cv2.cvtColor(alpha_mask, cv2.COLOR_BGR2GRAY)
            
            # Apply background blur
            blur_processor = BackgroundBlur()
            result_img = blur_processor.apply_background_blur(current_img, alpha_mask, config)
            
            # Save result
            tmp = str(Path(working).with_name("blurred_bg.png"))
            cv2.imwrite(tmp, result_img)
            working = tmp
            
        except Exception as e:
            print("Skipping background blur:", e)
            import traceback
            traceback.print_exc()

    # ------------------------------------
    # 7. COLOR GRADING
    # ------------------------------------
    if do_color_grade:
        print("STEP 7 – Color Grading")
        try:
            cg_cfg = params.get("color_grading_config", {})
            
            # Use preset if specified
            if cg_cfg.get("filter_name"):
                config = ColorGradingConfig(
                    filter_name=cg_cfg["filter_name"],
                    intensity=cg_cfg.get("intensity", 1.0)
                )
            else:
                print("No color grading filter specified, skipping...")
                config = None
            
            if config:
                tmp = str(Path(working).with_name("color_graded.png"))
                grader = ColorGradingFilters()
                grader.apply_color_grading(working, tmp, config)
                working = tmp
                
        except Exception as e:
            print("Skipping color grading:", e)
            import traceback
            traceback.print_exc()

    # ------------------------------------
    # 8. BACKGROUND REPLACEMENT
    # ------------------------------------
    if do_bg_replace:
        print("STEP 8 – Background Replacement")
        try:
            bg_cfg = params.get("background_config", {})
            bg_path = bg_cfg.get("background_path")
            
            if bg_path and os.path.exists(bg_path):
                # Load foreground and background
                from PIL import Image
                foreground_img = cv2.imread(working)
                background_img = cv2.imread(bg_path)
                
                # Load alpha mask
                if alpha_mask_path:
                    alpha_mask = cv2.imread(alpha_mask_path, cv2.IMREAD_UNCHANGED)
                    
                    # Extract alpha channel
                    if len(alpha_mask.shape) == 3 and alpha_mask.shape[2] == 4:
                        alpha_mask = alpha_mask[:, :, 3]
                    elif len(alpha_mask.shape) == 3:
                        alpha_mask = cv2.cvtColor(alpha_mask, cv2.COLOR_BGR2GRAY)
                    
                    # Normalize mask to 0-1
                    foreground_mask = alpha_mask.astype(np.float32) / 255.0
                    
                    # Apply background replacement
                    result_img = apply_background_replacement(
                        foreground_img, 
                        background_img, 
                        foreground_mask
                    )
                    
                    # Save result
                    tmp = str(Path(working).with_name("bg_replaced.png"))
                    cv2.imwrite(tmp, result_img)
                    working = tmp
                else:
                    print("No alpha mask available for background replacement")
            else:
                print(f"Background image not found: {bg_path}")
                
        except Exception as e:
            print("Skipping background replacement:", e)
            import traceback
            traceback.print_exc()

    # ------------------------------------
    # 9. SAVE FINAL RESULT
    # ------------------------------------
    print("STEP 9 – Saving final result")
    
    # Copy or rename final working file to output path
    if working != output_path:
        import shutil
        shutil.copy(working, output_path)
    
    print(f"✓ Pipeline complete! Final output: {output_path}")


if __name__ == "__main__":
    # Example usage
    example_params = {
        "features": {
            "enhancement": False,
            "makeup": True,
            "reshape": True,
            "smoothing": True,
            "smile_transfer": True,
            "background_blur": True,
            "color_grading": True,
            "background_replacement": False
        },
        "beauty_config": {
            "preset": "natural"
        },
        "reshape_config": {
            "preset": "subtle"
        },
        "makeup_config": {
            "preset": "natural"
        },
        "smile_config": {
            "preset": "natural",
            "intensity": 0.7
        },
        "blur_config": {
            "preset": "portrait"
        },
        "color_grading_config": {
            "filter_name": "cinematic",
            "intensity": 0.8
        }
    }
    
    run_aura_pipeline(
        input_path="input.jpg",
        output_path="output.jpg",
        params=example_params
    )