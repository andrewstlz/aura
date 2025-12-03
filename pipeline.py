# pipeline.py
import os
from pathlib import Path

from skin_smoothing import SkinSmoothingFilter
from face_reshape import OpticalFlowReshaper, ReshapeConfig
from virtual_makeup import VirtualMakeup, MakeupConfig
from background_removal import load_model as load_bg_model, process_image as remove_bg
from blend_background import apply_background_replacement
from smile_transfer import transfer_smile  # must exist


def run_aura_pipeline(input_path: str, output_path: str, params: dict):
    """
    Runs ALL Aura modules in sequence.
    Compatible with NLP_interface.py output.
    """

    features = params.get("features", {})
    do_enhance = features.get("enhancement", False)
    do_makeup = features.get("makeup", False)
    do_reshape = features.get("reshape", False)
    do_smoothing = features.get("smoothing", False)

    working = input_path

    # ------------------------------------
    # 1. BACKGROUND REMOVAL
    # ------------------------------------
    print("STEP 1 — Background Removal")
    try:
        model = load_bg_model("u2netp.pth")  # adjust if needed
        tmp = str(Path(working).with_name("fg.png"))
        remove_bg(model, working, tmp)
        working = tmp
    except Exception as e:
        print("Skipping background removal:", e)

    # ------------------------------------
    # 2. SKIN SMOOTHING
    # ------------------------------------
    if do_smoothing:
        print("STEP 2 — Skin Smoothing")
        tmp = str(Path(working).with_name("smooth.png"))
        
        smooth = SkinSmoothingFilter(device="cpu")
        
        smooth.apply_beauty_filter(
            working,
            tmp,
            config=None  # optional: fill from params["beauty_config"]
        )
        
        working = tmp

    # ------------------------------------
    # 3. FACE RESHAPE
    # ------------------------------------
    if do_reshape:
        print("STEP 3 — Face Reshape")
        
        rcfg = params.get("reshape_config", {})
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
        print("STEP 4 — Virtual Makeup")
        
        mc = params.get("makeup_config", {})

        mk = MakeupConfig(
            lipstick_color=tuple(mc.get("lipstick_color", (180, 38, 45))),
            lipstick_intensity=mc.get("lipstick_intensity", 0.0),
            blush_intensity=mc.get("blush_intensity", 0.0),
            eyeshadow_intensity=mc.get("eyeshadow_intensity", 0.0),
            eyeliner_thickness=mc.get("eyeliner_thickness", 0),
            eyebrow_intensity=mc.get("eyebrow_intensity", 0.0),
        )

        vm = VirtualMakeup(device="cpu")
        tmp = str(Path(working).with_name("makeup.png"))
        vm.apply_makeup(working, tmp, mk)
        working = tmp

    # ------------------------------------
    # 5. SMILE TRANSFER
    # ------------------------------------
    print("STEP 5 — Smile Transfer")
    try:
        from smile_transfer import SmileTransfer, SmileConfig

        smile_source = "assets/smile_ref.webp"  # << YOUR FILE

        if not os.path.exists(smile_source):
            raise FileNotFoundError(f"Smile source image not found at: {smile_source}")

        smiler = SmileTransfer()

        tmp = str(Path(working).with_name("smile.png"))

        config = SmileConfig(
            intensity=1.0,
            color_match=True,
            blend_method="seamless",
            source_image_path=smile_source
        )

        smiler.apply_smile(
            target_image_path=working,
            output_path=tmp,
            config=config
        )
        
        working = tmp

    except Exception as e:
        print("Skipping smile transfer:", e)

    # ------------------------------------
    # 6. Optional background blending
    # ------------------------------------
    # (you can later add: if params["background"] != None)
    # result = apply_background_replacement(...)

    # ------------------------------------
    # 7. SAVE FINAL RESULT
    # ------------------------------------
    print("STEP 6 — Saving final result")
    
    os.rename(working, output_path)
