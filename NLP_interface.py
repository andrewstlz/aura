"""
Natural Language Interface for Aura Photo Editing Pipeline
Converts natural language requests to structured parameters using GPT-4o-mini
"""

import json
import requests
from typing import Dict, Any


class NaturalLanguageInterface:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://api.openai.com/v1/chat/completions"
        
    def parse_request(self, user_request: str) -> Dict[str, Any]:
        system_prompt = """You are a professional photo editing parameter parser that understands the necessary components for editing photos. Convert natural language requests into JSON parameters.

Output ONLY valid JSON with this exact structure:
{
  "features": {
    "enhancement": boolean,
    "makeup": boolean,
    "reshape": boolean,
    "smoothing": boolean,
    "smile_transfer": boolean
  },
  "enhancement_params": {
    "scale": int (2-8),
    "amount": float (0.0-1.0),
    "chroma_boost": float (1.0-1.5)
  },
  "makeup_config": {
    "preset": string ("natural"/"glamorous"/"soft_pink"/"bold_red"/"subtle") or null,
    "lipstick_color": [int, int, int],
    "lipstick_intensity": float (0.0-1.0),
    "blush_intensity": float (0.0-1.0),
    "eyeshadow_intensity": float (0.0-1.0),
    "eyeliner_thickness": int (0-3),
    "eyebrow_intensity": float (0.0-1.0)
  },
  "reshape_config": {
    "preset": string ("subtle"/"natural"/"moderate") or null,
    "slim_factor": float (0.0-0.3),
    "eye_factor": float (0.0-0.3),
    "chin_factor": float (0.0-0.3),
    "nose_factor": float (0.0-0.2)
  },
  "beauty_config": {
    "preset": string ("subtle"/"natural"/"strong"/"intense") or null,
    "smooth_strength": float (0.0-3.0),
    "brightness_boost": float (1.0-1.15),
    "blemish_removal": boolean
  },
  "smile_config": {
    "intensity": float (0.0-1.0),
    "color_match": boolean,
    "blend_method": string ("seamless" or "alpha"),
    "source_image_path": string or null
  }
}
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_request}
        ]
        
        try:
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 800
                },
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            content = result['choices'][0]['message']['content'].strip()
            
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            params = json.loads(content)
            params["raw_text"] = user_request.lower()
            return params
            
        except Exception:
            return self._get_default_params()
    
    def _get_default_params(self) -> Dict[str, Any]:
        return {
            "features": {
                "enhancement": False,
                "makeup": False,
                "reshape": False,
                "smoothing": True,
                "smile_transfer": False
            },
            "enhancement_params": {},
            "makeup_config": {},
            "reshape_config": {},
            "beauty_config": {
                "preset": "natural",
                "smooth_strength": 0.6,
                "brightness_boost": 1.05,
                "blemish_removal": True
            },
            "smile_config": {
                "intensity": 0.5,
                "color_match": True,
                "blend_method": "seamless",
                "source_image_path": None
            }
        }
    
    def validate_and_normalize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if "features" not in params:
            params["features"] = {
                "enhancement": False,
                "makeup": False,
                "reshape": False,
                "smoothing": False,
                "smile_transfer": False
            }
        
        features = params["features"]
        for key in ["enhancement", "makeup", "reshape", "smoothing", "smile_transfer"]:
            if key not in features:
                features[key] = False

        text = params.get("raw_text", "")
        smile_keywords = [
            "smile", "slight smile", "subtle smile",
            "big smile", "make me smile", "show teeth"
        ]
        if any(k in text for k in smile_keywords):
            features["smile_transfer"] = True

        if "smile_config" not in params:
            params["smile_config"] = {}
        sc = params["smile_config"]
        sc.setdefault("intensity", 0.5)
        sc.setdefault("color_match", True)
        sc.setdefault("blend_method", "seamless")
        sc.setdefault("source_image_path", None)
        sc["intensity"] = max(0.0, min(1.0, float(sc["intensity"])))

        if "enhancement_params" in params and params["enhancement_params"]:
            ep = params["enhancement_params"]
            if "scale" in ep:
                ep["scale"] = max(2, min(8, int(ep["scale"])))
            if "amount" in ep:
                ep["amount"] = max(0.0, min(1.0, float(ep["amount"])))
            if "chroma_boost" in ep:
                ep["chroma_boost"] = max(1.0, min(1.5, float(ep["chroma_boost"])))

        if "makeup_config" in params and params["makeup_config"]:
            mc = params["makeup_config"]
            for key in ["lipstick_intensity", "blush_intensity",
                        "eyeshadow_intensity", "eyebrow_intensity"]:
                if key in mc:
                    mc[key] = max(0.0, min(1.0, float(mc[key])))

        if "reshape_config" in params and params["reshape_config"]:
            rc = params["reshape_config"]
            if "slim_factor" in rc:
                rc["slim_factor"] = max(0.0, min(0.3, float(rc["slim_factor"])))
            if "eye_factor" in rc:
                rc["eye_factor"] = max(0.0, min(0.3, float(rc["eye_factor"])))
            if "chin_factor" in rc:
                rc["chin_factor"] = max(0.0, min(0.3, float(rc["chin_factor"])))
            if "nose_factor" in rc:
                rc["nose_factor"] = max(0.0, min(0.2, float(rc["nose_factor"])))

        if "beauty_config" in params and params["beauty_config"]:
            bc = params["beauty_config"]
            if "smooth_strength" in bc:
                bc["smooth_strength"] = max(0.0, min(3.0, float(bc["smooth_strength"])))
            if "brightness_boost" in bc:
                bc["brightness_boost"] = max(1.0, min(1.15, float(bc["brightness_boost"])))
        
        return params
    
    def print_summary(self, params: Dict[str, Any]):
        print("\n" + "="*60)
        print("PARSED EDITING PARAMETERS")
        print("="*60)
        
        features = params.get("features", {})
        enabled = [k for k, v in features.items() if v]
        
        print(f"\nFeatures to apply: {len(enabled)}")
        for feature in enabled:
            print(f"  ✓ {feature.capitalize()}")
        
        if features.get("enhancement") and params.get("enhancement_params"):
            print("\nEnhancement:")
            for k, v in params["enhancement_params"].items():
                print(f"  {k}: {v}")
        
        if features.get("makeup") and params.get("makeup_config"):
            mc = params["makeup_config"]
            print("\nMakeup:")
            for k, v in mc.items():
                print(f"  {k}: {v}")
        
        if features.get("reshape") and params.get("reshape_config"):
            rc = params["reshape_config"]
            print("\nReshape:")
            for k, v in rc.items():
                print(f"  {k}: {v}")
        
        if features.get("smoothing") and params.get("beauty_config"):
            bc = params["beauty_config"]
            print("\nBeauty Filter:")
            for k, v in bc.items():
                print(f"  {k}: {v}")
        
        if features.get("smile_transfer") and params.get("smile_config"):
            sc = params["smile_config"]
            print("\nSmile Transfer:")
            for k, v in sc.items():
                print(f"  {k}: {v}")
        
        print("="*60 + "\n")
