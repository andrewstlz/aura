"""
Natural Language Interface for Aura Photo Editing Pipeline
Converts natural language requests to structured parameters using GPT-4.1-nano
"""

import json
import requests
from typing import Dict, Any


class NaturalLanguageInterface:
    """Parse natural language editing requests into structured parameters"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://api.openai.com/v1/chat/completions"
        
    def parse_request(self, user_request: str) -> Dict[str, Any]:
        """Convert natural language request to structured parameters"""
        
        system_prompt = """You are a professional photo editing parameter parser that understands the necessary components for editing photos. Convert natural language requests into JSON parameters.

Output ONLY valid JSON with this exact structure:
{
  "features": {
    "enhancement": boolean,
    "makeup": boolean,
    "reshape": boolean,
    "smoothing": boolean
  },
  "enhancement_params": {
    "scale": int (2-8),
    "amount": float (0.0-1.0),
    "chroma_boost": float (1.0-1.5)
  },
  "makeup_config": {
    "preset": string ("natural"/"glamorous"/"soft_pink"/"bold_red"/"subtle") or null,
    "lipstick_color": [int, int, int] RGB,
    "lipstick_intensity": float (0.0-1.0),
    "blush_intensity": float (0.0-1.0),
    "eyeshadow_intensity": float (0.0-1.0),
    "eyeliner_thickness": int (0-3, pixels),
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
  }
}

CRITICAL: Always include ALL fields in the response, even if they are 0, null, or false. This ensures consistency.

Examples:

Request: "slim my face a little, smooth my face hard, add pink lipstick"
Response: {"features":{"enhancement":false,"makeup":true,"reshape":true,"smoothing":true},"enhancement_params":{},"makeup_config":{"preset":null,"lipstick_color":[255,150,170],"lipstick_intensity":0.7,"blush_intensity":0.0,"eyeshadow_intensity":0.0,"eyeliner_thickness":0,"eyebrow_intensity":0.0},"reshape_config":{"preset":null,"slim_factor":0.08,"eye_factor":0.0,"chin_factor":0.0,"nose_factor":0.0},"beauty_config":{"preset":null,"smooth_strength":0.8,"brightness_boost":1.05,"blemish_removal":true}}

Request: "slim my face a little, smooth my face hard, add pink lipstick and draw eyeliner"
Response: {"features":{"enhancement":false,"makeup":true,"reshape":true,"smoothing":true},"enhancement_params":{},"makeup_config":{"preset":null,"lipstick_color":[255,150,170],"lipstick_intensity":0.7,"blush_intensity":0.0,"eyeshadow_intensity":0.0,"eyeliner_thickness":2,"eyebrow_intensity":0.0},"reshape_config":{"preset":null,"slim_factor":0.08,"eye_factor":0.0,"chin_factor":0.0,"nose_factor":0.0},"beauty_config":{"preset":null,"smooth_strength":0.8,"brightness_boost":1.05,"blemish_removal":true}}

Request: "add brown eyeshadow and enhance my eyebrows"
Response: {"features":{"enhancement":false,"makeup":true,"reshape":false,"smoothing":false},"enhancement_params":{},"makeup_config":{"preset":null,"lipstick_color":[180,38,45],"lipstick_intensity":0.0,"blush_intensity":0.0,"eyeshadow_intensity":0.5,"eyeliner_thickness":0,"eyebrow_intensity":0.4},"reshape_config":{},"beauty_config":{}}

Request: "apply blush on my cheeks"
Response: {"features":{"enhancement":false,"makeup":true,"reshape":false,"smoothing":false},"enhancement_params":{},"makeup_config":{"preset":null,"lipstick_color":[180,38,45],"lipstick_intensity":0.0,"blush_intensity":0.5,"eyeshadow_intensity":0.0,"eyeliner_thickness":0,"eyebrow_intensity":0.0},"reshape_config":{},"beauty_config":{}}

Request: "make me look glamorous with bold red lips"
Response: {"features":{"enhancement":false,"makeup":true,"reshape":false,"smoothing":true},"enhancement_params":{},"makeup_config":{"preset":"bold_red","lipstick_color":[160,20,30],"lipstick_intensity":0.85,"blush_intensity":0.35,"eyeshadow_intensity":0.5,"eyeliner_thickness":3,"eyebrow_intensity":0.4},"reshape_config":{},"beauty_config":{"preset":"natural","smooth_strength":0.6,"brightness_boost":1.05,"blemish_removal":true}}

Request: "enlarge my eyes slightly and smooth skin naturally"
Response: {"features":{"enhancement":false,"makeup":false,"reshape":true,"smoothing":true},"enhancement_params":{},"makeup_config":{},"reshape_config":{"preset":null,"slim_factor":0.0,"eye_factor":0.08,"chin_factor":0.0,"nose_factor":0.0},"beauty_config":{"preset":"natural","smooth_strength":0.6,"brightness_boost":1.05,"blemish_removal":true}}

Request: "refine my chin and slim my nose"
Response: {"features":{"enhancement":false,"makeup":false,"reshape":true,"smoothing":false},"enhancement_params":{},"makeup_config":{},"reshape_config":{"preset":null,"slim_factor":0.0,"eye_factor":0.0,"chin_factor":0.15,"nose_factor":0.1},"beauty_config":{}}

Request: "enhance image quality and add natural makeup"
Response: {"features":{"enhancement":true,"makeup":true,"reshape":false,"smoothing":false},"enhancement_params":{"scale":4,"amount":0.6,"chroma_boost":1.18},"makeup_config":{"preset":"natural","lipstick_color":[200,80,90],"lipstick_intensity":0.5,"blush_intensity":0.25,"eyeshadow_intensity":0.3,"eyeliner_thickness":2,"eyebrow_intensity":0.2},"reshape_config":{},"beauty_config":{}}

Request: "upscale the image and boost colors"
Response: {"features":{"enhancement":true,"makeup":false,"reshape":false,"smoothing":false},"enhancement_params":{"scale":4,"amount":0.6,"chroma_boost":1.3},"makeup_config":{},"reshape_config":{},"beauty_config":{}}

Request: "remove blemishes and brighten my skin"
Response: {"features":{"enhancement":false,"makeup":false,"reshape":false,"smoothing":true},"enhancement_params":{},"makeup_config":{},"reshape_config":{},"beauty_config":{"preset":null,"smooth_strength":0.5,"brightness_boost":1.08,"blemish_removal":true}}

Intensity mappings:
- "a little"/"slight"/"subtle": 0.1-0.3 or "subtle" preset
- "moderately"/"some": 0.4-0.6 or "natural" preset
- "hard"/"strong"/"very": 0.7-0.9 or "strong" preset
- "extremely"/"maximum": 0.9-1.0 or "intense" preset

Eyeliner keywords:
- "draw eyeliner"/"add eyeliner"/"eyeliner": eyeliner_thickness=2
- "thin eyeliner"/"subtle eyeliner": eyeliner_thickness=1
- "thick eyeliner"/"bold eyeliner"/"dramatic eyeliner": eyeliner_thickness=3
- If no eyeliner mentioned: eyeliner_thickness=0

Lipstick colors:
- "pink": [255,150,170]
- "red": [180,38,45]
- "bold red": [160,20,30]
- "nude": [220,120,130]
- "berry": [150,50,80]"""

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
                    "model": "gpt-4o-nano",
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 800
                },
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            content = result['choices'][0]['message']['content'].strip()
            
            # Extract JSON from response
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            params = json.loads(content)
            return params
            
        except Exception as e:
            print(f"⚠ NLP parsing failed: {e}")
            return self._get_default_params()
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Return safe default parameters"""
        return {
            "features": {
                "enhancement": False,
                "makeup": False,
                "reshape": False,
                "smoothing": True
            },
            "enhancement_params": {},
            "makeup_config": {},
            "reshape_config": {},
            "beauty_config": {
                "preset": "natural",
                "smooth_strength": 0.6,
                "brightness_boost": 1.05,
                "blemish_removal": True
            }
        }
    
    def validate_and_normalize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize parsed parameters"""
        
        if "features" not in params:
            params["features"] = {
                "enhancement": False,
                "makeup": False,
                "reshape": False,
                "smoothing": False
            }
        
        features = params["features"]
        for key in ["enhancement", "makeup", "reshape", "smoothing"]:
            if key not in features:
                features[key] = False
        
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
            for key in ["lipstick_intensity", "blush_intensity", "eyeshadow_intensity"]:
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
        """Print human-readable summary of parsed parameters"""
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
            if mc.get("preset"):
                print(f"  Preset: {mc['preset']}")
            else:
                if "lipstick_intensity" in mc and mc["lipstick_intensity"] > 0:
                    print(f"  Lipstick: RGB{mc.get('lipstick_color', [])}, intensity={mc['lipstick_intensity']:.2f}")
                if "blush_intensity" in mc and mc["blush_intensity"] > 0:
                    print(f"  Blush: intensity={mc['blush_intensity']:.2f}")
                if "eyeshadow_intensity" in mc and mc["eyeshadow_intensity"] > 0:
                    print(f"  Eyeshadow: intensity={mc['eyeshadow_intensity']:.2f}")
                if "eyeliner_thickness" in mc and mc["eyeliner_thickness"] > 0:
                    print(f"  Eyeliner: thickness={mc['eyeliner_thickness']}")
                if "eyebrow_intensity" in mc and mc["eyebrow_intensity"] > 0:
                    print(f"  Eyebrow enhancement: intensity={mc['eyebrow_intensity']:.2f}")
        
        if features.get("reshape") and params.get("reshape_config"):
            rc = params["reshape_config"]
            print("\nReshape:")
            if rc.get("preset"):
                print(f"  Preset: {rc['preset']}")
            else:
                for k, v in rc.items():
                    if k != "preset" and v > 0:
                        print(f"  {k}: {v:.2f}")
        
        if features.get("smoothing") and params.get("beauty_config"):
            bc = params["beauty_config"]
            print("\nBeauty Filter:")
            if bc.get("preset"):
                print(f"  Preset: {bc['preset']}")
            else:
                for k, v in bc.items():
                    if k != "preset":
                        print(f"  {k}: {v}")
        
        print("="*60 + "\n")