"""
Natural Language Interface for Aura Photo Editing Pipeline
"""

import json
import re
import requests
from typing import Dict, Any


class NaturalLanguageInterface:
    """Parse natural language editing requests into structured parameters"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://api.openai.com/v1/chat/completions"
        
    def parse_request(self, user_request: str) -> Dict[str, Any]:
        """Convert natural language request to structured parameters"""
        
        system_prompt = """You are a professional photo editing parameter parser. Convert natural language requests into JSON parameters.

You MUST respond with ONLY a valid JSON object, nothing else. No explanations, no markdown, no extra text.

JSON Structure:
{
  "features": {
    "enhancement": boolean,
    "makeup": boolean,
    "reshape": boolean,
    "smoothing": boolean,
    "color_grading": boolean,
    "background_replacement": boolean
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
  "color_grading_config": {
    "filter_name": string ("warm_sunset"/"cool_blue"/"vintage_film"/"high_contrast"/"soft_pastel"/"cinematic"/"moody_dark"/"bright_airy") or null,
    "intensity": float (0.0-1.0)
  },
  "background_config": {
    "background_path": string or null
  }
}

Examples:

Input: "slim my face a little, smooth my face hard, add pink lipstick"
Output: {"features":{"enhancement":false,"makeup":true,"reshape":true,"smoothing":true,"color_grading":false,"background_replacement":false},"enhancement_params":{},"makeup_config":{"preset":null,"lipstick_color":[255,150,170],"lipstick_intensity":0.7,"blush_intensity":0.0,"eyeshadow_intensity":0.0,"eyeliner_thickness":0,"eyebrow_intensity":0.0},"reshape_config":{"preset":null,"slim_factor":0.08,"eye_factor":0.0,"chin_factor":0.0,"nose_factor":0.0},"beauty_config":{"preset":null,"smooth_strength":0.8,"brightness_boost":1.05,"blemish_removal":true},"color_grading_config":{},"background_config":{}}

Input: "apply warm filter"
Output: {"features":{"enhancement":false,"makeup":false,"reshape":false,"smoothing":false,"color_grading":true,"background_replacement":false},"enhancement_params":{},"makeup_config":{},"reshape_config":{},"beauty_config":{},"color_grading_config":{"filter_name":"warm_sunset","intensity":0.8},"background_config":{}}

Input: "make it look cinematic"
Output: {"features":{"enhancement":false,"makeup":false,"reshape":false,"smoothing":false,"color_grading":true,"background_replacement":false},"enhancement_params":{},"makeup_config":{},"reshape_config":{},"beauty_config":{},"color_grading_config":{"filter_name":"cinematic","intensity":0.8},"background_config":{}}

Input: "vintage look with natural makeup"
Output: {"features":{"enhancement":false,"makeup":true,"reshape":false,"smoothing":false,"color_grading":true,"background_replacement":false},"enhancement_params":{},"makeup_config":{"preset":"natural","lipstick_color":[200,80,90],"lipstick_intensity":0.5,"blush_intensity":0.25,"eyeshadow_intensity":0.3,"eyeliner_thickness":2,"eyebrow_intensity":0.2},"reshape_config":{},"beauty_config":{},"color_grading_config":{"filter_name":"vintage_film","intensity":0.8},"background_config":{}}

Input: "replace background with beach"
Output: {"features":{"enhancement":false,"makeup":false,"reshape":false,"smoothing":false,"color_grading":false,"background_replacement":true},"enhancement_params":{},"makeup_config":{},"reshape_config":{},"beauty_config":{},"color_grading_config":{},"background_config":{"background_path":"beach"}}

Intensity: "a little"→0.1-0.3, "moderately"→0.4-0.6, "hard"/"strong"→0.7-0.9, "extremely"→0.9-1.0

Lipstick: "pink"→[255,150,170], "red"→[180,38,45], "bold red"→[160,20,30], "nude"→[220,120,130]

Color Grading Keywords:
- "warm"/"golden hour"/"sunset"/"orange tones": filter_name="warm_sunset"
- "cool"/"blue tones"/"cold"/"cyan": filter_name="cool_blue"
- "vintage"/"retro"/"film"/"old photo": filter_name="vintage_film"
- "high contrast"/"dramatic"/"punchy": filter_name="high_contrast"
- "pastel"/"soft colors"/"dreamy": filter_name="soft_pastel"
- "cinematic"/"movie"/"film look": filter_name="cinematic"
- "moody"/"dark"/"dramatic dark": filter_name="moody_dark"
- "bright"/"airy"/"light"/"clean": filter_name="bright_airy"

Background: "beach"/"nature"/"city"/"studio" or null

Remember: Output ONLY the JSON object, nothing else."""

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
                    "temperature": 0.2,
                    "max_tokens": 1500
                },
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            content = result['choices'][0]['message']['content'].strip()
            
            params = self._extract_json(content)
            
            if params is None:
                print(f"⚠ Could not parse JSON from response:")
                print(f"  Response preview: {content[:200]}...")
                return self._get_default_params()
            
            return params
            
        except requests.exceptions.RequestException as e:
            print(f"⚠ API request failed: {e}")
            return self._get_default_params()
        except Exception as e:
            print(f"⚠ NLP parsing failed: {e}")
            import traceback
            traceback.print_exc()
            return self._get_default_params()
    
    def _extract_json(self, content: str) -> Dict[str, Any]:
        """Extract JSON from response using multiple strategies"""
        
        # Strategy 1: Direct parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Remove markdown
        if '```json' in content:
            try:
                json_str = content.split('```json')[1].split('```')[0].strip()
                return json.loads(json_str)
            except:
                pass
        
        if '```' in content:
            try:
                json_str = content.split('```')[1].split('```')[0].strip()
                return json.loads(json_str)
            except:
                pass
        
        # Strategy 3: Find JSON with regex
        try:
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except:
            pass
        
        # Strategy 4: Extract valid JSON substring
        try:
            stack = []
            start_idx = None
            
            for i, char in enumerate(content):
                if char == '{':
                    if not stack:
                        start_idx = i
                    stack.append(char)
                elif char == '}':
                    if stack:
                        stack.pop()
                        if not stack and start_idx is not None:
                            json_str = content[start_idx:i+1]
                            try:
                                return json.loads(json_str)
                            except json.JSONDecodeError:
                                continue
        except Exception:
            pass
        
        return None
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Return safe default parameters"""
        return {
            "features": {
                "enhancement": False,
                "makeup": False,
                "reshape": False,
                "smoothing": True,
                "color_grading": False,
                "background_replacement": False
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
            "color_grading_config": {},
            "background_config": {}
        }
    
    def validate_and_normalize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize parsed parameters"""
        
        if "features" not in params:
            params["features"] = {
                "enhancement": False,
                "makeup": False,
                "reshape": False,
                "smoothing": False,
                "color_grading": False,
                "background_replacement": False
            }
        
        features = params["features"]
        for key in ["enhancement", "makeup", "reshape", "smoothing", "color_grading", "background_replacement"]:
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
            for key in ["lipstick_intensity", "blush_intensity", "eyeshadow_intensity", "eyebrow_intensity"]:
                if key in mc:
                    mc[key] = max(0.0, min(1.0, float(mc[key])))
            if "eyeliner_thickness" in mc:
                mc["eyeliner_thickness"] = max(0, min(3, int(mc["eyeliner_thickness"])))
        
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
        
        if "color_grading_config" in params and params["color_grading_config"]:
            cg = params["color_grading_config"]
            if "intensity" in cg:
                cg["intensity"] = max(0.0, min(1.0, float(cg["intensity"])))
        
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
            print(f"  ✓ {feature.replace('_', ' ').capitalize()}")
        
        if features.get("background_replacement") and params.get("background_config"):
            bg = params["background_config"]
            print("\nBackground Replacement:")
            if bg.get("background_path"):
                print(f"  Background: {bg['background_path']}")
            else:
                print(f"  Background: (manual path required)")
        
        if features.get("color_grading") and params.get("color_grading_config"):
            cg = params["color_grading_config"]
            print("\nColor Grading:")
            if cg.get("filter_name"):
                print(f"  Filter: {cg['filter_name']}")
            if "intensity" in cg:
                print(f"  Intensity: {cg['intensity']:.2f}")
        
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