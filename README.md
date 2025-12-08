# aura ✨
*your digital mirror, sprinkled with magic*  

## how it works 🫧
Aura integrates multiple pretrained computer vision models and advanced CV methods into a cohesive editing pipeline. Users input images through a web interface, describe desired enhancements via natural language or UI controls, and receive processed results in seconds. The natural language command is processed by GPT-4o-mini to map requests (e.g., "slim my face, add pink lipstick, blur the background") to specific processing modules. Each module operates independently, allowing users to apply single or combined effects while maintaining interactive performance.

## why aura is just different 🪄
- not just "another AI beauty app" — powered by U²-Net segmentation, FAN facial landmarks, and advanced blend mode compositing
- focused on natural enhancements, not artificial filters
- natural language interface — just describe what you want
- 8 professional color grading filters inspired by film cinematography
- portrait mode with realistic depth-of-field blur

## what you can expect 🌹
- **face reshaping** — slim face, enlarge eyes, refine chin and nose
- **skin smoothing** — blemish removal, guided filtering, tone balancing
- **virtual makeup** — lipstick, blush, eyeshadow, eyeliner, eyebrow enhancement
- **smile transfer** — add natural smiles using landmark-based warping
- **background blur** — portrait mode with gaussian, lens, and bokeh effects
- **color grading** — warm sunset, cool blue, vintage film, cinematic, and more
- **super-resolution** — edge-aware upscaling with Laplacian pyramids
- **background removal** — clean segmentation with transparent PNG output

## how we built it ⚙️
- **Backend / ML:** PyTorch, OpenCV, NumPy, SciPy, face-alignment (FAN)
- **NLP:** OpenAI GPT-4o-mini for intent parsing
- **Frontend:** React, Vite, TypeScript
- **Deployment:** FastAPI, Docker

## contributors
- [@wuIlxy](https://github.com/wuIlxy)
- [@andrewstlz](https://github.com/andrewstlz)
