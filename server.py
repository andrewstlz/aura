# server.py
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import tempfile
import json

from NLP_interface import NaturalLanguageInterface
from pipeline import run_aura_pipeline

# Replace with environment variable later
API_KEY = "YOUR_OPENAI_API_KEY"

# Initialize NLP engine
nlp_engine = NaturalLanguageInterface(api_key=API_KEY)

# FastAPI app
app = FastAPI()

# CORS so frontend can access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/nlp")
async def parse_nlp(request: dict):
    """
    Converts natural language instructions into structured editing parameters.
    """
    text = request.get("request", "")
    params = nlp_engine.parse_request(text)
    params = nlp_engine.validate_and_normalize(params)
    return params


@app.post("/process")
async def process_image(image: UploadFile = File(...), params: str = Form(...)):
    """
    Runs the full Aura pipeline on an uploaded image.
    """
    params = json.loads(params)

    # Save uploaded image temporarily
    tmp_in = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_in.write(await image.read())
    tmp_in.close()

    # Prepare output path
    tmp_out = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_out.close()

    # Execute pipeline
    run_aura_pipeline(
        input_path=tmp_in.name,
        output_path=tmp_out.name,
        params=params
    )

    # Return image as streaming response
    def iterfile():
        with open(tmp_out.name, "rb") as f:
            yield from f

    return StreamingResponse(iterfile(), media_type="image/png")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
