import streamlit as st
import os
import uuid
import cv2
from PIL import Image

from NLP_interface import NaturalLanguageInterface
from aura_pipeline import run_aura_pipeline   # You will create/adapt this

# ---------------------------------------------------
# Config
# ---------------------------------------------------
os.makedirs("streamlit_outputs", exist_ok=True)
API_KEY = "YOUR_OPENAI_API_KEY"
nlp = NaturalLanguageInterface(API_KEY)

OUTPUT_DIR = "streamlit_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------
# Streamlit UI
# ---------------------------------------------------
st.set_page_config(page_title="Aura Photo Editor", layout="wide")
st.title("Aura Photo Editing Demo")

st.write("Upload a photo and describe how you want it edited.")ss

# ---------------------------------------------------
# Image Upload
# ---------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload an image", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=False
)

if uploaded_file:
    input_image = Image.open(uploaded_file).convert("RGB")
    st.image(input_image, caption="Uploaded Photo", width=350)

# ---------------------------------------------------
# NLP Text Input
# ---------------------------------------------------
prompt = st.text_area(
    "Describe the edits you want:",
    placeholder="e.g., 'Make me smile a little, smooth skin naturally.'"
)

run_button = st.button("Run Aura 🚀")

# ---------------------------------------------------
# When Run Button is Clicked
# ---------------------------------------------------
if run_button:

    if uploaded_file is None:
        st.error("Please upload an image first.")
        st.stop()

    if not prompt.strip():
        st.error("Please enter an editing command.")
        st.stop()

    with st.spinner("Parsing natural language request..."):
        params = nlp.parse_request(prompt)
        params = nlp.validate_and_normalize(params)
    
    st.success("NLP Parsing Complete!")
    
    st.subheader("Interpreted Features")
    st.json(params)

    st.write("---")

    # Save temp input
    input_path = os.path.join(OUTPUT_DIR, f"{uuid.uuid4()}.jpg")
    input_image.save(input_path)

    st.info("Running Aura pipeline... This may take ~3–10 seconds.")

    # ---------------------------------------------------
    # Run the actual processing pipeline
    # ---------------------------------------------------
    with st.spinner("Generating enhanced image..."):
        result_path = run_aura_pipeline(
            prompt,
            input_path,
            smile_source_path=params["smile_config"]["source_image_path"]
        )

    st.success("Done!")

    # ---------------------------------------------------
    # Display before/after
    # ---------------------------------------------------
    st.subheader("📸 Before / After")

    col1, col2 = st.columns(2)

    with col1:
        st.image(input_path, caption="Before", use_column_width=True)

    with col2:
        st.image(result_path, caption="After", use_column_width=True)

    # ---------------------------------------------------
    # Download Button
    # ---------------------------------------------------
    with open(result_path, "rb") as f:
        st.download_button(
            label="Download Edited Image",
            data=f,
            file_name="aura_edited.jpg",
            mime="image/jpeg"
        )

