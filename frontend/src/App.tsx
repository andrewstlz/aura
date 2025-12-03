import React, { useState } from "react";
import { NLPParams } from "./types";
import NLPInput from "./components/NLPInput";
import ImageUpload from "./components/ImageUpload";
import PreviewPane from "./components/PreviewPane";
import DownloadButton from "./components/DownloadButton";

export default function App() {
  const [nlpParams, setNlpParams] = useState<NLPParams | null>(null);
  const [uploadedImage, setUploadedImage] = useState<File | null>(null);
  const [processedImage, setProcessedImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const runPipeline = async () => {
    if (!uploadedImage || !nlpParams) return;
    setLoading(true);

    const form = new FormData();
    form.append("image", uploadedImage);
    form.append("params", JSON.stringify(nlpParams));

    const res = await fetch("http://localhost:8000/process", {
      method: "POST",
      body: form,
    });

    const blob = await res.blob();
    setProcessedImage(URL.createObjectURL(blob));

    setLoading(false);
  };

  return (
    <div className="app-container">
      <h1>Aura ✨</h1>

      <div className="card">
        <ImageUpload onUpload={setUploadedImage} />
      </div>

      <div className="card">
        <NLPInput onParamsParsed={setNlpParams} />
      </div>

      <button disabled={!uploadedImage || !nlpParams} onClick={runPipeline}>
        {loading ? "Processing..." : "Apply Edits"}
      </button>

      <div className="card">
        <PreviewPane input={uploadedImage} output={processedImage} />
        {processedImage && <DownloadButton url={processedImage} />}
      </div>
    </div>
  );
}
