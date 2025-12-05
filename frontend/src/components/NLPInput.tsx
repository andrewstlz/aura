import React, { useState } from "react";
import { NLPParams } from "../types";

interface Props {
  onParamsParsed: (p: NLPParams) => void;
}

export default function NLPInput({ onParamsParsed }: Props) {
  const [text, setText] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const [parsed, setParsed] = useState<any>(null);

  const handleParse = async () => {
    setStatus("Parsing...");
    setParsed(null);
    try {
      const res = await fetch("http://localhost:8000/nlp", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ request: text }),
      });

      if (!res.ok) {
        throw new Error(`NLP failed (${res.status})`);
      }

      const data = (await res.json()) as NLPParams;
      setParsed(data);
      onParamsParsed(data);
      setStatus("Parsed");
    } catch (err: any) {
      console.error(err);
      setStatus(err?.message || "Parse failed");
    }
  };

  return (
    <div>
      <h3>✨ Describe your edits</h3>
      <textarea
        rows={3}
        placeholder="Example: slim my face, soften skin, add soft pink makeup..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <button onClick={handleParse}>Parse Request</button>
      {status && <p>{status}</p>}
      {parsed && (
        <pre style={{ background: "#f3f3f3", padding: "12px", borderRadius: 8 }}>
          {JSON.stringify(parsed, null, 2)}
        </pre>
      )}
    </div>
  );
}
