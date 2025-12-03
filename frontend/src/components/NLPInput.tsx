import React, { useState } from "react";
import { NLPParams } from "../types";

interface Props {
  onParamsParsed: (p: NLPParams) => void;
}

export default function NLPInput({ onParamsParsed }: Props) {
  const [text, setText] = useState("");

  const handleParse = async () => {
    const res = await fetch("http://localhost:8000/nlp", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ request: text }),
    });

    const data = (await res.json()) as NLPParams;
    onParamsParsed(data);
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
    </div>
  );
}
