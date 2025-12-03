interface Props {
  input: File | null;
  output: string | null;
}

export default function PreviewPane({ input, output }: Props) {
  return (
    <div>
      <h3>Results</h3>
      <div style={{ display: "flex", gap: 20 }}>
        {input && (
          <div>
            <p>Original</p>
            <img className="preview-img" src={URL.createObjectURL(input)} />
          </div>
        )}
        {output && (
          <div>
            <p>Edited</p>
            <img className="preview-img" src={output} />
          </div>
        )}
      </div>
    </div>
  );
}
