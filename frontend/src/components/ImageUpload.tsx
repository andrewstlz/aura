import React from "react";

interface Props {
  onUpload: (file: File | null) => void;
}

export default function ImageUpload({ onUpload }: Props) {
  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    onUpload(e.target.files?.[0] ?? null);
  };

  return (
    <div>
      <h3>📷 Upload Image</h3>
      <input type="file" accept="image/*" onChange={handleFile} />
    </div>
  );
}
