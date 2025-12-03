interface Props {
  url: string;
}

export default function DownloadButton({ url }: Props) {
  return (
    <a
      href={url}
      download="aura_output.png"
      style={{ display: "inline-block", marginTop: 20 }}
    >
      <button>Download Image</button>
    </a>
  );
}
