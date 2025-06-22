import React, { useState } from "react";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [result, setResult] = useState(null);
  const [maskUrl, setMaskUrl] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setResult(null);
    setMaskUrl(null);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!selectedFile) return;
    setLoading(true);
    const formData = new FormData();
    formData.append("file", selectedFile);

    const response = await fetch("/upload", {
      method: "POST",
      body: formData,
    });

    if (response.ok) {
      const data = await response.json();
      setResult(data.vit_label);
      setMaskUrl(data.mask_url);
    } else {
      setResult("Error processing image.");
    }
    setLoading(false);
  };

  return (
    <div style={{ maxWidth: 600, margin: "auto", padding: 20 }}>
      <h2>Building Mould Detection</h2>
      <form onSubmit={handleSubmit}>
        <input type="file" accept="image/*" onChange={handleFileChange} />
        <button type="submit" disabled={!selectedFile || loading}>
          {loading ? "Processing..." : "Upload"}
        </button>
      </form>
      {result && (
        <div style={{ marginTop: 20 }}>
          <p>
            <strong>ViT Predicted Class:</strong> {result}
          </p>
          {maskUrl && (
            <img
              src={maskUrl}
              alt="Combined Mask with Label"
              style={{ maxWidth: "100%" }}
            />
          )}
        </div>
      )}
    </div>
  );
}

export default App;