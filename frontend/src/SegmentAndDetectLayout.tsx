import React, { useState } from "react";
import axios from "axios";

const SegmentAndDetectLayout: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [original, setOriginal] = useState<string | null>(null);
  const [mask, setMask] = useState<string | null>(null);
  const [overlay, setOverlay] = useState<string | null>(null);
  const [yoloOverlay, setYoloOverlay] = useState<string | null>(null);
  const [numDetections, setNumDetections] = useState<number | null>(null);
  const [conf, setConf] = useState<number>(0.5);
  const [loading, setLoading] = useState<boolean>(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) setFile(e.target.files[0]);
  };

  const handleSubmit = async () => {
    if (!file) return;
    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("conf_threshold", conf.toString());

    try {
      const res = await axios.post(
        "http://127.0.0.1:8000/api/segment_and_detect",
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );

      setOriginal(`data:image/png;base64,${res.data.original}`);
      setMask(`data:image/png;base64,${res.data.mask}`);
      setOverlay(`data:image/png;base64,${res.data.overlay}`);
      setYoloOverlay(`data:image/png;base64,${res.data.yolo_overlay}`);
      setNumDetections(res.data.num_detections);
    } catch (err) {
      console.error(err);
      alert("Failed to process the image.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: "2rem" }}>
      <h1>UNet Segmentation + YOLO Detection</h1>

      <div style={{ display: "flex", gap: "1rem", alignItems: "center" }}>
        <input type="file" accept="image/*" onChange={handleFileChange} />
        <div>
          <label>YOLO Confidence: {conf}</label>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={conf}
            onChange={(e) => setConf(Number(e.target.value))}
          />
        </div>
        <button onClick={handleSubmit} disabled={loading}>
          {loading ? "Processing..." : "Run"}
        </button>
      </div>

      <div style={{ display: "flex", flexWrap: "wrap", marginTop: "2rem", gap: "2rem" }}>
        {original && (
          <div>
            <h4>Original</h4>
            <img src={original} alt="Original" style={{ maxWidth: "300px" }} />
          </div>
        )}
        {overlay && (
          <div>
            <h4>UNet Overlay</h4>
            <img src={overlay} alt="Overlay" style={{ maxWidth: "300px" }} />
          </div>
        )}
        {yoloOverlay && (
          <div>
            <h4>YOLO Overlay</h4>
            <img src={yoloOverlay} alt="YOLO Overlay" style={{ maxWidth: "300px" }} />
            <p>Detections: {numDetections}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default SegmentAndDetectLayout;
