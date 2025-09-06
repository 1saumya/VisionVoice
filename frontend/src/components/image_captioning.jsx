import React, { useState } from "react";
import axios from "axios";
import "./App.css";

export default function App() {
  const [file, setFile] = useState(null);
  const [caption, setCaption] = useState("");
  const [language, setLanguage] = useState("en");
  const [cameraStream, setCameraStream] = useState(null);
  const [loading, setLoading] = useState(false);
  const [preview, setPreview] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Please select an image first!");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("language", language);

    try {
      setLoading(true);
      setCaption("");
      const res = await axios.post("http://127.0.0.1:8000/predict/", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setCaption(res.data.caption);
    } catch (err) {
      console.error(err);
      alert("Error uploading file!");
    } finally {
      setLoading(false);
    }
  };

  const openCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      setCameraStream(stream);
    } catch (err) {
      alert("Camera not accessible!");
    }
  };

  return (
    <div className="app" style={{ textAlign: "center", padding: "20px" }}>
      <h1 style={{ marginBottom: "20px" }}>🔮 VisionVoice</h1>

      {/* Upload Box */}
      <input type="file" accept="image/*" onChange={handleFileChange} />

      <div style={{ marginTop: "10px" }}>
        <button onClick={openCamera}>📷 Open Camera</button>
      </div>

      {cameraStream && (
        <div style={{ marginTop: "10px" }}>
          <video
            autoPlay
            playsInline
            width="250"
            ref={(video) => {
              if (video) video.srcObject = cameraStream;
            }}
          />
          <div style={{ marginTop: "10px" }}>
            <button
              onClick={() => {
                const video = document.querySelector("video");
                const canvas = document.createElement("canvas");
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                const ctx = canvas.getContext("2d");
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

                canvas.toBlob((blob) => {
                  if (blob) {
                    const capturedFile = new File([blob], "captured.png", {
                      type: "image/png",
                    });
                    setFile(capturedFile);
                    setPreview(URL.createObjectURL(capturedFile));
                  }
                }, "image/png");
              }}
            >
              📸 Capture
            </button>
            <button
              style={{ marginLeft: "10px" }}
              onClick={() => {
                cameraStream.getTracks().forEach((track) => track.stop());
                setCameraStream(null);
              }}
            >
              ❌ Close
            </button>
          </div>
        </div>
      )}

      {/* Preview */}
      {preview && (
        <div style={{ marginTop: "20px" }}>
          <h3>Preview:</h3>
          <img
            src={preview}
            alt="Selected"
            style={{ maxWidth: "250px", borderRadius: "6px" }}
          />
        </div>
      )}

      {/* Language Options */}
      <div style={{ marginTop: "15px" }}>
        <label>
          <input
            type="radio"
            name="language"
            value="en"
            checked={language === "en"}
            onChange={(e) => setLanguage(e.target.value)}
          />
          English
        </label>
        <label style={{ marginLeft: "10px" }}>
          <input
            type="radio"
            name="language"
            value="hi"
            checked={language === "hi"}
            onChange={(e) => setLanguage(e.target.value)}
          />
          Hindi
        </label>
      </div>

      {/* Generate */}
      <div style={{ marginTop: "15px" }}>
        <button onClick={handleUpload}>✨ Generate Caption</button>
      </div>

      {/* Loading */}
      {loading && <p style={{ marginTop: "10px" }}>⏳ Analyzing...</p>}

      {/* Caption */}
      {caption && (
        <div style={{ marginTop: "15px" }}>
          <p>{caption}</p>
          <button
            onClick={() => {
              const utterance = new SpeechSynthesisUtterance(caption);
              utterance.lang = language === "hi" ? "hi-IN" : "en-US";
              speechSynthesis.speak(utterance);
            }}
          >
            🔊 Speak
          </button>
        </div>
      )}
    </div>
  );
}
