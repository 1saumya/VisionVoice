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
    } 
    catch (err) {
      console.error(err);
      alert("Error uploading file!");
    } 
    finally {
      setLoading(false);
    }
  };


  const openCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      setCameraStream(stream);
    } 
    catch (err) {
      alert("Camera not accessible!");
    }
  };

  return (
    <div className="app">
      <div className="card">
        {/* Logo / Title */}
        <h1 className="title">
          <span className="logo">🔮 VisionVoice</span>
        </h1>

        {/* Upload Box */}
        <div className="upload-box">
          <label className="label">Upload Image</label>
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
          />
        </div>


        <button className="btn secondary" onClick={openCamera}>
          📷 Open Camera
        </button>

        {cameraStream && (
        <div className="camera-container">
          <video
            autoPlay
            playsInline
            className="camera-view"
            ref={(video) => {
              if (video) video.srcObject = cameraStream;
            }}
          />
          
          {/* Capture Button */}
          <button
            className="btn primary"
            onClick={() => {
              const video = document.querySelector("video");
              const canvas = document.createElement("canvas");
              canvas.width = video.videoWidth;
              canvas.height = video.videoHeight;
              const ctx = canvas.getContext("2d");
              ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

              canvas.toBlob((blob) => {
                if (blob) {
                  const capturedFile = new File([blob], "captured.png", { type: "image/png" });
                  setFile(capturedFile);
                  setPreview(URL.createObjectURL(capturedFile)); // ✅ preview dikhega
                }
              }, "image/png");
            }}
          >
            📸 Capture Image
          </button>

          <button
            className="btn danger"
            onClick={() => {
              cameraStream.getTracks().forEach((track) => track.stop());
              setCameraStream(null);
            }}
          >
            ❌ Close Camera
          </button>
        </div>
      )}

        
        {/* Image preview */}
        {preview && (
          <div style={{ marginTop: "20px" }}>
            <h3>Preview:</h3>
            <img
              src={preview}
              alt="Selected"
              style={{ maxWidth: "80%", borderRadius: "10px", boxShadow: "0px 0px 15px rgba(0,0,0,0.3)" }}
            />
          </div>
        )}

        {/* Language Options */}
        <div className="lang-box">
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
          <label>
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

        {/* Generate Button */}
        <button className="btn primary" onClick={handleUpload}>
          ✨ Generate Caption
        </button>


        {/* Analyzing state */}
        {loading && (
          <div className="caption-box analyzing">
            <p>⏳ Analyzing the image...</p>
          </div>
        )}


        {/* Caption Box */}
        {caption && (
          <div className="caption-box">
            <p>{caption}</p>
            <button
              className="btn speak-btn"
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
    </div>
  );
}
