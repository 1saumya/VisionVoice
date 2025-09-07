import React, { useState, useRef } from "react";
import Webcam from "react-webcam";
import "./App.css"

export default function App() {
  const [voiceText, setVoiceText] = useState("");
  const [currentMode, setCurrentMode] = useState(null);
  const [result, setResult] = useState("");
  const [audioUrl, setAudioUrl] = useState("");
  const [file, setFile] = useState(null);
  const [cameraOn, setCameraOn] = useState(false);
  const [language, setLanguage] = useState("en")
  const [loading, setLoading] = useState(false);


  const webcamRef = useRef(null);

  // 🎤 Voice recognition
  const startListening = () => {
    const recognition = new (window.SpeechRecognition ||
      window.webkitSpeechRecognition)();
    recognition.lang = "en-US";

    recognition.onresult = (event) => {
      const text = event.results[0][0].transcript.toLowerCase();
      setVoiceText(text);

      if (text.includes("mood") || text.includes("emotion")) {
        setCurrentMode("face");
      } 
      else if (text.includes("image") || text.includes("photo")) {
        setCurrentMode("image");
      } 
      else if(text.includes("note") || text.includes("paisa")) {
        setCurrentMode("rupee")
      }
      else if(text.includes("colour") || text.includes("shade")) {
        setCurrentMode("colour")
      }
      else {
        alert("Say 'mood' for face recognition OR 'image' for captioning");
      }
    };

    recognition.start();
  };

  // File select
  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

   const captureImage = () => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();

      // Convert base64 → File object
      const byteString = atob(imageSrc.split(",")[1]);
      const mimeString = imageSrc.split(",")[0].split(":")[1].split(";")[0];
      const ab = new ArrayBuffer(byteString.length);
      const ia = new Uint8Array(ab);
      for (let i = 0; i < byteString.length; i++) {
        ia[i] = byteString.charCodeAt(i);
      }
      const blob = new Blob([ab], { type: mimeString });
      const file = new File([blob], "capture.png", { type: mimeString });
      setFile(file);
      alert("Image captured!");
    }
  };

  // Send to backend
  const sendFile = async () => {
    if (!file || !currentMode) {
      alert("Please record voice first and upload file.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("language", language);

    let endpoint = "";
    if (currentMode === "face") endpoint = "/recognize";
    else if (currentMode === "image") endpoint = "/predict";
    else if (currentMode === "rupee") endpoint = "/rupee"
    else endpoint = "/colour"

    try {
      setLoading(true);
      const res = await fetch("http://127.0.0.1:8000" + endpoint, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      
      setResult(data.caption);
      setAudioUrl(data.audio_url);

    } 
    catch (err) {
      console.error(err);
      setResult("Error calling API");
    }
    finally{
      setLoading(false);
    }
  };

  return (
    <div className="container">

      <div className="card" style={{background : '#054385ff',  padding: "20px", fontFamily: "sans-serif" }}>
        <h1 style={{textAlign : 'center', marginBottom : '45px', color : '#a8cef6ff'}}>👁️VisionVoice</h1>
        <button className="voice" onClick={startListening}>Start Voice</button>
        <p className="said"><b>You said:</b> {voiceText}</p>

        {/* Language */}
        <div className="lang">

          <h3>🌐 Select Language</h3>
          <select value={language} onChange={(e) => setLanguage(e.target.value)} name="" id="">
            <option value={"en"}>English</option>
            <option value={"hi"}>Hindi</option>
          </select>
        </div>

        <div className="upload">
          <div className="up">

            <h3>Upload File</h3>
            <input type="file" onChange={handleFileChange} />
          </div>

          <h3>OR</h3>

          <div className="camera">

            <h3>📷 Camera</h3>
              {!cameraOn ? (
                <button className="btn_camera" onClick={() => setCameraOn(true)}>Open Camera</button>
              ) : (
                <>
                  <Webcam
                    audio={false}
                    ref={webcamRef}
                    screenshotFormat="image/png"
                    style={{ width: "320px", borderRadius: "10px" }}
                  />
                  <div>
                    <button className="btn_capture" onClick={captureImage}>Capture Image</button>
                    <button className="btn_close_camera" onClick={() => setCameraOn(false)}>Close Camera</button>
                  </div>
                </>
              )}
          </div>

        </div>
          <button className="submit" onClick={sendFile}>Submit</button>

        {loading && (
          <p style={{ color: "yellow", fontWeight: "bold" }}>
            ⏳ Analyzing the image...
          </p>
        )}

        <h3>Result</h3>
        <p>{result}</p>
        {audioUrl && <audio controls src={audioUrl}></audio>}
      </div>
    </div>
  );
}
