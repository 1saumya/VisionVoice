from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from gtts import gTTS
from sklearn.cluster import KMeans
import numpy as np
import webcolors
import os
import uuid
from transformers import pipeline

app = FastAPI()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cpu"  # keep CPU only (Render free tier has no GPU)

@app.get("/")
def home():
    return {"message": "Unified API for Face Emotion + Image Captioning 🚀"}

# ---------------- GLOBALS (Lazy Init) ---------------- #
face_model = None
face_processor = None

caption_model = None
caption_extractor = None
caption_tokenizer = None

model_curr = None
train_classes = ['1Hundredrupeenote','2Hundredruppeenote','2Thousandruppeenote',
 '5Hundredruppeenote','Fiftyruppeenote','Tenruppeenote','Twentyruppeenote']

translator = None

# ---------------- UTILS ---------------- #
def get_translator():
    global translator
    if translator is None:
        translator = pipeline("translation_en_to_hi",
                              model="Helsinki-NLP/opus-mt-en-hi")
    return translator

def get_dominant_color(image_path, k=3):
    image = Image.open(image_path).convert("RGB").resize((150, 150))
    arr = np.array(image).reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, random_state=42).fit(arr)
    colors = kmeans.cluster_centers_
    counts = np.bincount(kmeans.labels_)
    dominant_color = colors[counts.argmax()]
    return "#{:02x}{:02x}{:02x}".format(int(dominant_color[0]),
                                        int(dominant_color[1]),
                                        int(dominant_color[2]))

def closest_color(requested_hex):
    try:
        return webcolors.hex_to_name(requested_hex)
    except ValueError:
        requested_rgb = webcolors.hex_to_rgb(requested_hex)
        min_dist, closest_name = float("inf"), None
        for name in webcolors.names("css3"):
            r_c, g_c, b_c = webcolors.hex_to_rgb(webcolors.name_to_hex(name))
            dist = (r_c - requested_rgb[0])**2 + (g_c - requested_rgb[1])**2 + (b_c - requested_rgb[2])**2
            if dist < min_dist:
                min_dist, closest_name = dist, name
        return closest_name

def save_tts(caption, lang="en"):
    filename = f"audio_{uuid.uuid4().hex}.mp3"
    filepath = os.path.join("static", filename)
    gTTS(caption, lang=lang).save(filepath)
    return f"/static/{filename}"

# ---------------- ENDPOINTS ---------------- #
@app.post("/recognize")
async def recognize(file: UploadFile = File(...), language: str = Form("en")):
    global face_model, face_processor
    try:
        if face_model is None:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            face_processor = AutoImageProcessor.from_pretrained("trpakov/vit-face-expression")
            face_model = AutoModelForImageClassification.from_pretrained("trpakov/vit-face-expression").to(device)

        image = Image.open(file.file).convert("RGB")
        inputs = face_processor(images=image, return_tensors="pt").to(device)
        outputs = face_model(**inputs)
        predicted_class = outputs.logits.argmax(-1).item()
        caption = face_model.config.id2label[predicted_class]

        if language == "hi":
            caption = get_translator()(caption)[0]["translation_text"]
            audio_url = save_tts(caption, "hi")
        else:
            audio_url = save_tts(caption, "en")

        return {"caption": caption, "audio_url": audio_url}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/predict")
async def predict(file: UploadFile = File(...), language: str = Form("en")):
    global caption_model, caption_extractor, caption_tokenizer
    try:
        if caption_model is None:
            from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
            caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
            caption_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            caption_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

        image = Image.open(file.file).convert("RGB")
        pixel_values = caption_extractor(images=image, return_tensors="pt").pixel_values.to(device)
        output_ids = caption_model.generate(pixel_values, max_length=16, num_beams=4)
        caption = caption_tokenizer.decode(output_ids[0], skip_special_tokens=True)

        if language == "hi":
            caption = get_translator()(caption)[0]["translation_text"]
            audio_url = save_tts(caption, "hi")
        else:
            audio_url = save_tts(caption, "en")

        return {"caption": caption, "audio_url": audio_url}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/rupee")
async def rupee(file: UploadFile = File(...), language: str = Form("en")):
    global model_curr
    try:
        if model_curr is None:
            model_curr = models.resnet50(pretrained=True)
            feat = model_curr.fc.in_features
            model_curr.fc = nn.Linear(feat, len(train_classes))
            state_dict = torch.load("model_weights.pth", map_location="cpu")
            model_curr.load_state_dict(state_dict)
            model_curr = model_curr.to(device).eval()

        image = Image.open(file.file).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model_curr(image)
            _, predicted = torch.max(output, 1)
        caption = train_classes[predicted.item()]

        if language == "hi":
            caption = get_translator()(caption)[0]["translation_text"]
            audio_url = save_tts(caption, "hi")
        else:
            audio_url = save_tts(caption, "en")

        return {"prediction": caption, "audio_url": audio_url}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/colour")
async def color(file: UploadFile = File(...), language: str = Form("en")):
    try:
        filepath = f"temp_{file.filename}"
        with open(filepath, "wb") as f:
            f.write(await file.read())

        dominant_color = get_dominant_color(filepath)
        res = closest_color(dominant_color)

        if language == "hi":
            caption = get_translator()(res, max_length=15)[0]["translation_text"]
            audio_url = save_tts(caption, "hi")
        else:
            caption = res
            audio_url = save_tts(caption, "en")

        return {"caption": caption, "audio_url": audio_url}
    except Exception as e:
        return {"error": str(e)}


@app.get("/get-audio/{filename}")
async def get_audio(filename: str):
    filepath = os.path.join("static", filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="audio/mpeg", filename=filename)
    return JSONResponse(content={"error": "File not found"}, status_code=404)


if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/assets", StaticFiles(directory="frontend/dist/assets"), name="assets")

@app.get("/{full_path:path}")
async def serve_react(full_path: str):
    return FileResponse("frontend/dist/index.html")
