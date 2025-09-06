from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from PIL import Image
from gtts import gTTS
import os
import uuid
from transformers import (
    AutoImageProcessor, AutoModelForImageClassification,
    VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer,
    pipeline
)


app = FastAPI()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

@app.get("/")
def home():
    return {"message": "Unified API for Face Emotion + Image Captioning 🚀"}

# Face Expression Model
face_processor = AutoImageProcessor.from_pretrained("trpakov/vit-face-expression")
face_model = AutoModelForImageClassification.from_pretrained("trpakov/vit-face-expression").to(device)

# Image Captioning Model
caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
caption_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
caption_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

#currency model
train_classes = ['1Hundredrupeenote',
 '2Hundredruppeenote',
 '2Thousandruppeenote',
 '5Hundredruppeenote',
 'Fiftyruppeenote',
 'Tenruppeenote',
 'Twentyruppeenote']

model_curr = models.resnet50(pretrained = True)
feat = model_curr.fc.in_features
model_curr.fc = nn.Linear(feat, len(train_classes))

state_dict = torch.load("model_weights.pth", map_location = torch.device("cpu"))
model_curr.load_state_dict(state_dict)
model_curr = model_curr.to(device)
model_curr.eval()

# Translator
translator = pipeline("translation_en_to_hi", model="Helsinki-NLP/opus-mt-en-hi")

# Static folder for audio files
if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")


# face_recognization_wala
@app.post("/recognize")
async def recognize(file: UploadFile = File(...), language: str = Form("en")):
    try:
        image = Image.open(file.file).convert("RGB")
        inputs = face_processor(images=image, return_tensors="pt").to(device)
        outputs = face_model(**inputs)
        predicted_class = outputs.logits.argmax(-1).item()
        caption = face_model.config.id2label[predicted_class]


        if language == "hi":
            res = translator(caption, max_length=50)
            caption = res[0]["translation_text"]
            tts = gTTS(caption, lang="hi")
        else:
            tts = gTTS(caption, lang="en")

        filename = f"audio_{uuid.uuid4().hex}.mp3"
        filepath = os.path.join("static", filename)
        tts.save(filepath)
        audio_url = f"http://127.0.0.1:8000/static/{filename}"

        return {"caption": caption, "audio_url": audio_url}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# image_caption_wala
@app.post("/predict")
async def predict(file: UploadFile = File(...), language: str = Form("en")):
    try:
        image = Image.open(file.file).convert("RGB")
        pixel_values = caption_extractor(images=image, return_tensors="pt").pixel_values.to(device)
        output_ids = caption_model.generate(pixel_values, max_length=16, num_beams=4)
        caption = caption_tokenizer.decode(output_ids[0], skip_special_tokens=True)

        
        if language == "hi":
            res = translator(caption, max_length=50)
            caption = res[0]["translation_text"]
            tts = gTTS(caption, lang="hi")
        else:
            tts = gTTS(caption, lang="en")

        filename = f"audio_{uuid.uuid4().hex}.mp3"
        filepath = os.path.join("static", filename)
        tts.save(filepath)
        audio_url = f"http://127.0.0.1:8000/static/{filename}"

        return {"caption": caption, "audio_url": audio_url}
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# rupee_wala
@app.post("/rupee")
async def rupee(file : UploadFile = File(...), language : str = Form("en")):

    try:
        image = Image.open(file.file).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model_curr(image)
            _, predicted = torch.max(output, 1)
        predicted_class = predicted.item()
        class_name = train_classes
        final_ans = class_name[predicted_class]

        if language == "hi":
            res_translated = translator(final_ans, max_length = 50)
            caption = res_translated[0]["translation_text"]
            tts = gTTS(caption, lang = "hi")
        
        else:
            caption = final_ans
            tts = gTTS(caption, lang = "en")
        
        filename = f"audio_{uuid.uuid4().hex}.mp3"
        filepath = os.path.join("static", filename)
        tts.save(filepath)
        audio_url = f"http://127.0.0.1:8000/static/{filename}"

        return {"prediction" : caption, "audio_url" : audio_url}
    
    except Exception as e:

        return JSONResponse(content = {"error" : str(e)}, status_code = 500)


@app.get("/get-audio/{filename}")
async def get_audio(filename: str):
    filepath = os.path.join("static", filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="audio/mpeg", filename=filename)
    return JSONResponse(content={"error": "File not found"}, status_code=404)
