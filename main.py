from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import whisper
import os

app = FastAPI()

# Create necessary directories if they don’t exist
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("temp", exist_ok=True)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve HTML templates
templates = Jinja2Templates(directory="templates")

# Load Whisper Model
model = whisper.load_model("base")

@app.get("/", response_class=HTMLResponse)
async def serve_homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/transcribe/")
async def transcribe_audio(files: list[UploadFile] = File(...)):
    transcriptions = {}

    for file in files:
        file_path = f"temp/{file.filename}"

        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Transcribe the audio
        result = model.transcribe(file_path)

        # Store transcription
        transcriptions[file.filename] = result["text"]

    return JSONResponse(content={"transcriptions": transcriptions})

