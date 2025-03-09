import os
import json
import numpy as np
import librosa
import joblib
import requests
import tempfile
import logging
import cv2
import speech_recognition as sr
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from pydub import AudioSegment
import noisereduce as nr
import re
import google.generativeai as genai
from dotenv import load_dotenv
import ffmpeg

load_dotenv()

app = FastAPI(title="HireWise AI Analysis API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models and scalers
audio_confidence_model = None
audio_fluency_model = None
cnn_model = None
scaler_confidence = None
scaler_fluency = None


@app.post("/process-video")
async def process_video(request: Request):
    data = await request.json()
    cloudinary_url = data.get("cloudinary_url")
    question = data.get("question")
    temp_media_path = None
    temp_audio_path = None

    try:
        # Step 1: Download video
        response = requests.get(cloudinary_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download media file")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_media:
            temp_media.write(response.content)
            temp_media_path = temp_media.name

        # Step 2: Process video frames for confidence
        video_confidence = round(float(process_video_frames(temp_media_path) * 100), 1)
        
        # Step 3: Extract and validate audio
        temp_audio_path = extract_audio(temp_media_path)
        if not temp_audio_path:
            return {
                "status": "failure",
                "message": "Audio extraction failed",
            }

        # Step 4: Analyze fluency, confidence & speech energy
        audio_features = process_audio_features(temp_audio_path)
        if audio_features["is_silent"]:
            return {
                "status": "failure",
                "message": "There is no audio",
            }

        # Step 5: Transcribe audio & detect filler words
        transcription = transcribe_audio(temp_audio_path)
        fluency = predict_fluency(audio_features)
        audio_conf = predict_confidence(audio_features)

        # Step 6: Analyze grammar, enhance response, check relevance
        analysis_result = analyze_response(question, transcription.get("transcription"))

        return {
            "status": "success",
            "message": "Processing complete",
            "video_confidence": video_confidence,
            "audio_confidence": round(float(audio_conf * 100), 1),
            "fluency_percentage": round(float(fluency * 100), 1),
            "transcription": transcription,
            "grammar": analysis_result
        }

    finally:
        # Cleanup temporary files
        if temp_media_path and os.path.exists(temp_media_path):
            os.remove(temp_media_path)
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

@app.on_event("startup")
def load_models():
    """Loads all ML models and scalers at startup"""
    global audio_confidence_model, audio_fluency_model, scaler_confidence, scaler_fluency, cnn_model
    
    try:

        audio_confidence_model = load_tflite_model(r"D:\Final Year Project\Final Year Project Code\hirewise python backend\app\confidence_model.tflite")
        audio_fluency_model = load_tflite_model(r"D:\Final Year Project\Final Year Project Code\hirewise python backend\app\fluency_model.tflite")
        cnn_model = load_tflite_model(r"D:\Final Year Project\Final Year Project Code\hirewise python backend\app\video_confidence.tflite")
        scaler_confidence = joblib.load(r"D:\Final Year Project\Final Year Project Code\hirewise python backend\app\confidence_scaler.pkl")
        scaler_fluency = joblib.load(r"D:\Final Year Project\Final Year Project Code\hirewise python backend\app\fluency_scaler.pkl")
        
        logger.info("✅ All models and scalers loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}")

def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()  # Allocate memory for the model
    return interpreter

import cv2
import numpy as np

def process_video_frames(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (128, 128)) / 255.0  # Resize and normalize
            frames.append(frame)
            for _ in range(9):  # Skip frames to reduce processing load
                cap.read()
                
        if not frames:
            return 0.0  # No valid frames, return 0 confidence
        
        frames = np.array(frames, dtype=np.float32)  # Convert list to NumPy array

        # Get model input shape dynamically
        input_shape = cnn_model.get_input_details()[0]['shape']  # (1, 128, 128, 3)
        batch_size = input_shape[0]  # Typically 1 for inference

        # Ensure correct shape (batch_size, 128, 128, 3)
        if frames.shape[0] != batch_size:
            frames = np.expand_dims(frames[0], axis=0)  # Use first frame only

        predictions = predict_with_tflite(cnn_model, frames)
        
        return float(np.mean(predictions))  # Average confidence across frames
        
    finally:
        cap.release()

def extract_audio(video_path: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        audio_path = temp_audio.name

    (
        ffmpeg.input(video_path)
        .output(audio_path, ac=1, ar=16000)
        .overwrite_output()
        .run(quiet=True)
    )

    # Validate extracted audio
    audio, sr = librosa.load(audio_path, sr=16000)
    
    if len(audio) == 0 or np.max(np.abs(audio)) < 0.001:
        os.remove(audio_path)
        return None

    return audio_path

def predict_with_tflite(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Ensure input data has the correct shape
    input_shape = input_details[0]['shape']
    input_data = np.array(input_data, dtype=np.float32).reshape(input_shape)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()  # Run inference

    output_data = interpreter.get_tensor(output_details[0]['index'])
    return float(np.mean(output_data))  # Return mean confidence score

def process_audio_features(audio_path: str) -> dict:
    audio, sr = librosa.load(audio_path, sr=16000, duration=5)
    audio = nr.reduce_noise(y=audio, sr=sr)
    rms_energy = librosa.feature.rms(y=audio).mean()
    
    silence_threshold = 0.005
    is_silent = rms_energy < silence_threshold

    if is_silent:
        return {
            "mfcc": np.zeros(40),
            "speech_rate": 0,
            "pitch": [0],
            "energy": rms_energy,
            "is_silent": True
        }

    return {
        "mfcc": librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).mean(axis=1),
        "speech_rate": len(librosa.effects.split(audio)[0]),
        "pitch": librosa.yin(audio, fmin=50, fmax=500) if len(audio) > 0 else [0],
        "energy": rms_energy,
        "is_silent": False
    }

def predict_fluency(features: dict) -> float:
    if features["is_silent"]:
        return 0.0

    feature_array = np.append(features["mfcc"], features["speech_rate"])
    scaled_features = scaler_fluency.transform([feature_array])
    reshaped_features = scaled_features.reshape((1, 41, 1))

    return float(predict_with_tflite(audio_fluency_model, reshaped_features))


def predict_confidence(features: dict) -> float:
    if features["is_silent"]:
        return 0.0

    feature_array = np.array([
        np.mean(features["pitch"]),
        np.std(features["pitch"]),
        features["energy"]
    ])
    scaled_features = scaler_confidence.transform([feature_array])
    reshaped_features = scaled_features.reshape((1, 3, 1))

    return float(predict_with_tflite(audio_confidence_model, reshaped_features))


# Initialize Gemini AI client
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def analyze_response(question: str, transcript: str) -> dict:
    """Sends the transcript and question to Gemini AI for grammar correction and enhancement."""
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = f"""
            Given the following interview question and candidate response (transcript), analyze the response using the following criteria:

            1. Identify and list grammar mistakes in an array called "grammar_mistakes."
            2. Provide a more powerful, concise version of the response under "enhanced_response."
            3. Offer constructive feedback in "feedback" to help the candidate improve their response.

            Return the result in a structured JSON format as follows:
            {{
              "grammar_mistakes": ["List of grammar mistakes"],
              "enhanced_response": "Improved and more concise version of the response",
              "feedback": ["List of specific improvement suggestions"]
            }}

            Interview Question: "{question}"
            Candidate Response (Transcript): "{transcript}"
        """


        response = model.generate_content(prompt)

        # Extract text response & clean potential formatting issues
        cleaned_text = response.text.strip().replace("```json", "").replace("```", "").strip()

        # Try parsing as JSON
        try:
            parsed_response = json.loads(cleaned_text)
            return {
                "grammar_mistakes": parsed_response.get("grammar_mistakes", []),
                "enhanced_response": parsed_response.get("enhanced_response", "").strip(),
                "feedback": parsed_response.get("feedback", [])
            }
        except json.JSONDecodeError:
            return {
                "grammar_mistakes": [],
                "enhanced_response": cleaned_text, 
                "feedback": []
            }

    except Exception as e:
        return {
            "grammar_mistakes": [],
            "enhanced_response": "Could not generate improved response.",
            "feedback": []
        }

# Define common filler words, including "aa"
FILLER_WORDS = ["um", "uh", "like", "you know", "so", "actually", "basically", "right", "okay", "hmm"]

def transcribe_audio(audio_path: str) -> dict:
    """Transcribes audio, detects filler words, and analyzes speech energy for missing fillers like 'aa'."""
    recognizer = sr.Recognizer()
    
    try:
        with sr.AudioFile(audio_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source)
            transcript = recognizer.recognize_google(audio_data).lower()  # Normalize case
            
            if not transcript.strip():
                return {"transcription": "Could not process audio", "filler_words": {}, "total_fillers": 0, "energy": 0.0}

            # Regex-based filler word detection
            filler_count = {word: len(re.findall(rf"\b{re.escape(word)}\b", transcript)) for word in FILLER_WORDS}
            
            total_fillers = sum(filler_count.values())

            return {
                "transcription": transcript,
                "filler_words": {k: v for k, v in filler_count.items() if v > 0},  # Remove unused words
                "total_fillers": total_fillers,
            }

    except (sr.UnknownValueError, sr.RequestError):
        return {"transcription": "Could not process audio", "filler_words": {}, "total_fillers": 0, "energy": 0.0}




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
