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

# Load environment variables
load_dotenv()

# Initialize FastAPI app
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

# Set up model paths dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(BASE_DIR, "app")

MODEL_PATHS = {
    "audio_confidence": os.path.join(APP_DIR, "confidence_model.tflite"),
    "audio_fluency": os.path.join(APP_DIR, "fluency_model.tflite"),
    "video_confidence": os.path.join(APP_DIR, "video_confidence.tflite"),
    "scaler_confidence": os.path.join(APP_DIR, "confidence_scaler.pkl"),
    "scaler_fluency": os.path.join(APP_DIR, "fluency_scaler.pkl"),
}

# Global variables for models and scalers
audio_confidence_model = None
audio_fluency_model = None
cnn_model = None
scaler_confidence = None
scaler_fluency = None


def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()  # Allocate memory for the model
    return interpreter

@app.on_event("startup")
def load_models():
    """Loads all ML models and scalers at startup."""
    global audio_confidence_model, audio_fluency_model, scaler_confidence, scaler_fluency, cnn_model

    try:
        audio_confidence_model = load_tflite_model(MODEL_PATHS["audio_confidence"])
        audio_fluency_model = load_tflite_model(MODEL_PATHS["audio_fluency"])
        cnn_model = load_tflite_model(MODEL_PATHS["video_confidence"])
        scaler_confidence = joblib.load(MODEL_PATHS["scaler_confidence"])
        scaler_fluency = joblib.load(MODEL_PATHS["scaler_fluency"])

        logger.info("✅ All models and scalers loaded successfully")

    except Exception as e:
        logger.error(f"❌ Model loading failed: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}")











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
        video_confidences = process_video_frames(temp_media_path)
        
        # Step 3: Extract and validate audio
        temp_audio_path = extract_audio(temp_media_path)
        if not temp_audio_path:
            return {
                "status": "failure",
                "message": "Audio extraction failed",
            }

        # # Step 4: Analyze fluency, confidence & speech energy
        fluency_scores, confidence_scores = process_audio_features(temp_audio_path)

        # Step 5: Transcribe audio & detect filler words
        transcription = transcribe_audio(temp_audio_path)

        # # Step 6: Analyze grammar, enhance response, check relevance
        analysis_result = analyze_response(question, transcription)
        return {
            "status": "success",
            "message": "Processing complete",
            "video_confidence": video_confidences,
            "audio_confidence": fluency_scores,
            "fluency_percentage": confidence_scores,
            "transcription": transcription,
            "grammar": analysis_result
        }

    finally:
        # Cleanup temporary files
        if temp_media_path and os.path.exists(temp_media_path):
            os.remove(temp_media_path)
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)














def process_video_frames(video_path: str, segment_duration: int = 5) -> list:
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get frames per second
    segment_frames = segment_duration * fps  # Number of frames per segment
    confidences = []

    frames = []
    frame_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (128, 128)) / 255.0  # Resize and normalize
            frames.append(frame)
            frame_count += 1

            # Process each 5-second segment
            if frame_count >= segment_frames:
                if frames:
                    frames = np.array(frames, dtype=np.float32)

                    # Ensure correct shape for model
                    frames = np.expand_dims(frames[0], axis=0)  # Use first frame of segment

                    segment_confidence = predict_with_tflite(cnn_model, frames)
                    confidences.append(segment_confidence)

                frames = []
                frame_count = 0  # Reset for the next segment

        return confidences  # List of confidence values per 5-sec segment

    finally:
        cap.release()

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


def process_audio_features(audio_path: str, segment_duration: int = 5) -> list:
    audio, sr = librosa.load(audio_path, sr=16000)
    total_duration = librosa.get_duration(y=audio, sr=sr)
    
    fluency_scores = []
    confidence_scores = []
    
    for start in range(0, int(total_duration), segment_duration):
        end = start + segment_duration
        audio_segment = audio[start * sr : end * sr]  
        
        if len(audio_segment) == 0:
            fluency_scores.append(0.0)
            confidence_scores.append(0.0)
            continue
        
        audio_segment = nr.reduce_noise(y=audio_segment, sr=sr)

        rms_energy = librosa.feature.rms(y=audio_segment).mean()
        silence_threshold = 0.005
        is_silent = rms_energy < silence_threshold

        if is_silent:
            fluency_scores.append(0.0)
            confidence_scores.append(0.0)
            continue

        features = {
            "mfcc": librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=40).mean(axis=1),
            "speech_rate": len(librosa.effects.split(audio_segment)[0]),
            "pitch": librosa.yin(audio_segment, fmin=50, fmax=500) if len(audio_segment) > 0 else [0],
            "energy": rms_energy,
            "is_silent": False
        }

        fluency_scores.append(predict_fluency(features))
        confidence_scores.append(predict_confidence(features))

    return fluency_scores, confidence_scores  


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
    """
    Sends the transcript and question to Gemini AI for grammar correction,
    enhancement, and filler word detection.
    """
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = f"""
        You are an advanced AI language model evaluating a candidate’s response to an interview question.  
        Your task is to analyze the response for **grammar correctness, fluency, conciseness, and filler word usage**.  

        ### **Instructions:**  
        - **Identify grammar mistakes**, categorizing them into specific types.  
        - **For each mistake, provide the incorrect version, the corrected version, and the type of mistake.**  
        - **Provide an improved, more concise, and impactful version of the response.**  

        ### **Output Format (JSON):**  
        {{
          "grammar_accuracy": "XX%",  # Percentage of grammatically correct sentences.
          "enhanced_response": "A refined, more effective version of the response"
        }}

        ### **Input Data:**  
        - **Interview Question:** "{question}"  
        - **Candidate Response:** "{transcript}"  

        Analyze the response and return structured JSON output.
        """


        response = model.generate_content(prompt)
        
        # Extract and clean the response
        cleaned_text = response.text.strip().replace("```json", "").replace("```", "").strip()

        # Try parsing as JSON
        try:
            parsed_response = json.loads(cleaned_text)
            return{
                "grammar_accuracy": parsed_response.get("grammar_accuracy", "N/A"),
                "enhanced_response": parsed_response.get("enhanced_response", "").strip()
            }
        except json.JSONDecodeError:
            return {
                "grammar_accuracy": "N/A",
                "enhanced_response": cleaned_text 
            }
    
    except Exception as e:
        return {
            "grammar_accuracy": "N/A",
            "enhanced_response": "Could not generate improved response."
        }














def transcribe_audio(audio_path: str) -> dict:
    recognizer = sr.Recognizer()
    
    try:
        with sr.AudioFile(audio_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source)
            transcript = recognizer.recognize_google(audio_data)

            return transcript

    except (sr.UnknownValueError, sr.RequestError):
        return "Could not process audio"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)