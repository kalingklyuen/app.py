from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import pymysql
import os
from datetime import datetime
import json

from google import genai

app = Flask(__name__)
CORS(app)

# ========================= CONFIG =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MYSQL_HOST     = os.getenv("MYSQL_HOST")
MYSQL_USER     = os.getenv("MYSQL_USER")
MYSQL_PASS     = os.getenv("MYSQL_PASS")
MYSQL_DB       = os.getenv("MYSQL_DB")
# ========================================================

client = genai.Client(api_key=GEMINI_API_KEY)

def cleanup_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray)
    enhanced = cv2.equalizeHist(denoised)
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cleaned = cv2.bitwise_not(thresh)
    _, buffer = cv2.imencode('.jpg', cleaned, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return buffer.tobytes()

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "No image"}), 400

    file = request.files['image']
    student_id = request.form.get('student_id', 'unknown')

    cleaned_bytes = cleanup_image(file.read())

    system_prompt = """You are an expert Simultaneous Equations tutor.
Analyze the handwritten image and return **ONLY** a valid JSON object. No explanation, no extra text.

The JSON must follow this exact structure:

{
  "problem": "Write the original simultaneous equations here",
  "extracted_steps": "Summarize the student's solving steps clearly",
  "level": 3,
  "level_name": "Multi-structural (Procedural Rigidity)",
  "justification": "One short sentence explaining the level",
  "feedback": "Write the scaffolding feedback here"
}

Classify level using Biggs & Collis SOLO Taxonomy:
1 = Pre-structural (Foundational Gap)
2 = Uni-structural (Isolated Step)
3 = Multi-structural (Procedural Rigidity)
4 = Relational (Strategic Explorer)
5 = Extended Abstract (Strategic Master)

Now analyze the image and output only the JSON."""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                system_prompt,
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64.b64encode(cleaned_bytes).decode("utf-8")
                    }
                }
            ]
        )

        raw_text = response.text.strip()

        # Aggressive JSON cleaning
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].strip()

        if "{" in raw_text and "}" in raw_text:
            start = raw_text.find("{")
            end = raw_text.rfind("}") + 1
            raw_text = raw_text[start:end]

        result = json.loads(raw_text)

    except Exception as e:
        print("Gemini Error:", str(e))
        result = {
            "problem": "Not detected",
            "extracted_steps": "Not extracted",
            "level": 0,
            "level_name": "Error",
            "justification": "AI failed to return valid JSON",
            "feedback": "The AI could not parse the handwriting properly. Please try a clearer photo with brighter lighting and darker handwriting."
        }

    # Save to MySQL
    try:
        conn = pymysql.connect(host=MYSQL_HOST, user=MYSQL_USER, password=MYSQL_PASS, db=MYSQL_DB)
        cur = conn.cursor()
        sql = """
            INSERT INTO mastery_trace
            (student_id, level, extracted_steps, feedback, timestamp)
            VALUES (%s, %s, %s, %s, %s)
        """
        cur.execute(sql, (student_id, result.get("level", 0), result.get("extracted_steps", ""), result.get("feedback", ""), datetime.now()))
        conn.commit()
        cur.close()
        conn.close()
   
