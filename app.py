# app.py  (Corrected version for Gemini 2026)

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import pymysql
import os
from datetime import datetime
import json

# New correct import for Gemini (2026 SDK)
from google import genai

app = Flask(__name__)
CORS(app)

# ========================= CONFIG =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASS = os.getenv("MYSQL_PASS")
MYSQL_DB   = os.getenv("MYSQL_DB")
# ========================================================

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

def cleanup_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cleaned = cv2.bitwise_not(thresh)
    
    _, buffer = cv2.imencode('.jpg', cleaned)
    return buffer.tobytes()

def save_to_mysql(student_id, level, extracted_steps, feedback):
    try:
        conn = pymysql.connect(host=MYSQL_HOST, user=MYSQL_USER, password=MYSQL_PASS, db=MYSQL_DB)
        cur = conn.cursor()
        sql = """
            INSERT INTO mastery_trace 
            (student_id, level, extracted_steps, feedback, timestamp)
            VALUES (%s, %s, %s, %s, %s)
        """
        cur.execute(sql, (student_id, level, extracted_steps, feedback, datetime.now()))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print("MySQL error:", e)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "No image"}), 400
   
    file = request.files['image']
    student_id = request.form.get('student_id', 'unknown')
   
    cleaned_bytes = cleanup_image(file.read())

    system_prompt = """You are an expert Simultaneous Equations tutor using Biggs & Collis (1982) SOLO Taxonomy.

Extract from the image:
- The original problem (two linear equations)
- The student's full handwritten steps

Classify strictly into ONE level (1-5) and return **clean JSON only**:

{
  "problem": "...",
  "extracted_steps": "...",
  "level": 3,
  "level_name": "Multi-structural (Procedural Rigidity)",
  "justification": "One-sentence reason",
  "feedback": "Detailed scaffolding..."
}

Feedback rules:
• Levels 1-2: Direct scaffolding on isolating variables & inverse operations
• Levels 3-4: Strategic efficiency & path of least resistance
• Level 5: Problem-posing for extended mastery
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",   # or "gemini-3-flash" if available
            contents=[system_prompt, {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(cleaned_bytes).decode('utf-8')}}]
        )

        raw_text = response.text.strip()

        # Clean JSON if Gemini adds markdown
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0]
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1]

        result = json.loads(raw_text)

    except Exception as e:
        print("Gemini Error:", str(e))
        result = {"error": "AI processing failed", "details": str(e)}

    save_to_mysql(
        student_id=student_id,
        level=result.get("level", 0),
        extracted_steps=result.get("extracted_steps", ""),
        feedback=result.get("feedback", "")
    )
   
    return jsonify(result)

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
