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
    """Improve image for better handwriting recognition"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray)
    # Increase contrast for handwriting
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

    system_prompt = """You are an expert Simultaneous Equations tutor using Biggs & Collis (1982) SOLO Taxonomy.

Analyze the handwritten photo carefully.
Extract:
- The original system of equations
- The student's solving steps

Classify strictly into ONE level 1-5.

Return **ONLY** clean JSON (no extra text):

{
  "problem": "the equations",
  "extracted_steps": "what the student did",
  "level": 3,
  "level_name": "Multi-structural (Procedural Rigidity)",
  "justification": "short reason",
  "feedback": "scaffolding according to the level"
}

Feedback rules:
- Levels 1-2: Direct hints on isolating variables and inverse operations
- Levels 3-4: Show efficient method vs student's path
- Level 5: Problem-posing prompts
"""

    try:
        image_part = {
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(cleaned_bytes).decode("utf-8")
            }
        }

        response = client.models.generate_content(
            model="gemini-2.5-flash",        # ← Changed to more stable model
            contents=[system_prompt, image_part]
        )

        raw_text = response.text.strip()

        # Clean JSON
        if "```" in raw_text:
            raw_text = raw_text.split("```")[1].strip()
            if raw_text.startswith("json"):
                raw_text = raw_text[4:].strip()

        result = json.loads(raw_text)

    except Exception as e:
        print("Gemini Error:", str(e))
        result = {
            "problem": "Image processing failed",
            "extracted_steps": "",
            "level": 0,
            "level_name": "Error",
            "justification": "AI failed to read image",
            "feedback": "The AI still cannot read the handwriting clearly. Please try these tips:\n1. Take photo in brighter light\n2. Make sure paper is flat and handwriting is dark\n3. Take photo from directly above (no angle)\n4. Avoid shadows on the paper"
        }

    # Save to MySQL (skip if error)
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
    except:
        pass

    return jsonify(result)

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
