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
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cleaned = cv2.bitwise_not(thresh)
    _, buffer = cv2.imencode('.jpg', cleaned, [cv2.IMWRITE_JPEG_QUALITY, 90])
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
        print("MySQL Error:", e)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "No image"}), 400

    file = request.files['image']
    student_id = request.form.get('student_id', 'unknown')

    cleaned_bytes = cleanup_image(file.read())

    system_prompt = """You are an expert math tutor specializing in Simultaneous Equations using Biggs & Collis SOLO Taxonomy (1982).

Look at the handwritten work carefully.
Extract:
- The original simultaneous equations
- The student's solving steps

Classify the student's mastery level into exactly one of these 1-5 levels.

Return **ONLY clean JSON**, nothing else:

{
  "problem": "the system of equations",
  "extracted_steps": "summary of student's method",
  "level": 4,
  "level_name": "Relational (Strategic Explorer)",
  "justification": "short reason",
  "feedback": "helpful scaffolding based on the level"
}

Feedback rules:
- Level 1-2: Basic hints on variables and inverse operations
- Level 3-4: Compare method with most efficient way
- Level 5: Ask student to create their own harder problem
"""

    try:
        # Improved way to send image to Gemini
        image_part = {
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(cleaned_bytes).decode("utf-8")
            }
        }

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[system_prompt, image_part]
        )

        raw_text = response.text.strip()

        # Clean JSON if Gemini adds extra text
        if "```" in raw_text:
            raw_text = raw_text.split("```")[1].strip()
            if raw_text.startswith("json"):
                raw_text = raw_text[4:].strip()

        result = json.loads(raw_text)

    except Exception as e:
        print("Gemini Error:", str(e))
        result = {
            "problem": "Cannot read the image",
            "extracted_steps": "",
            "level": 0,
            "level_name": "Error",
            "justification": "Image processing failed",
            "feedback": "The AI could not read the handwriting. Please try taking the photo again with brighter lighting and make sure all writing is clear."
        }

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
