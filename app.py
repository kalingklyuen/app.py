from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import pymysql
import os
from datetime import datetime
import json

from google import genai   # Correct for google-genai package

app = Flask(__name__)
CORS(app)

# ========================= CONFIG =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MYSQL_HOST     = os.getenv("MYSQL_HOST")      # e.g. mysql-kck8
MYSQL_USER     = os.getenv("MYSQL_USER")      # yk1
MYSQL_PASS     = os.getenv("MYSQL_PASS")
MYSQL_DB       = os.getenv("MYSQL_DB")        # mysql or tutor_db
# ========================================================

client = genai.Client(api_key=GEMINI_API_KEY)

def cleanup_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cleaned = cv2.bitwise_not(thresh)
    _, buffer = cv2.imencode('.jpg', cleaned, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return buffer.tobytes()

def save_to_mysql(student_id, level, extracted_steps, feedback):
    try:
        conn = pymysql.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASS,
            db=MYSQL_DB
        )
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

    system_prompt = """You are an expert Simultaneous Equations tutor using Biggs & Collis (1982) SOLO Taxonomy.
Analyze the handwritten image carefully.
Extract the original problem and student's steps.
Classify into ONE level 1-5 and return **clean JSON only**."""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                system_prompt,
                {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(cleaned_bytes).decode("utf-8")}}
            ]
        )

        raw_text = response.text.strip()
        # Clean markdown if present
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].strip()

        result = json.loads(raw_text)

    except Exception as e:
        print("Gemini Error:", str(e))
        result = {
            "problem": "Processing failed",
            "extracted_steps": "",
            "level": 0,
            "level_name": "Error",
            "justification": "AI could not process the image",
            "feedback": "Please try a clearer photo with better lighting."
        }

    save_to_mysql(student_id, result.get("level", 0), result.get("extracted_steps", ""), result.get("feedback", ""))

    return jsonify(result)

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
