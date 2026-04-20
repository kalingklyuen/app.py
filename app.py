# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import pymysql
import os
from datetime import datetime

import google.generativeai as genai   # New import for Gemini

app = Flask(__name__)
CORS(app)

# ========================= CONFIG =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")   # ← Change this
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASS = os.getenv("MYSQL_PASS")
MYSQL_DB   = os.getenv("MYSQL_DB")
# ========================================================

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)

def cleanup_image(image_bytes):
    """STEP 2: OpenCV Image Cleanup"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cleaned = cv2.bitwise_not(thresh)
    
    _, buffer = cv2.imencode('.jpg', cleaned)
    return buffer.tobytes()

def save_to_mysql(student_id, level, extracted_steps, feedback):
    """Adaptive Mastery Trace → MySQL"""
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
   
    # STEP 2: OpenCV cleanup
    cleaned_bytes = cleanup_image(file.read())
   
    # STEP 3: SOLO Taxonomy Classification using Gemini
    system_prompt = """You are an expert Simultaneous Equations tutor using Biggs & Collis (1982) SOLO Taxonomy.

Extract from the image:
- The original problem (two linear equations)
- The student's full handwritten steps

Classify strictly into ONE level (1 to 5):
1. Pre-structural (Foundational Gap)
2. Uni-structural (Isolated Step)
3. Multi-structural (Procedural Rigidity)
4. Relational (Strategic Explorer)
5. Extended Abstract (Strategic Master)

Return **clean JSON only** (no extra text):
{
  "problem": "...",
  "extracted_steps": "...",
  "level": 3,
  "level_name": "Multi-structural (Procedural Rigidity)",
  "justification": "One-sentence reason",
  "feedback": "Detailed scaffolding exactly matching the level..."
}

Feedback rules (follow strictly):
• Levels 1-2 → Direct scaffolding: hints on isolating variables & inverse operations
• Levels 3-4 → Strategic efficiency: show "Path of Least Resistance" comparison
• Level 5 → Extended mastery: problem-posing prompts (design your own complex system)
"""

    try:
        model = genai.GenerativeModel('gemini-3-flash')   # or 'gemini-2.5-flash' if available

        response = model.generate_content([
            system_prompt,
            {"mime_type": "image/jpeg", "data": base64.b64encode(cleaned_bytes).decode('utf-8')}
        ])

        raw_text = response.text.strip()
        
        # Try to extract JSON
        import json
        # Gemini sometimes adds extra text, so we clean it
        if raw_text.startswith("```json"):
            raw_text = raw_text.split("```json")[1].split("```")[0]
        elif raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
        
        result = json.loads(raw_text)
        
    except Exception as e:
        print("Gemini error:", e)
        result = {"error": "Failed to parse response", "raw": str(e)}

    # Save to MySQL
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
