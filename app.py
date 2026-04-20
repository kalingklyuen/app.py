# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import pymysql
import os
from datetime import datetime
from openai import OpenAI   # pip install openai flask flask-cors pymysql opencv-python numpy

app = Flask(__name__)
CORS(app)

# ========================= CONFIG =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")          # Put in Render environment variables
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASS = os.getenv("MYSQL_PASS")
MYSQL_DB   = os.getenv("MYSQL_DB")
# ========================================================

client = OpenAI(api_key=OPENAI_API_KEY)

def cleanup_image(image_bytes):
    """STEP 2: OpenCV Image Cleanup"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Classic cleanup pipeline
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cleaned = cv2.bitwise_not(thresh)  # invert for better OCR/LLM
    
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
    
    # Convert to base64 for vision LLM
    b64 = base64.b64encode(cleaned_bytes).decode('utf-8')
    
    # STEP 3: SOLO Taxonomy Classification + Scaffolding (Biggs & Collis 1982)
    system_prompt = """You are an expert Simultaneous Equations tutor using Biggs & Collis (1982) SOLO Taxonomy.

Extract from the image:
- The original problem (two linear equations)
- The student's full handwritten steps

Classify strictly into ONE level:
1. Pre-structural (Foundational Gap) → misses the point, cannot solve even 2x=10
2. Uni-structural (Isolated Step) → solves one equation but cannot link the two
3. Multi-structural (Procedural Rigidity) → solves both but only one rigid method
4. Relational (Strategic Explorer) → chooses optimal method most of the time
5. Extended Abstract (Strategic Master) → sees structure instantly, can generalize or reverse-engineer

Return clean JSON only:
{
  "problem": "2x + 3y = 8, 4x - y = 7",
  "extracted_steps": "Student wrote...",
  "level": 3,
  "level_name": "Multi-structural (Procedural Rigidity)",
  "justification": "One-sentence reason",
  "feedback": "Detailed scaffolding exactly matching the level in the flowchart"
}
Feedback rules:
• Levels 1-2 → Direct scaffolding: hints on isolating variables & inverse operations
• Levels 3-4 → Strategic efficiency: show "Path of Least Resistance" flowchart comparison
• Level 5 → Extended mastery: problem-posing prompts (design your own complex system)
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": "Analyze this handwritten simultaneous equation work and classify using SOLO Taxonomy."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]}
        ],
        temperature=0.3,
        max_tokens=800
    )
    
    try:
        result = eval(response.choices[0].message.content.strip())  # safe JSON from LLM
    except:
        result = {"error": "Parsing failed"}
    
    # Save to MySQL for teacher dashboard & longitudinal tracking
    save_to_mysql(
        student_id=student_id,
        level=result.get("level", 0),
        extracted_steps=result.get("extracted_steps", ""),
        feedback=result.get("feedback", "")
    )
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
