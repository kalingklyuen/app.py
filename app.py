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
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])

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
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    denoised = cv2.fastNlMeansDenoising(img, h=10)
    enhanced = cv2.equalizeHist(denoised)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    _, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cleaned = cv2.bitwise_not(thresh)
    _, buffer = cv2.imencode('.jpg', cleaned, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buffer.tobytes()

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "No image"}), 400

    file = request.files['image']
    student_id = request.form.get('student_id', 'unknown')

    cleaned_bytes = cleanup_image(file.read())

    # Stronger, more precise prompt
   system_prompt = """You are an expert Simultaneous Equations tutor using Biggs & Collis (1982) SOLO Taxonomy.

Classify the student's mastery level strictly into ONE of these 5 levels:

Level 1: Student misses the point. Cannot solve a single linear equation (e.g. 2x=10).
Level 2: Can solve one equation but cannot "link" them.
Level 3: Can do both equations but treats them as a list of steps. Uses only one method regardless of difficulty.
Level 4: Understands the relationship between the two equations. Chooses the optimal method most of the time.
Level 5: Can generalize. Sees the "structure" of the equation instantly and predicts the most efficient path.

For Levels 3, 4, and 5, also generate a simple text-based flow chart.

Return **ONLY** clean JSON:

{
  "problem": "the original two equations",
  "extracted_steps": "summary of student's solving steps",
  "level": number,
  "level_name": "exact level name",
  "justification": "short reason",
  "feedback": "helpful scaffolding",
  "flowchart": "text-based flow chart (only for level 3,4,5). Use simple arrows like → or |"
}

Now analyze the image and output only the JSON."""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                system_prompt,
                {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(cleaned_bytes).decode("utf-8")}}
            ]
        )

        raw_text = response.text.strip()

        if "```" in raw_text:
            raw_text = raw_text.split("```")[1].strip()
            if raw_text.startswith("json"):
                raw_text = raw_text[4:].strip()

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
            "justification": "AI failed to classify",
            "feedback": "The AI could not read the handwriting clearly. Please try a clearer photo with brighter lighting and darker pen."
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
    except Exception as db_e:
        print("MySQL Error:", db_e)

    return jsonify(result)


if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
