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

# Strong CORS for Wix
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

    # === YOUR EXACT SOLO TAXONOMY CRITERIA ===
    system_prompt = """You are an expert Simultaneous Equations tutor using Biggs & Collis (1982) SOLO Taxonomy.

Classify the student's mastery level strictly into ONE of these 5 levels based on the handwritten work:

Level 1 (Pre-structural / Foundational Gap):
Student misses the point. Cannot solve a single linear equation (e.g. 2x=10).

Level 2 (Uni-structural / Isolated Step):
Focuses on one relevant part. Can solve one equation but cannot "link" them.

Level 3 (Multi-structural / Procedural Rigidity):
Can do both equations but treats them as a list of steps. Uses only one method (e.g., Substitution) regardless of difficulty.

Level 4 (Relational / Strategic Explorer):
Understands the relationship between the two equations. Chooses the optimal method most of the time.

Level 5 (Extended Abstract / Strategic Master):
Can generalize. Sees the "structure" of the equation instantly and predicts the most efficient path.

Return **ONLY** clean JSON in this exact format. No extra text, no markdown, no explanation:

{
  "problem": "the original two equations clearly written",
  "extracted_steps": "clear summary of what the student wrote and did",
  "level": number from 1 to 5,
  "level_name": "exact level name as shown above",
  "justification": "one short sentence explaining why this level",
  "feedback": "helpful scaffolding and next steps according to the level"
}

Now carefully analyze the handwritten image and output only the JSON."""

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
            "justification": "AI failed to return valid JSON",
            "feedback": "The AI could not parse the handwriting properly. Please try a clearer photo with better lighting and darker handwriting."
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


# Dashboard Route
@app.route('/dashboard', methods=['GET'])
def dashboard():
    try:
        conn = pymysql.connect(host=MYSQL_HOST, user=MYSQL_USER, password=MYSQL_PASS, db=MYSQL_DB)
        cur = conn.cursor()
        cur.execute("""
            SELECT id, student_id, level, extracted_steps, feedback, timestamp 
            FROM mastery_trace 
            ORDER BY timestamp DESC
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        html = "<h2>📊 Student Mastery Trace Dashboard</h2><table border='1' cellpadding='8' cellspacing='0'><tr><th>ID</th><th>Student ID</th><th>Level</th><th>Steps</th><th>Feedback</th><th>Time</th></tr>"
        for row in rows:
            html += f"<tr><td>{row[0]}</td><td>{row[1]}</td><td>Level {row[2]}</td><td>{str(row[3])[:100]}...</td><td>{str(row[4])[:100]}...</td><td>{row[5]}</td></tr>"
        html += "</table>"
        return html
    except Exception as e:
        return f"<h2>Database Error:</h2><p>{str(e)}</p>"


if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
