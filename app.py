from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import pymysql
import os
from datetime import datetime
import json

# New imports for OCR fallback
import easyocr

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

# Initialize EasyOCR (English + Chinese)
reader = easyocr.Reader(['en', 'ch_sim'], gpu=False)

def cleanup_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    enhanced = cv2.equalizeHist(denoised)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    _, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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

Classify the student's mastery level strictly into ONE of these 5 levels:

Level 1: Student misses the point. Cannot solve a single linear equation (e.g. 2x=10).
Level 2: Can solve one equation but cannot "link" them.
Level 3: Can do both equations but treats them as a list of steps. Uses only one method regardless of difficulty.
Level 4: Understands the relationship between the two equations. Chooses the optimal method most of the time.
Level 5: Can generalize. Sees the "structure" of the equation instantly and predicts the most efficient path.

Return **ONLY** clean JSON:

{
  "problem": "the original two equations",
  "extracted_steps": "summary of student's steps",
  "level": number 1-5,
  "level_name": "exact level name",
  "justification": "short reason",
  "feedback": "helpful scaffolding"
}

Now analyze and output only the JSON."""

    try:
        # First try Gemini Vision
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                system_prompt,
                {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(cleaned_bytes).decode("utf-8")}}
            ]
        )

        raw_text = response.text.strip()

        # Clean JSON
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
        print("Gemini Vision failed, trying OCR fallback:", str(e))
        
        # Fallback: Use EasyOCR to extract text
        try:
            img_array = cv2.imdecode(np.frombuffer(cleaned_bytes, np.uint8), cv2.IMREAD_COLOR)
            ocr_result = reader.readtext(img_array, detail=0)
            extracted_text = " ".join(ocr_result)

            # Send extracted text to Gemini for classification
            text_prompt = f"""Here is the extracted text from student's handwritten simultaneous equations:

{extracted_text}

Classify the student's mastery level using the SOLO Taxonomy and return ONLY JSON in the same format as before."""

            response = client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=[{"role": "user", "content": text_prompt}]
            )

            raw_text = response.text.strip()
            result = json.loads(raw_text)

        except Exception as fallback_error:
            print("OCR Fallback also failed:", str(fallback_error))
            result = {
                "problem": "Not detected",
                "extracted_steps": "Not extracted",
                "level": 0,
                "level_name": "Error",
                "justification": "Both vision and OCR failed",
                "feedback": "The AI could not read the handwriting. Please try a clearer photo with brighter lighting, darker pen, and take the photo from directly above the paper."
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
