<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Simultaneous Equations AI Tutor</title>
  <style>
    body {
      font-family: 'Segoe UI', Arial, sans-serif;
      background: #f4f6f9;
      margin: 0;
      padding: 20px;
    }
    .container {
      max-width: 1000px;
      margin: auto;
      background: white;
      padding: 30px;
      border-radius: 16px;
      box-shadow: 0 8px 30px rgba(0,0,0,0.1);
    }
    h1 {
      text-align: center;
      color: #1e3a8a;
    }
    p {
      text-align: center;
      color: #555;
      margin-bottom: 25px;
    }
    .options {
      display: flex;
      gap: 15px;
      justify-content: center;
      margin: 25px 0;
      flex-wrap: wrap;
    }
    button {
      background: #0066ff;
      color: white;
      border: none;
      padding: 16px 24px;
      font-size: 17px;
      border-radius: 12px;
      cursor: pointer;
      min-width: 180px;
    }
    button:hover {
      background: #0055cc;
    }
    .reset-btn {
      background: #6b7280;
      margin-top: 30px;
    }
    #preview {
      text-align: center;
      margin: 25px 0;
      display: none;
    }
    #preview img {
      max-width: 100%;
      max-height: 480px;
      border: 3px solid #ddd;
      border-radius: 12px;
      box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .result {
      margin-top: 30px;
      padding: 25px;
      background: #f0f9ff;
      border-radius: 12px;
      border-left: 6px solid #3b82f6;
      display: none;
      line-height: 1.7;
    }
    .visual-steps {
      background: #f8fafc;
      padding: 18px;
      border-radius: 10px;
      border: 2px dashed #64748b;
      margin: 15px 0;
      font-family: monospace;
      white-space: pre-wrap;
      color: #334155;
    }
    .error {
      color: #dc2626;
      background: #fee2e2;
      padding: 15px;
      border-radius: 8px;
    }
    .level-title {
      font-size: 28px;
      color: #1e40af;
      margin: 20px 0 15px 0;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>🤖 Simultaneous Equations AI Tutor</h1>
    <p>Take a photo or upload your handwritten solving work</p>

    <div class="options">
      <button onclick="takePhoto()">📸 Take Photo</button>
      <button onclick="document.getElementById('fileInput').click()">📁 Upload Photo</button>
    </div>

    <input type="file" id="fileInput" accept="image/*" style="display:none;">

    <div id="preview">
      <img id="previewImg" alt="Preview of your work">
    </div>

    <button id="analyzeBtn" style="display:none; margin-top: 20px;">
      Analyze with AI Tutor →
    </button>

    <div id="result" class="result"></div>

    <button id="resetBtn" class="reset-btn" style="display:none;">
      Try Another Problem
    </button>
  </div>

  <script>
    let uploadedImageFile = null;
    const BACKEND_URL = "https://app-py-0q39.onrender.com/analyze";

    function takePhoto() {
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = 'image/*';
      input.capture = 'environment';
      input.onchange = function(e) {
        if (e.target.files[0]) handleFile(e.target.files[0]);
      };
      input.click();
    }

    document.getElementById('fileInput').addEventListener('change', function(e) {
      if (e.target.files[0]) handleFile(e.target.files[0]);
    });

    function handleFile(file) {
      uploadedImageFile = file;
      const reader = new FileReader();
      reader.onload = function(ev) {
        document.getElementById('previewImg').src = ev.target.result;
        document.getElementById('preview').style.display = 'block';
        document.getElementById('analyzeBtn').style.display = 'block';
      };
      reader.readAsDataURL(file);
    }

    document.getElementById('analyzeBtn').addEventListener('click', async function() {
      if (!uploadedImageFile) return alert("Please take or upload a photo first!");

      const btn = document.getElementById('analyzeBtn');
      btn.textContent = "Analyzing... (Please wait 15-30 seconds)";
      btn.disabled = true;

      const formData = new FormData();
      formData.append('image', uploadedImageFile);
      formData.append('student_id', 'student-' + Date.now());

      try {
        const response = await fetch(BACKEND_URL, { 
          method: 'POST', 
          body: formData 
        });

        if (!response.ok) throw new Error("Server error");

        const data = await response.json();
        showScaffolding(data);

      } catch (err) {
        console.error(err);
        document.getElementById('result').innerHTML = `
          <div class="error">
            <h3>❌ Connection Error</h3>
            <p>Could not connect to the AI backend.</p>
            <p>Please try again in a moment.</p>
          </div>`;
        document.getElementById('result').style.display = 'block';
      }

      btn.textContent = "Analyze with AI Tutor →";
      btn.disabled = false;
    });

    function showScaffolding(data) {
      const level = parseInt(data.level) || 0;

      let title = `SOLO Level ${level} • ${data.level_name || 'Unknown'}`;

      let scaffolding = '';
      let visualHTML = data.visual_steps ? 
        `<div class="visual-steps">${data.visual_steps}</div>` : '';

      switch(level) {
        case 1:
          title = "Level 1: Foundational Gap (Pre-structural)";
          scaffolding = `<strong>Description:</strong> Student misses the point. Cannot solve a single linear equation (e.g. 2x=10).<br><br>
                         <strong>Scaffolding:</strong> Direct scaffolding on basic linear equation concepts. Focus on isolating variables and inverse operations.`;
          break;
        case 2:
          title = "Level 2: Isolated Step (Uni-structural)";
          scaffolding = `<strong>Description:</strong> Can solve one equation but cannot "link" them.<br><br>
                         <strong>Scaffolding:</strong> Hints on how to link two independent equations.`;
          break;
        case 3:
          title = "Level 3: Procedural Rigidity (Multi-structural)";
          scaffolding = `<strong>Description:</strong> Can do both equations but treats them as a list of steps. Uses only one method regardless of difficulty.<br><br>
                         <strong>Scaffolding:</strong> Let's look at the steps one by one and find a more efficient way.`;
          break;
        case 4:
          title = "Level 4: Strategic Explorer (Relational)";
          scaffolding = `<strong>Description:</strong> Understands the relationship between the two equations. Chooses the optimal method most of the time.<br><br>
                         <strong>Scaffolding:</strong> Good understanding of relationships. Let's refine your strategy for even better efficiency.`;
          break;
        case 5:
          title = "Level 5: Strategic Master (Extended Abstract)";
          scaffolding = `<strong>Description:</strong> Can generalize. Sees the "structure" of the equation instantly and predicts the most efficient path.<br><br>
                         <strong>Scaffolding:</strong> Excellent mastery! Now try creating your own complex simultaneous equation system.`;
          break;
        default:
          title = "Level 0: Detection Error";
          scaffolding = `The AI had difficulty reading your handwriting.<br>Please try again with brighter lighting, darker pen, and take the photo from directly above the paper.`;
      }

      const html = `
        <h2 class="level-title">${title}</h2>
        <p><strong>Original Problem:</strong> ${data.problem || 'Not detected'}</p>
        <p><strong>Your Steps:</strong><br>${data.extracted_steps || 'Not extracted'}</p>
        <hr>
        <h3>AI Explanation & Feedback</h3>
        <p>${data.feedback || 'No feedback available'}</p>
        ${visualHTML ? `<hr><h3>Visual Step-by-Step Explanation</h3>${visualHTML}` : ''}
        <hr>
        <h3>Personalized Scaffolding</h3>
        <p>${scaffolding}</p>
      `;

      document.getElementById('result').innerHTML = html;
      document.getElementById('result').style.display = 'block';
      document.getElementById('resetBtn').style.display = 'block';
    }

    document.getElementById('resetBtn').addEventListener('click', () => location.reload());
  </script>
</body>
</html>
