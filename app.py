from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
import re
from helpers import extract_text, find_best_match, extract_best_sentence
from  sentence_transformers import SentenceTransformer,util


model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_amount(text):
    match = re.search(r'(₹|\$|INR|USD)\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?', text, re.IGNORECASE)
    if match:
        return match.group().strip()
    return None


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files or 'user_input' not in request.form:
        return jsonify({"error": "Missing file or input"}), 400

    file = request.files['file']
    user_input = request.form['user_input']

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        policy_text = extract_text(filepath)
        from helpers import chunk_text 
        chunks = chunk_text(policy_text)
        weighted_chunks = []
        for heading, chunk in chunks:
            weight = 1.0
            if heading and any(key in heading.lower() for key in ['exclusion', 'table of benefits']):
                weight = 1.5  # boost for important sections
            weighted_chunks.append((chunk, weight))

        best_match, score = find_best_match(weighted_chunks, user_input)
        decision = score > 0.80
        amount = extract_amount(best_match)
        return jsonify({
            "decision": decision,
            "amount": amount,
            "justification": extract_best_sentence(best_match, user_input, model)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':    
    import webbrowser
    import threading
    port = 5000
    threading.Timer(1.25, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    app.run(debug=True, port=port)
