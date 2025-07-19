from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
from helpers import extract_text, chunk_text, find_best_match

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
        chunks = chunk_text(policy_text)
        best_match, score = find_best_match(chunks, user_input)

        # Simple logic for demo purpose
        decision = "approved" if any(word in best_match.lower() for word in ["covered", "included", "approved", "claim admissible"]) else "denied"
        amount = None
        for token in best_match.split():
            if "₹" in token or "$" in token or token.replace(',', '').replace('.', '').isdigit():
                amount = token
                break

        return jsonify({
            "decision": decision,
            "amount": amount,
            "justification": best_match.strip()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    

    
    import webbrowser
    import threading
    port = 5000
    app.run(debug=True, port=port)
    threading.Timer(1.25, lambda: webbrowser.open(f"http://127.0.0.1:{port}"))
