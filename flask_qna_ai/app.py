from flask import Flask, render_template, request, jsonify
import os
import time
from werkzeug.utils import secure_filename
from qna2 import agent, agent_inst, log_messages, memory

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'pdf'}


pdf_path = []
is_interrupted = False
chat_history = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global pdf_path
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid or unsupported file'}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        file.save(save_path)
        
        print("="*10)
        print(f"[UPLOAD] File saved at: {save_path}")
        print("="*10)
        
        pdf_path.append(save_path)
        
        return jsonify({'filepath': save_path})
    except Exception as e:
        return jsonify({'error': f'File save failed: {str(e)}'}), 500
    
@app.route('/logs', methods=['GET'])
def get_logs():
    global log_messages
    logs_to_send = log_messages.copy()
    log_messages.clear()  # Optional: clear after sending
    return jsonify({'logs': logs_to_send})


@app.route('/ask', methods=['POST'])
def ask():
    global pdf_path, is_interrupted, chat_history
    
    data = request.get_json()
    question = data.get('question', '').strip()
#     pdf_path = data.get('filepath', None)
    path = pdf_path
    pdf_path = []
    
    if not question:
        return jsonify({'answer': '❗ Please enter a question.'}), 400
    
    try:
      
        print("="*10)
        print(f"[ASK] Question: {question}")
        print(f"[ASK] PDF Path: {path}")
        print("="*10)
        
        if path != []:
            agent_inst.parse_and_save(path[0])
            
        result = agent.run(question)
        chat_history = memory.buffer
        
        print("="*10)
        print(f"[ASK] Result: {result}")
        print(f"[ASK] Chat : {chat_history}")
        print("="*10)
        
        
        return jsonify({'answer': result})

    except Exception as e:

        return jsonify({'answer': f"❌ Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
