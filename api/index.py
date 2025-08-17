from flask import Flask, request, jsonify
import os
import sys
import tempfile
import asyncio
from werkzeug.utils import secure_filename

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from agent import DataAnalystAgent

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

@app.route('/api', methods=['POST'])
def analyze_data():
    try:
        # Handle file upload
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as tmp:
                content = file.read().decode('utf-8')
                tmp.write(content)
                tmp_path = tmp.name
            
            question_text = content
        
        elif request.is_json:
            data = request.get_json()
            question_text = data.get('question', '')
        
        else:
            # Handle raw text
            question_text = request.data.decode('utf-8')
        
        if not question_text:
            return jsonify({'error': 'No question provided'}), 400
        
        # Initialize agent
        agent = DataAnalystAgent()
        
        # Process the request
        result = asyncio.run(agent.process_request(question_text))
        
        # Clean up temp file if created
        if 'tmp_path' in locals():
            os.unlink(tmp_path)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/')
def home():
    return 'Hello, World,T!'

@app.route('/about')
def about():
    return 'About'


if __name__ == '__main__':
    # Run on port 5000 (Flask default)
    app.run(debug=True, host='0.0.0.0', port=5000)
    print("Flask app running on http://localhost:5000")
