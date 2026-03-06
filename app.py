import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot_service import get_chat_response, user_memory
import uuid

app = Flask(__name__)
CORS(app)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    
    # Allow passing user_id or generate a new one
    user_id = data.get('user_id')
    if not user_id:
        user_id = str(uuid.uuid4())
        
    message = data.get('message')
    if not message:
        return jsonify({"error": "Message is required"}), 400
        
    bot_response, updated_memory = get_chat_response(user_id, message)
    
    # Filter out history block to keep the frontend payload clean
    memory_clean = {k: v for k, v in updated_memory.items() if k != "history"}
    
    return jsonify({
        "user_id": user_id,
        "response": bot_response,
        "memory": memory_clean
    })

@app.route('/memory/<user_id>', methods=['GET'])
def get_memory(user_id):
    if user_id in user_memory:
        mem = user_memory[user_id]
        memory_clean = {k: v for k, v in mem.items() if k != "history"}
        return jsonify(memory_clean)
    else:
        return jsonify({"error": "User not found"}), 404

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
