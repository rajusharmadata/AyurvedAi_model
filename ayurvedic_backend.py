#!/usr/bin/env python3
"""
Ayurvedic AI Backend API
Flask server with streaming responses (ChatGPT-style typing effect)
"""

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import json
import time
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Global variables for model
model = None
tokenizer = None
device = None

def load_model():
    """Load the trained Ayurvedic model"""
    global model, tokenizer, device

    print("🔄 Loading Ayurvedic AI model...")

    model_path = "./ayurvedic_model_final"

    if not os.path.exists(model_path):
        raise Exception(f"Model not found at {model_path}")

    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    print(f"✅ Model loaded on {device.upper()}")

def generate_response_stream(health_issue, max_length=250, temperature=0.7):
    """
    Generate streaming response word by word (ChatGPT style)
    """
    prompt = f"Customer: {health_issue}\nAgent:"

    # Encode input
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    # Generate with streaming
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )

    # Decode the full response
    full_response = tokenizer.decode(output.sequences[0], skip_special_tokens=True)

    # Extract agent's response
    if "Agent:" in full_response:
        agent_response = full_response.split("Agent:")[1].split("Customer:")[0].strip()
    else:
        agent_response = full_response

    # Stream word by word
    words = agent_response.split()
    for i, word in enumerate(words):
        # Add space before word (except first word)
        if i > 0:
            yield " "
        yield word
        time.sleep(0.05)  # Delay for typing effect (50ms per word)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device
    })

@app.route('/chat', methods=['POST'])
def chat():
    """
    Main chat endpoint with streaming response

    Request body:
    {
        "message": "I have a headache"
    }

    Response: Server-Sent Events (SSE) stream
    """
    try:
        data = request.json
        health_issue = data.get('message', '').strip()

        if not health_issue:
            return jsonify({"error": "Message is required"}), 400

        # Return streaming response
        def generate():
            try:
                for chunk in generate_response_stream(health_issue):
                    # Send as Server-Sent Events format
                    yield f"data: {json.dumps({'content': chunk})}\n\n"

                # Send completion signal
                yield f"data: {json.dumps({'done': True})}\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/chat/simple', methods=['POST'])
def chat_simple():
    """
    Simple chat endpoint without streaming (returns full response at once)

    Request body:
    {
        "message": "I have a headache",
        "temperature": 0.7  // optional
    }

    Response:
    {
        "response": "For headache, I recommend...",
        "query": "I have a headache"
    }
    """
    try:
        data = request.json
        health_issue = data.get('message', '').strip()
        temperature = data.get('temperature', 0.7)

        if not health_issue:
            return jsonify({"error": "Message is required"}), 400

        # Generate full response
        prompt = f"Customer: {health_issue}\nAgent:"
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=250,
                temperature=temperature,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )

        full_response = tokenizer.decode(output[0], skip_special_tokens=True)

        # Extract agent's response
        if "Agent:" in full_response:
            agent_response = full_response.split("Agent:")[1].split("Customer:")[0].strip()
        else:
            agent_response = full_response

        return jsonify({
            "response": agent_response,
            "query": health_issue
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/info', methods=['GET'])
def info():
    """Get API information"""
    return jsonify({
        "name": "Ayurvedic AI API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "GET - Health check",
            "/chat": "POST - Streaming chat (SSE)",
            "/chat/simple": "POST - Simple chat (full response)",
            "/info": "GET - API information"
        },
        "model_info": {
            "loaded": model is not None,
            "device": device,
            "model_path": "./ayurvedic_model_final"
        }
    })

if __name__ == '__main__':
    # Load model at startup
    try:
        load_model()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Please ensure 'ayurvedic_model_final' folder exists")
        exit(1)

    print("\n" + "="*70)
    print("🌿 AYURVEDIC AI BACKEND SERVER")
    print("="*70)
    print("\n✅ Server starting...")
    print("\n📡 API Endpoints:")
    print("   - POST /chat           → Streaming response (ChatGPT style)")
    print("   - POST /chat/simple    → Full response at once")
    print("   - GET  /health         → Health check")
    print("   - GET  /info           → API information")
    print("\n🌐 Server running at: http://localhost:5000")
    print("\n💡 CORS enabled - can accept requests from any frontend")
    print("\n⚠️  Press Ctrl+C to stop")
    print("="*70 + "\n")

    # Run the server
    app.run(
        host='0.0.0.0',  # Allow external connections
        port=5000,
        debug=False,
        threaded=True
    )