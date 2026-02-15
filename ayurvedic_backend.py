#!/usr/bin/env python3
"""
Ayurvedic AI Backend API
Production-ready Flask server with streaming responses
"""

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import json
import time
import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from functools import wraps
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Application configuration"""
    # Server
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))  # Render sets PORT automatically
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

    # Model
    MODEL_PATH = os.getenv('MODEL_PATH', './ayurvedic_model_final')
    MAX_LENGTH = int(os.getenv('MAX_LENGTH', 250))
    DEFAULT_TEMPERATURE = float(os.getenv('DEFAULT_TEMPERATURE', 0.7))

    # Rate Limiting
    RATE_LIMIT = os.getenv('RATE_LIMIT', '30 per minute')

    # Streaming
    STREAMING_DELAY = float(os.getenv('STREAMING_DELAY', 0.05))  # seconds

    # CORS
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*')

    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'ayurvedic_api.log')

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    """Configure application logging"""
    log_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler with rotation
    file_handler = RotatingFileHandler(
        Config.LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(getattr(logging, Config.LOG_LEVEL))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(getattr(logging, Config.LOG_LEVEL))

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, Config.LOG_LEVEL))
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return logging.getLogger(__name__)

logger = setup_logging()

# ============================================================================
# FLASK APP INITIALIZATION
# ============================================================================

app = Flask(__name__)
app.config.from_object(Config)

# CORS configuration
CORS(app, origins=Config.CORS_ORIGINS)

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[Config.RATE_LIMIT],
    storage_uri="memory://"
)

# ============================================================================
# MODEL MANAGEMENT
# ============================================================================

class ModelManager:
    """Singleton class to manage the ML model"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not ModelManager._initialized:
            self.model = None
            self.tokenizer = None
            self.device = None
            ModelManager._initialized = True

    def load_model(self):
        """Load the trained Ayurvedic model"""
        if self.model is not None:
            logger.warning("Model already loaded")
            return

        logger.info("Loading Ayurvedic AI model...")

        if not os.path.exists(Config.MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {Config.MODEL_PATH}")

        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(Config.MODEL_PATH)
            self.model = GPT2LMHeadModel.from_pretrained(Config.MODEL_PATH)

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(self.device)
            self.model.eval()

            logger.info(f"Model loaded successfully on {self.device.upper()}")

        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            raise

    def is_loaded(self):
        """Check if model is loaded"""
        return self.model is not None and self.tokenizer is not None

    def generate_response(self, health_issue, max_length=None, temperature=None):
        """
        Generate response from the model

        Args:
            health_issue: User's health query
            max_length: Maximum response length
            temperature: Sampling temperature

        Returns:
            Generated response text
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")

        max_length = max_length or Config.MAX_LENGTH
        temperature = temperature or Config.DEFAULT_TEMPERATURE

        # Validate temperature
        temperature = max(0.1, min(2.0, temperature))

        prompt = f"Customer: {health_issue}\nAgent:"

        try:
            input_ids = self.tokenizer.encode(
                prompt,
                return_tensors='pt',
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            full_response = self.tokenizer.decode(output[0], skip_special_tokens=True)

            # Extract agent's response
            if "Agent:" in full_response:
                agent_response = full_response.split("Agent:")[1].split("Customer:")[0].strip()
            else:
                agent_response = full_response

            return agent_response

        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            raise

# Initialize model manager
model_manager = ModelManager()

# ============================================================================
# DECORATORS & UTILITIES
# ============================================================================

def log_request(f):
    """Decorator to log API requests"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()

        # Log request
        logger.info(f"Request: {request.method} {request.path} from {request.remote_addr}")

        # Execute function
        response = f(*args, **kwargs)

        # Log response time
        duration = time.time() - start_time
        logger.info(f"Response time: {duration:.3f}s")

        return response
    return decorated_function

def validate_input(required_fields):
    """Decorator to validate request JSON input"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not request.is_json:
                return jsonify({"error": "Content-Type must be application/json"}), 400

            data = request.json
            if not data:
                return jsonify({"error": "Request body is required"}), 400

            # Check required fields
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                return jsonify({
                    "error": f"Missing required fields: {', '.join(missing_fields)}"
                }), 400

            return f(*args, **kwargs)
        return decorated_function
    return decorator

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "error": "Endpoint not found",
        "message": "The requested URL was not found on the server"
    }), 404

@app.errorhandler(429)
def ratelimit_handler(error):
    """Handle rate limit errors"""
    return jsonify({
        "error": "Rate limit exceeded",
        "message": "Too many requests. Please try again later."
    }), 429

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {error}", exc_info=True)
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred"
    }), 500

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/', methods=['GET'])
def index():
    """API root endpoint"""
    return jsonify({
        "name": "Ayurvedic AI API",
        "version": "2.0.0",
        "status": "running",
        "documentation": "/info"
    })

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for monitoring

    Returns:
        JSON with health status
    """
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "model": {
                "loaded": model_manager.is_loaded(),
                "device": model_manager.device,
                "path": Config.MODEL_PATH
            },
            "server": {
                "debug": Config.DEBUG,
                "rate_limit": Config.RATE_LIMIT
            }
        }

        # If model not loaded, status is degraded
        if not model_manager.is_loaded():
            health_status["status"] = "degraded"
            health_status["message"] = "Model not loaded"
            return jsonify(health_status), 503

        return jsonify(health_status), 200

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 503

@app.route('/chat', methods=['POST'])
@log_request
@validate_input(['message'])
@limiter.limit(Config.RATE_LIMIT)
def chat_streaming():
    """
    Streaming chat endpoint with Server-Sent Events

    Request Body:
        {
            "message": "I have a headache",
            "temperature": 0.7  // optional, default 0.7
        }

    Response:
        Server-Sent Events stream with typing effect
    """
    try:
        data = request.json
        health_issue = data.get('message', '').strip()
        temperature = data.get('temperature', Config.DEFAULT_TEMPERATURE)

        # Validate message
        if not health_issue:
            return jsonify({"error": "Message cannot be empty"}), 400

        if len(health_issue) > 500:
            return jsonify({"error": "Message too long (max 500 characters)"}), 400

        logger.info(f"Chat request: '{health_issue[:100]}...'")

        # Generate response
        response_text = model_manager.generate_response(health_issue, temperature=temperature)

        def generate():
            """Generator for streaming response"""
            try:
                words = response_text.split()

                for i, word in enumerate(words):
                    # Add space before word (except first word)
                    chunk = word if i == 0 else f" {word}"

                    # Send as Server-Sent Events format
                    yield f"data: {json.dumps({'content': chunk})}\n\n"

                    # Typing effect delay
                    time.sleep(Config.STREAMING_DELAY)

                # Send completion signal
                yield f"data: {json.dumps({'done': True, 'total_words': len(words)})}\n\n"

                logger.info(f"Streaming completed: {len(words)} words")

            except Exception as e:
                logger.error(f"Streaming error: {e}", exc_info=True)
                yield f"data: {json.dumps({'error': 'Streaming failed'})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
                'Connection': 'keep-alive'
            }
        )

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        return jsonify({"error": "Failed to process request"}), 500

@app.route('/chat/simple', methods=['POST'])
@log_request
@validate_input(['message'])
@limiter.limit(Config.RATE_LIMIT)
def chat_simple():
    """
    Simple chat endpoint (full response at once)

    Request Body:
        {
            "message": "I have a headache",
            "temperature": 0.7  // optional, default 0.7
        }

    Response:
        {
            "response": "For headache, I recommend...",
            "query": "I have a headache",
            "timestamp": "2024-01-01T12:00:00Z",
            "model_info": {
                "temperature": 0.7,
                "device": "cuda"
            }
        }
    """
    try:
        data = request.json
        health_issue = data.get('message', '').strip()
        temperature = data.get('temperature', Config.DEFAULT_TEMPERATURE)

        # Validate message
        if not health_issue:
            return jsonify({"error": "Message cannot be empty"}), 400

        if len(health_issue) > 500:
            return jsonify({"error": "Message too long (max 500 characters)"}), 400

        logger.info(f"Simple chat request: '{health_issue[:100]}...'")

        # Generate response
        response_text = model_manager.generate_response(health_issue, temperature=temperature)

        logger.info(f"Response generated: {len(response_text)} characters")

        return jsonify({
            "response": response_text,
            "query": health_issue,
            "timestamp": datetime.utcnow().isoformat(),
            "model_info": {
                "temperature": temperature,
                "device": model_manager.device,
                "max_length": Config.MAX_LENGTH
            }
        }), 200

    except Exception as e:
        logger.error(f"Simple chat error: {e}", exc_info=True)
        return jsonify({"error": "Failed to generate response"}), 500

@app.route('/info', methods=['GET'])
def info():
    """
    Get comprehensive API information

    Returns:
        Detailed API documentation
    """
    return jsonify({
        "api": {
            "name": "Ayurvedic AI API",
            "version": "2.0.0",
            "description": "AI-powered Ayurvedic health assistant"
        },
        "endpoints": {
            "GET /": "API root",
            "GET /health": "Health check endpoint",
            "POST /chat": "Streaming chat (Server-Sent Events)",
            "POST /chat/simple": "Simple chat (full response)",
            "GET /info": "API information"
        },
        "model": {
            "loaded": model_manager.is_loaded(),
            "device": model_manager.device,
            "path": Config.MODEL_PATH,
            "max_length": Config.MAX_LENGTH,
            "default_temperature": Config.DEFAULT_TEMPERATURE
        },
        "configuration": {
            "rate_limit": Config.RATE_LIMIT,
            "cors_enabled": True,
            "cors_origins": Config.CORS_ORIGINS,
            "streaming_delay": Config.STREAMING_DELAY
        },
        "usage": {
            "chat_streaming": {
                "method": "POST",
                "endpoint": "/chat",
                "body": {
                    "message": "Your health query",
                    "temperature": 0.7
                },
                "response": "Server-Sent Events stream"
            },
            "chat_simple": {
                "method": "POST",
                "endpoint": "/chat/simple",
                "body": {
                    "message": "Your health query",
                    "temperature": 0.7
                },
                "response": "JSON with full response"
            }
        }
    }), 200

# ============================================================================
# APPLICATION STARTUP
# ============================================================================

def print_banner():
    """Print startup banner"""
    banner = """
╔════════════════════════════════════════════════════════════════════╗
║                    AYURVEDIC AI BACKEND SERVER                     ║
║                         Production Ready                           ║
╚════════════════════════════════════════════════════════════════════╝

✅ Server Configuration:
   • Host: {host}
   • Port: {port}
   • Debug: {debug}
   • Rate Limit: {rate_limit}

🤖 Model Configuration:
   • Path: {model_path}
   • Device: {device}
   • Max Length: {max_length}
   • Temperature: {temperature}

📡 API Endpoints:
   • GET  /              → API root
   • GET  /health        → Health check
   • POST /chat          → Streaming chat (SSE)
   • POST /chat/simple   → Simple chat
   • GET  /info          → API documentation

🔒 Security Features:
   • CORS enabled
   • Rate limiting enabled
   • Input validation
   • Error handling

📝 Logging:
   • Level: {log_level}
   • File: {log_file}

🌐 Server URL: http://{host}:{port}

⚠️  Press Ctrl+C to stop
╚════════════════════════════════════════════════════════════════════╝
    """.format(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG,
        rate_limit=Config.RATE_LIMIT,
        model_path=Config.MODEL_PATH,
        device=model_manager.device or 'Not loaded',
        max_length=Config.MAX_LENGTH,
        temperature=Config.DEFAULT_TEMPERATURE,
        log_level=Config.LOG_LEVEL,
        log_file=Config.LOG_FILE
    )
    print(banner)

if __name__ == '__main__':
    try:
        # Load model at startup
        logger.info("Starting Ayurvedic AI Backend Server...")
        model_manager.load_model()

        # Print startup banner
        print_banner()

        # Run the server
        app.run(
            host=Config.HOST,
            port=Config.PORT,
            debug=Config.DEBUG,
            threaded=True,
            use_reloader=False  # Disable reloader in production
        )

    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        print(f"\n❌ ERROR: {e}")
        print(f"Please ensure the model exists at: {Config.MODEL_PATH}")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)
        print(f"\n❌ FATAL ERROR: {e}")
        sys.exit(1)