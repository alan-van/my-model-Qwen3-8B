# Qwen3-8B Fine-tuning Application

A professional Python application for fine-tuning the Qwen3-8B model with support for multiple file formats and a comprehensive chat interface.

## 🚀 Features

- **Fine-tuning Qwen3-8B** with data from various file formats (.csv, .txt, .pdf, .docx, .xlsx)
- **REST API** for managing fine-tuning processes
- **Chat interface** with file upload capabilities
- **Database** storage for fine-tuning information
- **Web interface** with user-friendly design (Streamlit)
- **Real-time progress tracking** for fine-tuning jobs
- **Model management** with performance metrics
- **File processing** with automatic content extraction
- **File Upload System** supporting multiple formats
- **Session Management** for chat conversations

## 📋 System Requirements

- Python 3.8+
- RAM: 16GB+ (for fine-tuning)
- GPU: NVIDIA GPU with CUDA (recommended)
- Disk space: 20GB+ for models and data

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd qwen3-8b-finetuning
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment
```bash
cp env.example .env
# Edit .env file with your configuration
```

### 5. Test the application
```bash
python test_app.py
```

### 6. Initialize the application
```bash
python init_db.py
```

## 🚀 Quick Start

### Run the complete application
```bash
python run.py --both
```

This will start both the API server (port 8000) and Streamlit interface (port 8501).

### Run individual components
```bash
# API server only
python run.py --mode api --api-port 8000

# Streamlit interface only
python run.py --mode streamlit --streamlit-port 8501

# Manual startup
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
streamlit run app/streamlit_app.py
```

## 📖 Usage

### 1. Fine-tuning Process

1. **Upload Training Data**: Use the web interface to upload your training files
2. **Configure Parameters**: Set learning rate, batch size, epochs, etc.
3. **Start Fine-tuning**: Begin the fine-tuning process
4. **Monitor Progress**: Track real-time progress and metrics
5. **Download Model**: Access your fine-tuned model when complete

### 2. Chat Interface

1. **Select Model**: Choose from available models (base or fine-tuned)
2. **Start Chat**: Begin a conversation with the selected model
3. **Upload Context**: Add files for additional context
4. **View History**: Access chat history and session management

### 3. API Usage

The application provides a comprehensive REST API:

```bash
# Check API health
curl http://localhost:8000/health

# Get upload configuration
curl http://localhost:8000/api/upload/config

# Start fine-tuning
curl -X POST http://localhost:8000/api/finetune/start \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "my-model",
    "base_model": "Qwen/Qwen3-8B",
    "learning_rate": 2e-5,
    "batch_size": 4,
    "epochs": 3,
    "data_files": ["path/to/data.csv"]
  }'
```

## 🔌 API Usage Examples

### Using Curl

#### 1. Health Check
```bash
curl -X GET "http://localhost:8000/api/upload/config"
```

#### 2. Upload File
```bash
curl -X POST "http://localhost:8000/api/upload/file" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/file.csv"
```

#### 3. Start Fine-tuning
```bash
curl -X POST "http://localhost:8000/api/finetune/start" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "my-finetuned-model",
    "base_model": "Qwen/Qwen3-8B",
    "learning_rate": 2e-5,
    "batch_size": 4,
    "epochs": 3,
    "max_length": 512,
    "warmup_steps": 100,
    "data_files": ["path/to/uploaded/file.csv"]
  }'
```

#### 4. Check Fine-tuning Status
```bash
curl -X GET "http://localhost:8000/api/finetune/status/job_12345678"
```

#### 5. Create Chat Session
```bash
curl -X POST "http://localhost:8000/api/chat/session" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "model_12345678",
    "session_name": "My Chat Session"
  }'
```

#### 6. Send Chat Message
```bash
curl -X POST "http://localhost:8000/api/chat/send" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "session_12345678",
    "message": "Hello, how are you?",
    "model_id": "model_12345678"
  }'
```

#### 7. Get Model List
```bash
curl -X GET "http://localhost:8000/api/models"
```

#### 8. Get Model Details
```bash
curl -X GET "http://localhost:8000/api/models/model_12345678"
```

### Using Postman

#### 1. Setup Postman Collection

Create a new collection with base URL: `http://localhost:8000`

#### 2. Health Check Request
- **Method**: GET
- **URL**: `{{base_url}}/api/upload/config`
- **Headers**: None

#### 3. Upload File Request
- **Method**: POST
- **URL**: `{{base_url}}/api/upload/file`
- **Headers**: 
  - `Content-Type: multipart/form-data`
- **Body**: 
  - Type: `form-data`
  - Key: `file`
  - Type: `File`
  - Value: Select your file

#### 4. Start Fine-tuning Request
- **Method**: POST
- **URL**: `{{base_url}}/api/finetune/start`
- **Headers**:
  - `Content-Type: application/json`
- **Body** (raw JSON):
```json
{
  "model_name": "my-finetuned-model",
  "base_model": "Qwen/Qwen3-8B",
  "learning_rate": 2e-5,
  "batch_size": 4,
  "epochs": 3,
  "max_length": 512,
  "warmup_steps": 100,
  "data_files": ["path/to/uploaded/file.csv"]
}
```

#### 5. Check Status Request
- **Method**: GET
- **URL**: `{{base_url}}/api/finetune/status/{{job_id}}`
- **Headers**: None

#### 6. Create Chat Session Request
- **Method**: POST
- **URL**: `{{base_url}}/api/chat/session`
- **Headers**:
  - `Content-Type: application/json`
- **Body** (raw JSON):
```json
{
  "model_id": "model_12345678",
  "session_name": "My Chat Session"
}
```

#### 7. Send Chat Message Request
- **Method**: POST
- **URL**: `{{base_url}}/api/chat/send`
- **Headers**:
  - `Content-Type: application/json`
- **Body** (raw JSON):
```json
{
  "session_id": "session_12345678",
  "message": "Hello, how are you?",
  "model_id": "model_12345678"
}
```

#### 8. Get Models Request
- **Method**: GET
- **URL**: `{{base_url}}/api/models`
- **Headers**: None

### Postman Environment Variables

Create environment with variables:
- `base_url`: `http://localhost:8000`
- `job_id`: (will be set from response)
- `model_id`: (will be set from response)
- `session_id`: (will be set from response)

### Postman Test Scripts

#### Set Job ID from Fine-tuning Response
```javascript
// In Tests tab of Start Fine-tuning request
if (pm.response.code === 200) {
    const response = pm.response.json();
    pm.environment.set("job_id", response.job_id);
}
```

#### Set Model ID from Model List Response
```javascript
// In Tests tab of Get Models request
if (pm.response.code === 200) {
    const response = pm.response.json();
    if (response.models && response.models.length > 0) {
        pm.environment.set("model_id", response.models[0].model_id);
    }
}
```

#### Set Session ID from Create Session Response
```javascript
// In Tests tab of Create Chat Session request
if (pm.response.code === 200) {
    const response = pm.response.json();
    pm.environment.set("session_id", response.session_id);
}
```

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env`:

```env
# Database Configuration
DATABASE_URL=sqlite:///./finetuning.db

# Model Configuration
MODEL_NAME=Qwen/Qwen3-8B
BASE_MODEL_PATH=./models/base
FINETUNED_MODEL_PATH=./models/finetuned

# Hugging Face Configuration
HF_TOKEN=your_huggingface_token_here
HF_REPO_ID=your_username/your_model_name

# Security
SECRET_KEY=your_secret_key_here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# File Upload Configuration
MAX_FILE_SIZE=104857600
UPLOAD_DIR=./uploads
ALLOWED_EXTENSIONS=["csv", "txt", "pdf", "docx", "xlsx", "jpg", "jpeg", "png"]

# Fine-tuning Configuration
DEFAULT_LEARNING_RATE=2e-5
DEFAULT_BATCH_SIZE=4
DEFAULT_EPOCHS=3
DEFAULT_MAX_LENGTH=512
DEFAULT_WARMUP_STEPS=100

# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=True
```

## 📁 Project Structure

```
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── config.py              # Configuration settings
│   ├── database.py            # Database setup
│   ├── models/                # Database models
│   │   ├── finetune_job.py    # Fine-tuning job model
│   │   ├── model_info.py      # Model information
│   │   ├── chat_session.py    # Chat session
│   │   └── chat_message.py    # Chat messages
│   ├── schemas/               # Pydantic schemas
│   │   ├── finetune.py
│   │   ├── chat.py
│   │   └── model.py
│   ├── api/                   # API routes
│   │   ├── finetune.py        # Fine-tuning endpoints
│   │   ├── chat.py            # Chat endpoints
│   │   ├── model.py           # Model management
│   │   └── upload.py          # File upload
│   ├── services/              # Business logic
│   │   ├── finetune_service.py
│   │   ├── chat_service.py
│   │   └── model_service.py
│   ├── utils/                 # Utility functions
│   │   ├── file_processor.py
│   │   ├── data_processor.py
│   │   ├── model_utils.py
│   │   └── chat_utils.py
│   └── streamlit_app.py       # Streamlit interface
├── data/                      # Data storage
├── models/                    # Fine-tuned models
├── uploads/                   # Uploaded files
├── static/                    # Static files
├── requirements.txt           # Python dependencies
├── .env.example              # Environment template
├── init_db.py                # Database initialization
├── run.py                    # Application runner
├── test_app.py               # Test script
└── README.md                 # Documentation
```

## 🔌 API Endpoints

### Fine-tuning
- `POST /api/finetune/upload` - Upload files for fine-tuning
- `POST /api/finetune/start` - Start fine-tuning process
- `GET /api/finetune/status/{job_id}` - Check fine-tuning status
- `GET /api/finetune/history` - Get fine-tuning history
- `POST /api/finetune/cancel/{job_id}` - Cancel fine-tuning job

### Chat
- `POST /api/chat/send` - Send chat message
- `POST /api/chat/session` - Create chat session
- `GET /api/chat/session/{session_id}` - Get session info
- `GET /api/chat/session/{session_id}/history` - Get chat history
- `POST /api/chat/session/{session_id}/upload` - Upload context files
- `GET /api/chat/sessions` - List all sessions
- `DELETE /api/chat/session/{session_id}` - Delete session

### Models
- `GET /api/models` - List all models
- `GET /api/models/{model_id}` - Get model details
- `POST /api/models` - Create new model
- `PUT /api/models/{model_id}` - Update model
- `DELETE /api/models/{model_id}` - Delete model
- `GET /api/models/{model_id}/performance` - Get performance metrics
- `GET /api/models/statistics` - Get model statistics

### File Upload
- `POST /api/upload/file` - Upload single file
- `POST /api/upload/files` - Upload multiple files
- `GET /api/upload/file/{file_path}` - Download file
- `DELETE /api/upload/file/{file_path}` - Delete file
- `POST /api/upload/process` - Process file content
- `GET /api/upload/list` - List uploaded files
- `GET /api/upload/config` - Get upload configuration

## 🧪 Testing

Run tests to verify the application:
```bash
python test_app.py
```

Initialize database:
```bash
python init_db.py
```

## 🔧 Troubleshooting

### Common Issues:

1. **Import errors**: Ensure all dependencies are installed
2. **Database errors**: Run `python init_db.py` to initialize database
3. **Configuration errors**: Check `.env` file format
4. **Port conflicts**: Change ports in `.env` file if needed

### Logs:
- API logs: Terminal running uvicorn
- Streamlit logs: Terminal running streamlit

## 📊 Monitoring

### Health Check
```bash
curl -X GET "http://localhost:8000/api/upload/config"
```

### System Status
- Database connection
- Model availability
- File system status
- Memory usage

## 🔒 Security

### Best Practices:
1. Change default `SECRET_KEY`
2. Use HTTPS in production
3. Limit file access permissions
4. Validate input data
5. Log security events

## 📈 Scaling

### Production Deployment:
1. Use PostgreSQL instead of SQLite
2. Redis for caching
3. Load balancer for API
4. CDN for static files
5. Monitoring and logging

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 📞 Support

If you encounter issues:
1. Check logs
2. Run `python test_app.py`
3. Create issue with detailed error information
4. Provide environment details

## 📝 License

MIT License 