# Qwen3-8B Fine-tuning Application

Ứng dụng Python chuyên nghiệp để fine-tuning model Qwen3-8B với khả năng xử lý đa dạng định dạng file và chat interface.

## 🚀 Tính năng chính

- **Fine-tuning Qwen3-8B** với dữ liệu từ nhiều định dạng file (.csv, .txt, .pdf, .docx, .xlsx)
- **REST API** để quản lý quá trình fine-tuning
- **Chat interface** với khả năng upload file bổ sung
- **Database** lưu trữ thông tin fine-tuning
- **Web interface** thân thiện người dùng (Streamlit)
- **File Upload System** hỗ trợ nhiều định dạng file
- **Model Management** quản lý và theo dõi các model đã fine-tune

## 📋 Yêu cầu hệ thống

- Python 3.8+
- RAM: 16GB+ (cho fine-tuning)
- GPU: NVIDIA GPU với CUDA (khuyến nghị)
- Disk space: 20GB+ cho model và dữ liệu

## 🛠️ Cài đặt

```bash
# Clone repository
git clone <repository-url>
cd qwen3-8b-finetuning

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

## ⚙️ Cấu hình

1. Tạo file `.env`:
```bash
cp .env.example .env
```

2. Cập nhật các biến môi trường trong `.env`:
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

## 🚀 Sử dụng

### Khởi động API server
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
- API Documentation: http://localhost:8000/docs
- API Base URL: http://localhost:8000

### Khởi động Streamlit interface
```bash
streamlit run app/streamlit_app.py
```
- Web Interface: http://localhost:8501

### Chạy cả hai cùng lúc
```bash
python run.py --both
```

## 📡 API Endpoints

### Fine-tuning
- `POST /api/finetune/upload` - Upload file để fine-tuning
- `POST /api/finetune/start` - Bắt đầu fine-tuning
- `GET /api/finetune/status/{job_id}` - Kiểm tra trạng thái fine-tuning
- `GET /api/finetune/history` - Lịch sử fine-tuning

### Chat
- `POST /api/chat` - Chat với model đã fine-tune
- `POST /api/chat/upload` - Upload file bổ sung cho chat
- `POST /api/chat/session` - Tạo session chat mới

### Model Management
- `GET /api/models` - Danh sách models
- `GET /api/models/{model_id}` - Thông tin chi tiết model
- `DELETE /api/models/{model_id}` - Xóa model

### File Upload
- `POST /api/upload/file` - Upload một file
- `POST /api/upload/files` - Upload nhiều files
- `GET /api/upload/list` - Danh sách files đã upload
- `GET /api/upload/config` - Cấu hình upload

## 🔌 API Usage Examples

### Sử dụng với Curl

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

### Sử dụng với Postman

#### 1. Setup Postman Collection

Tạo collection mới với base URL: `http://localhost:8000`

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

### Environment Variables cho Postman

Tạo environment với các variables:
- `base_url`: `http://localhost:8000`
- `job_id`: (sẽ được set từ response)
- `model_id`: (sẽ được set từ response)
- `session_id`: (sẽ được set từ response)

### Test Scripts cho Postman

#### Set Job ID từ Fine-tuning Response
```javascript
// Trong Tests tab của Start Fine-tuning request
if (pm.response.code === 200) {
    const response = pm.response.json();
    pm.environment.set("job_id", response.job_id);
}
```

#### Set Model ID từ Model List Response
```javascript
// Trong Tests tab của Get Models request
if (pm.response.code === 200) {
    const response = pm.response.json();
    if (response.models && response.models.length > 0) {
        pm.environment.set("model_id", response.models[0].model_id);
    }
}
```

#### Set Session ID từ Create Session Response
```javascript
// Trong Tests tab của Create Chat Session request
if (pm.response.code === 200) {
    const response = pm.response.json();
    pm.environment.set("session_id", response.session_id);
}
```

## 📁 Cấu trúc thư mục

```
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── config.py              # Configuration
│   ├── database.py            # Database setup
│   ├── models/                # Database models
│   │   ├── finetune_job.py    # Fine-tuning job model
│   │   ├── model_info.py      # Model information
│   │   ├── chat_session.py    # Chat session
│   │   └── chat_message.py    # Chat messages
│   ├── schemas/               # Pydantic schemas
│   ├── api/                   # API routes
│   │   ├── finetune.py        # Fine-tuning endpoints
│   │   ├── chat.py            # Chat endpoints
│   │   ├── model.py           # Model management
│   │   └── upload.py          # File upload
│   ├── services/              # Business logic
│   ├── utils/                 # Utility functions
│   └── streamlit_app.py       # Streamlit interface
├── data/                      # Data storage
├── models/                    # Fine-tuned models
├── uploads/                   # Uploaded files
├── static/                    # Static files
├── requirements.txt
├── .env.example
├── run.py                     # Application runner
├── init_db.py                 # Database initialization
├── test_app.py                # Application tests
└── README.md
```

## 🧪 Testing

Chạy test để kiểm tra ứng dụng:
```bash
python test_app.py
```

Khởi tạo database:
```bash
python init_db.py
```

## 🔧 Troubleshooting

### Lỗi thường gặp:

1. **Import errors**: Đảm bảo đã cài đặt đầy đủ dependencies
2. **Database errors**: Chạy `python init_db.py` để khởi tạo database
3. **Configuration errors**: Kiểm tra file `.env` có đúng định dạng
4. **Port conflicts**: Thay đổi port trong file `.env` nếu cần

### Logs:
- API logs: Terminal nơi chạy uvicorn
- Streamlit logs: Terminal nơi chạy streamlit

## 📝 License

MIT License

## 🤝 Contributing

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push to branch
5. Tạo Pull Request

## 📞 Support

Nếu gặp vấn đề, hãy:
1. Kiểm tra logs
2. Chạy `python test_app.py`
3. Tạo issue với thông tin lỗi chi tiết 