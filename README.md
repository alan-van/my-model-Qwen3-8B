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
- **Automatic Tokenizer Fallback** - Tự động sử dụng local tokenizer nếu cần

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

# Cài đặt accelerate (bắt buộc cho device_map)
pip install accelerate
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

## ⚡️ Tải model về local để tăng tốc và tránh timeout

**Khuyến nghị:** Trước khi chạy backend lần đầu, hãy tải model về local để tránh timeout khi tải model lớn từ Hugging Face.

### Bước 1: Tải model về local (khuyến nghị dùng snapshot)

Chạy script sau để tải toàn bộ snapshot model về thư mục local (ví dụ cho Qwen/Qwen3-8B):

```bash
python download_snapshot.py --repo_id Qwen/Qwen3-8B --output_dir ./models/base/Qwen3-8B
```

> Nếu bạn chỉ muốn tải model/tokenizer (không phải toàn bộ snapshot), có thể dùng script download_model.py:
> ```bash
> python download_model.py --model_repo Qwen/Qwen3-8B --output_dir ./models/base/Qwen3-8B
> ```

### Bước 2: Cấu hình backend ưu tiên load model từ local

- Mặc định, backend sẽ tự động ưu tiên load model từ `./models/base/Qwen3-8B` nếu thư mục này tồn tại và không rỗng.
- Nếu muốn chỉ định đường dẫn khác, đặt biến môi trường:
  ```bash
  export BASE_MODEL_LOCAL_DIR=/duong/dan/den/thu_muc_model
  ```
- Sau đó khởi động lại backend.

## 🔧 Troubleshooting

### Lỗi thường gặp và cách khắc phục

#### 1. Lỗi "accelerate required"
```
Error: Using a `device_map`, `tp_plan`, `torch.device` context manager or setting `torch.set_default_device(device)` requires `accelerate`.
```
**Giải pháp:**
```bash
pip install accelerate
```

#### 2. Lỗi "Qwen2Tokenizer does not exist"
```
Error: Tokenizer class Qwen2Tokenizer does not exist or is not currently imported.
```
**Giải pháp:**
- Cập nhật transformers lên phiên bản mới nhất:
  ```bash
  pip install --upgrade transformers
  ```
- Ứng dụng đã có fallback mechanism để tự động sử dụng local tokenizer nếu cần.

#### 3. Lỗi bitsandbytes GPU support
```
Warning: The installed version of bitsandbytes was compiled without GPU support.
```
**Giải pháp:**
- Nếu bạn có GPU và muốn dùng quantization:
  ```bash
  pip uninstall bitsandbytes
  pip install bitsandbytes
  ```
- Nếu không có GPU, có thể bỏ qua cảnh báo này.

#### 4. Lỗi timeout khi tải model
**Giải pháp:**
- Tải model về local trước (xem phần "Tải model về local" ở trên)
- Hoặc tăng timeout trong cấu hình

#### 5. Lỗi CUDA out of memory
**Giải pháp:**
- Giảm batch_size trong cấu hình fine-tuning
- Sử dụng quantization (4-bit hoặc 8-bit)
- Giảm max_length

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

## 🧹 Xoá Job History (Lịch sử Fine-tune)

### Cách 1: Xoá trực tiếp trong database (SQLite)

Nếu bạn dùng SQLite (file `finetuning.db`), có thể xoá job history bằng lệnh:

```bash
sqlite3 finetuning.db
```
Sau đó trong prompt SQLite:
```sql
DELETE FROM finetune_jobs;
-- hoặc xoá từng job theo job_id:
DELETE FROM finetune_jobs WHERE job_id = 'your_job_id';
.exit
```

Hoặc dùng phần mềm DB Browser for SQLite để thao tác trực quan.

### Cách 2: Thêm API xoá job (tuỳ chọn)

Bạn có thể thêm endpoint vào `app/api/finetune.py`:
```python
@router.delete("/delete/{job_id}")
async def delete_finetune_job(job_id: str, db: Session = Depends(get_db)):
    job = db.query(FineTuneJob).filter(FineTuneJob.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    db.delete(job)
    db.commit()
    return {"message": "Job deleted successfully"}
```
Hoặc xoá toàn bộ:
```python
@router.delete("/delete_all")
async def delete_all_finetune_jobs(db: Session = Depends(get_db)):
    db.query(FineTuneJob).delete()
    db.commit()
    return {"message": "All jobs deleted successfully"}
```
Sau đó gọi API này bằng curl hoặc Postman.

### Cách 3: Xoá toàn bộ database

Chỉ cần xoá file `finetuning.db` rồi khởi động lại backend (sẽ mất toàn bộ dữ liệu jobs, models, ...). 