# Hướng dẫn sử dụng Qwen3-8B Fine-tuning Application

## 🚀 Khởi động nhanh

### 1. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 2. Khởi tạo ứng dụng
```bash
python init_db.py
```

### 3. Cấu hình
```bash
cp env.example .env
# Chỉnh sửa file .env với cấu hình của bạn
```

### 4. Test ứng dụng
```bash
python test_app.py
```

### 5. Chạy ứng dụng
```bash
# Chạy cả API và Streamlit interface
python run.py --both

# Hoặc chạy riêng lẻ
python run.py --mode api --api-port 8000
python run.py --mode streamlit --streamlit-port 8501

# Hoặc chạy thủ công
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
streamlit run app/streamlit_app.py
```

## 📋 Tính năng chính

### 🎯 Fine-tuning
- **Upload dữ liệu**: Hỗ trợ .csv, .txt, .pdf, .docx, .xlsx
- **Cấu hình training**: Learning rate, batch size, epochs, max length
- **Theo dõi tiến trình**: Real-time progress tracking
- **Lưu trữ model**: Tự động lưu model đã fine-tune
- **Job Management**: Quản lý và hủy jobs fine-tuning

### 💬 Chat Interface
- **Chat với model**: Tương tác với model đã fine-tune
- **Upload context files**: Thêm file bổ sung cho context
- **Lịch sử chat**: Lưu trữ và xem lại lịch sử chat
- **Session management**: Quản lý nhiều session chat
- **File attachments**: Upload ảnh và file bổ sung

### 📊 Model Management
- **Quản lý models**: Xem, tạo, cập nhật, xóa models
- **Performance metrics**: Theo dõi accuracy, loss, perplexity
- **Model statistics**: Thống kê tổng quan về models
- **Model comparison**: So sánh performance giữa các models

### 📁 File Upload
- **Upload files**: Hỗ trợ nhiều định dạng file
- **File processing**: Tự động xử lý và trích xuất nội dung
- **File management**: Quản lý files đã upload
- **File validation**: Kiểm tra kích thước và định dạng

## 🔧 API Endpoints

### Fine-tuning
```
POST /api/finetune/upload     - Upload files
POST /api/finetune/start      - Bắt đầu fine-tuning
GET  /api/finetune/status/{id} - Kiểm tra trạng thái
GET  /api/finetune/history    - Lịch sử fine-tuning
POST /api/finetune/cancel/{id} - Hủy job
POST /api/finetune/register/{id} - Đăng ký model
GET  /api/finetune/config     - Cấu hình mặc định
```

### Chat
```
POST /api/chat/send           - Gửi tin nhắn
POST /api/chat/session        - Tạo session
GET  /api/chat/session/{id}   - Lấy thông tin session
GET  /api/chat/session/{id}/history - Lịch sử chat
POST /api/chat/session/{id}/upload - Upload context files
GET  /api/chat/sessions       - Danh sách sessions
DELETE /api/chat/session/{id} - Xóa session
```

### Models
```
GET  /api/models              - Danh sách models
GET  /api/models/{id}         - Thông tin model
POST /api/models              - Tạo model
PUT  /api/models/{id}         - Cập nhật model
DELETE /api/models/{id}       - Xóa model
GET  /api/models/{id}/performance - Performance metrics
GET  /api/models/statistics   - Thống kê models
```

### File Upload
```
POST /api/upload/file         - Upload file
POST /api/upload/files        - Upload nhiều files
GET  /api/upload/file/{path}  - Download file
DELETE /api/upload/file/{path} - Xóa file
POST /api/upload/process      - Xử lý file
GET  /api/upload/list         - Danh sách files
GET  /api/upload/config       - Cấu hình upload
```

## 🔌 API Usage Examples

### Sử dụng với Curl

#### 1. Health Check & System Info
```bash
# Kiểm tra cấu hình upload
curl -X GET "http://localhost:8000/api/upload/config"

# Kiểm tra trạng thái hệ thống
curl -X GET "http://localhost:8000/health"
```

#### 2. Upload Files
```bash
# Upload một file
curl -X POST "http://localhost:8000/api/upload/file" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/data.csv"

# Upload nhiều files
curl -X POST "http://localhost:8000/api/upload/files" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@/path/to/file1.csv" \
  -F "files=@/path/to/file2.txt"

# Xem danh sách files đã upload
curl -X GET "http://localhost:8000/api/upload/list"
```

#### 3. Fine-tuning Workflow
```bash
# Bước 1: Upload dữ liệu training
curl -X POST "http://localhost:8000/api/upload/file" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@training_data.csv"

# Bước 2: Bắt đầu fine-tuning
curl -X POST "http://localhost:8000/api/finetune/start" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "my-custom-model",
    "base_model": "Qwen/Qwen3-8B",
    "learning_rate": 2e-5,
    "batch_size": 4,
    "epochs": 3,
    "max_length": 512,
    "warmup_steps": 100,
    "data_files": ["path/to/uploaded/training_data.csv"]
  }'

# Bước 3: Theo dõi tiến trình
curl -X GET "http://localhost:8000/api/finetune/status/job_12345678"

# Bước 4: Xem lịch sử fine-tuning
curl -X GET "http://localhost:8000/api/finetune/history"

# Bước 5: Hủy job nếu cần
curl -X POST "http://localhost:8000/api/finetune/cancel/job_12345678"
```

#### 4. Chat Workflow
```bash
# Bước 1: Xem danh sách models
curl -X GET "http://localhost:8000/api/models"

# Bước 2: Tạo session chat
curl -X POST "http://localhost:8000/api/chat/session" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "model_12345678",
    "session_name": "My Chat Session"
  }'

# Bước 3: Gửi tin nhắn
curl -X POST "http://localhost:8000/api/chat/send" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "session_12345678",
    "message": "Hello, how are you?",
    "model_id": "model_12345678"
  }'

# Bước 4: Upload context file cho chat
curl -X POST "http://localhost:8000/api/chat/session/session_12345678/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@context_document.pdf"

# Bước 5: Xem lịch sử chat
curl -X GET "http://localhost:8000/api/chat/session/session_12345678/history"

# Bước 6: Xem danh sách sessions
curl -X GET "http://localhost:8000/api/chat/sessions"
```

#### 5. Model Management
```bash
# Xem danh sách models
curl -X GET "http://localhost:8000/api/models"

# Xem chi tiết model
curl -X GET "http://localhost:8000/api/models/model_12345678"

# Xem performance metrics
curl -X GET "http://localhost:8000/api/models/model_12345678/performance"

# Xem thống kê models
curl -X GET "http://localhost:8000/api/models/statistics"

# Xóa model
curl -X DELETE "http://localhost:8000/api/models/model_12345678"
```

### Sử dụng với Postman

#### 1. Setup Postman Collection

**Tạo Collection mới:**
- Name: `Qwen3-8B Fine-tuning API`
- Description: `API collection for Qwen3-8B fine-tuning application`

**Tạo Environment:**
- Name: `Qwen3-8B Local`
- Variables:
  - `base_url`: `http://localhost:8000`
  - `job_id`: (empty - sẽ được set từ response)
  - `model_id`: (empty - sẽ được set từ response)
  - `session_id`: (empty - sẽ được set từ response)
  - `file_path`: (empty - sẽ được set từ response)

#### 2. Health Check Requests

**Get Upload Config:**
- Method: `GET`
- URL: `{{base_url}}/api/upload/config`
- Headers: None
- Tests:
```javascript
pm.test("Status code is 200", function () {
    pm.response.to.have.status(200);
});

pm.test("Response has required fields", function () {
    const response = pm.response.json();
    pm.expect(response).to.have.property('max_file_size');
    pm.expect(response).to.have.property('allowed_extensions');
    pm.expect(response).to.have.property('upload_dir');
});
```

**Health Check:**
- Method: `GET`
- URL: `{{base_url}}/health`
- Headers: None

#### 3. File Upload Requests

**Upload Single File:**
- Method: `POST`
- URL: `{{base_url}}/api/upload/file`
- Headers: None (Postman sẽ tự động set Content-Type)
- Body:
  - Type: `form-data`
  - Key: `file`
  - Type: `File`
  - Value: Select your file
- Tests:
```javascript
pm.test("Upload successful", function () {
    pm.response.to.have.status(200);
});

if (pm.response.code === 200) {
    const response = pm.response.json();
    pm.environment.set("file_path", response.file_path);
}
```

**Upload Multiple Files:**
- Method: `POST`
- URL: `{{base_url}}/api/upload/files`
- Body:
  - Type: `form-data`
  - Key: `files`
  - Type: `File`
  - Value: Select multiple files

**List Uploaded Files:**
- Method: `GET`
- URL: `{{base_url}}/api/upload/list`
- Headers: None

#### 4. Fine-tuning Requests

**Start Fine-tuning:**
- Method: `POST`
- URL: `{{base_url}}/api/finetune/start`
- Headers:
  - `Content-Type: application/json`
- Body (raw JSON):
```json
{
  "model_name": "my-custom-model",
  "base_model": "Qwen/Qwen3-8B",
  "learning_rate": 2e-5,
  "batch_size": 4,
  "epochs": 3,
  "max_length": 512,
  "warmup_steps": 100,
  "data_files": ["{{file_path}}"]
}
```
- Tests:
```javascript
pm.test("Fine-tuning started", function () {
    pm.response.to.have.status(200);
});

if (pm.response.code === 200) {
    const response = pm.response.json();
    pm.environment.set("job_id", response.job_id);
}
```

**Check Fine-tuning Status:**
- Method: `GET`
- URL: `{{base_url}}/api/finetune/status/{{job_id}}`
- Headers: None
- Tests:
```javascript
pm.test("Status check successful", function () {
    pm.response.to.have.status(200);
});

const response = pm.response.json();
pm.test("Job status is valid", function () {
    pm.expect(response).to.have.property('status');
    pm.expect(['pending', 'running', 'completed', 'failed']).to.include(response.status);
});
```

**Get Fine-tuning History:**
- Method: `GET`
- URL: `{{base_url}}/api/finetune/history`
- Headers: None

**Cancel Fine-tuning:**
- Method: `POST`
- URL: `{{base_url}}/api/finetune/cancel/{{job_id}}`
- Headers: None

#### 5. Chat Requests

**Create Chat Session:**
- Method: `POST`
- URL: `{{base_url}}/api/chat/session`
- Headers:
  - `Content-Type: application/json`
- Body (raw JSON):
```json
{
  "model_id": "{{model_id}}",
  "session_name": "My Chat Session"
}
```
- Tests:
```javascript
if (pm.response.code === 200) {
    const response = pm.response.json();
    pm.environment.set("session_id", response.session_id);
}
```

**Send Chat Message:**
- Method: `POST`
- URL: `{{base_url}}/api/chat/send`
- Headers:
  - `Content-Type: application/json`
- Body (raw JSON):
```json
{
  "session_id": "{{session_id}}",
  "message": "Hello, how are you?",
  "model_id": "{{model_id}}"
}
```

**Upload Context File for Chat:**
- Method: `POST`
- URL: `{{base_url}}/api/chat/session/{{session_id}}/upload`
- Body:
  - Type: `form-data`
  - Key: `file`
  - Type: `File`
  - Value: Select context file

**Get Chat History:**
- Method: `GET`
- URL: `{{base_url}}/api/chat/session/{{session_id}}/history`
- Headers: None

**List Chat Sessions:**
- Method: `GET`
- URL: `{{base_url}}/api/chat/sessions`
- Headers: None

#### 6. Model Management Requests

**Get Models List:**
- Method: `GET`
- URL: `{{base_url}}/api/models`
- Headers: None
- Tests:
```javascript
if (pm.response.code === 200) {
    const response = pm.response.json();
    if (response.models && response.models.length > 0) {
        pm.environment.set("model_id", response.models[0].model_id);
    }
}
```

**Get Model Details:**
- Method: `GET`
- URL: `{{base_url}}/api/models/{{model_id}}`
- Headers: None

**Get Model Performance:**
- Method: `GET`
- URL: `{{base_url}}/api/models/{{model_id}}/performance`
- Headers: None

**Get Models Statistics:**
- Method: `GET`
- URL: `{{base_url}}/api/models/statistics`
- Headers: None

**Delete Model:**
- Method: `DELETE`
- URL: `{{base_url}}/api/models/{{model_id}}`
- Headers: None

#### 7. Advanced Postman Features

**Pre-request Scripts:**
```javascript
// Set timestamp for unique names
pm.environment.set("timestamp", new Date().getTime());
```

**Collection Variables:**
- `default_model_name`: `my-model-{{timestamp}}`
- `default_learning_rate`: `2e-5`
- `default_batch_size`: `4`

**Environment Switching:**
- Local: `http://localhost:8000`
- Development: `http://dev-server:8000`
- Production: `https://api.production.com`

**Request Chaining:**
1. Upload file → Get file_path
2. Start fine-tuning → Get job_id
3. Check status → Monitor progress
4. Create session → Get session_id
5. Send message → Get response

## 📝 Cấu trúc dữ liệu

### Fine-tuning Request
```json
{
  "model_name": "my-finetuned-model",
  "base_model": "Qwen/Qwen3-8B",
  "learning_rate": 2e-5,
  "batch_size": 4,
  "epochs": 3,
  "max_length": 512,
  "warmup_steps": 100,
  "data_files": ["path/to/file1.csv", "path/to/file2.txt"]
}
```

### Chat Request
```json
{
  "model_id": "model_12345678",
  "message": "Hello, how are you?",
  "session_id": "session_12345678",
  "system_prompt": "You are a helpful assistant.",
  "context_files": ["path/to/context.pdf"]
}
```

### Model Info
```json
{
  "model_id": "model_12345678",
  "model_name": "My Fine-tuned Model",
  "model_type": "finetuned",
  "base_model": "Qwen/Qwen3-8B",
  "model_path": "./models/finetuned/model_12345678",
  "accuracy": 0.85,
  "loss": 0.15,
  "training_data_size": 1000,
  "training_epochs": 3,
  "is_active": true
}
```

## 🎨 Streamlit Interface

### Dashboard
- **Overview**: Thông tin tổng quan về ứng dụng
- **Statistics**: Thống kê models và jobs
- **Recent Activity**: Hoạt động gần đây
- **System Status**: Trạng thái hệ thống

### Fine-tuning
- **Start Fine-tuning**: Tạo job fine-tuning mới
- **Job Status**: Theo dõi tiến trình training
- **History**: Xem lịch sử fine-tuning
- **Job Management**: Hủy và quản lý jobs

### Chat
- **Model Selection**: Chọn model để chat
- **Chat Interface**: Giao diện chat thân thiện
- **Context Files**: Upload files bổ sung
- **Session Management**: Quản lý sessions
- **File Attachments**: Upload ảnh và file

### Models
- **Model List**: Danh sách và quản lý models
- **Performance**: Biểu đồ performance metrics
- **Model Actions**: Tạo, cập nhật, xóa models
- **Model Comparison**: So sánh models

### File Upload
- **Upload Files**: Upload files với drag & drop
- **File List**: Quản lý files đã upload
- **File Processing**: Xem nội dung files
- **File Validation**: Kiểm tra định dạng và kích thước

## 🔧 Cấu hình nâng cao

### Environment Variables
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

## 🚀 Workflow sử dụng

### 1. Chuẩn bị dữ liệu
```bash
# Upload dữ liệu training
curl -X POST "http://localhost:8000/api/upload/file" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_data.csv"
```

### 2. Fine-tune model
```bash
# Bắt đầu fine-tuning
curl -X POST "http://localhost:8000/api/finetune/start" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "my-model",
    "data_files": ["path/to/uploaded/file.csv"],
    "learning_rate": 2e-5,
    "batch_size": 4,
    "epochs": 3
  }'
```

### 3. Theo dõi tiến trình
```bash
# Kiểm tra trạng thái
curl -X GET "http://localhost:8000/api/finetune/status/job_id"
```

### 4. Chat với model
```bash
# Tạo session chat
curl -X POST "http://localhost:8000/api/chat/session" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "model_id",
    "session_name": "My Chat Session"
  }'

# Gửi tin nhắn
curl -X POST "http://localhost:8000/api/chat/send" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "session_id",
    "message": "Hello, how are you?"
  }'
```

## 🔧 Troubleshooting

### Lỗi thường gặp:

1. **Import errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **Database errors**
   ```bash
   python init_db.py
   ```

3. **Configuration errors**
   - Kiểm tra file `.env` có đúng định dạng
   - Đảm bảo `ALLOWED_EXTENSIONS` là JSON array
   - Đảm bảo `MAX_FILE_SIZE` là số nguyên

4. **Port conflicts**
   - Thay đổi port trong file `.env`
   - Kiểm tra port đang được sử dụng

5. **File upload errors**
   - Kiểm tra kích thước file
   - Kiểm tra định dạng file được hỗ trợ
   - Đảm bảo thư mục upload có quyền ghi

6. **API connection errors**
   - Kiểm tra API server có đang chạy không
   - Kiểm tra URL và port
   - Kiểm tra firewall settings

### Logs và Debug:

- **API logs**: Terminal nơi chạy uvicorn
- **Streamlit logs**: Terminal nơi chạy streamlit
- **Database logs**: Kiểm tra file `finetuning.db`

### Performance Optimization:

1. **GPU Usage**: Đảm bảo CUDA được cài đặt cho fine-tuning
2. **Memory**: Tăng RAM nếu gặp lỗi out of memory
3. **Batch Size**: Giảm batch size nếu gặp lỗi memory
4. **File Size**: Chia nhỏ file dữ liệu nếu quá lớn

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
1. Thay đổi `SECRET_KEY` mặc định
2. Sử dụng HTTPS trong production
3. Giới hạn quyền truy cập file
4. Validate input data
5. Log security events

## 📈 Scaling

### Production Deployment:
1. Sử dụng PostgreSQL thay vì SQLite
2. Redis cho caching
3. Load balancer cho API
4. CDN cho static files
5. Monitoring và logging

## 🤝 Contributing

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push to branch
5. Tạo Pull Request

## 📞 Support

Nếu gặp vấn đề:
1. Kiểm tra logs
2. Chạy `python test_app.py`
3. Tạo issue với thông tin lỗi chi tiết
4. Cung cấp environment details 