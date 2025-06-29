from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class FineTuneStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class FineTuneRequest(BaseModel):
    name: str = Field(..., description="Tên model sau khi fine-tune")
    base_model: str = Field(default="Qwen/Qwen3-8B", description="Base model để fine-tune")
    learning_rate: float = Field(default=2e-5, description="Learning rate")
    batch_size: int = Field(default=4, description="Batch size")
    epochs: int = Field(default=3, description="Số epochs")
    max_length: int = Field(default=512, description="Maximum sequence length")
    warmup_steps: int = Field(default=100, description="Warmup steps")
    data_files: List[str] = Field(..., description="Danh sách file dữ liệu")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Cấu hình bổ sung")

class FineTuneResponse(BaseModel):
    job_id: str
    status: FineTuneStatus
    message: str
    created_at: datetime

class FineTuneStatusResponse(BaseModel):
    job_id: str
    status: FineTuneStatus
    progress: float = Field(..., ge=0.0, le=1.0)
    current_epoch: int
    current_step: int
    total_steps: int
    final_loss: Optional[float] = None
    final_accuracy: Optional[float] = None
    path: Optional[str] = None
    hf_repo_url: Optional[str] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class FineTuneHistoryItem(BaseModel):
    id: int
    job_id: str
    name: str
    base_model: str
    status: FineTuneStatus
    progress: float
    final_loss: Optional[float]
    final_accuracy: Optional[float]
    created_at: datetime
    completed_at: Optional[datetime]

class FineTuneHistory(BaseModel):
    jobs: List[FineTuneHistoryItem]
    total: int
    page: int
    size: int 