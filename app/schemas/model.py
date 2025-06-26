from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class ModelInfoResponse(BaseModel):
    model_id: str
    model_name: str
    model_type: str
    base_model: str
    model_path: str
    hf_repo_url: Optional[str]
    model_size: Optional[str]
    accuracy: Optional[float]
    loss: Optional[float]
    perplexity: Optional[float]
    training_data_size: Optional[int]
    training_epochs: Optional[int]
    training_steps: Optional[int]
    description: Optional[str]
    tags: Optional[List[str]]
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime]

class ModelListResponse(BaseModel):
    models: List[ModelInfoResponse]
    total: int
    page: int
    size: int

class ModelCreateRequest(BaseModel):
    model_name: str = Field(..., description="Tên model")
    model_type: str = Field(..., description="Loại model (base, finetuned)")
    base_model: str = Field(..., description="Base model")
    model_path: str = Field(..., description="Đường dẫn đến model")
    hf_repo_url: Optional[str] = Field(default=None, description="Hugging Face repo URL")
    model_size: Optional[str] = Field(default=None, description="Kích thước model")
    description: Optional[str] = Field(default=None, description="Mô tả model")
    tags: Optional[List[str]] = Field(default=None, description="Tags")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Cấu hình model") 