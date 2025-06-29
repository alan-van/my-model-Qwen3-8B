import os
from typing import List, Union
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # Database
    database_url: str = "sqlite:///./finetuning.db"
    
    # Model Configuration
    model_name: str = "Qwen/Qwen3-8B"
    base_model_path: str = "./models/base"
    finetuned_model_path: str = "./models/finetuned"
    
    # Hugging Face
    hf_token: str = "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"
    hf_repo_id: str = "=Qwen/Qwen3-8B"
    
    # Security
    secret_key: str = "tx2A4kso2qq5Jo4yCZSBoHGa1I_sOYHS0kS37-riE-g"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # File Upload
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    upload_dir: str = "./uploads"
    allowed_extensions: List[str] = ["csv", "txt", "pdf", "docx", "xlsx", "jpg", "jpeg", "png"]
    
    # Fine-tuning Defaults
    default_learning_rate: float = 2e-5
    default_batch_size: int = 4
    default_epochs: int = 3
    default_max_length: int = 512
    default_warmup_steps: int = 100
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        protected_namespaces=("settings_",)
    )
    
    @field_validator('allowed_extensions', mode='before')
    @classmethod
    def parse_allowed_extensions(cls, v):
        if isinstance(v, str):
            return [ext.strip() for ext in v.split(',') if ext.strip()]
        return v

settings = Settings()

# Ensure directories exist
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.base_model_path, exist_ok=True)
os.makedirs(settings.finetuned_model_path, exist_ok=True)
os.makedirs("./data", exist_ok=True) 