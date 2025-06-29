from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # Database configuration
    database_url: str = Field("sqlite:///./finetuning.db", alias="DATABASE_URL")
    
    # Model configuration
    model_name: str = Field("Qwen/Qwen3-8B", alias="MODEL_NAME")
    base_model_path: str = Field("Qwen/Qwen3-8B", alias="BASE_MODEL_PATH")
    finetuned_model_path: str = Field("./models/finetuned", alias="FINETUNED_MODEL_PATH")
    
    # Security configuration
    secret_key: str = Field(..., alias="SECRET_KEY")
    algorithm: str = Field("HS256", alias="ALGORITHM")
    access_token_expire_minutes: int = Field(30, alias="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Training defaults
    default_learning_rate: float = Field(2e-5, alias="DEFAULT_LEARNING_RATE")
    default_batch_size: int = Field(4, alias="DEFAULT_BATCH_SIZE")
    default_epochs: int = Field(3, alias="DEFAULT_EPOCHS")
    default_max_length: int = Field(512, alias="DEFAULT_MAX_LENGTH")
    default_warmup_steps: int = Field(100, alias="DEFAULT_WARMUP_STEPS")
    
    # Server configuration
    host: str = Field("0.0.0.0", alias="HOST")
    port: int = Field(8000, alias="PORT")
    debug: bool = True
    
    # Hugging Face configuration
    hf_token: str = Field("", alias="HF_TOKEN")
    hf_repo_id: str = Field("", alias="HF_REPO_ID")
    
    # File upload configuration
    upload_dir: str = Field("./uploads", alias="UPLOAD_DIR")
    max_file_size: int = Field(100 * 1024 * 1024, alias="MAX_FILE_SIZE")
    allowed_extensions: str = Field('["csv", "txt", "pdf", "docx", "xlsx", "jpg", "jpeg", "png"]', alias="ALLOWED_EXTENSIONS")
    
    # API configuration
    api_prefix: str = "/api/v1"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=("settings_",))

settings = Settings()

# Hướng dẫn clone model Qwen2-7B về local:
# git lfs install
# git clone https://huggingface.co/Qwen/Qwen2-7B ./models/base/Qwen2-7B 