import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from app.database import engine, Base
from app.api import finetune_router, chat_router, model_router, upload_router
from app.config.settings import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Qwen3-8B Fine-tuning Application")
    
    # Create database tables
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Qwen3-8B Fine-tuning Application")

# Create FastAPI app
app = FastAPI(
    title="Qwen3-8B Fine-tuning API",
    description="API để fine-tuning model Qwen3-8B và chat với model đã fine-tune",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production, chỉ định domain cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(finetune_router)
app.include_router(chat_router)
app.include_router(model_router)
app.include_router(upload_router)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Qwen3-8B Fine-tuning API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "API is running"
    }

@app.get("/info")
async def get_info():
    """Get application information"""
    return {
        "app_name": "Qwen3-8B Fine-tuning Application",
        "version": "1.0.0",
        "model_name": settings.model_name,
        "base_model_path": settings.base_model_path,
        "fastapi": "0.104.1",
        "finetuned_model_path": settings.finetuned_model_path,
        "upload_dir": settings.upload_dir,
        "allowed_extensions": settings.allowed_extensions,
        "max_file_size": settings.max_file_size
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    ) 