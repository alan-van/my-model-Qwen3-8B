#!/usr/bin/env python3
"""
Script để khởi tạo database và dữ liệu mẫu cho ứng dụng Qwen3-8B Fine-tuning
"""

import os
import sys
import logging
from sqlalchemy.orm import Session

# Add app to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.database import engine, Base, get_db
from app.models import FineTuneJob, ModelInfo, ChatSession, ChatMessage
from app.services.model_service import ModelService
from app.config import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_tables():
    """Tạo các bảng database"""
    try:
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Error creating database tables: {e}")
        return False

def create_sample_data():
    """Tạo dữ liệu mẫu"""
    try:
        logger.info("Creating sample data...")
        
        # Get database session
        db = next(get_db())
        
        # Create base model
        model_service = ModelService()
        
        # Register base Qwen3-8B model
        base_model_request = {
            "name": "Qwen3-8B Base",
            "type": "base",
            "base_model": "Qwen/Qwen3-8B",
            "path": "Qwen/Qwen3-8B",
            "size": "8B",
            "description": "Base Qwen3-8B model from Alibaba",
            "tags": ["base", "qwen", "8b"],
            "is_active": True
        }
        
        try:
            base_model = model_service.create_model(db, base_model_request)
            logger.info(f"✅ Created base model: {base_model.id}")
        except Exception as e:
            logger.warning(f"Base model might already exist: {e}")
        
        # Create sample fine-tuned model
        finetuned_model_request = {
            "name": "Qwen3-8B Fine-tuned Sample",
            "type": "finetuned",
            "base_model": "Qwen/Qwen3-8B",
            "path": "./models/sample_finetuned",
            "size": "8B",
            "description": "Sample fine-tuned model for demonstration",
            "tags": ["finetuned", "qwen", "sample"],
            "is_active": True,
            "accuracy": 0.85,
            "loss": 0.15,
            "training_data_size": 1000,
            "training_epochs": 3
        }
        
        try:
            finetuned_model = model_service.create_model(db, finetuned_model_request)
            logger.info(f"✅ Created sample fine-tuned model: {finetuned_model.id}")
        except Exception as e:
            logger.warning(f"Sample fine-tuned model might already exist: {e}")
        
        # Create sample chat session
        sample_session = ChatSession(
            session_id="sample_session_001",
            model_id="sample_model_id",
            session_name="Sample Chat Session",
            system_prompt="You are a helpful AI assistant.",
            message_count=2,
            created_at=datetime.now(),
            last_activity=datetime.now()
        )
        
        try:
            db.add(sample_session)
            db.commit()
            logger.info("✅ Created sample chat session")
        except Exception as e:
            logger.warning(f"Sample chat session might already exist: {e}")
            db.rollback()
        
        # Create sample messages
        sample_messages = [
            ChatMessage(
                session_id="sample_session_001",
                message_id="msg_001",
                role="user",
                content="Hello, how are you?",
                content_type="text",
                timestamp=datetime.now()
            ),
            ChatMessage(
                session_id="sample_session_001",
                message_id="msg_002",
                role="assistant",
                content="Hello! I'm doing well, thank you for asking. How can I help you today?",
                content_type="text",
                timestamp=datetime.now()
            )
        ]
        
        try:
            for msg in sample_messages:
                db.add(msg)
            db.commit()
            logger.info("✅ Created sample chat messages")
        except Exception as e:
            logger.warning(f"Sample messages might already exist: {e}")
            db.rollback()
        
        db.close()
        logger.info("✅ Sample data created successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating sample data: {e}")
        return False

def create_directories():
    """Tạo các thư mục cần thiết"""
    try:
        logger.info("Creating necessary directories...")
        
        directories = [
            settings.upload_dir,
            settings.base_model_path,
            settings.finetuned_model_path,
            "./data",
            "./static"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"✅ Created directory: {directory}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating directories: {e}")
        return False

def create_sample_files():
    """Tạo các file mẫu"""
    try:
        logger.info("Creating sample files...")
        
        # Sample training data
        sample_training_data = """Q: What is machine learning?
A: Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.

Q: How does fine-tuning work?
A: Fine-tuning is a technique where a pre-trained model is further trained on a specific dataset to adapt it to a particular task or domain.

Q: What is Qwen3-8B?
A: Qwen3-8B is an 8-billion parameter language model developed by Alibaba Cloud, designed for various natural language processing tasks.

Q: What are the benefits of fine-tuning?
A: Fine-tuning allows models to perform better on specific tasks, reduces training time compared to training from scratch, and requires less data than full training.

Q: How do you evaluate a fine-tuned model?
A: Fine-tuned models are typically evaluated using metrics like accuracy, loss, perplexity, and task-specific metrics on a validation dataset."""
        
        # Write sample training file
        sample_file_path = os.path.join("./data", "sample_training_data.txt")
        with open(sample_file_path, "w", encoding="utf-8") as f:
            f.write(sample_training_data)
        
        logger.info(f"✅ Created sample training data: {sample_file_path}")
        
        # Sample CSV data
        sample_csv_data = """question,answer
What is Python?,Python is a high-level programming language known for its simplicity and readability.
What is FastAPI?,FastAPI is a modern web framework for building APIs with Python based on standard Python type hints.
What is SQLAlchemy?,SQLAlchemy is a SQL toolkit and Object-Relational Mapping library for Python.
What is Streamlit?,Streamlit is an open-source app framework for creating web applications for data science and machine learning.
What is Hugging Face?,Hugging Face is a company that develops tools for building machine learning applications and hosts a large collection of pre-trained models."""
        
        sample_csv_path = os.path.join("./data", "sample_qa_data.csv")
        with open(sample_csv_path, "w", encoding="utf-8") as f:
            f.write(sample_csv_data)
        
        logger.info(f"✅ Created sample CSV data: {sample_csv_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating sample files: {e}")
        return False

def main():
    """Main function"""
    logger.info("🚀 Initializing Qwen3-8B Fine-tuning Application...")
    
    # Create directories
    if not create_directories():
        logger.error("Failed to create directories")
        return False
    
    # Create database tables
    if not create_tables():
        logger.error("Failed to create database tables")
        return False
    
    # Create sample data
    if not create_sample_data():
        logger.error("Failed to create sample data")
        return False
    
    # Create sample files
    if not create_sample_files():
        logger.error("Failed to create sample files")
        return False
    
    logger.info("🎉 Application initialization completed successfully!")
    logger.info("📋 Next steps:")
    logger.info("1. Copy env.example to .env and update configuration")
    logger.info("2. Install dependencies: pip install -r requirements.txt")
    logger.info("3. Start API server: python -m uvicorn app.main:app --reload")
    logger.info("4. Start Streamlit interface: streamlit run app/streamlit_app.py")
    
    return True

if __name__ == "__main__":
    from datetime import datetime
    success = main()
    sys.exit(0 if success else 1) 