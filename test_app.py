#!/usr/bin/env python3
"""
Test script để kiểm tra ứng dụng Qwen3-8B Fine-tuning
"""

import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test import các module chính"""
    try:
        logger.info("Testing imports...")
        
        # Test core dependencies
        import fastapi
        import uvicorn
        import streamlit
        import torch
        import transformers
        import sqlalchemy
        import pydantic
        import pydantic_settings
        
        logger.info("✅ Core dependencies imported successfully")
        
        # Test app modules
        from app.config import settings
        from app.database import engine, Base
        from app.models import FineTuneJob, ModelInfo, ChatSession, ChatMessage
        
        logger.info("✅ App modules imported successfully")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False

def test_database():
    """Test database connection"""
    try:
        logger.info("Testing database...")
        
        from app.database import engine, Base
        
        # Test database connection
        with engine.connect() as conn:
            logger.info("✅ Database connection successful")
        
        # Test table creation
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database error: {e}")
        return False

def test_config():
    """Test configuration"""
    try:
        logger.info("Testing configuration...")
        
        from app.config import settings
        
        logger.info(f"Model name: {settings.model_name}")
        logger.info(f"Database URL: {settings.database_url}")
        logger.info(f"Upload directory: {settings.upload_dir}")
        
        logger.info("✅ Configuration loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration error: {e}")
        return False

def test_api_routes():
    """Test API routes"""
    try:
        logger.info("Testing API routes...")
        
        from app.main import app
        
        # Check if routes are registered
        routes = [route.path for route in app.routes]
        logger.info(f"Found {len(routes)} routes")
        
        # Check for key routes
        key_routes = [
            "/api/finetune",
            "/api/chat", 
            "/api/models",
            "/api/upload"
        ]
        
        for route in key_routes:
            if any(route in r for r in routes):
                logger.info(f"✅ Found route: {route}")
            else:
                logger.warning(f"⚠️ Missing route: {route}")
        
        logger.info("✅ API routes test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ API routes error: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🧪 Starting Qwen3-8B Fine-tuning Application Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Database", test_database),
        ("API Routes", test_api_routes)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} test failed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 Test Results Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("🎉 All tests passed! Application is ready to run.")
        return True
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 