#!/usr/bin/env python3
"""
Script để chạy ứng dụng Qwen3-8B Fine-tuning
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Kiểm tra dependencies"""
    try:
        import fastapi
        import uvicorn
        import streamlit
        import torch
        import transformers
        import sqlalchemy
        logger.info("✅ All dependencies are installed")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.info("Please install dependencies: pip install -r requirements.txt")
        return False

def check_env_file():
    """Kiểm tra file .env"""
    env_file = Path(".env")
    if not env_file.exists():
        logger.warning("⚠️ .env file not found")
        logger.info("Creating .env file from env.example...")
        
        env_example = Path("env.example")
        if env_example.exists():
            import shutil
            shutil.copy("env.example", ".env")
            logger.info("✅ Created .env file from env.example")
            logger.info("Please update .env file with your configuration")
        else:
            logger.error("❌ env.example file not found")
            return False
    
    return True

def run_api_server(host="0.0.0.0", port=8000, reload=True):
    """Chạy API server"""
    logger.info(f"🚀 Starting API server on {host}:{port}")
    
    cmd = [
        sys.executable, "-m", "uvicorn",
        "app.main:app",
        "--host", host,
        "--port", str(port)
    ]
    
    if reload:
        cmd.append("--reload")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("🛑 API server stopped")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Error starting API server: {e}")
        return False
    
    return True

def run_streamlit(port=8501):
    """Chạy Streamlit interface"""
    logger.info(f"🚀 Starting Streamlit interface on port {port}")
    
    cmd = [
        sys.executable, "-m", "streamlit",
        "run", "app/streamlit_app.py",
        "--server.port", str(port),
        "--server.address", "0.0.0.0"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("🛑 Streamlit interface stopped")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Error starting Streamlit interface: {e}")
        return False
    
    return True

def run_both(api_port=8000, streamlit_port=8501):
    """Chạy cả API server và Streamlit interface"""
    import threading
    import time
    
    logger.info("🚀 Starting both API server and Streamlit interface")
    
    # Start API server in a separate thread
    api_thread = threading.Thread(
        target=run_api_server,
        args=("0.0.0.0", api_port, True),
        daemon=True
    )
    api_thread.start()
    
    # Wait a bit for API server to start
    time.sleep(3)
    
    # Start Streamlit interface
    run_streamlit(streamlit_port)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Qwen3-8B Fine-tuning Application")
    parser.add_argument(
        "--mode",
        choices=["api", "streamlit", "both"],
        default="both",
        help="Mode to run the application"
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="Port for API server"
    )
    parser.add_argument(
        "--streamlit-port",
        type=int,
        default=8501,
        help="Port for Streamlit interface"
    )
    parser.add_argument(
        "--api-host",
        default="0.0.0.0",
        help="Host for API server"
    )
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Disable auto-reload for API server"
    )
    
    args = parser.parse_args()
    
    logger.info("🤖 Qwen3-8B Fine-tuning Application")
    logger.info("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check environment file
    if not check_env_file():
        sys.exit(1)
    
    # Run based on mode
    if args.mode == "api":
        run_api_server(args.api_host, args.api_port, not args.no_reload)
    elif args.mode == "streamlit":
        run_streamlit(args.streamlit_port)
    elif args.mode == "both":
        run_both(args.api_port, args.streamlit_port)

if __name__ == "__main__":
    main() 