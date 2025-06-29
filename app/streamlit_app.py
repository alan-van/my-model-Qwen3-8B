import streamlit as st
import requests
import json
import time
import os
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configuration
API_BASE_URL = "http://localhost:8000"
PAGE_CONFIG = {
    "page_title": "Qwen3-8B Fine-tuning App",
    "page_icon": "🤖",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

st.set_page_config(**PAGE_CONFIG)

def check_api_health():
    """Kiểm tra API health"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        return response.status_code == 200
    except:
        return False

def main():
    st.title("🤖 Qwen3-8B Fine-tuning Application")
    st.markdown("---")
    
    # Check API health
    if not check_api_health():
        st.error("❌ Không thể kết nối đến API server. Vui lòng khởi động server trước.")
        st.info("Chạy lệnh: `python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`")
        return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Chọn trang",
        ["🏠 Dashboard", "🎯 Fine-tuning", "💬 Chat", "📊 Models", "📁 File Upload"]
    )
    
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "🎯 Fine-tuning":
        show_finetune_page()
    elif page == "💬 Chat":
        show_chat_page()
    elif page == "📊 Models":
        show_models_page()
    elif page == "📁 File Upload":
        show_upload_page()

def show_dashboard():
    """Hiển thị dashboard"""
    st.header("📊 Dashboard")
    
    # Get API info
    try:
        response = requests.get(f"{API_BASE_URL}/info")
        info = response.json()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Model", info.get("model_name", "Qwen3-8B"))
        
        with col2:
            st.metric("Version", info.get("version", "1.0.0"))
        
        with col3:
            st.metric("Max File Size", f"{info.get('max_file_size', 0) // (1024*1024)}MB")
        
        with col4:
            st.metric("Supported Extensions", len(info.get("allowed_extensions", [])))
        
        # Get statistics
        try:
            response = requests.get(f"{API_BASE_URL}/api/models/statistics")
            stats = response.json()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Model Statistics")
                fig = go.Figure(data=[
                    go.Bar(x=['Total', 'Base', 'Fine-tuned', 'Active'], 
                           y=[stats.get('total_models', 0), stats.get('base_models', 0), 
                              stats.get('finetuned_models', 0), stats.get('active_models', 0)])
                ])
                fig.update_layout(title="Model Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📋 Recent Activity")
                # Get recent finetune jobs
                response = requests.get(f"{API_BASE_URL}/api/finetune/history?size=5")
                if response.status_code == 200:
                    jobs = response.json().get("jobs", [])
                    if jobs:
                        df = pd.DataFrame(jobs)
                        st.dataframe(df[["name", "status", "progress", "created_at"]])
                    else:
                        st.info("No recent fine-tuning jobs")
                else:
                    st.info("No recent activity")
        
        except Exception as e:
            st.error(f"Error loading statistics: {e}")
    
    except Exception as e:
        st.error(f"Error loading dashboard: {e}")

def show_finetune_page():
    """Hiển thị trang fine-tuning"""
    st.header("🎯 Fine-tuning")
    
    tab1, tab2, tab3 = st.tabs(["🚀 Start Fine-tuning", "📊 Job Status", "📋 History"])
    
    with tab1:
        st.subheader("Start New Fine-tuning Job")
        
        # Get configuration
        try:
            response = requests.get(f"{API_BASE_URL}/api/finetune/config")
            config = response.json()
        except:
            config = {}
        
        with st.form("finetune_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                model_name = st.text_input("Model Name", "my-finetuned-model")
                base_model = st.selectbox("Base Model", ["Qwen/Qwen3-8B"])
                learning_rate = st.number_input("Learning Rate", value=config.get("default_learning_rate", 2e-5), format="%.2e")
                batch_size = st.number_input("Batch Size", value=config.get("default_batch_size", 4), min_value=1, max_value=32)
            
            with col2:
                epochs = st.number_input("Epochs", value=config.get("default_epochs", 3), min_value=1, max_value=10)
                max_length = st.number_input("Max Length", value=config.get("default_max_length", 512), min_value=128, max_value=2048)
                warmup_steps = st.number_input("Warmup Steps", value=config.get("default_warmup_steps", 100), min_value=0)
            
            # Parse allowed_extensions
            allowed_ext = config.get("allowed_extensions", ["csv", "txt", "pdf", "docx", "xlsx"])
            if isinstance(allowed_ext, str):
                try:
                    allowed_ext = json.loads(allowed_ext)
                except Exception:
                    allowed_ext = ["csv", "txt", "pdf", "docx", "xlsx"]

            # File upload
            st.subheader("📁 Training Data Files")
            st.info("Supported formats: CSV, TXT, PDF, DOCX, XLSX")
            
            uploaded_files = st.file_uploader(
                "Upload Training Data",
                type=allowed_ext,
                accept_multiple_files=True,
                help="Select one or more files for training"
            )
            
            # Debug info
            if uploaded_files:
                st.success(f"✅ {len(uploaded_files)} file(s) selected:")
                for file in uploaded_files:
                    st.write(f"📄 {file.name} ({file.size} bytes)")
            else:
                st.info("ℹ️ No files selected yet")
            
            submitted = st.form_submit_button("Start Fine-tuning")
            
            if submitted and uploaded_files:
                # Upload files first
                files_data = []
                with st.spinner("Uploading files..."):
                    for file in uploaded_files:
                        try:
                            files = {"file": (file.name, file.getvalue(), file.type)}
                            response = requests.post(f"{API_BASE_URL}/api/upload/file", files=files)
                            if response.status_code == 200:
                                result = response.json()
                                files_data.append(result["file_path"])
                                st.success(f"✅ Uploaded: {file.name}")
                            else:
                                st.error(f"❌ Failed to upload {file.name}: {response.text}")
                        except Exception as e:
                            st.error(f"❌ Error uploading {file.name}: {str(e)}")
                
                if files_data:
                    # Start fine-tuning
                    with st.spinner("Starting fine-tuning job..."):
                        finetune_data = {
                            "name": model_name,
                            "base_model": base_model,
                            "learning_rate": learning_rate,
                            "batch_size": batch_size,
                            "epochs": epochs,
                            "max_length": max_length,
                            "warmup_steps": warmup_steps,
                            "data_files": files_data
                        }
                        
                        try:
                            response = requests.post(f"{API_BASE_URL}/api/finetune/start", json=finetune_data)
                            if response.status_code == 200:
                                result = response.json()
                                st.success(f"✅ Fine-tuning job started! Job ID: {result['job_id']}")
                                st.info(f"Status: {result['status']}")
                            else:
                                st.error(f"❌ Error starting fine-tuning: {response.text}")
                        except Exception as e:
                            st.error(f"❌ Error starting fine-tuning: {str(e)}")
                else:
                    st.error("❌ No files were uploaded successfully")
            elif submitted and not uploaded_files:
                st.error("❌ Please upload training data files")
    
    with tab2:
        st.subheader("Job Status")
        
        job_id = st.text_input("Enter Job ID")
        if job_id:
            if st.button("Check Status"):
                response = requests.get(f"{API_BASE_URL}/api/finetune/status/{job_id}")
                if response.status_code == 200:
                    status = response.json()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Status", status["status"])
                        st.metric("Progress", f"{status['progress']:.1%}")
                        st.metric("Current Epoch", status["current_epoch"])
                    
                    with col2:
                        st.metric("Current Step", status["current_step"])
                        st.metric("Total Steps", status["total_steps"])
                        if status.get("final_loss"):
                            st.metric("Final Loss", f"{status['final_loss']:.4f}")
                    
                    # Progress bar
                    st.progress(status["progress"])
                    
                    if status.get("error_message"):
                        st.error(f"Error: {status['error_message']}")
                    
                    if status["status"] == "completed":
                        st.success("✅ Job completed successfully!")
                        if st.button("Register Model"):
                            response = requests.post(f"{API_BASE_URL}/api/finetune/register/{job_id}")
                            if response.status_code == 200:
                                st.success("✅ Model registered successfully!")
                            else:
                                st.error(f"❌ Error registering model: {response.text}")
                else:
                    st.error(f"❌ Job not found: {response.text}")
    
    with tab3:
        st.subheader("Fine-tuning History")
        
        response = requests.get(f"{API_BASE_URL}/api/finetune/history")
        if response.status_code == 200:
            history = response.json()
            jobs = history.get("jobs", [])
            
            if jobs:
                df = pd.DataFrame(jobs)
                st.dataframe(df)
                
                # Chart
                fig = px.line(df, x="created_at", y="progress", color="name", 
                             title="Fine-tuning Progress Over Time")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No fine-tuning history found")
        else:
            st.error("Error loading history")

def show_chat_page():
    """Hiển thị trang chat"""
    st.header("💬 Chat with Fine-tuned Model")
    
    # Get available models
    try:
        response = requests.get(f"{API_BASE_URL}/api/models?model_type=finetuned&is_active=true")
        models = response.json().get("models", [])
    except:
        models = []
    
    if not models:
        st.warning("⚠️ No fine-tuned models available. Please fine-tune a model first.")
        return
    
    # Model selection
    model_options = {f"{m['name']} ({m['model_id']})": m['model_id'] for m in models}
    selected_model = st.selectbox("Select Model", list(model_options.keys()))
    model_id = model_options[selected_model]
    
    # Chat interface
    st.subheader("Chat Interface")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                chat_data = {
                    "model_id": model_id,
                    "message": prompt,
                    "session_id": st.session_state.session_id
                }
                
                response = requests.post(f"{API_BASE_URL}/api/chat/send", json=chat_data)
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.session_id = result["session_id"]
                    st.session_state.messages.append({"role": "assistant", "content": result["response"]})
                    st.markdown(result["response"])
                    
                    # Show metadata
                    with st.expander("Response Info"):
                        st.write(f"Tokens used: {result.get('tokens_used', 'N/A')}")
                        st.write(f"Response time: {result.get('response_time', 'N/A'):.2f}s")
                else:
                    st.error(f"Error: {response.text}")
    
    # Sidebar controls
    st.sidebar.subheader("Chat Controls")
    
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()
    
    if st.sidebar.button("New Session"):
        st.session_state.session_id = None
        st.info("New session started")

def show_models_page():
    """Hiển thị trang quản lý models"""
    st.header("📊 Model Management")
    
    tab1, tab2 = st.tabs(["📋 Model List", "📈 Performance"])
    
    with tab1:
        st.subheader("Available Models")
        
        response = requests.get(f"{API_BASE_URL}/api/models")
        if response.status_code == 200:
            models_data = response.json()
            models = models_data.get("models", [])
            
            if models:
                df = pd.DataFrame(models)
                
                # Filter options
                col1, col2 = st.columns(2)
                with col1:
                    model_type_filter = st.selectbox("Filter by Type", ["All"] + list(df["type"].unique()))
                with col2:
                    active_filter = st.selectbox("Filter by Status", ["All", True, False])
                
                # Apply filters
                if model_type_filter != "All":
                    df = df[df["type"] == model_type_filter]
                if active_filter != "All":
                    df = df[df["is_active"] == active_filter]
                
                st.dataframe(df)
                
                # Model actions
                st.subheader("Model Actions")
                selected_model_id = st.selectbox("Select Model", df["id"].tolist())
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("View Details"):
                        response = requests.get(f"{API_BASE_URL}/api/models/{selected_model_id}")
                        if response.status_code == 200:
                            model = response.json()
                            st.json(model)
                
                with col2:
                    if st.button("Delete Model"):
                        st.session_state["confirm_delete_model"] = selected_model_id
                    if (
                        st.session_state.get("confirm_delete_model") == selected_model_id
                    ):
                        st.warning("Are you sure you want to delete this model?")
                        if st.button("Yes, delete model", key="confirm_delete_model_btn"):
                            response = requests.delete(f"{API_BASE_URL}/api/models/{selected_model_id}")
                            if response.status_code == 200:
                                st.success("Model deleted successfully!")
                                st.session_state["confirm_delete_model"] = None
                                st.rerun()
                            else:
                                st.error(f"Error deleting model: {response.text}")
                                st.session_state["confirm_delete_model"] = None
                        if st.button("Cancel", key="cancel_delete_model_btn"):
                            st.session_state["confirm_delete_model"] = None
                
                with col3:
                    if st.button("Refresh"):
                        st.rerun()
            else:
                st.info("No models found")
        else:
            st.error("Error loading models")
    
    with tab2:
        st.subheader("Model Performance")
        
        # Get model statistics
        response = requests.get(f"{API_BASE_URL}/api/models/statistics")
        if response.status_code == 200:
            stats = response.json()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Models", stats.get("total_models", 0))
            with col2:
                st.metric("Base Models", stats.get("base_models", 0))
            with col3:
                st.metric("Fine-tuned Models", stats.get("finetuned_models", 0))
            with col4:
                st.metric("Active Models", stats.get("active_models", 0))
            
            # Performance chart
            if stats.get("total_models", 0) > 0:
                models_response = requests.get(f"{API_BASE_URL}/api/models")
                if models_response.status_code == 200:
                    models = models_response.json().get("models", [])
                    if models:
                        df = pd.DataFrame(models)
                        
                        # Filter models with performance data
                        perf_df = df[df["accuracy"].notna() | df["loss"].notna()]
                        
                        if not perf_df.empty:
                            fig = px.scatter(perf_df, x="accuracy", y="loss", 
                                           color="type", size="size",
                                           hover_data=["name"],
                                           title="Model Performance Comparison")
                            st.plotly_chart(fig, use_container_width=True)

def show_upload_page():
    """Hiển thị trang upload files"""
    st.header("📁 File Upload")
    
    tab1, tab2 = st.tabs(["📤 Upload Files", "📋 File List"])
    
    with tab1:
        st.subheader("Upload Files")
        
        # Get upload config
        response = requests.get(f"{API_BASE_URL}/api/upload/config")
        if response.status_code == 200:
            config = response.json()
        else:
            config = {}
        
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=config.get("allowed_extensions", ["csv", "txt", "pdf", "docx", "xlsx"]),
            accept_multiple_files=True,
            help=f"Maximum file size: {config.get('max_file_size', 0) // (1024*1024)}MB"
        )
        
        if uploaded_files:
            if st.button("Upload Files"):
                with st.spinner("Uploading files..."):
                    files_data = []
                    for file in uploaded_files:
                        files = {"file": (file.name, file.getvalue(), file.type)}
                        response = requests.post(f"{API_BASE_URL}/api/upload/file", files=files)
                        if response.status_code == 200:
                            result = response.json()
                            files_data.append(result)
                        else:
                            st.error(f"Error uploading {file.name}: {response.text}")
                    
                    if files_data:
                        st.success(f"✅ Successfully uploaded {len(files_data)} files!")
                        
                        # Show uploaded files
                        df = pd.DataFrame(files_data)
                        st.dataframe(df)
    
    with tab2:
        st.subheader("Uploaded Files")
        
        response = requests.get(f"{API_BASE_URL}/api/upload/list")
        if response.status_code == 200:
            files_data = response.json()
            files = files_data.get("files", [])
            
            if files:
                df = pd.DataFrame(files)
                st.dataframe(df)
                
                # File actions
                selected_file = st.selectbox("Select File", df["filename"].tolist())
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Process File"):
                        file_path = df[df["filename"] == selected_file]["file_path"].iloc[0]
                        response = requests.post(f"{API_BASE_URL}/api/upload/process", json={"file_path": file_path})
                        if response.status_code == 200:
                            result = response.json()
                            st.json(result)
                
                with col2:
                    if st.button("Download File"):
                        file_path = df[df["filename"] == selected_file]["file_path"].iloc[0]
                        response = requests.get(f"{API_BASE_URL}/api/upload/file/{file_path}")
                        if response.status_code == 200:
                            st.download_button(
                                label="Download",
                                data=response.content,
                                file_name=selected_file
                            )
                
                with col3:
                    if st.button("Delete File"):
                        st.session_state["confirm_delete_file"] = selected_file
                    if (
                        st.session_state.get("confirm_delete_file") == selected_file
                    ):
                        st.warning("Are you sure you want to delete this file?")
                        if st.button("Yes, delete file", key="confirm_delete_file_btn"):
                            file_name = selected_file  # Only the filename, not the full path
                            response = requests.delete(f"{API_BASE_URL}/api/upload/file/{file_name}")
                            if response.status_code == 200:
                                st.success("File deleted successfully!")
                                st.session_state["confirm_delete_file"] = None
                                st.rerun()
                            else:
                                st.error(f"Error deleting file: {response.text}")
                                st.session_state["confirm_delete_file"] = None
                        if st.button("Cancel", key="cancel_delete_file_btn"):
                            st.session_state["confirm_delete_file"] = None
            else:
                st.info("No files uploaded yet")
        else:
            st.error("Error loading files")

if __name__ == "__main__":
    main() 