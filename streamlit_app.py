import os
import subprocess
import sys
import time
import socket
import streamlit as st

st.set_page_config(page_title="Legal Guidance Platform", layout="centered")

# Helper function to check if the background Flask app is awake
def is_port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

# 1. Boot up your original Flask application in the background
@st.cache_resource
def start_flask_backend():
    flask_file = os.path.join(os.path.dirname(__file__), "app.py")
    process = subprocess.Popen(
        [sys.executable, flask_file],
        env=dict(os.environ, FLASK_RUN_PORT="8080"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return process

backend_process = start_flask_backend()

# 2. Check if the AI models and FAISS are still building
if not is_port_open(8080):
    st.title("⚖️ Legal Guidance AI Platform")
    st.info("⏳ Initializing Machine Learning Components...")
    st.caption("Please allow up to 2 minutes while the server safely downloads the 'all-MiniLM-L6-v2' model layers and maps the local text vectors.")
    
    with st.spinner("Compiling database frameworks..."):
        while not is_port_open(8080):
            time.sleep(3)
    st.success("✅ Application engine loaded successfully!")
    time.sleep(1)
    st.rerun()

# 3. Present a direct gateway launch screen (Bypasses all browser security blocks)
st.title("⚖️ Legal Guidance AI Platform")
st.success("🎉 Your core engine has successfully initialized on the free server!")

st.write("To protect your privacy and ensure secure sessions, click the launch button below to open the application in a full browser interface.")

# This generates a dynamic link pointing directly to your background server instance
st.link_button(
    "🚀 Launch Full Legal App UI", 
    "http://127.0.0.1:8080", 
    type="primary", 
    use_container_width=True
)

st.divider()
st.caption("ℹ️ Note: Clicking this button opens your original custom-designed multi-page HTML forms, language controls, and document generators directly from the background process.")
