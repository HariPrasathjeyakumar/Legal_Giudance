import os
import subprocess
import sys
import time
import socket
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Legal Guidance Platform", layout="wide")

# Helper function to check if the Flask port is finally active
def is_port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

# 1. Start your original Flask app in the background
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

# Fire up the background server
backend_process = start_flask_backend()

# 2. Wait dynamically for the Model Download & FAISS Bootup
if not is_port_open(8080):
    st.info("⏳ Initializing Machine Learning Layers...")
    st.caption("Please wait up to 2-3 minutes while the server downloads the 'all-MiniLM-L6-v2' transformer model weights and builds the FAISS index database for the first time.")
    
    # Create an animated progress spinner while checking every 5 seconds
    with st.spinner("Downloading weights and mapping vector indices..."):
        while not is_port_open(8080):
            time.sleep(5)
    st.success("✅ Systems Ready! Loading user dashboard interface...")
    time.sleep(1)
    st.rerun()

# 3. Inject CSS to hide default Streamlit headers/padding to match your original layout
st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .block-container {padding: 0px !important; margin: 0px !important;}
        .stAppViewMain {padding: 0px !important;}
    </style>
    """,
    unsafe_allow_html=True
)

# 4. Safely render your Flask app inside the safe Streamlit Component iframe
components.iframe("http://127.0.0.1:8080", height=900, scrolling=True)
