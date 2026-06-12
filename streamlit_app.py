import os
import subprocess
import sys
import time
import streamlit as st

st.set_page_config(page_title="Legal Guidance Platform", layout="wide")

# 1. Start your exact Flask app in the background
@st.cache_resource
def start_flask_backend():
    # Looks for your original app.py file
    flask_file = os.path.join(os.path.dirname(__file__), "app.py")
    
    # Run the Flask app on port 8501 (or another free port) via a background process
    process = subprocess.Popen(
        [sys.executable, flask_file],
        env=dict(os.environ, FLASK_RUN_PORT="8080"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    # Give the heavy FAISS database and model 5 seconds to boot up
    time.sleep(5)
    return process

# Trigger the background startup sequence
backend_process = start_flask_backend()

# 2. Display your EXACT Flask UI inside Streamlit using an Iframe
st.markdown(
    """
    <style>
        /* Remove all default Streamlit padding, headers, and footers */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .block-container {padding: 0px !important;}
        iframe {border: none; width: 100%; height: 100vh;}
    </style>
    <iframe src="http://127.0.0.1:8080"></iframe>
    """,
    unsafe_allow_html=True
)
