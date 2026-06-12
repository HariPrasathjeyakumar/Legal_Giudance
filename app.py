import os
import json
import shutil
import secrets
from datetime import datetime
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from flask import Flask, request, jsonify, render_template_string, send_file
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Global in-memory session storage
sessions = {}

# ── 1) AUTOMATED PATH BALANCING ─────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Check for both standard naming patterns to avoid strict crashing
KB_PATH = os.path.join(PROJECT_DIR, "legal_kb.json")
if not os.path.exists(KB_PATH):
    KB_PATH = os.path.join(PROJECT_DIR, "legal_jason.json")

if not os.path.exists(KB_PATH):
    raise FileNotFoundError("Critical Error: 'legal_kb.json' or 'legal_jason.json' must be present.")

with open(KB_PATH, "r", encoding="utf-8") as f:
    kb = json.load(f)
print(f"✅ Knowledge Base Loaded Successfully — {len(kb)} scenarios found.")

# ── 2) FAISS INDEX GENERATION ───────────────────────────────────────────
print("⏳ Initializing Sentence-Transformers embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
scenarios = [item["scenario"] for item in kb]
embeddings = embed_model.encode(scenarios, convert_to_numpy=True).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
print(f"✅ FAISS Index successfully built with {index.ntotal} vector entries.")

# ── 3) SECURE ENVIRONMENT RESOLUTION ───────────────────────────────────
# Strip any potential accidental spaces from the environment dashboard input
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY:
    GROQ_API_KEY = GROQ_API_KEY.strip()

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ── 4) CONSTANTS & HELPERS ──────────────────────────────────────────────
LEGAL_TERMS = ["liability", "negligence", "contract", "breach", "indemnity", "jurisdiction", "tort"]
SENSITIVE_KEYWORDS = ["suicide", "self-harm", "violence", "threat", "abuse", "extortion"]

def search_kb(query, k=2):
    query_vector = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_vector, k)
    results = []
    for idx in indices[0]:
        if idx < len(kb):
            results.append(kb[idx])
    return results

def check_sensitivity(text):
    return any(word in text.lower() for word in SENSITIVE_KEYWORDS)

# ── 5) INTERFACE HTML TEMPLATE ──────────────────────────────────────────
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Legal Guidance Platform</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css" rel="stylesheet">
    <style>
        .chat-container { height: 450px; }
    </style>
</head>
<body class="bg-gray-50 font-sans">
    <div class="max-w-4xl mx-auto my-10 p-6 bg-white rounded-xl shadow-lg">
        <header class="border-b pb-4 mb-6">
            <h1 class="text-3xl font-bold text-blue-900">Legal Guidance & Complaint Assistant</h1>
            <p class="text-gray-600 text-sm mt-1">Automated AI Analysis, Knowledge Base RAG Retrieval, and PDF Generation</p>
        </header>

        {% if not has_api_key %}
        <div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4">
            <strong class="font-bold">Missing/Invalid Configuration:</strong> Please verify GROQ_API_KEY in the environment secrets settings.
        </div>
        {% endif %}

        <div class="mb-6">
            <label class="block text-gray-700 font-semibold mb-2">Describe your issue or legal scenario:</label>
            <div id="chatBox" class="chat-container border rounded-lg p-4 overflow-y-auto bg-gray-50 space-y-3 mb-4">
                <div class="text-gray-500 italic text-sm">System: Provide details about your situation to begin analysis...</div>
            </div>
            
            <div class="flex space-x-2">
                <input type="text" id="userInput" class="flex-1 border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500" placeholder="Type your situation here...">
                <button onclick="sendMessage()" class="bg-blue-800 hover:bg-blue-900 text-white font-bold px-6 py-2 rounded-lg transition">Analyze</button>
            </div>
        </div>

        <div id="actionPanel" class="hidden border-t pt-4 flex justify-between items-center">
            <button onclick="downloadPDF()" class="bg-green-700 hover:bg-green-800 text-white text-sm font-bold px-4 py-2 rounded transition">
                📥 Download Legal Analysis Document (PDF)
            </button>
            <button onclick="saveSession()" class="bg-gray-600 hover:bg-gray-700 text-white text-sm font-bold px-4 py-2 rounded transition">
                💾 Save Session State
            </button>
        </div>
    </div>

    <script>
        let currentSessionId = "";

        async function sendMessage() {
            const input = document.getElementById('userInput');
            const chatBox = document.getElementById('chatBox');
            const text = input.value.trim();
            if (!text) return;

            chatBox.innerHTML += `<div class="text-right"><span class="bg-blue-100 text-blue-900 px-3 py-1.5 rounded-lg inline-block max-w-xl">${text}</span></div>`;
            input.value = "";
            chatBox.scrollTop = chatBox.scrollHeight;

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ message: text, session_id: currentSessionId })
                });
                const data = await response.json();
                
                if (data.error) {
                    chatBox.innerHTML += `<div class="text-left"><div class="bg-red-100 text-red-700 px-4 py-2 rounded-lg inline-block">Error: ${data.error}</div></div>`;
                    chatBox.scrollTop = chatBox.scrollHeight;
                    return;
                }

                if (data.session_id) currentSessionId = data.session_id;
                chatBox.innerHTML += `<div class="text-left"><div class="bg-gray-200 text-gray-800 px-4 py-2 rounded-lg inline-block max-w-xl whitespace-pre-line"><strong>Analysis:</strong><br>${data.reply}</div></div>`;
                
                if(data.is_sensitive) {
                    chatBox.innerHTML += `<div class="text-left"><div class="bg-red-50 border border-red-200 text-red-700 px-4 py-2 rounded-lg inline-block text-xs">⚠️ <strong>Notice:</strong> High priority terms detected.</div></div>`;
                }

                document.getElementById('actionPanel').classList.remove('hidden');
            } catch (err) {
                chatBox.innerHTML += `<div class="text-red-500 text-sm">Error connecting to server.</div>`;
            }
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        function downloadPDF() {
            if(!currentSessionId) return alert("Please run an analysis first.");
            window.location.href = `/download_pdf?session_id=${currentSessionId}`;
        }

        async function saveSession() {
            if(!currentSessionId) return alert("No active session.");
            const response = await fetch('/save_session', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ session_id: currentSessionId })
            });
            const data = await response.json();
            alert(data.message || data.error);
        }
    </script>
</body>
</html>
"""

# ── 6) ROUTING CONTROLLERS ──────────────────────────────────────────────
@app.route("/")
def index_route():
    return render_template_string(HTML_TEMPLATE, has_api_key=(groq_client is not None))

@app.route('/chat', methods=['POST'])
def chat_route():
    try:
        data = request.json or {}
        user_message = data.get("message", "").strip()
        session_id = data.get("session_id", "")

        if not user_message:
            return jsonify({"error": "Message is empty"}), 400

        if not session_id or session_id not in sessions:
            session_id = secrets.token_hex(8)
            sessions[session_id] = {
                "history": [],
                "created_at": datetime.now().isoformat(),
                "is_sensitive": False
            }
        
        session_data = sessions[session_id]
        session_data["history"].append({"role": "user", "content": user_message})

        if check_sensitivity(user_message):
            session_data["is_sensitive"] = True

        relevant_docs = search_kb(user_message, k=2)
        context_str = "\n".join([f"- Scenario: {d['scenario']}\n  Guidance: {d.get('guidance', d.get('remedy', ''))}" for d in relevant_docs])

        system_prompt = (
            "You are an expert legal information assistant. Provide educational resources and structural legal clarity based on context facts.\n"
            "Do not officially state or establish an unverified lawyer-client contract.\n\n"
            f"Relevant Context Reference:\n{context_str}"
        )

        if groq_client:
            try:
                chat_completion = groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    model="llama3-8b-8192",
                    temperature=0.3
                )
                ai_reply = chat_completion.choices[0].message.content
            except Exception as e:
                ai_reply = f"Inference Warning (API processing issue): {str(e)}\n\nFallback context database match:\n{context_str}"
        else:
            ai_reply = f"Local Fallback Match (Groq Key missing/unconfigured):\n\n{context_str}"

        session_data["history"].append({"role": "assistant", "content": ai_reply})
        return jsonify({
            "reply": ai_reply,
            "session_id": session_id,
            "is_sensitive": session_data["is_sensitive"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/download_pdf", methods=["GET"])
def download_pdf():
    try:
        session_id = request.args.get("session_id", "")
        if not session_id or session_id not in sessions:
            return "Invalid session token context parameter.", 404
        
        session_data = sessions[session_id]
        pdf_filename = f"legal_analysis_{session_id}.pdf"
        pdf_path = os.path.join(PROJECT_DIR, pdf_filename)
        
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontSize=20, leading=24, spaceAfter=20)
        body_style = ParagraphStyle('DocBody', parent=styles['Normal'], fontSize=11, leading=16, spaceAfter=12)
        
        story = [
            Paragraph("Preliminary Legal Analysis Documentation Report", title_style),
            Paragraph(f"Generated Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style),
            Paragraph(f"Reference Identifier Code: REF-{session_id.upper()}", body_style),
            Spacer(1, 15)
        ]
        
        for dialogue in session_data["history"]:
            speaker = "User Statement" if dialogue["role"] == "user" else "System Analysis Assessment Guidance"
            story.append(Paragraph(f"<b>{speaker}:</b>", body_style))
            story.append(Paragraph(dialogue["content"], body_style))
            story.append(Spacer(1, 8))
            
        doc.build(story)
        return send_file(pdf_path, as_attachment=True)
    except Exception as e:
        return str(e), 500

@app.route("/save_session", methods=["POST"])
def save_session():
    try:
        data = request.json or {}
        session_id = data.get("session_id", "")
        if session_id not in sessions:
            return jsonify({"error": "Active session profile identifier not found."}), 400
            
        path = os.path.join(PROJECT_DIR, f"session_{session_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sessions[session_id], f, ensure_ascii=False, indent=2)
        return jsonify({"success": True, "message": f"Saved. Session ID: {session_id}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── 7) PRODUCTION LAUNCH ENVIRONMENT ────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("FLASK_RUN_PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
