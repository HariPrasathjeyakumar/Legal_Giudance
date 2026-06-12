---
title: Legal Guidance Platform
emoji: ⚖️
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# Legal Guidance Platform

A Flask-based legal intake and complaint-drafting assistant for India, using
semantic search (FAISS + sentence-transformers) over a knowledge base of
legal scenarios, and Groq's LLM API for intake questions, summaries, and
complaint drafting.

## Deploying on Hugging Face Spaces (Docker)

1. **Create a new Space**
   - Go to https://huggingface.co/new-space
   - Choose **Docker** as the Space SDK (this README's YAML header already
     sets `sdk: docker` and `app_port: 7860`, which is what the Dockerfile
     listens on).

2. **Upload these files** to the Space repo (via the web UI, `git push`, or
   the Spaces "Files" tab):
   - `app.py`
   - `Dockerfile`
   - `requirements.txt`
   - `legal_kb.json`
   - `.dockerignore`
   - `README.md` (this file — the YAML header is required by Spaces)

3. **Add your Groq API key as a secret**
   - In the Space, go to **Settings → Variables and secrets**
   - Add a new **secret**: `GROQ_API_KEY` = `your_actual_groq_api_key`
   - (Optional) Add `FLASK_SECRET_KEY` for a stable Flask session secret.

4. **Build & run**
   - The Space will automatically build the Docker image and start the
     container. The first build can take a few minutes because it installs
     `torch` + `sentence-transformers`.
   - On first request, the app downloads the `all-MiniLM-L6-v2` embedding
     model (~80MB) and builds a FAISS index over `legal_kb.json`.

5. **Access the app**
   - Once the build finishes, the Space will show the running app at its
     public URL (`https://<your-username>-<space-name>.hf.space`).

## Customizing the legal knowledge base

`legal_kb.json` is an array of scenario objects. Each entry needs:

```json
{
  "scenario": "Free-text description used for semantic matching",
  "title": "Short case title",
  "category": "Civil | Criminal",
  "documents": ["Document 1", "Document 2"],
  "steps": ["Step 1: ...", "Step 2: ..."],
  "provisions": ["Relevant law/section", "..."],
  "authority": "Authority A → Authority B → Authority C"
}
```

The `authority` field uses `→` to define an escalation chain — this is used
both to display the suggested authority and to auto-generate the
"Escalation Path" section.

Add as many scenarios as you like; the FAISS index is rebuilt automatically
each time the container starts, based on the contents of `legal_kb.json`.

## Environment variables

| Variable           | Required | Purpose                                         |
|--------------------|----------|--------------------------------------------------|
| `GROQ_API_KEY`     | Yes      | Used for all LLM calls (questions, summaries, complaint drafting, translation, dynamic portals). |
| `FLASK_SECRET_KEY` | No       | Sets a stable Flask `secret_key`. If unset, a random one is generated on each restart (sessions in `sessions` dict are in-memory anyway and reset on restart). |

## Notes & limitations

- **In-memory sessions**: the `sessions` dict lives in process memory. If the
  Space restarts (e.g., after a period of inactivity / "sleeping"), all
  in-progress sessions are lost. For persistence across restarts, you'd need
  to swap this for a database (e.g., a small SQLite file or an external
  store).
- **Free Spaces sleep when idle** and take ~10-30 seconds to wake up on the
  next request (the embedding model needs to reload).
- **Saved sessions** via `/save_session` are written to local files inside
  the container (`session_<id>.json`). These are **not persisted** across
  rebuilds/restarts on Spaces — treat this as a debugging feature, not real
  storage.
- The "Official Portals" section is generated dynamically by the LLM per
  case, with a hard-coded fallback (`_get_portals_fallback`) if the LLM call
  fails or returns nothing usable.

## Running locally (for testing before deploying)

```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here
python app.py
# App runs at http://localhost:5000
```

Or with Docker (same image used on Spaces):

```bash
docker build -t legal-guidance-app .
docker run -p 7860:7860 -e GROQ_API_KEY=your_key_here legal-guidance-app
# App runs at http://localhost:7860
```
