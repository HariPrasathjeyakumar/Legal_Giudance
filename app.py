# =========================================================
# LEGAL GUIDANCE PLATFORM — v5 (Render-ready Flask app)
# =========================================================

# ── 1) IMPORTS & SETUP ───────────────────────────────────
import os
import json
import secrets
from datetime import datetime

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = BASE_DIR

KB_PATH = os.path.join(BASE_DIR, "legal_kb.json")
with open(KB_PATH, "r", encoding="utf-8") as f:
    kb = json.load(f)
print(f"✅ KB loaded — {len(kb)} scenarios.")

# ── 4) FAISS INDEX ──────────────────────────────────────
import faiss, numpy as np
from sentence_transformers import SentenceTransformer

print("⏳ Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
scenarios   = [item["scenario"] for item in kb]
embeddings  = embed_model.encode(scenarios, convert_to_numpy=True).astype("float32")
index       = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
print(f"✅ FAISS index built — {index.ntotal} scenarios.")

# ── 5) GROQ ─────────────────────────────────────────────
from groq import Groq

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    print("⚠️  WARNING: GROQ_API_KEY environment variable is not set. "
          "Set it in your Render service's Environment settings.")

groq_client = Groq(api_key=GROQ_API_KEY)
print("✅ Groq configured.")

# ── 6) CONSTANTS ─────────────────────────────────────────
LEGAL_TERMS = {
    "FIR":             "First Information Report — the first formal police complaint document.",
    "Legal Notice":    "A formal written warning sent before taking stronger legal action.",
    "Jurisdiction":    "The court or authority legally allowed to hear your case.",
    "Affidavit":       "A written statement confirmed as true, signed before an authorized person.",
    "Annexure":        "A supporting document attached to a complaint or petition.",
    "Acknowledgment":  "Official proof that your complaint was received.",
    "Cognizable":      "An offence where police can register a case without court permission.",
    "Escalation":      "Taking your complaint to a higher authority if the first one doesn't act."
}

MODEL = "llama-3.3-70b-versatile"

# ── SENSITIVE CASE DETECTION ─────────────────────────────
SENSITIVE_KEYWORDS = [
    "sexual assault","rape","molestation","pocso","minor","child abuse",
    "child sexual","trafficking","domestic violence","acid attack",
    "sexual harassment","stalking","revenge porn","intimate image",
    "ncrb","ndps minor","juvenile","underage","sexual offence"
]

def is_sensitive_case(title, category, problem=""):
    combined = (title + " " + category + " " + problem).lower()
    return any(kw in combined for kw in SENSITIVE_KEYWORDS)

# ── 7) CORE HELPERS ──────────────────────────────────────

def search(query):
    qv = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    dists, idxs = index.search(qv, 1)
    dist    = float(dists[0][0])
    matched = kb[int(idxs[0][0])]
    print(f"\n🔍 Match: '{matched['title']}' | dist: {dist:.4f}")
    return None if dist > 1.25 else matched


def groq_chat(prompt, max_tokens=1500, temperature=0.2):
    r = groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"user","content":prompt}],
        max_tokens=max_tokens,
        temperature=temperature
    )
    return r.choices[0].message.content.strip()


def parse_json(text):
    text = text.replace("```json","").replace("```","").strip()
    s = text.find("{"); e = text.rfind("}") + 1
    if s != -1 and e > s:
        text = text[s:e]
    return json.loads(text)


def translate_text(text, language):
    if language == "English" or not str(text).strip():
        return text
    try:
        return groq_chat(
            f"Translate exactly to {language}. Keep legal terms and section numbers unchanged. Return only translation:\n{text}",
            max_tokens=800, temperature=0.1
        )
    except:
        return text


def translate_list(items, language):
    return [translate_text(str(x), language) for x in items] if language != "English" else items


def format_questions(raw):
    """Convert raw question list (strings or dicts) into structured format."""
    out = []
    for q in raw:
        if isinstance(q, dict):
            question   = q.get("question","").strip()
            input_type = q.get("input_type","text")
            options    = q.get("options",[])
            if not question:
                continue
            if input_type == "select" and isinstance(options, list) and options:
                out.append({"question":question,"input_type":"select","options":options})
            else:
                out.append({"question":question,"input_type":"text"})
            continue
        if not isinstance(q, str):
            continue
        q = q.strip()
        ql = q.lower()
        if ql.startswith(("did ","do ","have ","has ","is ","are ","was ","were ","can ","will ")):
            out.append({"question":q,"input_type":"select","options":["Yes","No","Not Sure"]})
        elif any(x in ql for x in ["which upi","which app","which payment"]):
            out.append({"question":q,"input_type":"select","options":["PhonePe","Google Pay","Paytm","Bank App","Other"]})
        elif "other party involved" in ql:
            out.append({"question":q,"input_type":"select","options":["Person","Company","Govt. Authority","Employer","Not Sure"]})
        else:
            out.append({"question":q,"input_type":"text"})
    return out[:8]


# ── DYNAMIC QUESTION GENERATION ──────────────────────────
def get_initial_questions(user_input, matched_case, language="English"):
    sensitive = is_sensitive_case(matched_case.get("title",""), matched_case.get("category",""), user_input)
    try:
        steps_text = "\n".join(matched_case["steps"])
        docs_text  = "\n".join(f"- {d}" for d in matched_case["documents"])
        laws_text  = "\n".join(f"- {l}" for l in matched_case["provisions"])
        auth_text  = matched_case["authority"]

        already_known = []
        ul = user_input.lower()
        if any(w in ul for w in ["₹","rs.","rupee","amount","thousand","lakh"]):
            already_known.append("approximate amount/financial value")
        if any(w in ul for w in ["yesterday","today","last week","last month","ago","date"]):
            already_known.append("approximate time of incident")
        if any(w in ul for w in ["police","fir","complaint","filed"]):
            already_known.append("whether complaint was already filed")
        if any(w in ul for w in ["screenshot","proof","evidence","receipt","video","photo"]):
            already_known.append("some evidence exists")

        already_str = ", ".join(already_known) if already_known else "nothing specific yet"
        lang_note = f"\nIMPORTANT: Write ALL questions in {language} language only." if language != "English" else ""

        # Sensitive case: never ask for name, exact address, identity-revealing info
        sensitive_note = ""
        if sensitive:
            sensitive_note = """
CRITICAL — SENSITIVE CASE RULES:
- NEVER ask for the complainant's full name, home address, exact location, or any identity-revealing information.
- Do NOT ask "What is your name?" or "Where do you live?" or "What is your exact address?"
- You MAY ask for the district or state (not street/house address) only if needed for jurisdiction.
- Focus only on facts of the incident, evidence, timeline, and support sought.
- Be gentle and trauma-informed in phrasing. Avoid clinical or blunt language.
"""

        prompt = f"""You are a senior Indian lawyer doing a client intake for this EXACT case.
{lang_note}
{sensitive_note}

USER'S PROBLEM (read carefully):
"{user_input}"

MATCHED LEGAL CASE:
Category : {matched_case['category']}
Title    : {matched_case['title']}

REQUIRED DOCUMENTS FOR THIS CASE:
{docs_text}

LEGAL STEPS THE CLIENT MUST TAKE:
{steps_text}

APPLICABLE LAWS:
{laws_text}

AUTHORITY TO APPROACH:
{auth_text}

ALREADY KNOWN FROM USER'S DESCRIPTION (DO NOT ASK AGAIN):
{already_str}

YOUR TASK:
Generate 6-8 intake questions SPECIFIC to this exact case type and situation.

RULES:
1. Every question must relate to a specific document, step, or law listed above
2. Ask about jurisdiction (state/district) unless sensitive case restricts it
3. Ask about urgency/ongoing risk
4. Ask about specific documents listed above
5. NEVER ask something generic
6. NEVER repeat what the user already told you
{"7. DO NOT ask for name, home address, or identity details (sensitive case)" if sensitive else ""}

OUTPUT as JSON only:
{{
  "complexity": "simple | medium | complex",
  "reason": "one sentence why this complexity level",
  "questions": [
    {{"question": "...", "input_type": "text"}},
    {{"question": "...", "input_type": "select", "options": ["Yes","No","Not Sure"]}}
  ]
}}"""

        raw  = groq_chat(prompt, max_tokens=1500, temperature=0.2)
        data = parse_json(raw)

        if "questions" not in data or not isinstance(data["questions"], list):
            raise ValueError("Bad response")

        # Filter out identity questions for sensitive cases
        if sensitive:
            identity_phrases = ["your name","full name","home address","house address","street address",
                                "where do you live","residential address","address of","your address"]
            filtered = []
            for q in data["questions"]:
                q_lower = q.get("question","").lower() if isinstance(q, dict) else str(q).lower()
                if not any(phrase in q_lower for phrase in identity_phrases):
                    filtered.append(q)
            data["questions"] = filtered

        data["questions"] = format_questions(data["questions"])

        if language != "English":
            translated = []
            for q in data["questions"]:
                q2 = dict(q)
                q2["question"] = translate_text(q2["question"], language)
                if q2.get("options"):
                    q2["options"] = translate_list(q2["options"], language)
                translated.append(q2)
            data["questions"]  = translated
            data["reason"]     = translate_text(data.get("reason",""), language)

        data["sensitive"] = sensitive
        if sensitive:
            data["complexity"] = "complex"
            data["reason"] = "Sensitive case requiring priority handling and careful legal support."

        print(
            f"✅ Generated {len(data['questions'])} questions "
            f"[{data.get('complexity','?')}] | Sensitive: {sensitive}"
        )

        return data

    except Exception as e:
        print(f"❌ Question gen error: {e}")
        fallback_questions = _build_kb_fallback(matched_case, user_input, language, sensitive)
        return {
            "complexity": "medium",
            "reason": "Questions based on case requirements.",
            "questions": fallback_questions,
            "sensitive": sensitive
        }


def _build_kb_fallback(matched_case, user_input, language="English", sensitive=False):
    questions = []
    docs   = matched_case.get("documents", [])
    steps  = matched_case.get("steps", [])

    if not sensitive:
        questions.append({"question": "In which state and district did this incident occur?", "input_type": "text"})
    else:
        questions.append({"question": "In which state (general region) did this incident occur?", "input_type": "text"})

    for doc in docs[:2]:
        questions.append({
            "question": f"Do you currently have the '{doc}'?",
            "input_type": "select",
            "options": ["Yes, I have it", "No, I need to get it", "Not Sure"]
        })

    if steps:
        first_step = steps[0].replace("Step 1:","").strip()
        questions.append({"question": f"Regarding: {first_step} — have you done this yet?", "input_type": "select", "options": ["Yes","No","Partially"]})

    questions += [
        {"question": "When did this incident occur? Please give the date or approximate time.", "input_type": "text"},
        {"question": "Have you already filed any formal complaint or taken any legal action?", "input_type": "select", "options": ["Yes","No","Not Sure"]},
        {"question": "Is this problem still ongoing or causing you harm right now?", "input_type": "select", "options": ["Yes, still ongoing","No, it has stopped","Not Sure"]},
        {"question": "What is the financial loss or damage amount involved (if any)?", "input_type": "text"}
    ]

    if not sensitive:
        questions.append({"question": "Who is the other party involved (name, designation, or company name if known)?", "input_type": "text"})

    final = format_questions(questions[:8])

    if language != "English":
        translated = []
        for q in final:
            q2 = dict(q)
            q2["question"] = translate_text(q2["question"], language)
            if q2.get("options"):
                q2["options"] = translate_list(q2["options"], language)
            translated.append(q2)
        return translated
    return final


def evaluate_answers_and_followup(user_input, matched_case, conversation_history, language="English"):
    try:
        history_text = "\n\n".join([f"Q: {h['question']}\nA: {h['answer']}" for h in conversation_history])
        docs_text    = "\n".join(f"- {d}" for d in matched_case["documents"])

        prompt = f"""You are a legal intake reviewer.

Case: {matched_case['title']}
Problem: "{user_input}"

Required documents:
{docs_text}

Interview so far:
{history_text}

Is there enough information to draft a complaint and suggest authority?
If yes → ready: true
If a CRITICAL fact is missing → ask max 2 very specific follow-up questions.
Prefer ready:true unless truly critical info is missing.

Return ONLY JSON:
{{"ready": true}}
OR
{{"ready": false, "followups": [
  {{"question":"...", "input_type":"text"}},
  {{"question":"...", "input_type":"select","options":["Yes","No","Not Sure"]}}
]}}"""

        result = parse_json(groq_chat(prompt, max_tokens=400, temperature=0.2))

        if not result.get("ready") and "followups" in result:
            result["followups"] = format_questions(result["followups"][:2])
            if language != "English":
                translated = []
                for q in result["followups"]:
                    q2 = dict(q)
                    q2["question"] = translate_text(q2["question"], language)
                    if q2.get("options"):
                        q2["options"] = translate_list(q2["options"], language)
                    translated.append(q2)
                result["followups"] = translated

        return result
    except Exception as e:
        print(f"Follow-up error: {e}")
        return {"ready": True}


def generate_final_analysis(user_input, matched_case, history, language="English"):
    try:
        history_text = "\n".join([f"Q: {h['question']}\nA: {h['answer']}" for h in history])
        prompt = f"""You are a practical Indian legal assistant.

User issue: "{user_input}"
Case: {matched_case['category']} — {matched_case['title']}

Interview answers:
{history_text}

Write a 5-6 sentence plain language summary for a common person.
- Explain what kind of legal issue this is
- Mention their strongest fact or evidence
- Tell them the most important first action TODAY
- Reassure them about their legal rights
- Warn about any time-sensitive steps
- No jargon. No promises of success. Speak directly to them."""

        text = groq_chat(prompt, max_tokens=500, temperature=0.5)
        return translate_text(text, language)
    except Exception:
        fallback = (
            f"Your issue is a {matched_case['category']} matter — {matched_case['title']}. "
            f"You should organize all facts, documents, and evidence immediately. "
            f"Do not delay the first complaint step if the issue is ongoing. "
            f"Keep copies of everything you submit."
        )
        return translate_text(fallback, language)


def get_location_from_history(history):
    for h in history:
        q = h["question"].lower()
        if any(w in q for w in ["state","district","location","where","city"]):
            return h["answer"].strip()
    return ""


def calculate_evidence_strength(history, problem):
    text = (problem + " " + " ".join([h["answer"] for h in history])).lower()
    score = 0
    found = []
    evidence_map = {
        "screenshot":2,"receipt":2,"invoice":2,"recording":3,"audio":3,
        "video":3,"witness":2,"message":2,"chat":2,"email":2,
        "call log":1,"transaction id":2,"bank statement":2,
        "document":1,"agreement":2,"proof":1,"cctv":3,"fir":2
    }
    for k, pts in evidence_map.items():
        if k in text:
            score += pts
            found.append(k)
    if score >= 8:
        level = "Strong"; note = "You appear to have multiple useful forms of evidence."
    elif score >= 4:
        level = "Moderate"; note = "Some useful evidence exists. Collecting more will strengthen the case."
    else:
        level = "Weak"; note = "Very little concrete evidence visible. Collect stronger proof if possible."
    return {"level":level,"score":score,"note":note,"evidence_found":sorted(list(set(found)))[:8]}


def detect_missing_info(matched_case, history, problem):
    text  = (problem + " " + " ".join([h["answer"] for h in history])).lower()
    q_text = " ".join([h["question"].lower() for h in history])
    missing = []
    checks = [
        ("Full name / complainant identity",         ["name"]),
        ("Incident date or timeline",                 ["date","when","year"]),
        ("Location / state / district",              ["state","district","location","where"]),
        ("Other party / accused / respondent",       ["other party","officer","seller","respondent","who","name of"]),
        ("Evidence details",                          ["proof","evidence","screenshot","receipt","witness","recording"]),
        ("Prior complaint / FIR status",             ["already filed","complaint","fir","police"]),
        ("Relief sought",                             ["result","want","relief","seeking","outcome"]),
    ]
    for label, kws in checks:
        if not any(k in q_text or k in text for k in kws):
            missing.append(label)
    for h in history:
        q = h["question"].lower()
        a = h["answer"].strip().lower()
        if a in ["","not sure","no idea","unknown","n/a","na"]:
            if any(w in q for w in ["state","district","location"]) and "Location" not in " ".join(missing):
                missing.append("Location / state / district")
            if any(w in q for w in ["date","when"]) and "Incident date" not in " ".join(missing):
                missing.append("Incident date or timeline")
    return missing[:5]


def infer_high_risk(matched_case, history, problem):
    text  = (problem + " " + " ".join([h["answer"] for h in history])).lower()
    title = matched_case.get("title","").lower()
    danger_words = [
        "threat","kill","violence","urgent","ongoing","blackmail","suicide",
        "assault","sexual","abuse","harassment","attack","child","minor",
        "still happening","right now","emergency","scared","fear"
    ]
    if any(w in text for w in danger_words):
        return True
    if any(w in title for w in ["domestic violence","sexual","threat","harassment","acid","murder","stalking"]):
        return True
    return False


def build_urgency(matched_case, history, problem):

    title = matched_case.get("title", "")
    category = matched_case.get("category", "")

    if is_sensitive_case(title, category, problem):
        return {
            "high_risk": True,
            "label": "🔴 Sensitive & High Priority",
            "timeline": [
                "Immediately preserve all evidence.",
                "Contact police or relevant authority as soon as possible.",
                "Seek legal and emotional support if required.",
                "Maintain records of all communications and events."
            ]
        }

    high = infer_high_risk(matched_case, history, problem)

    if high:
        return {
            "high_risk": True,
            "label": "🔴 High Priority / Urgent",
            "timeline": [
                "Immediately: secure personal safety and preserve all evidence.",
                "Today: contact the most relevant authority or emergency support.",
                "Within 24 hours: submit formal complaint and document every action."
            ]
        }

    return {
        "high_risk": False,
        "label": "🟡 Normal Priority",
        "timeline": [
            "Today: organize facts, documents, and evidence.",
            "Within 1-3 days: submit written complaint to the correct authority.",
            "After submission: keep all copies and track progress regularly."
        ]
    }


# ── DYNAMIC PORTAL GENERATION (LLM-powered) ──────────────
def get_dynamic_portals(title, category, matched_case, problem=""):
    """Generate portals dynamically using LLM based on exact case classification."""
    try:
        steps_text = "\n".join(matched_case.get("steps",[])[:3])
        authority  = matched_case.get("authority","")
        prompt = f"""You are an Indian legal expert. For this EXACT legal case, provide the most relevant official Indian government portals/websites.

Case Title: {title}
Category: {category}
Authority: {authority}
Steps summary: {steps_text}
Problem context: {problem[:200]}

Return ONLY a JSON array of 2-4 portals most relevant to THIS specific case:
[
  {{"label": "Portal Name", "url": "https://actual-url.gov.in", "description": "one line what to do here"}},
  ...
]

Rules:
- URLs must be real Indian government portals (use .gov.in, .nic.in, .gov domains)
- Pick portals WHERE THE USER ACTUALLY FILES/TRACKS their specific complaint
- Be specific: prefer eDaakhil over generic consumer ministry, SCORES over generic SEBI, etc.
- Include the primary filing portal AND one reference/helpline portal
- No fake URLs. If unsure of exact URL, use the known official domain.
Return only valid JSON array."""

        raw = groq_chat(prompt, max_tokens=400, temperature=0.1)
        raw = raw.replace("```json","").replace("```","").strip()
        s = raw.find("["); e = raw.rfind("]") + 1
        if s != -1 and e > s:
            portals = json.loads(raw[s:e])
            if isinstance(portals, list) and len(portals) > 0:
                return portals[:4]
    except Exception as ex:
        print(f"Dynamic portal gen error: {ex}")

    # Minimal fallback — still case-aware
    return _get_portals_fallback(title, category)


def _get_portals_fallback(title, category):
    tl = title.lower()
    if any(w in tl for w in ["cyber","upi","fraud","online","hack","phish"]):
        return [{"label":"National Cyber Crime Portal","url":"https://cybercrime.gov.in","description":"File cyber fraud complaint"},
                {"label":"RBI Sachet","url":"https://sachet.rbi.org.in","description":"Report banking fraud"}]
    if any(w in tl for w in ["consumer","product","service","refund","defect"]):
        return [{"label":"eDaakhil Consumer Portal","url":"https://edaakhil.nic.in","description":"File consumer complaint"},
                {"label":"National Consumer Helpline","url":"https://consumerhelpline.gov.in","description":"Helpline & grievance"}]
    if any(w in tl for w in ["labour","salary","employment","epf","gratuity"]):
        return [{"label":"Shram Suvidha Portal","url":"https://shramsuvidha.gov.in","description":"Labour grievance"},
                {"label":"EPFO Portal","url":"https://www.epfindia.gov.in","description":"PF complaints"}]
    if any(w in tl for w in ["rti","information","public"]):
        return [{"label":"RTI Online Portal","url":"https://rtionline.gov.in","description":"File RTI application"}]
    if any(w in tl for w in ["sebi","stock","broker","mutual fund"]):
        return [{"label":"SEBI SCORES Portal","url":"https://scores.sebi.gov.in","description":"File investor complaint"}]
    if any(w in tl for w in ["rera","builder","flat","construction"]):
        return [{"label":"RERA Portal","url":"https://rera.gov.in","description":"Builder/property complaint"}]
    if any(w in tl for w in ["bribery","corruption","vigilance"]):
        return [{"label":"Central Vigilance Commission","url":"https://www.cvc.gov.in","description":"Corruption complaint"},
                {"label":"CBI","url":"https://cbi.gov.in","description":"Serious corruption cases"}]
    if any(w in tl for w in ["tax","gst","income tax"]):
        return [{"label":"Income Tax Portal","url":"https://www.incometax.gov.in","description":"Tax grievances"},
                {"label":"GST Portal","url":"https://www.gst.gov.in","description":"GST issues"}]
    if any(w in tl for w in ["domestic violence","women","harassment"]):
        return [{"label":"National Commission for Women","url":"https://ncw.nic.in","description":"Women's rights complaint"},
                {"label":"WCD Ministry","url":"https://wcd.nic.in","description":"Women & child welfare"}]
    if any(w in tl for w in ["insurance"]):
        return [{"label":"IRDAI Bima Bharosa","url":"https://bimabharosa.irdai.gov.in","description":"Insurance complaint"}]
    if any(w in tl for w in ["passport"]):
        return [{"label":"Passport Seva","url":"https://www.passportindia.gov.in","description":"Passport grievance"}]
    return [{"label":"India Government Services","url":"https://www.india.gov.in","description":"Central government portal"},
            {"label":"CPGRAMS Grievance Portal","url":"https://pgportal.gov.in","description":"Public grievance filing"}]


def build_guidance(matched_case, history, problem, location=""):
    title     = matched_case.get("title","")
    category  = matched_case.get("category","Civil")
    steps     = matched_case.get("steps",[])
    docs      = matched_case.get("documents",[])
    laws      = matched_case.get("provisions",[])
    authority = matched_case.get("authority","Concerned Authority")

    step_help = []
    for i, step in enumerate(steps[:5]):
        clean = step.replace(f"Step {i+1}:","").replace(f"Step {i+1}.","").strip()
        doc_hint = docs[i] if i < len(docs) else "relevant identity and case documents"
        step_help.append({
            "title":   clean[:80] + ("..." if len(clean) > 80 else ""),
            "how":     clean,
            "carry":   doc_hint,
            "outcome": f"This moves your case forward under {laws[0] if laws else 'applicable law'}."
        })

    # Dynamic portals via LLM
    portals = get_dynamic_portals(title, category, matched_case, problem)

    safety = _get_safety_advice(title, category, history, problem)

    authority_chain = authority.split("→")
    escalation = []
    for i in range(len(authority_chain)-1):
        curr = authority_chain[i].strip()
        nxt  = authority_chain[i+1].strip()
        escalation.append(f"If {curr} does not respond or resolve the issue, escalate to {nxt} with copies of all prior complaints.")

    if location:
        authority = f"{authority} (in your area: {location})"

    urgent_actions = _get_urgent_actions(title, category)

    return {
        "portals":       portals,
        "step_help":     step_help,
        "safety_advice": safety,
        "escalation":    escalation,
        "urgent_actions":urgent_actions,
        "authority_name":authority,
        "template_type": f"{category} — {title}"
    }


def _get_safety_advice(title, category, history, problem):
    tl   = title.lower()
    text = (problem + " " + " ".join([h["answer"] for h in history])).lower()

    advice = [
        "Keep copies of all documents, complaints, and communications.",
        "Use written/email communication rather than only phone calls for a paper trail.",
        "Do not share sensitive personal information with unknown parties.",
        "Track all complaint reference numbers and submission dates."
    ]

    if any(w in tl for w in ["cyber","fraud","hack","phish","upi"]):
        advice = [
            "Do not share OTP, PIN, or remote access with anyone claiming to be from a bank.",
            "Immediately report to your bank and block the affected account if money is missing.",
            "Preserve all digital evidence — do not delete messages, screenshots, or call logs.",
            "Report to cybercrime.gov.in as soon as possible — speed improves recovery chances."
        ]
    elif any(w in tl for w in ["domestic","violence","abuse","assault","harassment","sexual"]):
        advice = [
            "Prioritize your personal safety above everything else.",
            "Document injuries with medical reports and photographs.",
            "Reach out to a trusted person or shelter if you are in immediate danger.",
            "Women's Helpline: 181 | Emergency: 100 | POCSO Helpline: 1098"
        ]
    elif any(w in tl for w in ["bribery","corruption"]):
        advice = [
            "Do not inform the accused that you are planning to file a complaint.",
            "Keep detailed records of every demand — amount, date, purpose, and the officer's details.",
            "Never pay voluntarily unless part of a coordinated lawful process under official advice.",
            "Preserve any messages, call records, or written demands."
        ]
    elif any(w in tl for w in ["cheque bounce"]):
        advice = [
            "The demand notice must be sent within 30 days of receiving cheque return memo.",
            "If no payment after 15 days of notice, file a criminal complaint under Section 138 NI Act.",
            "Keep the original cheque and bank return memo safely — these are primary evidence.",
            "Time limits are strict in cheque bounce cases — act promptly."
        ]
    elif any(w in tl for w in ["property","land","encroachment"]):
        advice = [
            "Get an official land survey done before any confrontation.",
            "Do not remove any physical markers or structures without legal process.",
            "Keep all title documents, mutation records, and tax receipts safely.",
            "Approach Revenue Court for record corrections in addition to Civil Court."
        ]

    return advice


def _get_urgent_actions(title, category):
    tl = title.lower()
    if any(w in tl for w in ["cyber","fraud","hack","upi"]):
        return ["Today: contact bank, block account, preserve evidence.", "Within 24 hours: file cyber complaint online."]
    if any(w in tl for w in ["cheque bounce"]):
        return ["Within 30 days of cheque return memo: send legal demand notice.", "Act before limitation period expires."]
    if any(w in tl for w in ["domestic","violence","abuse","sexual"]):
        return ["Immediately: ensure safety and get medical attention.", "Today: approach police or Protection Officer."]
    return ["Today: organize facts and evidence.", "Within 1-3 days: submit written complaint."]


def generate_complaint(matched_case, history, problem, guidance, language="English"):
    authority = guidance["authority_name"]
    subject   = f"Complaint regarding {matched_case.get('title','legal issue')}"

    def get_answer(kws):
        for h in history:
            if all(k in h["question"].lower() for k in kws):
                return h["answer"].strip()
        return ""

    name      = get_answer(["name"]) or "Complainant"
    date_inc  = get_answer(["date"]) or get_answer(["when"]) or "As per records"
    location  = get_answer(["state","district"]) or get_answer(["location"]) or "As per records"
    other     = get_answer(["other party"]) or get_answer(["who"]) or "As per records"
    evidence  = get_answer(["proof"]) or get_answer(["evidence"]) or "Documents available with complainant"
    relief    = get_answer(["relief"]) or get_answer(["result"]) or get_answer(["want"]) or "Appropriate legal action as per law"

    complaint = f"""To,
The {authority}

Subject: {subject}

Respected Sir/Madam,

I, {name}, respectfully submit this complaint regarding the legal matter described below.

1. Nature of Issue:
{matched_case.get('title','')} ({matched_case.get('category','')} matter)

2. Date / Timeline of Incident:
{date_inc}

3. Location of Incident:
{location}

4. Details of Other Party / Respondent:
{other}

5. Facts of the Case:
{problem.strip()}

6. Evidence Available:
{evidence}

7. Applicable Legal Provisions:
{", ".join(matched_case.get("provisions",[])[:3])}

8. Relief / Action Requested:
{relief}

I request your office to take prompt legal action as per law. I am ready to provide all supporting documents and further information as needed.

I affirm that the above facts are true to the best of my knowledge.

Yours faithfully,
{name}
Date: {datetime.now().strftime('%d-%m-%Y')}

Note: This is a draft. Please review and add any additional facts before submission.
"""
    return translate_text(complaint, language)


def get_editable_fields(history, problem, language="English"):
    def ga(kws):
        for h in history:
            if all(k in h["question"].lower() for k in kws):
                return h["answer"].strip()
        return ""

    fields = [
        {"key":"name",            "label":"Complainant Name",          "value": ga(["name"]) or ""},
        {"key":"date_of_incident","label":"Incident Date / Timeline",   "value": ga(["date"]) or ga(["when"]) or ""},
        {"key":"location",        "label":"Location / State / District","value": ga(["state","district"]) or ga(["location"]) or ""},
        {"key":"other_party",     "label":"Other Party / Respondent",   "value": ga(["other party"]) or ga(["who"]) or ""},
        {"key":"evidence",        "label":"Available Evidence",         "value": ga(["proof"]) or ga(["evidence"]) or ""},
        {"key":"relief",          "label":"Relief Requested",           "value": ga(["relief"]) or ga(["result"]) or ""},
    ]
    if language != "English":
        for f in fields:
            f["label"] = translate_text(f["label"], language)
    return fields


def evaluate_readiness(uploaded_docs, documents_required):
    n = len(uploaded_docs)
    t = len(documents_required)
    s = int((n/t)*100) if t > 0 else 0
    if s >= 100:
        return {"score":s,"status":"✅ All Documents Uploaded","color":"green","message":"All required documents are marked uploaded. You are ready to proceed."}
    elif s >= 50:
        return {"score":s,"status":"⚠️ Partially Ready","color":"orange","message":"Some documents uploaded. Collect the remaining ones before proceeding."}
    else:
        return {"score":s,"status":"⛔ Not Ready","color":"red","message":"Please collect and upload the required documents before filing."}


def build_finalized_payload(sess):
    problem  = sess["problem"]
    matched  = sess["matched"]
    history  = sess["history"]
    language = sess.get("language","English")
    uploaded = sess.get("uploaded_docs",[])
    location = get_location_from_history(history)

    friendly   = generate_final_analysis(problem, matched, history, language)
    guidance   = build_guidance(matched, history, problem, location)
    missing    = detect_missing_info(matched, history, problem)
    evidence   = calculate_evidence_strength(history, problem)
    urgency    = build_urgency(matched, history, problem)
    readiness  = evaluate_readiness(uploaded, matched["documents"])
    complaint  = generate_complaint(matched, history, problem, guidance, language)
    editable   = sess.get("editable_fields") or get_editable_fields(history, problem, language)

    if language != "English":
        guidance["authority_name"] = translate_text(guidance["authority_name"], language)
        guidance["safety_advice"]  = translate_list(guidance["safety_advice"], language)
        guidance["escalation"]     = translate_list(guidance["escalation"], language)
        for step in guidance["step_help"]:
            step["title"]   = translate_text(step["title"], language)
            step["how"]     = translate_text(step["how"], language)
            step["carry"]   = translate_text(step["carry"], language)
            step["outcome"] = translate_text(step["outcome"], language)
        missing             = translate_list(missing, language)
        evidence["note"]    = translate_text(evidence["note"], language)
        urgency["label"]    = translate_text(urgency["label"], language)
        urgency["timeline"] = translate_list(urgency["timeline"], language)
        readiness["status"] = translate_text(readiness["status"], language)
        readiness["message"]= translate_text(readiness["message"], language)

    terms = [{"term":k,"explanation":v} for k,v in list(LEGAL_TERMS.items())[:6]]

    return {
        "category":        matched["category"],
        "title":           matched["title"],
        "friendly":        friendly,
        "documents":       matched["documents"],
        "provisions":      matched["provisions"],
        "authority":       guidance["authority_name"],
        "readiness":       readiness,
        "safety_advice":   guidance["safety_advice"],
        "step_help":       guidance["step_help"],
        "complaint_text":  complaint,
        "missing_info":    missing,
        "evidence":        evidence,
        "urgency":         urgency,
        "portals":         guidance["portals"],
        "escalation":      guidance["escalation"],
        "legal_terms":     terms,
        "editable_fields": editable,
        "progress_tracker":[
            {"key":"facts_ready",       "label":"Facts organized"},
            {"key":"docs_ready",        "label":"Documents collected"},
            {"key":"complaint_drafted", "label":"Complaint draft prepared"},
            {"key":"authority_found",   "label":"Authority identified"},
            {"key":"complaint_filed",   "label":"Complaint submitted"},
            {"key":"followup",          "label":"Follow-up done"},
        ],
        "next_actions": [
            {"label":"View Steps",        "type":"scroll_steps"},
            {"label":"View Documents",    "type":"scroll_docs"},
            {"label":"Draft Complaint",   "type":"open_complaint"},
        ]
    }


# ── 8) FLASK APP ─────────────────────────────────────────
from flask import Flask, request, jsonify, render_template_string

app            = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(16))
sessions       = {}

# ─── COMPLAINT PAGE HTML ─────────────────────────────────
COMPLAINT_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Complaint Draft — Legal Guidance Platform</title>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{
  --navy:#0B1F45;--navy-light:#1a3461;--gold:#C9A84C;--gold-light:#e8c97a;
  --cream:#F7F5F0;--white:#ffffff;--gray-50:#F9FAFB;--gray-100:#F3F4F6;
  --gray-200:#E5E7EB;--gray-400:#9CA3AF;--gray-600:#4B5563;--gray-800:#1F2937;
  --red:#DC2626;--green:#16A34A;--blue:#1D4ED8;--shadow-sm:0 1px 3px rgba(0,0,0,0.08);
  --shadow-md:0 4px 16px rgba(0,0,0,0.10);--shadow-lg:0 8px 32px rgba(0,0,0,0.12);
  --radius:12px;--radius-sm:8px;--radius-lg:18px;
}
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:'Inter',sans-serif;background:var(--cream);color:var(--gray-800);min-height:100vh;}

/* Header */
.header{background:var(--navy);padding:0;border-bottom:3px solid var(--gold);position:sticky;top:0;z-index:100;box-shadow:var(--shadow-md);}
.header-inner{max-width:1100px;margin:0 auto;padding:14px 28px;display:flex;align-items:center;gap:16px;}
.header-logo{font-size:24px;}
.header-brand{flex:1;}
.header-brand h1{font-family:'Playfair Display',serif;color:white;font-size:18px;letter-spacing:.3px;}
.header-brand p{color:#94a3b8;font-size:12px;margin-top:2px;}
.header-back{display:flex;align-items:center;gap:7px;color:#cbd5e1;text-decoration:none;font-size:13px;font-weight:600;padding:8px 14px;border:1px solid rgba(255,255,255,0.15);border-radius:8px;transition:all .2s;}
.header-back:hover{background:rgba(255,255,255,0.08);color:white;}

.page{max-width:960px;margin:0 auto;padding:32px 24px 80px;}

/* Page title bar */
.page-title-bar{background:white;border-radius:var(--radius-lg);padding:24px 28px;margin-bottom:24px;box-shadow:var(--shadow-sm);border:1px solid var(--gray-200);display:flex;align-items:center;gap:16px;}
.page-title-bar .icon{width:48px;height:48px;background:linear-gradient(135deg,var(--navy),var(--navy-light));border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:22px;flex-shrink:0;}
.page-title-bar h2{font-family:'Playfair Display',serif;font-size:22px;color:var(--navy);margin-bottom:4px;}
.page-title-bar p{font-size:13px;color:var(--gray-600);line-height:1.5;}

/* Cards */
.card{background:white;border-radius:var(--radius-lg);padding:28px;margin-bottom:20px;box-shadow:var(--shadow-sm);border:1px solid var(--gray-200);}
.card-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;padding-bottom:14px;border-bottom:1px solid var(--gray-100);}
.card-title{font-family:'Playfair Display',serif;font-size:17px;color:var(--navy);display:flex;align-items:center;gap:10px;}
.card-title-icon{width:34px;height:34px;border-radius:8px;background:linear-gradient(135deg,#EEF2FF,#E0E7FF);display:flex;align-items:center;justify-content:center;font-size:16px;}

/* Edit Fields Grid */
.edit-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
@media(max-width:680px){.edit-grid{grid-template-columns:1fr;}}
.field-group{display:flex;flex-direction:column;gap:6px;}
.field-label{font-size:12px;font-weight:700;color:var(--navy);text-transform:uppercase;letter-spacing:.6px;}
.field-input{padding:11px 14px;border:1.5px solid var(--gray-200);border-radius:var(--radius-sm);font-size:14px;color:var(--gray-800);background:var(--gray-50);transition:all .2s;font-family:'Inter',sans-serif;}
.field-input:focus{outline:none;border-color:var(--navy);background:white;box-shadow:0 0 0 3px rgba(11,31,69,0.08);}
.field-input.full{width:100%;}

/* Complaint textarea */
.complaint-editor{width:100%;min-height:420px;padding:20px;border:1.5px solid var(--gray-200);border-radius:var(--radius);font-size:14px;line-height:1.85;color:var(--gray-800);font-family:'Courier New',monospace;background:var(--gray-50);resize:vertical;transition:border-color .2s;}
.complaint-editor:focus{outline:none;border-color:var(--navy);background:white;box-shadow:0 0 0 3px rgba(11,31,69,0.08);}

/* Action bar */
.action-bar{display:flex;gap:12px;flex-wrap:wrap;align-items:center;}
.btn{display:inline-flex;align-items:center;gap:8px;padding:12px 22px;border-radius:10px;font-size:14px;font-weight:700;cursor:pointer;transition:all .2s;border:none;font-family:'Inter',sans-serif;}
.btn-primary{background:var(--navy);color:white;}
.btn-primary:hover{background:var(--navy-light);box-shadow:0 4px 14px rgba(11,31,69,0.35);}
.btn-gold{background:var(--gold);color:white;}
.btn-gold:hover{background:var(--gold-light);color:var(--navy);}
.btn-outline{background:transparent;color:var(--navy);border:1.5px solid var(--navy);}
.btn-outline:hover{background:var(--navy);color:white;}
.btn-green{background:#16a34a;color:white;}
.btn-green:hover{background:#15803d;}
.btn-ghost{background:var(--gray-100);color:var(--gray-600);border:1px solid var(--gray-200);}
.btn-ghost:hover{background:var(--gray-200);color:var(--gray-800);}

/* Toast */
.toast{position:fixed;bottom:28px;right:28px;background:var(--navy);color:white;padding:13px 20px;border-radius:10px;font-size:14px;font-weight:600;box-shadow:var(--shadow-lg);display:none;align-items:center;gap:10px;z-index:9999;animation:slideUp .3s ease;}
.toast.show{display:flex;}
.toast.success{background:#16a34a;}
.toast.error{background:var(--red);}
@keyframes slideUp{from{opacity:0;transform:translateY(16px);}to{opacity:1;transform:translateY(0);}}

/* Note box */
.note-box{background:#FEF9C3;border:1px solid #FDE68A;border-radius:var(--radius-sm);padding:12px 16px;font-size:13px;color:#78350F;display:flex;gap:10px;align-items:flex-start;margin-bottom:16px;}
.note-box-icon{font-size:16px;flex-shrink:0;margin-top:1px;}

/* Loading overlay */
.loading-overlay{display:none;position:fixed;inset:0;background:rgba(11,31,69,0.7);z-index:9998;align-items:center;justify-content:center;flex-direction:column;gap:16px;}
.loading-overlay.show{display:flex;}
.spinner{width:44px;height:44px;border:3px solid rgba(255,255,255,0.25);border-top-color:var(--gold);border-radius:50%;animation:spin .75s linear infinite;}
.loading-overlay p{color:white;font-size:15px;font-weight:600;}
@keyframes spin{to{transform:rotate(360deg);}}
</style>
</head>
<body>
<div class="header">
  <div class="header-inner">
    <div class="header-logo">⚖️</div>
    <div class="header-brand">
      <h1>Legal Guidance Platform</h1>
      <p>Complaint Draft &amp; Export</p>
    </div>
    <a class="header-back" href="javascript:history.back()">← Back to Summary</a>
  </div>
</div>

<div class="page">
  <div class="page-title-bar">
    <div class="icon">🧾</div>
    <div>
      <h2>Smart Complaint Draft</h2>
      <p>Edit the fields below and regenerate your complaint. Download as PDF when ready.</p>
    </div>
  </div>

  <!-- Edit Fields -->
  <div class="card">
    <div class="card-header">
      <div class="card-title"><div class="card-title-icon">✍️</div> Edit Complaint Details</div>
    </div>
    <div class="note-box"><span class="note-box-icon">💡</span>Fill in the fields below accurately. Click "Update Draft" to regenerate the complaint with your changes.</div>
    <div class="edit-grid" id="edit-fields-grid"></div>
    <div style="margin-top:20px;">
      <button class="btn btn-primary" onclick="updateDraft()">🔄 Update Draft</button>
    </div>
  </div>

  <!-- Complaint Editor -->
  <div class="card">
    <div class="card-header">
      <div class="card-title"><div class="card-title-icon">📄</div> Complaint Text</div>
      <div class="action-bar">
        <button class="btn btn-ghost" onclick="copyText()">📋 Copy</button>
        <button class="btn btn-gold" onclick="downloadPDF()">⬇️ Download PDF</button>
      </div>
    </div>
    <textarea class="complaint-editor" id="complaint-editor" spellcheck="true"></textarea>
    <div style="margin-top:16px;" class="action-bar">
      <button class="btn btn-green" onclick="downloadPDF()">⬇️ Download as PDF</button>
      <button class="btn btn-outline" onclick="copyText()">📋 Copy to Clipboard</button>
      <button class="btn btn-ghost" onclick="window.print()">🖨️ Print</button>
    </div>
  </div>
</div>

<!-- Toast -->
<div class="toast" id="toast"></div>

<!-- Loading -->
<div class="loading-overlay" id="loading-overlay">
  <div class="spinner"></div>
  <p id="loading-msg">Generating PDF...</p>
</div>

<script>
// Inject data from parent page via sessionStorage
const data = JSON.parse(sessionStorage.getItem("complaint_page_data") || "{}");
const sessionId = data.session_id || "";

// Render editable fields
const fields = data.editable_fields || [];
const grid = document.getElementById("edit-fields-grid");
fields.forEach((f, idx) => {
  const div = document.createElement("div");
  div.className = "field-group";
  div.innerHTML = `<label class="field-label">${escH(f.label)}</label>
    <input class="field-input" id="ef_${idx}" value="${escH(f.value || "")}" placeholder="Enter ${escH(f.label).toLowerCase()}">`;
  grid.appendChild(div);
});

// Set complaint text
document.getElementById("complaint-editor").value = data.complaint_text || "";

function escH(s){ return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;"); }

function showToast(msg, type=""){
  const t = document.getElementById("toast");
  t.textContent = msg; t.className = "toast show " + type;
  setTimeout(()=>{ t.className="toast"; }, 3200);
}

function showLoading(msg){ document.getElementById("loading-msg").textContent=msg; document.getElementById("loading-overlay").classList.add("show"); }
function hideLoading(){ document.getElementById("loading-overlay").classList.remove("show"); }

async function updateDraft(){
  const updated = fields.map((f, idx) => ({
    key: f.key, label: f.label, value: document.getElementById("ef_"+idx).value.trim()
  }));
  showLoading("Regenerating complaint...");
  try{
    const res = await fetch("/update_complaint_fields", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ session_id: sessionId, editable_fields: updated })
    });
    const d = await res.json();
    if(res.ok && d.complaint_text){
      document.getElementById("complaint-editor").value = d.complaint_text;
      showToast("✅ Draft updated successfully!", "success");
    } else { showToast("Update failed. Please try again.", "error"); }
  } catch(e){ showToast("Network error. Please try again.", "error"); }
  hideLoading();
}

function copyText(){
  const text = document.getElementById("complaint-editor").value;
  navigator.clipboard.writeText(text)
    .then(()=>showToast("✅ Complaint copied to clipboard!", "success"))
    .catch(()=>showToast("Copy failed. Please select and copy manually.", "error"));
}

async function downloadPDF(){
  const text = document.getElementById("complaint-editor").value;
  if(!text.trim()){ showToast("No complaint text to export.", "error"); return; }
  showLoading("Generating PDF...");
  try{
    const res = await fetch("/generate_pdf", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ complaint_text: text, session_id: sessionId })
    });
    if(!res.ok) throw new Error("PDF generation failed");
    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    a.download = "legal_complaint_draft.pdf";
    document.body.appendChild(a); a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast("✅ PDF downloaded!", "success");
  } catch(e){ showToast("PDF generation failed. Please try again.", "error"); }
  hideLoading();
}
</script>
</body>
</html>
"""

# ─── MAIN PAGE HTML ──────────────────────────────────────
HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Legal Guidance Platform</title>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{
  --navy:#0B1F45;--navy-light:#1a3461;--gold:#C9A84C;--gold-light:#e8c97a;
  --cream:#F7F5F0;--white:#ffffff;--gray-50:#F9FAFB;--gray-100:#F3F4F6;
  --gray-200:#E5E7EB;--gray-300:#D1D5DB;--gray-400:#9CA3AF;--gray-600:#4B5563;--gray-800:#1F2937;
  --red:#DC2626;--red-light:#FEE2E2;--green:#16A34A;--green-light:#DCFCE7;
  --blue:#1D4ED8;--blue-light:#EFF6FF;--orange:#EA580C;--orange-light:#FFF7ED;
  --shadow-xs:0 1px 2px rgba(0,0,0,0.06);--shadow-sm:0 1px 4px rgba(0,0,0,0.08);
  --shadow-md:0 4px 16px rgba(0,0,0,0.10);--shadow-lg:0 8px 32px rgba(0,0,0,0.12);
  --radius-sm:8px;--radius:12px;--radius-lg:18px;--radius-xl:24px;
}
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:'Inter',sans-serif;background:var(--cream);color:var(--gray-800);min-height:100vh;}

/* HEADER */
.header{background:var(--navy);border-bottom:3px solid var(--gold);position:sticky;top:0;z-index:100;box-shadow:0 2px 12px rgba(0,0,0,0.25);}
.header-inner{max-width:1100px;margin:0 auto;padding:14px 28px;display:flex;align-items:center;gap:16px;}
.header-logo-wrap{width:42px;height:42px;background:linear-gradient(135deg,var(--gold),var(--gold-light));border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:22px;flex-shrink:0;}
.header-text{flex:1;}
.header-text h1{font-family:'Playfair Display',serif;color:white;font-size:19px;letter-spacing:.2px;}
.header-text p{color:#94a3b8;font-size:12px;margin-top:2px;}
.lang-select{background:rgba(255,255,255,0.1);color:white;border:1px solid rgba(255,255,255,0.2);border-radius:var(--radius-sm);padding:8px 13px;font-size:13px;cursor:pointer;outline:none;font-family:'Inter',sans-serif;transition:background .2s;}
.lang-select:hover{background:rgba(255,255,255,0.18);}
.lang-select option{background:var(--navy);color:white;}

/* MAIN */
.main{max-width:960px;margin:0 auto;padding:28px 24px 80px;}

/* PROGRESS */
.progress-wrap{margin-bottom:24px;}
.progress-meta{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;}
.progress-label{font-size:12px;font-weight:700;color:var(--navy);text-transform:uppercase;letter-spacing:.5px;}
.progress-pct{font-size:12px;font-weight:700;color:var(--gold);}
.progress-track{height:6px;background:var(--gray-200);border-radius:999px;overflow:hidden;}
.progress-fill{height:100%;background:linear-gradient(90deg,var(--navy),var(--gold));border-radius:999px;transition:width .5s ease;}

/* CARDS */
.card{background:white;border-radius:var(--radius-lg);padding:28px;margin-bottom:20px;box-shadow:var(--shadow-sm);border:1px solid var(--gray-200);}
.card-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:18px;padding-bottom:14px;border-bottom:1px solid var(--gray-100);}
.section-title{font-family:'Playfair Display',serif;font-size:17px;color:var(--navy);display:flex;align-items:center;gap:10px;}
.title-icon{width:34px;height:34px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:15px;flex-shrink:0;}
.ti-blue{background:#EEF2FF;}
.ti-gold{background:#FEF3C7;}
.ti-green{background:#DCFCE7;}
.ti-red{background:#FEE2E2;}
.ti-orange{background:#FFF7ED;}
.ti-purple{background:#F3E8FF;}

/* STEP 1 */
.step1-hero{font-family:'Playfair Display',serif;font-size:15px;color:var(--gray-600);line-height:1.7;margin-bottom:20px;}
.problem-textarea{width:100%;padding:16px;font-size:15px;line-height:1.7;border:1.5px solid var(--gray-200);border-radius:var(--radius);background:var(--gray-50);color:var(--gray-800);height:150px;resize:vertical;font-family:'Inter',sans-serif;transition:all .2s;}
.problem-textarea:focus{outline:none;border-color:var(--navy);background:white;box-shadow:0 0 0 3px rgba(11,31,69,0.08);}

/* BUTTONS */
.btn{display:inline-flex;align-items:center;gap:8px;padding:12px 22px;border-radius:10px;font-size:14px;font-weight:700;cursor:pointer;transition:all .2s;border:none;font-family:'Inter',sans-serif;letter-spacing:.1px;}
.btn-primary{background:var(--navy);color:white;width:100%;justify-content:center;margin-top:16px;font-size:15px;padding:14px;}
.btn-primary:hover{background:var(--navy-light);box-shadow:0 4px 14px rgba(11,31,69,0.3);}
.btn-outline{background:transparent;color:var(--navy);border:1.5px solid var(--navy);}
.btn-outline:hover{background:var(--navy);color:white;}
.btn-chip{background:var(--blue-light);color:var(--blue);border:1px solid #BFDBFE;padding:9px 16px;border-radius:999px;font-size:13px;}
.btn-chip:hover{background:#DBEAFE;}
.btn-sm{padding:8px 14px;font-size:13px;border-radius:8px;}
.btn-ghost{background:var(--gray-100);color:var(--gray-600);border:1px solid var(--gray-200);}
.btn-ghost:hover{background:var(--gray-200);}
.inline-btns{display:flex;gap:10px;flex-wrap:wrap;margin-top:14px;}

/* QUESTIONS */
.complexity-tag{display:inline-flex;align-items:center;gap:7px;padding:6px 14px;border-radius:999px;font-size:12px;font-weight:800;letter-spacing:.4px;text-transform:uppercase;margin-bottom:18px;}
.cx-simple{background:#DCFCE7;color:#166534;}
.cx-medium{background:#FEF9C3;color:#854D0E;}
.cx-complex{background:#FEE2E2;color:#991B1B;}
.round-label{font-size:12px;color:var(--gray-400);font-weight:700;text-transform:uppercase;letter-spacing:.5px;margin-bottom:20px;padding:8px 14px;background:var(--gray-50);border-radius:8px;border-left:3px solid var(--gold);}
.qa-item{margin-bottom:20px;}
.qa-label{display:flex;align-items:flex-start;gap:10px;margin-bottom:9px;}
.q-num{min-width:26px;height:26px;background:var(--navy);color:white;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:800;flex-shrink:0;margin-top:1px;}
.q-text{font-size:14px;font-weight:600;color:var(--gray-800);line-height:1.5;}
.qa-input,.qa-select{width:100%;padding:11px 14px;font-size:14px;border:1.5px solid var(--gray-200);border-radius:var(--radius-sm);background:var(--gray-50);color:var(--gray-800);font-family:'Inter',sans-serif;transition:all .2s;}
.qa-input:focus,.qa-select:focus{outline:none;border-color:var(--navy);background:white;box-shadow:0 0 0 3px rgba(11,31,69,0.08);}
.sensitive-notice{display:flex;gap:12px;align-items:flex-start;background:#FEF3C7;border:1px solid #FDE68A;border-radius:10px;padding:13px 16px;margin-bottom:20px;font-size:13px;color:#78350F;line-height:1.5;}

/* LOADING */
.loading{display:none;text-align:center;padding:50px 20px;}
.spinner{width:44px;height:44px;border:3px solid var(--gray-200);border-top-color:var(--navy);border-radius:50%;animation:spin .75s linear infinite;margin:0 auto 18px;}
@keyframes spin{to{transform:rotate(360deg);}}
.loading p{color:var(--gray-600);font-size:15px;font-weight:600;margin-bottom:4px;}
.loading small{color:var(--gray-400);font-size:13px;}

/* RESULT */
.result-section{display:none;}

/* BADGE */
.case-badge{display:inline-flex;align-items:center;padding:4px 12px;border-radius:20px;font-size:11px;font-weight:800;letter-spacing:.6px;text-transform:uppercase;}
.badge-criminal{background:#FEE2E2;color:#991B1B;}
.badge-civil{background:#DBEAFE;color:#1E40AF;}
.case-title-main{font-family:'Playfair Display',serif;font-size:20px;color:var(--navy);margin:10px 0 14px;}
.friendly-text{font-size:15px;line-height:1.8;color:var(--gray-800);background:linear-gradient(135deg,var(--gray-50),#F0F4FF);border-left:3px solid var(--gold);padding:16px 20px;border-radius:0 10px 10px 0;white-space:pre-wrap;}

/* INFO BOXES */
.info-box{border-radius:10px;padding:14px 18px;font-size:14px;line-height:1.7;}
.ib-blue{background:var(--blue-light);border-left:4px solid var(--blue);}
.ib-orange{background:var(--orange-light);border-left:4px solid var(--orange);}
.ib-red{background:var(--red-light);border-left:4px solid var(--red);}
.ib-green{background:var(--green-light);border-left:4px solid var(--green);}
.info-list{padding-left:18px;}
.info-list li{margin-bottom:8px;}

/* GRID */
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:20px;}
@media(max-width:680px){.grid2{grid-template-columns:1fr;}}

/* BADGES */
.badge-mini{display:inline-block;padding:3px 10px;border-radius:999px;font-size:11px;font-weight:800;letter-spacing:.3px;text-transform:uppercase;}
.bm-red{background:#FEE2E2;color:#991B1B;}
.bm-green{background:#DCFCE7;color:#166534;}
.bm-orange{background:#FFEDD5;color:#9A3412;}

/* AUTHORITY */
.authority-box{background:linear-gradient(135deg,#EEF2FF,#E0E7FF);border-radius:12px;padding:16px 20px;font-size:15px;color:var(--navy);font-weight:700;border:1px solid #C7D2FE;line-height:1.6;}

/* READINESS */
.readiness-box{border-radius:12px;padding:16px 20px;}
.rb-red{background:var(--red-light);border-left:4px solid var(--red);}
.rb-orange{background:var(--orange-light);border-left:4px solid var(--orange);}
.rb-green{background:var(--green-light);border-left:4px solid var(--green);}
.readiness-status{font-size:15px;font-weight:800;margin-bottom:4px;}
.readiness-msg{font-size:13px;color:var(--gray-600);}

/* CHECKLIST */
.step-item{padding:14px 0;border-bottom:1px solid var(--gray-100);}
.step-item:last-child{border-bottom:none;}
.step-top{display:flex;align-items:flex-start;gap:12px;}
.step-cb{width:18px;height:18px;min-width:18px;accent-color:var(--navy);cursor:pointer;margin-top:3px;}
.step-label{font-size:14px;font-weight:600;color:var(--gray-800);line-height:1.5;flex:1;}
.step-label.done{text-decoration:line-through;color:var(--gray-400);}
.step-detail{margin-left:30px;margin-top:10px;background:var(--gray-50);border-radius:10px;padding:12px 14px;border:1px solid var(--gray-200);}
.step-detail-line{font-size:13px;line-height:1.6;color:var(--gray-600);margin-bottom:5px;}
.doc-item{display:flex;align-items:center;gap:10px;padding:10px 0;border-bottom:1px solid var(--gray-100);flex-wrap:wrap;}
.doc-item:last-child{border-bottom:none;}
.doc-icon{width:30px;height:30px;background:#EEF2FF;border-radius:7px;display:flex;align-items:center;justify-content:center;font-size:15px;flex-shrink:0;}
.doc-name{font-size:14px;color:var(--gray-800);flex:1;font-weight:500;}
.doc-status{font-size:11px;padding:3px 9px;border-radius:999px;font-weight:800;}
.ds-pending{background:#FEF9C3;color:#854D0E;}
.ds-uploaded{background:#DCFCE7;color:#166534;}
.doc-upload-btn{padding:5px 12px;background:var(--navy);color:white;border:none;border-radius:6px;font-size:12px;cursor:pointer;font-weight:700;}
.doc-filename{font-size:11px;color:var(--gray-400);width:100%;padding-left:40px;}

/* PROGRESS BARS */
.sub-progress{font-size:12px;color:var(--gray-400);font-weight:600;margin-bottom:10px;}
.mini-track{height:4px;background:var(--gray-200);border-radius:999px;overflow:hidden;margin-top:5px;}
.mini-fill-navy{height:100%;background:var(--navy);border-radius:999px;transition:width .3s;}
.mini-fill-gold{height:100%;background:var(--gold);border-radius:999px;transition:width .3s;}

/* PORTALS */
.portal-item{padding:10px 0;border-bottom:1px solid var(--gray-100);}
.portal-item:last-child{border-bottom:none;}
.portal-link{color:var(--blue);text-decoration:none;font-weight:700;font-size:14px;}
.portal-link:hover{text-decoration:underline;}
.portal-desc{font-size:12px;color:var(--gray-400);margin-top:2px;}
.portal-url{font-size:11px;color:var(--gray-300);}

/* LAWS / TERMS */
.list-item{padding:9px 0;border-bottom:1px solid var(--gray-100);font-size:14px;color:var(--gray-800);}
.list-item:last-child{border-bottom:none;}

/* EDITABLE FIELDS */
.edit-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;}
@media(max-width:680px){.edit-grid{grid-template-columns:1fr;}}
.edit-field label{display:block;font-size:11px;font-weight:700;color:var(--navy);text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px;}
.edit-input{width:100%;padding:10px 13px;border:1.5px solid var(--gray-200);border-radius:8px;font-size:14px;color:var(--gray-800);background:var(--gray-50);font-family:'Inter',sans-serif;transition:all .2s;}
.edit-input:focus{outline:none;border-color:var(--navy);background:white;box-shadow:0 0 0 3px rgba(11,31,69,0.08);}

/* PROGRESS TRACKER */
.tracker-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;}
@media(max-width:680px){.tracker-grid{grid-template-columns:1fr;}}
.tracker-item{display:flex;align-items:center;gap:10px;padding:11px 14px;border:1px solid var(--gray-200);border-radius:10px;background:var(--gray-50);font-size:14px;font-weight:500;cursor:pointer;}
.tracker-item input{width:16px;height:16px;accent-color:var(--navy);}

/* ERROR */
.error-box{display:none;background:#FFF1F2;border-left:4px solid #E11D48;padding:12px 16px;border-radius:0 8px 8px 0;font-size:14px;color:#9F1239;margin-top:12px;}

/* DIVIDER */
.section-divider{border:none;border-top:1px solid var(--gray-100);margin:4px 0 20px;}

/* EMERGENCY CARD */
.emergency-card{display:none;background:linear-gradient(135deg,#FEE2E2,#FFF1F2);border:2px solid var(--red);border-radius:var(--radius-lg);padding:22px;margin-bottom:20px;}
.emergency-title{font-size:15px;font-weight:800;color:var(--red);margin-bottom:8px;display:flex;align-items:center;gap:8px;}
.emergency-text{font-size:14px;color:#7F1D1D;line-height:1.7;}

/* COMPLAINT DRAFT BUTTON */
.draft-complaint-btn{background:linear-gradient(135deg,var(--navy),var(--navy-light));color:white;padding:14px 24px;border-radius:12px;font-size:15px;font-weight:800;cursor:pointer;border:none;display:inline-flex;align-items:center;gap:10px;font-family:'Inter',sans-serif;transition:all .2s;box-shadow:0 4px 14px rgba(11,31,69,0.3);}
.draft-complaint-btn:hover{transform:translateY(-1px);box-shadow:0 6px 20px rgba(11,31,69,0.4);}

.fade-in{animation:fadeIn .4s ease forwards;}
@keyframes fadeIn{from{opacity:0;transform:translateY(12px);}to{opacity:1;transform:translateY(0);}}
</style>
</head>
<body>
<div class="header">
  <div class="header-inner">
    <div class="header-logo-wrap">⚖️</div>
    <div class="header-text">
      <h1>Legal Guidance Platform</h1>
      <p>Personalized Indian legal guidance — step by step</p>
    </div>
    <select class="lang-select" id="language-select">
      <option value="English">🇬🇧 English</option>
      <option value="Hindi">🇮🇳 Hindi</option>
      <option value="Tamil">🇮🇳 Tamil</option>
      <option value="Telugu">🇮🇳 Telugu</option>
      <option value="Kannada">🇮🇳 Kannada</option>
      <option value="Malayalam">🇮🇳 Malayalam</option>
      <option value="Marathi">🇮🇳 Marathi</option>
      <option value="Bengali">🇮🇳 Bengali</option>
    </select>
  </div>
</div>

<div class="main">
  <div class="progress-wrap">
    <div class="progress-meta">
      <span class="progress-label" id="progress-text">Step 1 of 3 — Describe your problem</span>
      <span class="progress-pct" id="progress-pct">0%</span>
    </div>
    <div class="progress-track"><div class="progress-fill" id="progress-fill" style="width:5%"></div></div>
  </div>

  <!-- STEP 1 -->
  <div id="step1" class="card fade-in">
    <div class="card-header">
      <div class="section-title"><div class="title-icon ti-blue">📝</div> Describe Your Legal Problem</div>
    </div>
    <p class="step1-hero">Explain what happened in your own words. Include names, dates, amounts, location, and any evidence you have. The more detail you provide, the more specific the guidance.</p>
    <textarea class="problem-textarea" id="problem" placeholder="Example: My employer has not paid my salary for 3 months despite multiple requests. I have salary slips showing ₹28,000/month. My last payment was in October 2024. The company is in Pune, Maharashtra..."></textarea>
    <div class="error-box" id="err1"></div>
    <button class="btn btn-primary" onclick="submitProblem()">Continue →</button>
  </div>

  <!-- STEP 2 -->
  <div id="step2" style="display:none;">
    <div class="card fade-in">
      <div class="card-header">
        <div class="section-title"><div class="title-icon ti-blue">🧑‍⚖️</div> Legal Intake Interview</div>
      </div>
      <p style="font-size:14px;color:var(--gray-600);margin-bottom:16px;" id="step2-sub"></p>
      <div id="complexity-tag-wrap"></div>
      <div id="sensitive-notice-wrap"></div>
      <div class="round-label" id="round-label"></div>
      <div id="questions-container"></div>
      <div class="error-box" id="err2"></div>
      <button class="btn btn-primary" onclick="submitAnswers()">Submit Answers →</button>
    </div>
  </div>

  <!-- Loading -->
  <div class="loading" id="loading">
    <div class="spinner"></div>
    <p id="loading-text">Analyzing your problem...</p>
    <small id="loading-sub"></small>
  </div>

  <!-- STEP 3 RESULTS -->
  <div id="step3" class="result-section">

    <!-- Summary Card -->
    <div class="card fade-in">
      <div class="card-header">
        <div class="section-title"><div class="title-icon ti-blue">📋</div> Case Summary</div>
        <span class="case-badge" id="badge"></span>
      </div>
      <div class="case-title-main" id="case-title"></div>
      <div class="friendly-text" id="friendly"></div>
      <div class="inline-btns" id="next-actions"></div>
      <div class="inline-btns" style="margin-top:10px;">
        <button class="btn btn-ghost btn-sm" onclick="saveCurrentSession()">💾 Save Progress</button>
      </div>
    </div>

    <!-- Emergency -->
    <div class="emergency-card" id="emergency-card">
      <div class="emergency-title">🚨 High-Risk / Urgent Alert</div>
      <div class="emergency-text" id="emergency-text"></div>
    </div>

    <!-- Urgency -->
    <div class="card fade-in">
      <div class="card-header"><div class="section-title"><div class="title-icon ti-orange">⏱️</div> Urgency &amp; Timeline</div></div>
      <div class="info-box ib-blue">
        <div id="urgency-label" style="font-weight:800;margin-bottom:10px;font-size:15px;"></div>
        <ul class="info-list" id="timeline-list"></ul>
      </div>
    </div>

    <!-- Evidence + Missing -->
    <div class="grid2">
      <div class="card fade-in">
        <div class="card-header"><div class="section-title"><div class="title-icon ti-green">🧠</div> Evidence Strength</div></div>
        <div id="evidence-strength-wrap"></div>
      </div>
      <div class="card fade-in">
        <div class="card-header"><div class="section-title"><div class="title-icon ti-red">❗</div> Missing Information</div></div>
        <div id="missing-info-wrap"></div>
      </div>
    </div>

    <!-- Safety -->
    <div class="card fade-in">
      <div class="card-header"><div class="section-title"><div class="title-icon ti-orange">⚠️</div> Safety &amp; Risk Advice</div></div>
      <div class="info-box ib-orange"><ul class="info-list" id="safety-list"></ul></div>
    </div>

    <!-- Authority -->
    <div class="card fade-in">
      <div class="card-header"><div class="section-title"><div class="title-icon ti-purple">🏛️</div> Suggested Authority</div></div>
      <div class="authority-box" id="authority"></div>
    </div>

    <!-- Portals + Escalation -->
    <div class="grid2">
      <div class="card fade-in">
        <div class="card-header"><div class="section-title"><div class="title-icon ti-blue">🌐</div> Official Portals</div></div>
        <ul id="portal-list" style="list-style:none;"></ul>
      </div>
      <div class="card fade-in">
        <div class="card-header"><div class="section-title"><div class="title-icon ti-orange">⬆️</div> Escalation Path</div></div>
        <ul class="info-list" id="escalation-list"></ul>
      </div>
    </div>

    <!-- Steps Checklist -->
    <div class="card fade-in" id="steps-section">
      <div class="card-header">
        <div class="section-title"><div class="title-icon ti-blue">👣</div> Legal Workflow Checklist</div>
      </div>
      <div class="sub-progress" id="step-progress-label">0 of 0 steps completed</div>
      <div class="mini-track"><div class="mini-fill-navy" id="step-progress-fill" style="width:0%"></div></div>
      <ul id="steps" style="list-style:none;margin-top:16px;"></ul>
    </div>

    <!-- Case Tracker -->
    <div class="card fade-in">
      <div class="card-header"><div class="section-title"><div class="title-icon ti-green">📌</div> Case Progress Tracker</div></div>
      <div class="tracker-grid" id="progress-tracker"></div>
    </div>

    <!-- Documents -->
    <div class="card fade-in" id="docs-section">
      <div class="card-header"><div class="section-title"><div class="title-icon ti-gold">📁</div> Document Checklist</div></div>
      <div class="sub-progress" id="doc-progress-label">0 of 0 documents uploaded</div>
      <div class="mini-track"><div class="mini-fill-gold" id="doc-progress-fill" style="width:0%"></div></div>
      <ul id="docs" style="list-style:none;margin-top:16px;"></ul>
    </div>

    <!-- Document Readiness -->
    <div class="card fade-in">
      <div class="card-header"><div class="section-title"><div class="title-icon ti-green">✅</div> Document Readiness</div></div>
      <div class="readiness-box rb-red" id="readiness-box">
        <div class="readiness-status" id="readiness-status"></div>
        <div class="readiness-msg" id="readiness-msg"></div>
      </div>
    </div>

    <!-- Laws -->
    <div class="card fade-in">
      <div class="card-header"><div class="section-title"><div class="title-icon ti-purple">📜</div> Applicable Laws</div></div>
      <ul id="laws" style="list-style:none;"></ul>
    </div>

    <!-- Legal Terms -->
    <div class="card fade-in">
      <div class="card-header"><div class="section-title"><div class="title-icon ti-blue">📘</div> Plain-Language Legal Terms</div></div>
      <ul id="term-list" style="list-style:none;"></ul>
    </div>

    <!-- Complaint Draft CTA -->
    <div class="card fade-in" style="text-align:center;padding:36px 28px;">
      <div style="font-family:'Playfair Display',serif;font-size:20px;color:var(--navy);margin-bottom:8px;">Ready to draft your complaint?</div>
      <p style="font-size:14px;color:var(--gray-600);margin-bottom:24px;line-height:1.6;">Edit your complaint details, regenerate the draft, and download it as a PDF — all in one place.</p>
      <button class="draft-complaint-btn" onclick="openComplaintPage()">🧾 Open Complaint Draft Page</button>
    </div>

    <div style="text-align:center;margin-top:16px;">
      <button class="btn btn-outline" onclick="restart()">← Start a New Query</button>
    </div>
  </div>
</div>

<script>
let sessionId = Math.random().toString(36).substring(2);
let currentRound = 1;
let totalSteps = 0, doneSteps = 0;
let totalDocs = 0, uploadedDocs = 0;
let currentEditableFields = [];
let currentLanguage = "English";
let currentResultData = null;

document.getElementById("language-select").addEventListener("change", function(){
  currentLanguage = this.value;
});

function setProgress(pct, label){
  document.getElementById("progress-fill").style.width = pct + "%";
  document.getElementById("progress-pct").textContent = pct + "%";
  document.getElementById("progress-text").textContent = label;
}
function showLoading(text, sub){
  document.getElementById("loading-text").textContent = text;
  document.getElementById("loading-sub").textContent = sub || "";
  document.getElementById("loading").style.display = "block";
}
function hideLoading(){ document.getElementById("loading").style.display = "none"; }

async function submitProblem(){
  const problem = document.getElementById("problem").value.trim();
  const err = document.getElementById("err1");
  if(!problem){ err.textContent = "Please describe your problem first."; err.style.display = "block"; return; }
  err.style.display = "none";
  document.getElementById("step1").style.display = "none";
  showLoading("Analyzing your case and generating specific questions...", "");
  setProgress(15, "Step 2 of 3 — Answering questions");
  try{
    const res = await fetch("/start_session", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({problem, session_id: sessionId, language: currentLanguage})
    });
    const d = await res.json();
    if(res.status === 422 || d.error === "unrecognized"){
      hideLoading();
      document.getElementById("step1").style.display = "block";
      err.textContent = "We couldn't identify a matching legal case. Please describe the issue in more detail.";
      err.style.display = "block"; return;
    }
    if(!res.ok) throw new Error("Server error");
    hideLoading();
    renderQuestions(d.questions, d.complexity, d.reason, d.category, d.title, 1, d.sensitive);
  } catch(e){
    hideLoading();
    document.getElementById("step1").style.display = "block";
    err.textContent = "Something went wrong. Please try again."; err.style.display = "block";
  }
}

function renderQuestions(questions, complexity, reason, category, title, round, sensitive){
  currentRound = round;
  if(round === 1){
    let cx = complexity || "medium";
let label = reason || "";

if (sensitive) {
    cx = "complex";
    label = "Sensitive case requiring priority handling";
}

const icons = {
    simple:"🟢",
    medium:"🟡",
    complex:"🔴"
};

document.getElementById("complexity-tag-wrap").innerHTML =
  `<div class="complexity-tag cx-${cx}">${icons[cx]} ${cx === "complex"? "HIGH PRIORITY SENSITIVE CASE": cx.toUpperCase() + " CASE"} — ${label}</div>`;
    document.getElementById("step2-sub").textContent =
      `Identified as a ${category} case: "${title}". Questions below are specific to your situation.`;

    // Sensitive notice
    const sw = document.getElementById("sensitive-notice-wrap");
    if(sensitive){
      sw.innerHTML = `<div class="sensitive-notice">
        <span style="font-size:18px;flex-shrink:0">🔒</span>
        <span><strong>Confidential Case:</strong> This is a sensitive matter. We will NOT ask for your name, home address, or any identity-revealing details. Only case-relevant facts are requested.</span>
      </div>`;
    } else { sw.innerHTML = ""; }
  }
  document.getElementById("round-label").textContent =
    round === 1 ? `📋 Initial intake — ${questions.length} questions` : `🔍 Round ${round} — ${questions.length} clarifying question(s)`;
  const container = document.getElementById("questions-container");
  container.innerHTML = "";
  questions.forEach((qObj, i) => {
    const qText = qObj.question, inputType = qObj.input_type || "text", options = qObj.options || [];
    const div = document.createElement("div"); div.className = "qa-item";
    let inputHTML = inputType === "select"
      ? `<select id="ans${i}" class="qa-select"><option value="">Select...</option>${options.map(o=>`<option value="${o}">${o}</option>`).join("")}</select>`
      : `<input type="text" id="ans${i}" class="qa-input" placeholder="Type your answer here..." />`;
    div.innerHTML = `<div class="qa-label"><span class="q-num">${i+1}</span><span class="q-text">${qText}</span></div>${inputHTML}`;
    container.appendChild(div);
  });
  document.getElementById("step2").style.display = "block";
  window.scrollTo({top:0, behavior:"smooth"});
}

async function submitAnswers(){
  const container = document.getElementById("questions-container");
  const fields = container.querySelectorAll("input, select");
  const labels = container.querySelectorAll(".q-text");
  let answers = {}, hasEmpty = false;
  fields.forEach((field, i) => {
    const val = field.value.trim(), qText = labels[i] ? labels[i].textContent.trim() : "";
    if(!val) hasEmpty = true;
    else answers[qText] = val;
  });
  if(hasEmpty){
    const e = document.getElementById("err2");
    e.textContent = "Please answer all questions before continuing."; e.style.display = "block"; return;
  }
  document.getElementById("err2").style.display = "none";
  document.getElementById("step2").style.display = "none";
  showLoading("Reviewing your answers...", "");
  setProgress(Math.min(30 + currentRound * 15, 75), "Step 2 — Processing answers");
  try{
    const res = await fetch("/submit_answers", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({answers, session_id: sessionId})
    });
    const d = await res.json();
    if(!res.ok) throw new Error("Server error");
    hideLoading();
    if(d.ready){
      setProgress(85, "Step 3 — Generating legal guidance");
      showLoading("Generating your personalized legal guidance...", "This may take a moment.");
      const res2 = await fetch("/finalize", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({session_id: sessionId})
      });
      const result = await res2.json();
      if(!res2.ok) throw new Error("Server error");
      hideLoading(); renderResults(result);
    } else {
      renderQuestions(d.followups, d.complexity, d.reason, d.category, d.title, currentRound + 1, false);
    }
  } catch(e){
    hideLoading();
    document.getElementById("step2").style.display = "block";
    document.getElementById("err2").textContent = "Something went wrong. Please try again.";
    document.getElementById("err2").style.display = "block";
  }
}

function renderResults(d){
  currentResultData = d;
  setProgress(100, "Analysis complete ✓");

  const badge = document.getElementById("badge");
  badge.textContent = d.category;
  badge.className = "case-badge " + (String(d.category).toLowerCase() === "criminal" ? "badge-criminal" : "badge-civil");
  document.getElementById("case-title").textContent = d.title;
  document.getElementById("friendly").textContent = d.friendly;
  document.getElementById("authority").textContent = d.authority;
  document.getElementById("readiness-status").textContent = d.readiness.status;
  document.getElementById("readiness-msg").textContent = d.readiness.message;
  const rb = document.getElementById("readiness-box");
  rb.className = "readiness-box rb-" + (d.readiness.color || "red");

  if(d.urgency && d.urgency.high_risk){
    const ec = document.getElementById("emergency-card");
    ec.style.display = "block";
    document.getElementById("emergency-text").textContent = "This case involves urgent or high-risk elements. Prioritize your safety, preserve all evidence, and contact the relevant authority immediately. Emergency: 100 | Women's Helpline: 181";
  }

  document.getElementById("urgency-label").textContent = d.urgency.label;
  const tl = document.getElementById("timeline-list"); tl.innerHTML = "";
  (d.urgency.timeline||[]).forEach(item => {
    const li = document.createElement("li"); li.textContent = item; li.style.marginBottom = "6px"; tl.appendChild(li);
  });

  // Evidence
  const eWrap = document.getElementById("evidence-strength-wrap");
  const lvlClass = d.evidence.level === "Strong" ? "bm-green" : (d.evidence.level === "Moderate" ? "bm-orange" : "bm-red");
  eWrap.innerHTML = `<div style="margin-bottom:10px;"><span class="badge-mini ${lvlClass}">${d.evidence.level}</span></div>
    <div style="font-size:14px;line-height:1.65;color:var(--gray-600);margin-bottom:8px;">${d.evidence.note}</div>
    <div style="font-size:13px;color:var(--gray-400);"><b>Detected:</b> ${(d.evidence.evidence_found||[]).join(", ")||"None detected yet"}</div>`;

  // Missing
  const mWrap = document.getElementById("missing-info-wrap");
  mWrap.innerHTML = d.missing_info && d.missing_info.length
    ? `<ul class="info-list" style="color:var(--gray-600);">${d.missing_info.map(x=>`<li style="margin-bottom:6px;">${x}</li>`).join("")}</ul>`
    : `<div class="info-box ib-green" style="font-size:14px;">No major missing information detected.</div>`;

  // Safety
  const sl = document.getElementById("safety-list"); sl.innerHTML = "";
  (d.safety_advice||[]).forEach(item => {
    const li = document.createElement("li"); li.textContent = item; li.style.marginBottom = "6px"; sl.appendChild(li);
  });

  // Portals (dynamic)
  const pl = document.getElementById("portal-list"); pl.innerHTML = "";
  (d.portals||[]).forEach(p => {
    const li = document.createElement("li"); li.className = "portal-item";
    li.innerHTML = `<a class="portal-link" href="${p.url}" target="_blank">${p.label}</a>
      ${p.description ? `<div class="portal-desc">${p.description}</div>` : ""}
      <div class="portal-url">${p.url}</div>`;
    pl.appendChild(li);
  });

  // Escalation
  const el = document.getElementById("escalation-list"); el.innerHTML = "";
  (d.escalation||[]).forEach(item => {
    const li = document.createElement("li"); li.textContent = item; li.style.marginBottom = "8px"; el.appendChild(li);
  });

  // Next actions
  const na = document.getElementById("next-actions"); na.innerHTML = "";
  (d.next_actions||[]).forEach(action => {
    const btn = document.createElement("button"); btn.className = "btn btn-chip";
    btn.textContent = action.label;
    btn.onclick = () => handleAction(action.type);
    na.appendChild(btn);
  });

  // Steps
  totalSteps = d.step_help.length; doneSteps = 0;
  const stepsList = document.getElementById("steps"); stepsList.innerHTML = "";
  d.step_help.forEach((step, i) => {
    const li = document.createElement("li"); li.className = "step-item";
    li.innerHTML = `<div class="step-top">
      <input type="checkbox" class="step-cb" id="sc${i}" onchange="stepToggle(${i})">
      <span class="step-label" id="st${i}">${step.title}</span>
    </div>
    <div class="step-detail">
      <div class="step-detail-line"><b>How:</b> ${step.how}</div>
      <div class="step-detail-line"><b>Carry:</b> ${step.carry}</div>
      <div class="step-detail-line"><b>Outcome:</b> ${step.outcome}</div>
    </div>`;
    stepsList.appendChild(li);
  });
  updateStepProgress();

  // Progress tracker
  const pt = document.getElementById("progress-tracker"); pt.innerHTML = "";
  (d.progress_tracker||[]).forEach((item, idx) => {
    const wrap = document.createElement("div"); wrap.className = "tracker-item";
    wrap.innerHTML = `<input type="checkbox" id="pt${idx}"><label for="pt${idx}" style="cursor:pointer;">${item.label}</label>`;
    pt.appendChild(wrap);
  });

  // Documents
  totalDocs = d.documents.length; uploadedDocs = 0;
  const docsList = document.getElementById("docs"); docsList.innerHTML = "";
  d.documents.forEach((doc, i) => {
    const li = document.createElement("li"); li.className = "doc-item";
    li.innerHTML = `<div class="doc-icon">📄</div>
      <span class="doc-name">${doc}</span>
      <span class="doc-status ds-pending" id="ds${i}">Pending</span>
      <label style="cursor:pointer;">
        <input type="file" id="df${i}" style="display:none" onchange="docUploaded(${i},this)">
        <button class="doc-upload-btn" onclick="document.getElementById('df${i}').click();event.preventDefault();">Upload</button>
      </label>
      <span class="doc-filename" id="dfn${i}"></span>`;
    docsList.appendChild(li);
  });
  updateDocProgress();

  // Laws
  const lawsList = document.getElementById("laws"); lawsList.innerHTML = "";
  (d.provisions||[]).forEach(law => {
    const li = document.createElement("li"); li.className = "list-item"; li.textContent = law; lawsList.appendChild(li);
  });

  // Legal Terms
  const termList = document.getElementById("term-list"); termList.innerHTML = "";
  (d.legal_terms||[]).forEach(term => {
    const li = document.createElement("li"); li.className = "list-item";
    li.innerHTML = `<b>${term.term}</b><div style="font-size:13px;color:var(--gray-600);margin-top:3px;">${term.explanation}</div>`;
    termList.appendChild(li);
  });

  currentEditableFields = d.editable_fields || [];

  document.getElementById("step3").style.display = "block";
  window.scrollTo({top:0, behavior:"smooth"});
}

function handleAction(type){
  const targets = {scroll_steps:"steps-section", scroll_docs:"docs-section", open_complaint:null};
  if(type === "open_complaint"){ openComplaintPage(); return; }
  if(targets[type]) document.getElementById(targets[type]).scrollIntoView({behavior:"smooth"});
}

function openComplaintPage(){
  if(!currentResultData){ alert("Please complete the analysis first."); return; }
  const pageData = {
    session_id: sessionId,
    complaint_text: currentResultData.complaint_text || "",
    editable_fields: currentEditableFields
  };
  sessionStorage.setItem("complaint_page_data", JSON.stringify(pageData));
  window.open("/complaint_page", "_blank");
}

function stepToggle(idx){
  const cb = document.getElementById("sc"+idx), text = document.getElementById("st"+idx);
  if(cb.checked){ doneSteps++; text.classList.add("done"); }
  else { doneSteps--; text.classList.remove("done"); }
  updateStepProgress();
}
function updateStepProgress(){
  const pct = totalSteps > 0 ? Math.round((doneSteps/totalSteps)*100) : 0;
  document.getElementById("step-progress-label").textContent = `${doneSteps} of ${totalSteps} steps completed`;
  document.getElementById("step-progress-fill").style.width = pct + "%";
}

async function docUploaded(idx, input){
  if(!input.files || !input.files[0]) return;
  const fname = input.files[0].name;
  const status = document.getElementById("ds"+idx), fnameEl = document.getElementById("dfn"+idx);
  if(!status.classList.contains("ds-uploaded")){
    uploadedDocs++;
    status.textContent = "Uploaded ✓";
    status.className = "doc-status ds-uploaded";
  }
  fnameEl.textContent = "📎 " + fname;
  updateDocProgress();
  const uploadedNames = [];
  for(let i = 0; i < totalDocs; i++){
    const s = document.getElementById("ds"+i);
    const name = document.querySelectorAll("#docs .doc-name")[i]?.textContent || "";
    if(s && s.classList.contains("ds-uploaded")) uploadedNames.push(name);
  }
  try{
    const res = await fetch("/update_uploaded_docs", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({session_id: sessionId, uploaded_docs: uploadedNames})
    });
    const data = await res.json();
    if(data.readiness){
      document.getElementById("readiness-status").textContent = data.readiness.status;
      document.getElementById("readiness-msg").textContent = data.readiness.message;
      document.getElementById("readiness-box").className = "readiness-box rb-" + (data.readiness.color || "red");
    }
  } catch(e){}
}
function updateDocProgress(){
  const pct = totalDocs > 0 ? Math.round((uploadedDocs/totalDocs)*100) : 0;
  document.getElementById("doc-progress-label").textContent = `${uploadedDocs} of ${totalDocs} documents uploaded`;
  document.getElementById("doc-progress-fill").style.width = pct + "%";
}

async function saveCurrentSession(){
  try{
    const res = await fetch("/save_session", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({session_id: sessionId})
    });
    const d = await res.json();
    alert(d.message || "Progress saved.");
  } catch(e){ alert("Failed to save."); }
}

function restart(){
  sessionId = Math.random().toString(36).substring(2);
  currentRound = 1; totalSteps = 0; doneSteps = 0;
  totalDocs = 0; uploadedDocs = 0; currentEditableFields = []; currentResultData = null;
  document.getElementById("problem").value = "";
  document.getElementById("step1").style.display = "block";
  document.getElementById("step2").style.display = "none";
  document.getElementById("step3").style.display = "none";
  document.getElementById("loading").style.display = "none";
  document.getElementById("complexity-tag-wrap").innerHTML = "";
  document.getElementById("sensitive-notice-wrap").innerHTML = "";
  document.getElementById("emergency-card").style.display = "none";
  setProgress(5, "Step 1 of 3 — Describe your problem");
  window.scrollTo({top:0, behavior:"smooth"});
}
</script>
</body>
</html>
"""


# ── 9) ROUTES ────────────────────────────────────────────
@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "kb_size": len(kb)})


@app.route("/complaint_page")
def complaint_page():
    return render_template_string(COMPLAINT_HTML)


@app.route("/generate_pdf", methods=["POST"])
def generate_pdf():
    """Generate a PDF from the complaint text using reportlab."""
    from io import BytesIO
    from flask import make_response
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
    from reportlab.lib.enums import TA_LEFT, TA_CENTER

    try:
        data = request.json or {}
        complaint_text = (data.get("complaint_text") or "").strip()
        if not complaint_text:
            return jsonify({"error": "No complaint text provided"}), 400

        buf = BytesIO()
        doc = SimpleDocTemplate(
            buf, pagesize=A4,
            leftMargin=2.8*cm, rightMargin=2.5*cm,
            topMargin=2.5*cm, bottomMargin=2.5*cm
        )

        navy       = colors.HexColor("#0B1F45")
        gold       = colors.HexColor("#C9A84C")
        gray       = colors.HexColor("#4B5563")
        dark       = colors.HexColor("#1F2937")
        light_gray = colors.HexColor("#E5E7EB")

        title_style = ParagraphStyle("LGTitle",
            fontName="Helvetica-Bold", fontSize=16, textColor=navy,
            spaceAfter=4, alignment=TA_CENTER)
        sub_style = ParagraphStyle("LGSub",
            fontName="Helvetica", fontSize=10, textColor=gray,
            spaceAfter=12, alignment=TA_CENTER)
        body_style = ParagraphStyle("LGBody",
            fontName="Helvetica", fontSize=11, textColor=dark,
            spaceAfter=5, leading=17, alignment=TA_LEFT, wordWrap="LTR")
        note_style = ParagraphStyle("LGNote",
            fontName="Helvetica-Oblique", fontSize=9, textColor=gray,
            spaceAfter=0, alignment=TA_LEFT)

        story = []
        story.append(Paragraph("Legal Complaint Draft", title_style))
        story.append(Paragraph(
            "Generated on: " + datetime.now().strftime("%d %B %Y"), sub_style))
        story.append(HRFlowable(width="100%", thickness=1.5, color=gold, spaceAfter=10))
        story.append(Spacer(1, 4))

        for line in complaint_text.split("\n"):
            stripped = line.strip()
            if not stripped:
                story.append(Spacer(1, 4))
            else:
                # Convert to ASCII safely — keeps numbers, punctuation, section refs intact
                safe = stripped.encode("ascii", "replace").decode("ascii")
                # Escape XML special characters required by ReportLab Paragraph
                safe = safe.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                story.append(Paragraph(safe, body_style))

        story.append(Spacer(1, 16))
        story.append(HRFlowable(width="100%", thickness=0.5, color=light_gray, spaceAfter=6))
        story.append(Paragraph(
            "This document was generated by the Legal Guidance Platform. "
            "Please review and verify all details before submission.",
            note_style
        ))

        doc.build(story)
        pdf_bytes = buf.getvalue()

        response = make_response(pdf_bytes)
        response.headers["Content-Type"]        = "application/pdf"
        response.headers["Content-Disposition"] = "attachment; filename=legal_complaint_draft.pdf"
        response.headers["Content-Length"]      = str(len(pdf_bytes))
        response.headers["Cache-Control"]       = "no-cache, no-store, must-revalidate"
        return response

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "PDF generation failed: " + str(e)}), 500


@app.route("/start_session", methods=["POST"])
def start_session():
    try:
        data       = request.json or {}
        problem    = data.get("problem","")[:2000]
        session_id = data.get("session_id", secrets.token_hex(8))
        language   = data.get("language","English")

        matched = search(problem)
        if matched is None:
            return jsonify({"error":"unrecognized"}), 422

        result = get_initial_questions(problem, matched, language)

        sessions[session_id] = {
            "problem":       problem,
            "matched":       matched,
            "history":       [],
            "complexity":    result.get("complexity","medium"),
            "language":      language,
            "uploaded_docs": [],
            "editable_fields": []
        }

        return jsonify({
            "questions":  result.get("questions",[]),
            "complexity": result.get("complexity","medium"),
            "reason":     result.get("reason",""),
            "category":   matched["category"],
            "title":      matched["title"],
            "sensitive":  result.get("sensitive", False)
        })
    except Exception as e:
        return jsonify({"error":str(e)}), 500


@app.route("/submit_answers", methods=["POST"])
def submit_answers():
    try:
        data       = request.json or {}
        session_id = data.get("session_id","")
        answers    = data.get("answers",{})

        if session_id not in sessions:
            return jsonify({"error":"Session not found"}), 400

        sess     = sessions[session_id]
        problem  = sess["problem"]
        matched  = sess["matched"]
        language = sess.get("language","English")

        for q, a in answers.items():
            sess["history"].append({"question":q,"answer":a})

        if len(sess["history"]) >= 8:
            return jsonify({"ready":True})

        eval_result = evaluate_answers_and_followup(problem, matched, sess["history"], language)

        if eval_result.get("ready", True):
            return jsonify({"ready":True})
        else:
            return jsonify({
                "ready":      False,
                "followups":  eval_result.get("followups",[])[:2],
                "complexity": sess["complexity"],
                "reason":     "Clarifying missing facts",
                "category":   matched["category"],
                "title":      matched["title"]
            })
    except Exception as e:
        return jsonify({"error":str(e)}), 500


@app.route("/finalize", methods=["POST"])
def finalize():
    try:
        data       = request.json or {}
        session_id = data.get("session_id","")
        if session_id not in sessions:
            return jsonify({"error":"Session not found"}), 400
        sess    = sessions[session_id]
        payload = build_finalized_payload(sess)
        sess["finalized_result"] = payload
        return jsonify(payload)
    except Exception as e:
        return jsonify({"error":str(e)}), 500


@app.route("/update_uploaded_docs", methods=["POST"])
def update_uploaded_docs():
    try:
        data       = request.json or {}
        session_id = data.get("session_id","")
        uploaded   = data.get("uploaded_docs",[])
        if session_id not in sessions:
            return jsonify({"error":"Session not found"}), 400
        sess = sessions[session_id]
        sess["uploaded_docs"] = uploaded
        readiness = evaluate_readiness(uploaded, sess["matched"]["documents"])
        lang = sess.get("language","English")
        if lang != "English":
            readiness["status"]  = translate_text(readiness["status"], lang)
            readiness["message"] = translate_text(readiness["message"], lang)
        return jsonify({"success":True,"readiness":readiness})
    except Exception as e:
        return jsonify({"error":str(e)}), 500


@app.route("/update_complaint_fields", methods=["POST"])
def update_complaint_fields():
    try:
        data            = request.json or {}
        session_id      = data.get("session_id","")
        editable_fields = data.get("editable_fields",[])
        if session_id not in sessions:
            return jsonify({"error":"Session not found"}), 400
        sess = sessions[session_id]
        sess["editable_fields"] = editable_fields
        fm = {item["key"]:item["value"] for item in editable_fields}
        matched   = sess["matched"]
        authority = matched.get("authority","Concerned Authority")
        subject   = f"Complaint regarding {matched.get('title','legal issue')}"
        complaint = f"""To,
The {authority}

Subject: {subject}

Respected Sir/Madam,

I, {fm.get('name','Complainant')}, respectfully submit this complaint.

1. Nature of Issue:
{matched.get('title','')} ({matched.get('category','')})

2. Date / Timeline:
{fm.get('date_of_incident','Not specified')}

3. Location:
{fm.get('location','Not specified')}

4. Other Party:
{fm.get('other_party','Not specified')}

5. Facts:
{sess['problem']}

6. Evidence:
{fm.get('evidence','Not specified')}

7. Relief Requested:
{fm.get('relief','Appropriate legal action')}

Yours faithfully,
{fm.get('name','Complainant')}
Date: {datetime.now().strftime('%d-%m-%Y')}"""
        lang = sess.get("language","English")
        if lang != "English":
            complaint = translate_text(complaint, lang)
        return jsonify({"success":True,"complaint_text":complaint})
    except Exception as e:
        return jsonify({"error":str(e)}), 500


@app.route("/save_session", methods=["POST"])
def save_session():
    try:
        data       = request.json or {}
        session_id = data.get("session_id","")
        if session_id not in sessions:
            return jsonify({"error":"Session not found"}), 400
        sess = sessions[session_id]
        if "finalized_result" not in sess:
            try: sess["finalized_result"] = build_finalized_payload(sess)
            except: pass
        path = os.path.join(PROJECT_DIR, f"session_{session_id}.json")
        with open(path,"w",encoding="utf-8") as f:
            json.dump(sess, f, ensure_ascii=False, indent=2)
        return jsonify({"success":True,"message":f"Saved. Session ID: {session_id}"})
    except Exception as e:
        return jsonify({"error":str(e)}), 500


# ── 10) LAUNCH (local dev only — Render uses gunicorn via Procfile) ──────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
