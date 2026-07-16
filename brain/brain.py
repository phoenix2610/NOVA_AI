#!/usr/bin/env python3
"""
NOVA Brain — Dual-Model Orchestrator
Receives classified intent from router, injects live context + RAG,
calls the appropriate LLM via Ollama (Primary: Llama 3.2 1B for speed,
Secondary: Mistral 7B for depth), handles proactive follow-up,
hands off to executioner, drives n8n browser automations.
"""

import os
import re
import sys
import json
import time
import socket
import logging
import threading
import subprocess
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv  # pip install python-dotenv

# Load project-level .env (~/nova/.env)
load_dotenv(os.path.expanduser("~/nova/.env"))

# ── LLM Router (Llama 3.2 1B ↔ Mistral 7B) ───────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from llm_router import select_model, model_label, PRIMARY_MODEL, SECONDARY_MODEL

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [BRAIN] %(message)s",
    handlers=[
        logging.FileHandler(os.path.expanduser("~/nova/logs/brain.log")),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("brain")

# ── Paths / Config ────────────────────────────────────────────────────────────
OLLAMA_URL       = "http://127.0.0.1:11434/api/generate"
TTS_SOCKET       = "/tmp/nova_tts.sock"
EXEC_SOCKET      = "/tmp/nova_executioner.sock"
RAG_SOCKET       = "/tmp/nova_rag.sock"
CHAT_LOG         = os.path.expanduser("~/nova/logs/chat.jsonl")
BRAIN_SOCKET     = "/tmp/nova_brain.sock"
MEMORY_DIR       = os.path.expanduser("~/nova/memory")
PROFILE_PATH     = os.path.join(MEMORY_DIR, "user_profile.json")

# ── N8N Automation ────────────────────────────────────────────────────────────
# Webhook base URL — defaults to localhost n8n instance
N8N_BASE_URL         = os.environ.get("NOVA_N8N_BASE", "http://localhost:5678")
N8N_API_KEY          = os.environ.get("n8n_API", "")   # loaded from ~/nova/.env
N8N_WEB_SEARCH_PATH  = "/webhook/nova-web-search"
N8N_RECORDS_PATH     = "/webhook/nova-personal-records"

os.makedirs(os.path.expanduser("~/nova/logs/"), exist_ok=True)

# ── Time-sensitive keywords (always trigger web RAG) ─────────────────────────
TIME_SENSITIVE_KEYWORDS = [
    "current", "today", "tonight", "now", "latest", "recent",
    "price", "stock", "market", "news", "weather", "score",
    "live", "right now", "this week", "this month", "update"
]

# ── Proactivity triggers — maps intent/topic to follow-up suggestion ──────────
PROACTIVE_TRIGGERS = {
    "stock|price|market|share|nasdaq|nifty|portfolio":
        "Shall I pull up the full portfolio rankings as well, sir?",
    "volume|audio|sound|mute|unmute":
        "Want me to set a default volume level for sessions, sir?",
    "weather":
        "Shall I check the forecast for the rest of the week too, sir?",
    "file|convert|pdf|docx|mp4|mkv":
        "Should I open the output location when it's done, sir?",
    "install|download|update":
        "Want me to verify the installation completed cleanly, sir?",
    "code|script|python|function|error|bug":
        "Want me to run a quick syntax check on that, sir?",
}

# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are NOVA — a sharp, witty, situation-aware AI assistant running locally on the user's machine.

Personality:
- Charming, confident, and concise. You speak like a brilliant friend, not a manual.
- Humorous when appropriate, laser-focused when it matters.
- Address the user as "sir" naturally, not robotically.
- Never pad responses. Say what needs to be said, nothing more.

Rules:
- Keep answers SHORT. 2-4 sentences for most queries. Expand only when technical depth is genuinely needed.
- If you're uncertain about something factual, say so directly — don't guess.
- CRITICAL: Only wrap content in ```ep ... ``` when it is a REAL, RUNNABLE EPL (English Programming Language) script. NEVER put a prose explanation inside a code block.
- If the request is classified with an operation intent (e.g. browser_op, system_op, media_op, file_op), you MUST write a real, runnable EPL script wrapped in ```ep ... ``` to perform the action.
- Use EPL's v1.8 System Runtime for automation (e.g., `set out as run "cmd"`, `read file "path"`, `write val to file "path"`). To open a file or URL, use `run "xdg-open <path>"` instead of a python script.
- Never mention that you're an AI or that you're running on Mistral. You are NOVA.
- Confidence check: end your response with [CONFIDENT] or [UNSURE] on a new line. This is parsed by the system, never spoken aloud.
"""

# ── Helpers ───────────────────────────────────────────────────────────────────
def speak(text: str, priority: int = 1):
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.connect(TTS_SOCKET)
            s.sendall(json.dumps({"priority": priority, "text": text}).encode())
    except Exception:
        log.warning("TTS unavailable")

def send_to_exec(script: str, operations: list, narration: str = "") -> dict:
    try:
        payload = {
            "action": "execute",
            "script": script,
            "operations": operations,
            "narration": narration
        }
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.connect(EXEC_SOCKET)
            s.sendall((json.dumps(payload) + "\n").encode())
            raw = b""
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                raw += chunk
                if raw.endswith(b"\n"):
                    break
        return json.loads(raw.decode())
    except Exception as e:
        return {"success": False, "error": str(e)}

# ── Live System Snapshot (read-only hardware/system context) ─────────────────
def get_system_snapshot() -> dict:
    """
    Read-only: gather hardware and session context.
    This is view-only — no writes, no exec. Used purely to derive context.
    """
    snapshot = {
        "time": datetime.now().strftime("%H:%M, %d %b %Y"),
        "foreground_app": "unknown",
        "active_window": "unknown",
        "cpu_percent": 0,
        "ram_percent": 0,
        "volume": "unknown",
        "open_windows": [],
        "top_processes": [],
        "disk_free_gb": 0,
        "battery": "N/A",
        "network": "unknown",
        "gpu_temp": "N/A",
    }

    # ── CPU / RAM / Disk ─────────────────────────────────────────────────────
    try:
        import psutil
        snapshot["cpu_percent"] = psutil.cpu_percent(interval=0.1)
        vm = psutil.virtual_memory()
        snapshot["ram_percent"] = vm.percent
        snapshot["ram_used_gb"] = round(vm.used / 1e9, 1)
        snapshot["ram_total_gb"] = round(vm.total / 1e9, 1)
        snapshot["disk_free_gb"] = round(psutil.disk_usage('/').free / 1e9, 1)
        # Top 3 CPU-hungry processes (view-only)
        procs = sorted(psutil.process_iter(["name", "cpu_percent"]),
                       key=lambda p: p.info.get("cpu_percent") or 0, reverse=True)[:3]
        snapshot["top_processes"] = [p.info["name"] for p in procs]
        # Battery
        batt = psutil.sensors_battery()
        if batt:
            snapshot["battery"] = f"{int(batt.percent)}% {'charging' if batt.power_plugged else 'discharging'}"
    except Exception:
        pass

    # ── Active window / all open clients via hyprctl ──────────────────────────
    try:
        r = subprocess.run(["hyprctl", "activewindow", "-j"],
                           capture_output=True, text=True, timeout=2)
        if r.returncode == 0:
            w = json.loads(r.stdout)
            snapshot["active_window"] = w.get("title", "unknown")
            snapshot["foreground_app"] = w.get("class", "unknown")
    except Exception:
        pass

    try:
        r = subprocess.run(["hyprctl", "clients", "-j"],
                           capture_output=True, text=True, timeout=2)
        if r.returncode == 0:
            clients = json.loads(r.stdout)
            snapshot["open_windows"] = [c.get("title", "")[:60] for c in clients[:8]]
    except Exception:
        pass

    # ── Volume ────────────────────────────────────────────────────────────────
    try:
        r = subprocess.run(["wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"],
                           capture_output=True, text=True, timeout=2)
        snapshot["volume"] = r.stdout.strip()
    except Exception:
        pass

    # ── Network interface ─────────────────────────────────────────────────────
    try:
        r = subprocess.run(["ip", "route", "get", "1.1.1.1"],
                           capture_output=True, text=True, timeout=2)
        if r.returncode == 0:
            for part in r.stdout.split():
                if part.startswith("dev"):
                    snapshot["network"] = r.stdout.split()[r.stdout.split().index("dev") + 1]
                    break
    except Exception:
        pass

    # ── GPU temp (amd/intel integrated via hwmon) ─────────────────────────────
    try:
        r = subprocess.run(["sensors", "-j"], capture_output=True, text=True, timeout=2)
        if r.returncode == 0:
            sensors = json.loads(r.stdout)
            for chip, data in sensors.items():
                for key, val in data.items():
                    if "temp" in key.lower() and isinstance(val, dict):
                        for k2, v2 in val.items():
                            if "input" in k2 and isinstance(v2, (int, float)):
                                snapshot["gpu_temp"] = f"{v2}°C"
                                break
    except Exception:
        pass

    return snapshot


# ── User Profile Context ───────────────────────────────────────────────────────
def get_profile_context() -> str:
    """Load user_profile.json for persistent context injection."""
    try:
        with open(PROFILE_PATH) as f:
            p = json.load(f)
        lines = ["[User profile]"]
        for k, v in p.items():
            if not k.startswith("_") and v:
                lines.append(f"• {k}: {v}")
        return "\n".join(lines)
    except Exception:
        return ""

# ── RAG Query (parallel, non-blocking) ───────────────────────────────────────
def query_rag_async(query: str, callback):
    """Fire RAG query in background thread. Calls callback(result) when done."""
    def _run():
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.connect(RAG_SOCKET)
                s.sendall((json.dumps({"query": query}) + "\n").encode())
                raw = b""
                while True:
                    chunk = s.recv(4096)
                    if not chunk:
                        break
                    raw += chunk
                    if raw.endswith(b"\n"):
                        break
                result = json.loads(raw.decode())
                callback(result)
        except Exception:
            callback(None)  # RAG unavailable, silent skip
    threading.Thread(target=_run, daemon=True).start()

# ── Proactivity Check ─────────────────────────────────────────────────────────
def get_proactive_followup(query: str, response: str) -> str | None:
    combined = (query + " " + response).lower()
    for pattern, suggestion in PROACTIVE_TRIGGERS.items():
        if re.search(pattern, combined):
            return suggestion
    return None

# ── N8N Web RAG ───────────────────────────────────────────────────────────────
def trigger_web_rag(query: str, mode: str = "search", url: str = "") -> str:
    """
    Trigger the n8n browser-automation workflow.
    mode: 'search' | 'fetch_url' | 'news'
    Authenticates using the n8n_API key from ~/nova/.env.
    """
    try:
        import urllib.request
        webhook_url = f"{N8N_BASE_URL}{N8N_WEB_SEARCH_PATH}"
        payload = json.dumps({"query": query, "mode": mode, "url": url}).encode()
        headers = {"Content-Type": "application/json"}
        if N8N_API_KEY:
            headers["X-N8N-API-KEY"] = N8N_API_KEY
        req = urllib.request.Request(
            webhook_url, data=payload, headers=headers, method="POST"
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode())
            return result.get("summary", "")
    except Exception as e:
        log.warning(f"N8N web RAG unavailable: {e}")
        return ""


# ── N8N Personal Records ──────────────────────────────────────────────────────
def n8n_records(action: str, **kwargs) -> dict:
    """
    Read/write personal records via the n8n personal-records workflow.
    action: 'read_notes' | 'write_note' | 'read_reminders' | 'write_reminder'
    """
    try:
        import urllib.request
        webhook_url = f"{N8N_BASE_URL}{N8N_RECORDS_PATH}"
        payload = json.dumps({"action": action, **kwargs}).encode()
        headers = {"Content-Type": "application/json"}
        if N8N_API_KEY:
            headers["X-N8N-API-KEY"] = N8N_API_KEY
        req = urllib.request.Request(
            webhook_url, data=payload, headers=headers, method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        log.warning(f"N8N records unavailable: {e}")
        return {}

# ── Ollama Call — model is now a parameter ────────────────────────────────────
def call_ollama(prompt: str, system: str = SYSTEM_PROMPT, model: str = PRIMARY_MODEL) -> str:
    """
    Call Ollama with the specified model.
    Primary (llama3.2:1b)       → fast, ~2-5s
    Secondary (mistral:7b-...) → thorough, ~15-30s
    """
    import urllib.request
    # 1B model benefits from slightly fewer tokens to stay crisp
    max_tokens = 200 if model == PRIMARY_MODEL else 400
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": max_tokens,
            "top_p": 0.9,
            "repeat_penalty": 1.1
        }
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    # 1B model is much faster; give secondary more time
    timeout = 60 if model == PRIMARY_MODEL else 120
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
        return data.get("response", "").strip()

# ── EPL code heuristic — must have at least one executable keyword ─────────
_EPL_KEYWORDS = re.compile(
    r"\b(set |run |read |write |append |serve |reply |send |fetch |post |json |now)\b"
)

# ── Response Parser ───────────────────────────────────────────────────────────
def parse_response(raw: str) -> tuple[str, bool, str | None]:
    """Returns (spoken_text, is_confident, script_or_none)"""
    confident = True
    if "[UNSURE]" in raw:
        confident = False
    raw = raw.replace("[CONFIDENT]", "").replace("[UNSURE]", "").strip()

    # Extract script if present — ONLY if it actually looks like EPL code.
    script = None
    script_match = re.search(r"```(?:ep|epl)\n(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    if script_match:
        candidate = script_match.group(1).strip()
        if _EPL_KEYWORDS.search(candidate):
            # Genuine executable code
            script = candidate
            raw = raw[:script_match.start()].strip()
        else:
            # Prose masquerading as code — strip the fences and speak it
            log.warning("parse_response: code block contained prose, treating as spoken text")
            raw = raw[:script_match.start()].strip() + " " + candidate
            raw = raw.strip()

    return raw, confident, script

# ── Chat Log ──────────────────────────────────────────────────────────────────
def log_exchange(session_id: str, intent: str, query: str,
                 response: str, tags: list, rag_hit: bool):
    entry = {
        "id": f"{session_id}_{int(time.time())}",
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "intent": intent,
        "topic_tags": tags,
        "user": query,
        "assistant": response,
        "rag_hit": rag_hit
    }
    with open(CHAT_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")

# ── Tag Extractor (simple keyword → tags) ────────────────────────────────────
def extract_tags(text: str) -> list:
    stopwords = {"the","a","an","is","are","was","were","what","how",
                 "why","when","where","who","can","could","would","should",
                 "do","does","did","i","my","me","you","your","it","its",
                 "this","that","and","or","but","in","on","at","to","for"}
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    return list({w for w in words if w not in stopwords})[:8]

# ── Main Brain Handler ────────────────────────────────────────────────────────
def handle(request: dict, session_id: str) -> None:
    query      = request.get("query", "").strip()
    intent     = request.get("intent", "conversation")
    confidence = float(request.get("confidence", 1.0))

    if not query:
        return

    log.info(f"[{intent}] {query[:60]}")

    snapshot  = get_system_snapshot()
    tags      = extract_tags(query)
    rag_result = {"hit": False, "context": ""}

    # ── Parallel RAG ─────────────────────────────────────────────────────────
    rag_event = threading.Event()
    rag_data  = {}

    def on_rag(result):
        if result and result.get("hit"):
            rag_data.update(result)
        rag_event.set()

    query_rag_async(query, on_rag)

    # ── Decide web RAG ────────────────────────────────────────────────────────
    query_lower = query.lower()
    needs_web   = any(kw in query_lower for kw in TIME_SENSITIVE_KEYWORDS)

    web_context = ""
    if needs_web:
        speak("Give me a moment sir, let me check that for you.", priority=0)
        web_context = trigger_web_rag(query)
        if web_context:
            log.info("N8N web RAG returned context")

    # ── Pick model (Llama 1B vs Mistral 7B) ──────────────────────────────────
    chosen_model = select_model(
        intent=intent,
        query=query,
        has_web_context=bool(web_context)
    )
    log.info(f"Using {model_label(chosen_model)} ({chosen_model})")

    # ── Load user profile context ─────────────────────────────────────────────
    profile_ctx = get_profile_context()

    # ── Build prompt ──────────────────────────────────────────────────────────
    context_block = ""
    if profile_ctx:
        context_block += f"\n{profile_ctx}\n"
    if web_context:
        context_block += f"\n[Web context]\n{web_context}\n"

    intent_block = ""
    if intent not in ("conversation", "knowledge_query"):
        intent_block = f"\n[Intent]\n{intent}\n"

    prompt = f"""[System snapshot]
Time: {snapshot['time']}
Active app: {snapshot['foreground_app']} — {snapshot['active_window']}
CPU: {snapshot['cpu_percent']}% | RAM: {snapshot['ram_percent']}%
Volume: {snapshot['volume']}
Top processes: {', '.join(snapshot.get('top_processes', []))}
Disk free: {snapshot.get('disk_free_gb', '?')} GB | Battery: {snapshot.get('battery', 'N/A')}
Open windows: {', '.join(snapshot.get('open_windows', [])[:4])}
{intent_block}{context_block}
[User query]
{query}"""

    # ── Call primary brain ────────────────────────────────────────────────────
    try:
        raw = call_ollama(prompt, model=chosen_model)
    except Exception as e:
        log.error(f"Ollama error ({model_label(chosen_model)}): {e}")
        speak("Sorry sir, the brain is unavailable right now.")
        return

    spoken, is_confident, script = parse_response(raw)

    # ── Primary unsure → escalate to secondary ────────────────────────────────
    if not is_confident and chosen_model == PRIMARY_MODEL:
        log.info("Primary brain unsure — escalating to Secondary (Mistral 7B)")
        speak("Let me think harder on that, sir.", priority=0)
        try:
            raw = call_ollama(prompt, model=SECONDARY_MODEL)
            spoken, is_confident, script = parse_response(raw)
            log.info(f"Secondary brain responded (confident={is_confident})")
        except Exception as e:
            log.warning(f"Secondary escalation failed: {e}")
            # Keep primary's answer

    # ── If still unsure and no web context — trigger n8n web RAG ─────────────
    if not is_confident and not web_context:
        log.info("Still unsure — triggering N8N web RAG")
        speak("Let me double-check that for you, sir.", priority=0)
        web_context = trigger_web_rag(query)
        if web_context:
            prompt_retry = prompt + f"\n[Additional context]\n{web_context}\n\nRevise your answer with this context."
            try:
                raw = call_ollama(prompt_retry, model=SECONDARY_MODEL)
                spoken, is_confident, script = parse_response(raw)
            except Exception:
                pass  # keep answer we have

    # ── Wait for RAG (max 2s — don't block TTS) ───────────────────────────────
    rag_event.wait(timeout=2.0)

    # ── Speak primary answer ──────────────────────────────────────────────────
    speak(spoken)

    # ── RAG context append ────────────────────────────────────────────────────
    if rag_data.get("hit") and rag_data.get("context"):
        rag_context = rag_data["context"]
        speak(rag_context, priority=1)
        rag_result = {"hit": True, "context": rag_context}

    # ── Execute script if brain generated one ─────────────────────────────────
    if script:
        log.info("Brain generated script — sending to executioner")
        speak("On it, sir.", priority=0)
        exec_result = send_to_exec(
            script,
            operations=["write_workspace", "network_fetch"],
            narration=f"Sir, executing: {query[:50]}"
        )
        if exec_result.get("success") and exec_result.get("stdout"):
            speak(exec_result["stdout"])
        elif not exec_result.get("success"):
            speak(f"Sir, the execution ran into an issue. {exec_result.get('stderr','')[:80]}")

    # ── Proactive follow-up ───────────────────────────────────────────────────
    followup = get_proactive_followup(query, spoken)
    if followup:
        time.sleep(0.8)  # brief natural pause
        speak(followup, priority=1)

    # ── Log exchange ──────────────────────────────────────────────────────────
    log_exchange(session_id, intent, query, spoken, tags, rag_result["hit"])

# ── Socket Server ─────────────────────────────────────────────────────────────
def serve():
    if os.path.exists(BRAIN_SOCKET):
        os.unlink(BRAIN_SOCKET)

    import uuid
    session_id = str(uuid.uuid4())[:8]
    log.info(f"Brain online | session: {session_id}")

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(BRAIN_SOCKET)
    os.chmod(BRAIN_SOCKET, 0o600)
    server.listen(3)

    try:
        while True:
            conn, _ = server.accept()
            try:
                raw = b""
                while True:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    raw += chunk
                    if raw.endswith(b"\n"):
                        break
                request = json.loads(raw.decode())
                conn.close()
                # Handle in thread so socket stays responsive
                threading.Thread(
                    target=handle,
                    args=(request, session_id),
                    daemon=True
                ).start()
            except Exception as e:
                log.error(f"Request error: {e}")
                try:
                    conn.close()
                except Exception:
                    pass
    except KeyboardInterrupt:
        log.info("Brain shutting down")
    finally:
        server.close()
        if os.path.exists(BRAIN_SOCKET):
            os.unlink(BRAIN_SOCKET)

if __name__ == "__main__":
    # Can also be called directly for testing:
    # echo '{"query":"how is chocolate made","intent":"knowledge_query"}' | python3 brain.py
    import select
    _piped_data = None
    if not sys.stdin.isatty():
        # Non-interactive: check if stdin actually has content (pipe vs /dev/null)
        _ready, _, _ = select.select([sys.stdin], [], [], 0.05)
        if _ready:
            _piped_data = sys.stdin.read().strip()

    if _piped_data:
        # Data piped in — one-shot test mode
        try:
            data = json.loads(_piped_data)
            handle(data, "test_session")
        except Exception as e:
            print(f"Error: {e}")
    else:
        # Normal daemon mode (systemd or interactive terminal)
        serve()

