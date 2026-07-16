#!/usr/bin/env python3
"""
NOVA Router Bridge — Pipeline glue
STT final transcript → Router (DeBERTa) → Brain or Executioner
Loads router, classifies, routes, then ejects itself.
"""

import os
import sys
import json
import socket
import logging
import threading
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ROUTER] %(message)s",
    handlers=[
        logging.FileHandler(os.path.expanduser("~/nova/logs/router.log")),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("router")

ROUTER_MODEL_PATH = os.path.expanduser("~/models/nova-router-v2")
STT_SOCKET        = "/tmp/nova_stt.sock"
BRAIN_SOCKET      = "/tmp/nova_brain.sock"
EXEC_SOCKET       = "/tmp/nova_executioner.sock"
TTS_SOCKET        = "/tmp/nova_tts.sock"
ROUTER_SOCKET     = "/tmp/nova_router.sock"

CONFIDENCE_THRESHOLD = 0.80

# ── Known system ops (direct to executioner, no brain needed) ─────────────────
SYSTEM_OP_PATTERNS = {
    "volume up":        "import subprocess; subprocess.run(['wpctl','set-volume','@DEFAULT_AUDIO_SINK@','5%+'])",
    "volume down":      "import subprocess; subprocess.run(['wpctl','set-volume','@DEFAULT_AUDIO_SINK@','5%-'])",
    "mute":             "import subprocess; subprocess.run(['wpctl','set-mute','@DEFAULT_AUDIO_SINK@','toggle'])",
    "unmute":           "import subprocess; subprocess.run(['wpctl','set-mute','@DEFAULT_AUDIO_SINK@','0'])",
}

def speak(text: str, priority: int = 1):
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.connect(TTS_SOCKET)
            s.sendall(json.dumps({"priority": priority, "text": text}).encode())
    except Exception:
        pass

def send_to_brain(query: str, intent: str, confidence: float):
    try:
        payload = {"query": query, "intent": intent, "confidence": confidence}
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.connect(BRAIN_SOCKET)
            s.sendall((json.dumps(payload) + "\n").encode())
    except Exception as e:
        log.error(f"Brain unreachable: {e}")
        speak("Sir, the brain is offline.")

def send_to_exec(script: str, ops: list, narration: str = ""):
    try:
        payload = {"action": "execute", "script": script,
                   "operations": ops, "narration": narration}
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

# ── Load router model (done once, stays in memory while routing) ──────────────
_model = None
_tokenizer = None
_labels = None

def load_router():
    global _model, _tokenizer, _labels
    if _model is not None:
        return
    log.info("Loading router model...")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    _tokenizer = AutoTokenizer.from_pretrained(
        ROUTER_MODEL_PATH, local_files_only=True, use_fast=False
    )
    _model = AutoModelForSequenceClassification.from_pretrained(
        ROUTER_MODEL_PATH, local_files_only=True
    )
    _model.eval()

    label_map_path = os.path.join(ROUTER_MODEL_PATH, "label_map.json")
    with open(label_map_path) as f:
        _labels = json.load(f)

    log.info(f"Router loaded | {len(_labels)} intents")

def classify(query: str) -> tuple[str, float]:
    import torch
    load_router()
    inputs = _tokenizer(
        query, return_tensors="pt",
        truncation=True, padding=True, max_length=64
    )
    with torch.no_grad():
        logits = _model(**inputs).logits
        probs  = torch.softmax(logits, dim=-1)
        top_id = probs.argmax().item()
        conf   = probs[0][top_id].item()

    intent = _labels[str(top_id)]
    return intent, conf

# ── Multi-intent detection (simple conjunction split) ────────────────────────
def split_intents(query: str) -> list[str]:
    """Split compound queries on conjunctions."""
    import re
    parts = re.split(
        r'\band\b|\balso\b|\bplus\b|\badditionally\b|\bas well\b',
        query, flags=re.IGNORECASE
    )
    return [p.strip() for p in parts if len(p.strip()) > 3]

# ── Quick system op lookup ────────────────────────────────────────────────────
def match_system_op(query: str) -> str | None:
    q = query.lower()
    for pattern, script in SYSTEM_OP_PATTERNS.items():
        if pattern in q:
            return script
    return None

# ── Route a single query segment ─────────────────────────────────────────────
def route_single(query: str):
    intent, conf = classify(query)
    log.info(f"'{query[:50]}' → {intent} ({conf:.2%})")

    # High-confidence system op — check for direct script match first
    if intent == "system_op" and conf >= CONFIDENCE_THRESHOLD:
        script = match_system_op(query)
        if script:
            result = send_to_exec(
                script,
                ops=["audio_control"],
                narration=f"Sir, {query.lower()}."
            )
            if result.get("success"):
                speak(f"Done, sir.")
            else:
                speak(f"Sir, that didn't work. {result.get('stderr','')[:60]}")
            return

    # Everything else — brain handles it
    if conf >= CONFIDENCE_THRESHOLD:
        send_to_brain(query, intent, conf)
    else:
        log.info(f"Low confidence ({conf:.2%}) — sending to brain as conversation")
        send_to_brain(query, "conversation", conf)

# ── Main route handler ────────────────────────────────────────────────────────
def route(query: str):
    if not query.strip():
        return

    parts = split_intents(query)

    if len(parts) > 1:
        log.info(f"Multi-intent detected: {len(parts)} parts")
        # Fire all parts — system ops immediately, brain tasks queued
        threads = []
        for part in parts:
            t = threading.Thread(target=route_single, args=(part,), daemon=True)
            t.start()
            threads.append(t)
        # Don't join — fire and forget, pipeline is async
    else:
        route_single(query)

# ── Socket server (listens for STT final transcripts) ────────────────────────
def serve():
    if os.path.exists(ROUTER_SOCKET):
        os.unlink(ROUTER_SOCKET)

    load_router()
    log.info("Router bridge online")

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(ROUTER_SOCKET)
    os.chmod(ROUTER_SOCKET, 0o600)
    server.listen(5)

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
                conn.close()
                data  = json.loads(raw.decode())
                query = data.get("transcript", data.get("query", ""))
                threading.Thread(target=route, args=(query,), daemon=True).start()
            except Exception as e:
                log.error(f"Route error: {e}")
                try:
                    conn.close()
                except Exception:
                    pass
    except KeyboardInterrupt:
        log.info("Router shutting down")
    finally:
        server.close()
        if os.path.exists(ROUTER_SOCKET):
            os.unlink(ROUTER_SOCKET)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Quick CLI test: python3 router_bridge.py "open youtube"
        query = " ".join(sys.argv[1:])
        load_router()
        intent, conf = classify(query)
        print(f"Intent: {intent} | Confidence: {conf:.2%}")
    else:
        serve()
