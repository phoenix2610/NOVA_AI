"""
NOVA OS — watchdog.py
Always-resident process manager. Monitors system health, manages the
load/eject lifecycle of every non-resident component, and ensures
STT, TTS, and Ollama services stay alive.

Responsibilities:
  1. Health monitoring — STT, TTS, Ollama user services
  2. Auto-restart dead services via systemctl --user
  3. RAM guard — blocks component loads if available RAM is below threshold
  4. Component registry — tracks what is currently loaded
  5. IPC socket — pipeline components request load/eject through here
  6. Chat log rotation — rolls over logs when they exceed size limit

Socket: /tmp/nova_watchdog.sock
Protocol: newline-terminated JSON requests/responses

Request types:
  {"action": "status"}                        → full system status
  {"action": "request_load", "component": X}  → load component if RAM allows
  {"action": "eject", "component": X}         → eject component from RAM
  {"action": "ram_check"}                     → current RAM snapshot
  {"action": "ping"}                          → heartbeat check
"""

import os
import gc
import sys
import json
import time
import signal
import socket
import logging
import threading
import subprocess
from pathlib import Path
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
SOCKET_PATH     = "/tmp/nova_watchdog.sock"
LOG_PATH        = "/home/tathya/nova/logs/watchdog.log"
CHAT_LOG_DIR    = "/home/tathya/nova/chat"
CHAT_LOG_MAX_MB = 50        # rotate chat log after 50MB

# RAM thresholds (GB)
RAM_FLOOR_GB        = 3.0   # never let available RAM drop below this
RAM_WARN_GB         = 5.0   # warn TTS if RAM getting tight

# Health check interval (seconds)
HEALTH_CHECK_SEC    = 15

# Managed user services — watchdog keeps these alive always
RESIDENT_SERVICES = [
    "nova-tts",
    "nova-stt",
    "ollama",
]

# Component RAM footprint estimates (GB) — used for pre-load RAM check
COMPONENT_RAM = {
    "router":    1.0,
    "brain":     5.0,    # Mistral 7B int4 via Ollama
    "whisper":   0.2,    # loaded inside STT daemon, tracked here for accounting
    "rag":       0.3,
    "embedding": 0.1,
}

# TTS socket for narration
TTS_SOCKET = "/tmp/nova_tts.sock"

# ── Logging ───────────────────────────────────────────────────────────────────
Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(CHAT_LOG_DIR).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [WDG] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("nova.watchdog")

# ── Global state ──────────────────────────────────────────────────────────────
_shutdown         = threading.Event()
_loaded_components = set()   # what is currently in RAM
_state_lock       = threading.Lock()

# ── RAM monitoring ────────────────────────────────────────────────────────────
def ram_snapshot() -> dict:
    """Read /proc/meminfo and return available/total in GB."""
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            parts = line.split()
            if parts[0] in ("MemTotal:", "MemAvailable:", "SwapTotal:", "SwapFree:"):
                info[parts[0].rstrip(":")] = int(parts[1]) / 1024 / 1024  # GB
    return {
        "total_gb":     round(info.get("MemTotal",     0), 2),
        "available_gb": round(info.get("MemAvailable", 0), 2),
        "swap_total_gb":round(info.get("SwapTotal",    0), 2),
        "swap_free_gb": round(info.get("SwapFree",     0), 2),
        "used_gb":      round(info.get("MemTotal", 0) - info.get("MemAvailable", 0), 2),
    }

def ram_ok_for(component: str) -> tuple[bool, str]:
    """Check if enough RAM is available to load a component."""
    needed = COMPONENT_RAM.get(component, 1.0)
    snap   = ram_snapshot()
    avail  = snap["available_gb"]
    # Must have component footprint + floor headroom
    if avail - needed < RAM_FLOOR_GB:
        return False, f"Insufficient RAM: need {needed}GB + {RAM_FLOOR_GB}GB floor, have {avail}GB"
    return True, f"RAM OK: {avail}GB available, loading {component} ({needed}GB)"

# ── TTS narration ─────────────────────────────────────────────────────────────
def narrate(text: str, priority: int = 0):
    """Send a status message to the TTS daemon."""
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(TTS_SOCKET)
        sock.sendall((json.dumps({"priority": priority, "text": text}) + "\n").encode())
        sock.close()
    except Exception as e:
        log.warning(f"TTS narration failed: {e}")

# ── Service health management ─────────────────────────────────────────────────
def service_is_active(name: str) -> bool:
    result = subprocess.run(
        ["systemctl", "--user", "is-active", name],
        capture_output=True, text=True
    )
    return result.stdout.strip() == "active"

def restart_service(name: str):
    log.warning(f"Service {name} is down — restarting...")
    subprocess.run(["systemctl", "--user", "restart", name], check=False)
    time.sleep(3)
    if service_is_active(name):
        log.info(f"Service {name} restarted successfully.")
        if name == "nova-tts":
            time.sleep(2)   # give TTS time to warm up before narrating
        narrate(f"Service {name} has been restored, sir.", priority=0)
    else:
        log.error(f"Service {name} failed to restart.")

def health_check_loop():
    """Periodically verify all resident services are alive."""
    log.info("Health check loop started.")
    while not _shutdown.is_set():
        for svc in RESIDENT_SERVICES:
            if not service_is_active(svc):
                restart_service(svc)
        # RAM warning
        snap = ram_snapshot()
        if snap["available_gb"] < RAM_WARN_GB:
            log.warning(f"RAM low: {snap['available_gb']:.1f}GB available.")
            narrate(f"Sir, available memory is low at {snap['available_gb']:.1f} gigabytes.", priority=0)
        _shutdown.wait(HEALTH_CHECK_SEC)
    log.info("Health check loop exiting.")

# ── Component lifecycle ───────────────────────────────────────────────────────
def handle_request_load(component: str) -> dict:
    with _state_lock:
        if component in _loaded_components:
            return {"status": "ok", "message": f"{component} already loaded."}

        ok, msg = ram_ok_for(component)
        if not ok:
            log.warning(msg)
            narrate(f"Sir, cannot load {component}. {msg}", priority=0)
            return {"status": "error", "message": msg}

        log.info(f"Approving load: {component}. {msg}")
        _loaded_components.add(component)
        return {"status": "ok", "message": msg, "component": component}

def handle_eject(component: str) -> dict:
    with _state_lock:
        if component in _loaded_components:
            _loaded_components.discard(component)
            log.info(f"Component ejected: {component}")
            return {"status": "ok", "message": f"{component} ejected."}
        return {"status": "ok", "message": f"{component} was not loaded."}

def handle_status() -> dict:
    snap = ram_snapshot()
    services = {s: service_is_active(s) for s in RESIDENT_SERVICES}
    return {
        "status":     "ok",
        "ram":        snap,
        "loaded":     list(_loaded_components),
        "services":   services,
        "timestamp":  datetime.now().isoformat(),
    }

# ── Chat log rotation ─────────────────────────────────────────────────────────
def rotate_chat_log_if_needed():
    """Rotate chat log if it exceeds CHAT_LOG_MAX_MB."""
    log_file = Path(CHAT_LOG_DIR) / "nova_chat.jsonl"
    if not log_file.exists():
        return
    size_mb = log_file.stat().st_size / 1024 / 1024
    if size_mb >= CHAT_LOG_MAX_MB:
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive  = Path(CHAT_LOG_DIR) / f"nova_chat_{ts}.jsonl"
        log_file.rename(archive)
        log.info(f"Chat log rotated → {archive.name}")

def chat_log_rotation_loop():
    while not _shutdown.is_set():
        rotate_chat_log_if_needed()
        _shutdown.wait(300)   # check every 5 minutes

# ── IPC socket server ─────────────────────────────────────────────────────────
def handle_client(conn: socket.socket):
    try:
        data = b""
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break

        if not data:
            return

        msg    = json.loads(data.decode().strip())
        action = msg.get("action", "")

        if action == "ping":
            response = {"status": "ok", "message": "pong"}
        elif action == "status":
            response = handle_status()
        elif action == "ram_check":
            response = {"status": "ok", "ram": ram_snapshot()}
        elif action == "request_load":
            component = msg.get("component", "")
            response  = handle_request_load(component)
        elif action == "eject":
            component = msg.get("component", "")
            response  = handle_eject(component)
        else:
            response = {"status": "error", "message": f"Unknown action: {action}"}

        conn.sendall((json.dumps(response) + "\n").encode())

    except Exception as e:
        log.warning(f"Client handler error: {e}")
        try:
            conn.sendall((json.dumps({"status": "error", "message": str(e)}) + "\n").encode())
        except Exception:
            pass
    finally:
        conn.close()

def socket_server():
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    os.chmod(SOCKET_PATH, 0o600)
    server.listen(8)
    server.settimeout(1.0)
    log.info(f"Watchdog socket listening at {SOCKET_PATH}")

    while not _shutdown.is_set():
        try:
            conn, _ = server.accept()
            threading.Thread(target=handle_client, args=(conn,), daemon=True).start()
        except socket.timeout:
            continue
        except Exception as e:
            log.error(f"Socket error: {e}")

    server.close()
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)
    log.info("Watchdog socket closed.")

# ── Signal handling ───────────────────────────────────────────────────────────
def shutdown(signum, frame):
    log.info("Shutdown signal received.")
    _shutdown.set()

signal.signal(signal.SIGTERM, shutdown)
signal.signal(signal.SIGINT,  shutdown)

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    log.info("NOVA Watchdog starting...")
    log.info(f"RAM snapshot: {ram_snapshot()}")

    # Verify resident services are up before declaring ready
    for svc in RESIDENT_SERVICES:
        if not service_is_active(svc):
            log.warning(f"{svc} not active on watchdog start — attempting restart.")
            restart_service(svc)

    narrate("NOVA systems nominal. Watchdog online, sir.", priority=0)

    # Start background threads
    health_thread  = threading.Thread(target=health_check_loop,     daemon=True)
    chatlog_thread = threading.Thread(target=chat_log_rotation_loop, daemon=True)

    health_thread.start()
    chatlog_thread.start()

    socket_server()   # blocks until shutdown

    _shutdown.set()
    health_thread.join(timeout=5)
    chatlog_thread.join(timeout=5)
    log.info("NOVA Watchdog stopped.")

if __name__ == "__main__":
    main()
