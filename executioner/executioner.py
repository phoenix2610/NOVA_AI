#!/usr/bin/env python3
"""
NOVA Executioner — Sandboxed script runner with tiered permission manifest
Listens on /tmp/nova_executioner.sock
"""

import os
import sys
import json
import time
import socket
import logging
import hashlib
import tempfile
import threading
import subprocess
from pathlib import Path
from datetime import datetime

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [EXEC] %(message)s",
    handlers=[
        logging.FileHandler(os.path.expanduser("~/nova/logs/executioner.log")),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("executioner")

# ── Paths ─────────────────────────────────────────────────────────────────────
SOCKET_PATH     = "/tmp/nova_executioner.sock"
PERMISSIONS_FILE = os.path.expanduser("~/nova/executioner/permissions.json")
TTS_SOCKET      = "/tmp/nova_tts.sock"
WORKSPACE       = os.path.expanduser("~/nova/workspace/")
SCRIPT_LOG      = os.path.expanduser("~/nova/logs/exec_history.jsonl")

os.makedirs(WORKSPACE, exist_ok=True)
os.makedirs(os.path.expanduser("~/nova/logs/"), exist_ok=True)

# ── Permission Manifest ───────────────────────────────────────────────────────
DEFAULT_PERMISSIONS = {
    "tier1_auto": {
        "description": "Auto-approved, silent execution",
        "allowed_modules": [
            "os", "sys", "subprocess", "pathlib", "shutil", "glob",
            "json", "re", "math", "datetime", "time", "tempfile",
            "ffmpeg", "pandoc", "convert", "wpctl", "pactl",
            "requests", "urllib", "http"
        ],
        "allowed_write_paths": [
            "~/nova/workspace/",
            "~/Documents/",
            "~/Downloads/",
            "~/nova/logs/"
        ],
        "operations": [
            "audio_control", "brightness", "system_stats",
            "network_fetch", "read_any", "write_workspace",
            "file_conversion", "process_list"
        ]
    },
    "tier2_narrate": {
        "description": "Narrated to user, auto-approved",
        "operations": [
            "open_browser", "write_documents", "send_notification",
            "create_file_outside_workspace", "clipboard"
        ]
    },
    "tier3_confirm": {
        "description": "Requires verbal YES from user",
        "operations": [
            "credentials", "write_outside_safe_paths",
            "install_packages", "modify_system_config",
            "delete_files", "network_post_external",
            "modify_systemd_services"
        ]
    },
    "banned_operations": [
        "rm -rf /",
        "mkfs",
        "dd if=",
        ":(){:|:&};:",
        "chmod 777 /",
        "chown -R root"
    ],
    "timeout_seconds": {
        "tier1": 30,
        "tier2": 60,
        "tier3": 120
    }
}

def load_permissions():
    if os.path.exists(PERMISSIONS_FILE):
        with open(PERMISSIONS_FILE) as f:
            return json.load(f)
    os.makedirs(os.path.dirname(PERMISSIONS_FILE), exist_ok=True)
    with open(PERMISSIONS_FILE, "w") as f:
        json.dump(DEFAULT_PERMISSIONS, f, indent=2)
    return DEFAULT_PERMISSIONS

# ── TTS Narration ─────────────────────────────────────────────────────────────
def speak(text: str, priority: int = 1):
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.connect(TTS_SOCKET)
            s.sendall(json.dumps({"priority": priority, "text": text}).encode())
    except Exception:
        pass  # TTS unavailable, continue silently

# ── Static Safety Check ───────────────────────────────────────────────────────
def static_check(script: str, permissions: dict) -> tuple[bool, str]:
    """Check script against banned operations before execution."""
    banned = permissions.get("banned_operations", [])
    for pattern in banned:
        if pattern in script:
            return False, f"Banned operation detected: {pattern}"

    # Block writes to system paths
    dangerous_paths = ["/etc/", "/usr/", "/boot/", "/sys/", "/proc/"]
    for path in dangerous_paths:
        if f'"{path}' in script or f"'{path}" in script:
            return False, f"Blocked write to system path: {path}"

    return True, "ok"

# ── Tier Classification ───────────────────────────────────────────────────────
def classify_tier(script: str, declared_ops: list) -> int:
    permissions = load_permissions()

    tier3_ops = permissions["tier3_confirm"]["operations"]
    tier2_ops = permissions["tier2_narrate"]["operations"]

    for op in declared_ops:
        if op in tier3_ops:
            return 3
    for op in declared_ops:
        if op in tier2_ops:
            return 2
    return 1

# ── Pending Confirmations (Tier 3) ────────────────────────────────────────────
pending_confirmations = {}

def await_confirmation(script_id: str, timeout: int = 30) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if script_id in pending_confirmations:
            result = pending_confirmations.pop(script_id)
            return result
        time.sleep(0.5)
    return False  # Timed out = denied

# ── Script Execution ──────────────────────────────────────────────────────────
def run_script(script: str, tier: int, script_id: str, env_extra: dict = None) -> dict:
    permissions = load_permissions()
    timeout = permissions["timeout_seconds"].get(f"tier{tier}", 30)

    # Write script to temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix=f"nova_{script_id}_",
        dir=WORKSPACE, delete=False
    ) as f:
        f.write(script)
        script_path = f.name

    env = os.environ.copy()
    env["NOVA_WORKSPACE"] = WORKSPACE
    env["NOVA_SCRIPT_ID"] = script_id

    # ── Inject display/session vars so xdg-open / GUI apps reach the compositor
    # systemd --user inherits DISPLAY & WAYLAND_DISPLAY but NOT XDG_SESSION_TYPE.
    # Without it, xdg-open picks X11 mode on a Wayland session and crashes.
    display_defaults = {
        "DISPLAY":                  ":1",
        "WAYLAND_DISPLAY":          "wayland-1",
        "XDG_SESSION_TYPE":         "wayland",
        "XDG_CURRENT_DESKTOP":      "Hyprland",
        "XDG_RUNTIME_DIR":          f"/run/user/{os.getuid()}",
        "DBUS_SESSION_BUS_ADDRESS": f"unix:path=/run/user/{os.getuid()}/bus",
    }
    for k, v in display_defaults.items():
        env.setdefault(k, v)   # only set if not already present

    if env_extra:
        env.update(env_extra)

    result = {
        "script_id":   script_id,
        "tier":        tier,
        "timestamp":   datetime.now().isoformat(),
        "script":      script[:400],   # log first 400 chars for auditability
        "stdout":      "",
        "stderr":      "",
        "returncode":  -1,
        "success":     False,
        "duration_ms": 0
    }

    t_start = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=WORKSPACE
        )
        result["stdout"]     = proc.stdout.strip()
        result["stderr"]     = proc.stderr.strip()
        result["returncode"] = proc.returncode
        result["success"]    = proc.returncode == 0
    except subprocess.TimeoutExpired:
        result["stderr"]  = f"Script timed out after {timeout}s"
        result["success"] = False
    except Exception as e:
        result["stderr"]  = str(e)
        result["success"] = False
    finally:
        result["duration_ms"] = int((time.time() - t_start) * 1000)
        try:
            os.unlink(script_path)
        except Exception:
            pass

    # Log to history
    with open(SCRIPT_LOG, "a") as f:
        f.write(json.dumps(result) + "\n")

    return result

# ── Request Handler ───────────────────────────────────────────────────────────
def handle_request(conn: socket.socket):
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
        action  = request.get("action", "execute")

        # ── Confirm (tier 3 verbal yes/no) ────────────────────────────────────
        if action == "confirm":
            script_id = request.get("script_id")
            answer    = request.get("answer", "").lower() in ("yes", "y", "confirm")
            if script_id:
                pending_confirmations[script_id] = answer
            conn.sendall(json.dumps({"status": "ok"}).encode())
            return

        # ── Execute ───────────────────────────────────────────────────────────
        script      = request.get("script", "")
        declared_ops = request.get("operations", [])
        script_id   = hashlib.md5((script + str(time.time())).encode()).hexdigest()[:8]

        if not script.strip():
            conn.sendall(json.dumps({"error": "Empty script"}).encode())
            return

        permissions = load_permissions()
        safe, reason = static_check(script, permissions)
        if not safe:
            log.warning(f"[{script_id}] BLOCKED: {reason}")
            speak(f"Sir, I blocked a script. {reason}")
            conn.sendall(json.dumps({"error": reason, "blocked": True}).encode())
            return

        tier = classify_tier(script, declared_ops)
        log.info(f"[{script_id}] Tier {tier} | ops: {declared_ops}")

        # Tier 2 — narrate then execute
        if tier == 2:
            narration = request.get("narration", "Sir, executing a task.")
            speak(narration)

        # Tier 3 — ask and wait
        elif tier == 3:
            question = request.get("narration", "Sir, this requires your confirmation. Say yes to proceed.")
            speak(question, priority=0)
            log.info(f"[{script_id}] Awaiting verbal confirmation...")
            approved = await_confirmation(script_id, timeout=30)
            if not approved:
                speak("Understood sir, action cancelled.")
                conn.sendall(json.dumps({"error": "Denied by user", "cancelled": True}).encode())
                return

        result = run_script(script, tier, script_id)

        if result["success"]:
            log.info(f"[{script_id}] ✓ {result['duration_ms']}ms")
        else:
            log.warning(f"[{script_id}] ✗ {result['stderr'][:120]}")

        conn.sendall((json.dumps(result) + "\n").encode())

    except json.JSONDecodeError:
        conn.sendall(json.dumps({"error": "Invalid JSON"}).encode())
    except Exception as e:
        log.error(f"Handler error: {e}")
        conn.sendall(json.dumps({"error": str(e)}).encode())
    finally:
        conn.close()

# ── Main Server ───────────────────────────────────────────────────────────────
def main():
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)

    permissions = load_permissions()
    log.info("Executioner online")
    log.info(f"Workspace: {WORKSPACE}")
    log.info(f"Permissions loaded from: {PERMISSIONS_FILE}")

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    os.chmod(SOCKET_PATH, 0o600)
    server.listen(5)

    speak("Executioner online, sir.", priority=1)

    try:
        while True:
            conn, _ = server.accept()
            t = threading.Thread(target=handle_request, args=(conn,), daemon=True)
            t.start()
    except KeyboardInterrupt:
        log.info("Executioner shutting down")
    finally:
        server.close()
        if os.path.exists(SOCKET_PATH):
            os.unlink(SOCKET_PATH)

if __name__ == "__main__":
    main()
