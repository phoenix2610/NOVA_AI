"""
NOVA OS — tts_daemon.py
Always-resident TTS daemon. Claims PipeWire output at startup and never releases it.
Exposes a Unix socket at /tmp/nova_tts.sock.
Clients send newline-terminated JSON: {"priority": 0-2, "text": "..."}
Priority 0 = status narration (fires immediately, interrupts nothing)
Priority 1 = result speech (queued, spoken in order)
Priority 2 = confirmation required (spoken, then waits for STT ack — future)
"""

import os
import sys
import json
import signal
import socket
import logging
import tempfile
import threading
import subprocess
import numpy as np
import soundfile as sf
from queue import PriorityQueue
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
SOCKET_PATH   = "/tmp/nova_tts.sock"
SPEAKER_SINK  = "alsa_output.pci-0000_05_00.6.analog-stereo"
SPEAKER_NODE  = "57"
KOKORO_LANG   = "b"          # British English
KOKORO_VOICE  = "bm_lewis"
KOKORO_SPEED  = 1.0
LOG_PATH      = "/home/tathya/nova/logs/tts_daemon.log"
PIPER_MODEL   = "/home/tathya/nova/tts/en_US-ryan-medium.onnx"

# ── Logging ───────────────────────────────────────────────────────────────────
Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TTS] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("nova.tts")

# ── Global state ──────────────────────────────────────────────────────────────
tts_queue     = PriorityQueue()   # (priority, sequence, text)
_seq_counter  = 0
_seq_lock     = threading.Lock()
_shutdown     = threading.Event()
_kokoro_pipe  = None             # loaded once, reused

# ── Kokoro loader — load once, keep warm ─────────────────────────────────────
def load_kokoro():
    global _kokoro_pipe
    try:
        from kokoro import KPipeline
        _kokoro_pipe = KPipeline(lang_code=KOKORO_LANG)
        log.info("Kokoro pipeline loaded and warm.")
    except ImportError:
        log.warning("Kokoro not found — will use Piper fallback.")
        _kokoro_pipe = None

def synthesise(text: str) -> str:
    """Synthesise text to a temp WAV file. Returns path."""
    tmp = tempfile.mktemp(suffix="_nova_tts.wav")

    if _kokoro_pipe is not None:
        samples = []
        for _, _, audio in _kokoro_pipe(text, voice=KOKORO_VOICE, speed=KOKORO_SPEED):
            samples.append(audio)
        if samples:
            audio_out = np.concatenate(samples)
            sf.write(tmp, audio_out, 24000)
            return tmp

    # Piper fallback
    subprocess.run(
        ["piper-tts", "--model", PIPER_MODEL,
         "--length-scale", "1.2", "-f", tmp],
        input=text.encode(), check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return tmp

def play(wav_path: str):
    """Play WAV through PipeWire, blocking until done."""
    subprocess.run(
        ["paplay", f"--device={SPEAKER_SINK}", wav_path],
        check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    os.unlink(wav_path)

# ── Speaker thread — drains queue ─────────────────────────────────────────────
def speaker_thread():
    log.info("Speaker thread started.")
    while not _shutdown.is_set():
        try:
            priority, seq, text = tts_queue.get(timeout=0.5)
            log.info(f"Speaking (p{priority}): {text[:60]}")
            wav = synthesise(text)
            play(wav)
            tts_queue.task_done()
        except Exception:
            continue   # timeout or synthesis error — keep looping
    log.info("Speaker thread exiting.")

# ── Socket server — accepts client connections ────────────────────────────────
def enqueue(text: str, priority: int = 1):
    global _seq_counter
    with _seq_lock:
        _seq_counter += 1
        seq = _seq_counter
    tts_queue.put((priority, seq, text))

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
        if data:
            msg = json.loads(data.decode().strip())
            text     = msg.get("text", "").strip()
            priority = int(msg.get("priority", 1))
            if text:
                enqueue(text, priority)
                conn.sendall(b'{"status":"queued"}\n')
    except Exception as e:
        log.warning(f"Client error: {e}")
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
    log.info(f"TTS socket listening at {SOCKET_PATH}")

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
    log.info("Socket server closed.")

# ── Signal handling ───────────────────────────────────────────────────────────
def shutdown(signum, frame):
    log.info("Shutdown signal received.")
    _shutdown.set()

signal.signal(signal.SIGTERM, shutdown)
signal.signal(signal.SIGINT,  shutdown)

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    log.info("NOVA TTS daemon starting...")
    load_kokoro()

    # Speak boot confirmation
    enqueue("NOVA voice online, sir.", priority=0)

    spk = threading.Thread(target=speaker_thread, daemon=True)
    spk.start()

    socket_server()   # blocks until shutdown

    _shutdown.set()
    spk.join(timeout=5)
    log.info("NOVA TTS daemon stopped.")

if __name__ == "__main__":
    main()
