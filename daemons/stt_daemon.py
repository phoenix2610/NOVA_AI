"""
NOVA OS — stt_daemon.py
Always-resident STT daemon. Claims PipeWire mic at startup and never releases it.
Runs openwakeword continuously. On wake word detection, activates VAD recording
and live faster-whisper transcription, then pushes final transcript to
/tmp/nova_stt.sock as newline-terminated JSON.

Downstream consumers (pipeline orchestrator) connect to the socket and receive:
{"type": "wake"}               — wake word detected
{"type": "partial", "text": "..."} — live partial transcript
{"type": "final", "text": "..."}   — final transcript ready for routing
"""

import os
import gc
import sys
import json
import math
import signal
import socket
import struct
import logging
import tempfile
import threading
import warnings
import wave
from pathlib import Path
from queue import Queue

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
from math import gcd

# ── Suppress third-party warnings ─────────────────────────────────────────────
warnings.filterwarnings("ignore", message="dropout option adds dropout after all but last recurrent layer", category=UserWarning)
warnings.filterwarnings("ignore", message=r"`torch\.nn\.utils\.weight_norm` is deprecated", category=FutureWarning)
warnings.filterwarnings("ignore", message="You are sending unauthenticated requests to the HF Hub")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["HF_HUB_OFFLINE"]        = "1"

# ── Config ────────────────────────────────────────────────────────────────────
SOCKET_PATH      = "/tmp/nova_stt.sock"
MIC_DEVICE_IDX   = 4                   # ALC257 Analog hw:1,0
MIC_NODE         = "58"
RECORD_RATE      = 44100
SAMPLE_RATE      = 16000
CHUNK            = 1280
WAKE_MODEL       = "/home/tathya/.local/lib/python3.14/site-packages/openwakeword/resources/models/hey_jarvis_v0.1.onnx"
WAKE_THRESHOLD   = 0.5
WHISPER_MODEL    = "base"
VAD_FRAME_MS     = 30
VAD_SILENCE_TIMEOUT = 2.0
VAD_MAX_DURATION    = 30
VAD_ENERGY_THRESHOLD = 500
STREAM_CHUNK_SEC    = 1.5
STREAM_CONTEXT_SEC  = 6.0
LOG_PATH         = "/home/tathya/nova/logs/stt_daemon.log"

# ── Logging ───────────────────────────────────────────────────────────────────
Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [STT] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("nova.stt")

# ── Global state ──────────────────────────────────────────────────────────────
_shutdown    = threading.Event()
_clients     = []
_clients_lock = threading.Lock()

# ── Resample constants ────────────────────────────────────────────────────────
_g   = gcd(RECORD_RATE, SAMPLE_RATE)
UP   = SAMPLE_RATE // _g
DOWN = RECORD_RATE // _g

# ── Client broadcast ──────────────────────────────────────────────────────────
def broadcast(msg: dict):
    """Send JSON message to all connected downstream clients."""
    data = (json.dumps(msg) + "\n").encode()
    with _clients_lock:
        dead = []
        for c in _clients:
            try:
                c.sendall(data)
            except Exception:
                dead.append(c)
        for c in dead:
            _clients.remove(c)

# ── Audio helpers ─────────────────────────────────────────────────────────────
def downsample(pcm_44k: np.ndarray) -> np.ndarray:
    pcm_f   = pcm_44k.astype(np.float32)
    pcm_16k = resample_poly(pcm_f, UP, DOWN)
    return np.clip(pcm_16k, -32768, 32767).astype(np.int16)

def noise_gate_normalize(samples: np.ndarray) -> np.ndarray:
    lst  = samples.tolist()
    fsz  = int(SAMPLE_RATE * 0.02)
    gated = []
    for i in range(0, len(lst), fsz):
        c  = lst[i:i + fsz]
        r  = (sum(s * s for s in c) / max(len(c), 1)) ** 0.5
        db = 20 * math.log10(r / 32768.0) if r > 0 else -96.0
        gated.extend(c if db > -60.0 else [0] * len(c))
    peak = max(abs(s) for s in gated) if gated else 1
    if peak > 0:
        gain  = 10 ** ((-14.0 - 20 * math.log10(peak / 32768.0)) / 20)
        gated = [max(-32768, min(32767, int(s * gain))) for s in gated]
    return np.array(gated, dtype=np.int16)

def write_wav(samples: np.ndarray) -> str:
    tmp = tempfile.mktemp(suffix="_nova_stt.wav")
    with wave.open(tmp, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(struct.pack(f"<{len(samples)}h", *samples.tolist()))
    return tmp

# ── Wake word listener ────────────────────────────────────────────────────────
def wake_loop():
    import openwakeword
    oww        = openwakeword.Model(wakeword_model_paths=[WAKE_MODEL])
    wake_block = int(CHUNK * RECORD_RATE / SAMPLE_RATE)
    log.info("Wake word listener active — say 'Hey Jarvis'")

    while not _shutdown.is_set():
        try:
            with sd.InputStream(samplerate=RECORD_RATE, channels=1, dtype="int16",
                                device=MIC_DEVICE_IDX, blocksize=wake_block,
                                latency="low") as stream:
                while not _shutdown.is_set():
                    audio, _ = stream.read(wake_block)
                    pcm_16k  = downsample(audio[:, 0])
                    result   = oww.predict(pcm_16k)
                    if any(v > WAKE_THRESHOLD for v in result.values()):
                        log.info("Wake word detected.")
                        broadcast({"type": "wake"})
                        record_and_transcribe()
        except Exception as e:
            log.error(f"Wake loop error: {e} — restarting in 2s")
            import time; time.sleep(2)

# ── VAD + live transcription ──────────────────────────────────────────────────
def record_and_transcribe():
    from faster_whisper import WhisperModel

    frame_samples = int(RECORD_RATE * VAD_FRAME_MS / 1000)
    silence_limit = int(VAD_SILENCE_TIMEOUT * 1000 / VAD_FRAME_MS)
    max_frames    = int(VAD_MAX_DURATION    * 1000 / VAD_FRAME_MS)
    chunk_samples = int(STREAM_CHUNK_SEC   * SAMPLE_RATE)
    ctx_samples   = int(STREAM_CONTEXT_SEC * SAMPLE_RATE)

    audio_queue = Queue()

    # Producer: mic → downsample → queue
    def producer():
        speech_started = False
        silence_frames = 0
        with sd.InputStream(samplerate=RECORD_RATE, channels=1, dtype="int16",
                            device=MIC_DEVICE_IDX, blocksize=frame_samples,
                            latency="low") as stream:
            for _ in range(max_frames):
                if _shutdown.is_set():
                    break
                audio, _  = stream.read(frame_samples)
                pcm_44k   = audio[:, 0].astype(np.float32)
                pcm_16k   = downsample(audio[:, 0])
                rms       = float(np.sqrt(np.mean(pcm_44k ** 2)))
                is_speech = rms > VAD_ENERGY_THRESHOLD

                if is_speech:
                    if not speech_started:
                        log.info("Speech detected.")
                    speech_started = True
                    silence_frames = 0
                    audio_queue.put(("audio", pcm_16k))
                elif speech_started:
                    silence_frames += 1
                    audio_queue.put(("audio", pcm_16k))
                    if silence_frames >= silence_limit:
                        log.info("Speech ended.")
                        break
        audio_queue.put(("done", None))

    log.info("Loading Whisper for live transcription...")
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

    accumulated        = np.array([], dtype=np.int16)
    samples_since_last = 0
    last_transcript    = ""

    prod = threading.Thread(target=producer, daemon=True)
    prod.start()

    while True:
        try:
            tag, chunk = audio_queue.get(timeout=5.0)
        except Exception:
            break

        if tag == "done":
            break

        accumulated        = np.concatenate([accumulated, chunk])
        samples_since_last += len(chunk)

        if samples_since_last >= chunk_samples and len(accumulated) > SAMPLE_RATE // 2:
            samples_since_last = 0
            window  = accumulated[-ctx_samples:]
            cleaned = noise_gate_normalize(window)
            tmp     = write_wav(cleaned)
            try:
                segs, _ = model.transcribe(tmp, language="en", beam_size=3,
                                           vad_filter=False, without_timestamps=True)
                partial = " ".join(s.text for s in segs).strip()
            except Exception:
                partial = last_transcript
            finally:
                os.unlink(tmp)

            if partial and partial != last_transcript:
                last_transcript = partial
                broadcast({"type": "partial", "text": partial})
                log.info(f"Partial: {partial}")

    prod.join(timeout=3.0)

    # Final high-accuracy pass
    if len(accumulated) > SAMPLE_RATE // 4:
        cleaned    = noise_gate_normalize(accumulated)
        tmp        = write_wav(cleaned)
        try:
            segs, _ = model.transcribe(tmp, language="en", beam_size=5,
                                       vad_filter=False, without_timestamps=True)
            final_text = " ".join(s.text for s in segs).strip()
        except Exception:
            final_text = last_transcript
        finally:
            os.unlink(tmp)
    else:
        final_text = last_transcript

    del model
    gc.collect()

    if final_text:
        broadcast({"type": "final", "text": final_text})
        log.info(f"Final: {final_text}")

# ── Socket server — downstream clients connect here ───────────────────────────
def socket_server():
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    os.chmod(SOCKET_PATH, 0o600)
    server.listen(8)
    server.settimeout(1.0)
    log.info(f"STT socket listening at {SOCKET_PATH}")

    while not _shutdown.is_set():
        try:
            conn, _ = server.accept()
            with _clients_lock:
                _clients.append(conn)
            log.info("Downstream client connected.")
        except socket.timeout:
            continue
        except Exception as e:
            log.error(f"Socket error: {e}")

    server.close()
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)

# ── Signal handling ───────────────────────────────────────────────────────────
def shutdown(signum, frame):
    log.info("Shutdown signal received.")
    _shutdown.set()

signal.signal(signal.SIGTERM, shutdown)
signal.signal(signal.SIGINT,  shutdown)

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    log.info("NOVA STT daemon starting...")

    sock_thread = threading.Thread(target=socket_server, daemon=True)
    sock_thread.start()

    wake_loop()   # blocks — restarts automatically on error

    sock_thread.join(timeout=3)
    log.info("NOVA STT daemon stopped.")

if __name__ == "__main__":
    main()
