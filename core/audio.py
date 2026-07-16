"""
NOVA OS — audio.py
Clean mic capture: noise gate + RNNoise + normalization.
Returns clean wav path for Whisper.
"""

import gc
import os
import struct
import subprocess
import tempfile
import wave

# ── Config ────────────────────────────────────────────────────────────────────
MIC_DEVICE      = "alsa_input.pci-0000_05_00.6.analog-stereo"
MIC_VOLUME      = 0.4          # wpctl volume — tune if needed
SAMPLE_RATE     = 16000
CHANNELS        = 1
SAMPLE_WIDTH    = 2            # s16le = 2 bytes
NOISE_GATE_DB   = -40.0        # frames below this are silenced
TARGET_DB       = -14.0        # normalization target
FRAME_DURATION  = 0.02         # 20ms frames for noise gate


# ── Ensure mic volume is set ──────────────────────────────────────────────────
def init_mic():
    subprocess.run(["wpctl", "set-mute", "58", "0"], check=True)
    subprocess.run(["wpctl", "set-volume", "58", str(MIC_VOLUME)], check=True)


# ── Record raw audio via parecord ─────────────────────────────────────────────
def record_raw(duration: int) -> str:
    tmp = tempfile.mktemp(suffix=".wav")
    print(f"[MIC] Recording {duration}s...")
    proc = subprocess.Popen([
        "parecord",
        f"--device={MIC_DEVICE}",
        "--rate=16000",
        "--channels=1",
        "--format=s16le",
        tmp
    ])
    import time
    time.sleep(duration)
    proc.terminate()
    proc.wait()
    return tmp


# ── Read wav samples ──────────────────────────────────────────────────────────
def read_wav(path: str):
    with wave.open(path, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        n_frames = wf.getnframes()
        rate = wf.getframerate()
    samples = list(struct.unpack(f"<{n_frames}h", frames))
    return samples, rate


# ── Write wav samples ─────────────────────────────────────────────────────────
def write_wav(path: str, samples: list, rate: int):
    frames = struct.pack(f"<{len(samples)}h", *samples)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(rate)
        wf.writeframes(frames)


# ── RMS of a chunk ────────────────────────────────────────────────────────────
def rms(chunk: list) -> float:
    if not chunk:
        return 0.0
    mean_sq = sum(s * s for s in chunk) / len(chunk)
    return mean_sq ** 0.5


# ── RMS → dB ──────────────────────────────────────────────────────────────────
def to_db(r: float) -> float:
    if r == 0:
        return -96.0
    import math
    return 20 * math.log10(r / 32768.0)


# ── Noise gate ────────────────────────────────────────────────────────────────
def noise_gate(samples: list, rate: int, threshold_db: float = NOISE_GATE_DB) -> list:
    frame_size = int(rate * FRAME_DURATION)
    output = []
    for i in range(0, len(samples), frame_size):
        chunk = samples[i:i + frame_size]
        if to_db(rms(chunk)) > threshold_db:
            output.extend(chunk)
        else:
            output.extend([0] * len(chunk))
    return output


# ── Normalize to target dB ────────────────────────────────────────────────────
def normalize(samples: list, target_db: float = TARGET_DB) -> list:
    import math
    peak = max(abs(s) for s in samples) if samples else 1
    if peak == 0:
        return samples
    current_db = 20 * math.log10(peak / 32768.0)
    gain = 10 ** ((target_db - current_db) / 20)
    normalized = [max(-32768, min(32767, int(s * gain))) for s in samples]
    return normalized


# ── RNNoise via ffmpeg (if available) ─────────────────────────────────────────
def rnnoise_filter(input_path: str) -> str:
    output_path = tempfile.mktemp(suffix=".wav")
    result = subprocess.run([
        "ffmpeg", "-y",
        "-i", input_path,
        "-af", "arnndn=m=/usr/share/rnnoise/bd.rnnn",
        "-ar", "16000",
        "-ac", "1",
        output_path
    ], capture_output=True)
    if result.returncode == 0:
        print("[AUDIO] RNNoise applied.")
        return output_path
    else:
        print("[AUDIO] RNNoise model not found, skipping.")
        return input_path


# ── Main: record + process → clean wav ───────────────────────────────────────
def capture_clean(duration: int = 5) -> str:
    """
    Record mic, apply noise gate + normalize.
    Returns path to clean wav for Whisper.
    """
    raw_path = record_raw(duration)

    # Try RNNoise first
    filtered_path = rnnoise_filter(raw_path)
    if filtered_path != raw_path:
        os.unlink(raw_path)
        raw_path = filtered_path

    # Load, gate, normalize
    samples, rate = read_wav(raw_path)
    samples = noise_gate(samples, rate)
    samples = normalize(samples)

    # Write clean output
    clean_path = tempfile.mktemp(suffix="_clean.wav")
    write_wav(clean_path, samples, rate)

    os.unlink(raw_path)
    gc.collect()
    print(f"[AUDIO] Clean audio ready: {clean_path}")
    return clean_path


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_mic()
    print("Speak for 5 seconds...")
    clean = capture_clean(5)
    subprocess.run([
        "paplay",
        "--device=alsa_output.pci-0000_05_00.6.analog-stereo",
        clean
    ])
    os.unlink(clean)
