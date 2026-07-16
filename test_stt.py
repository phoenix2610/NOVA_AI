"""
NOVA — test_stt.py
Standalone mic + STT diagnostic.
Records 5 seconds at 44100Hz, downsamples to 16kHz, transcribes with faster-whisper.
Run this, speak, and verify transcription before integrating into pipeline.
"""

import sounddevice as sd
import numpy as np
import tempfile, wave, struct, math
from scipy.signal import resample_poly
from math import gcd

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE       = 4        # HD-Audio Generic: ALC257 Analog
RECORD_RATE  = 44100    # hardware native
SAMPLE_RATE  = 16000    # whisper target
DURATION     = 5        # seconds to record
WHISPER_SIZE = "base"

# ── Resample constants ────────────────────────────────────────────────────────
_g   = gcd(RECORD_RATE, SAMPLE_RATE)
UP   = SAMPLE_RATE  // _g
DOWN = RECORD_RATE  // _g

# ── Record ────────────────────────────────────────────────────────────────────
print(f"[MIC] Recording {DURATION}s at {RECORD_RATE}Hz on device {DEVICE}...")
print("      Speak now!")
audio_44k = sd.rec(int(DURATION * RECORD_RATE), samplerate=RECORD_RATE,
                   channels=1, dtype="int16", device=DEVICE)
sd.wait()
print("[MIC] Done recording.")

samples_44k = audio_44k[:, 0].astype(np.float32)

# ── Stats ─────────────────────────────────────────────────────────────────────
peak = np.max(np.abs(samples_44k))
rms  = np.sqrt(np.mean(samples_44k ** 2))
db   = 20 * math.log10(rms / 32768.0) if rms > 0 else -96.0
print(f"[MIC] Peak: {peak:.0f}  RMS: {rms:.1f}  dB: {db:.1f} dBFS")

# ── Downsample 44100 → 16000 ──────────────────────────────────────────────────
print(f"[DSP] Resampling {RECORD_RATE}Hz → {SAMPLE_RATE}Hz...")
samples_16k = resample_poly(samples_44k, UP, DOWN)
samples_16k = np.clip(samples_16k, -32768, 32767).astype(np.int16)
print(f"[DSP] Resampled: {len(samples_44k)} → {len(samples_16k)} samples")

# ── Noise gate ────────────────────────────────────────────────────────────────
fsz    = int(SAMPLE_RATE * 0.02)
gated  = []
passed = 0
muted  = 0
for i in range(0, len(samples_16k), fsz):
    c  = samples_16k[i:i+fsz].tolist()
    r  = (sum(s*s for s in c) / max(len(c), 1)) ** 0.5
    db_frame = 20 * math.log10(r / 32768.0) if r > 0 else -96.0
    if db_frame > -60.0:
        gated.extend(c)
        passed += 1
    else:
        gated.extend([0] * len(c))
        muted += 1

print(f"[GATE] Frames passed: {passed}  muted: {muted}")

# ── Normalize ─────────────────────────────────────────────────────────────────
peak_g = max(abs(s) for s in gated) if gated else 1
if peak_g > 0:
    gain  = 10 ** ((-14.0 - 20 * math.log10(peak_g / 32768.0)) / 20)
    gated = [max(-32768, min(32767, int(s * gain))) for s in gated]

# ── Save WAV ──────────────────────────────────────────────────────────────────
tmp = tempfile.mktemp(suffix="_test.wav")
with wave.open(tmp, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(struct.pack(f"<{len(gated)}h", *gated))
print(f"[WAV] Saved to: {tmp}")

# ── Transcribe ────────────────────────────────────────────────────────────────
print("[STT] Loading faster-whisper...")
from faster_whisper import WhisperModel
model = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
segments, info = model.transcribe(tmp, language="en", beam_size=5)
text = " ".join(seg.text for seg in segments).strip()

print(f"\n{'='*50}")
print(f"  TRANSCRIPTION: {text}")
print(f"{'='*50}\n")

if not text:
    print("[WARN] Empty transcription — mic may not be capturing or noise gate too aggressive.")
