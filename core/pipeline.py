"""
NOVA OS — pipeline.py (Phase 2.5)
Sequential voice pipeline. Load → use → unload.
Resident: nova-router + openwakeword only.

Phase 2.5 upgrades:
  1. keep_alive: 0  — Ollama ejects model from RAM after each call
  2. System prompt  — NOVA personality (witty, loyal, sharp)
  3. Context history — last 3 exchanges, clears on intent change
  4. VAD recording  — dynamic end-of-speech, no fixed 6s window
  5. Kokoro TTS     — natural British male voice (bm_lewis)
  6. faster-whisper — chunked streaming transcription
"""

import os
import gc
import tempfile
import subprocess
import time
import warnings

# ── Suppress third-party warnings we cannot fix upstream ─────────────────────
# Kokoro LSTM: dropout=0.2 with num_layers=1 (harmless, model just ignores dropout)
warnings.filterwarnings(
    "ignore",
    message="dropout option adds dropout after all but last recurrent layer",
    category=UserWarning,
    module="torch.nn.modules.rnn",
)
# PyTorch weight_norm deprecation — comes from Kokoro internals, not our code
warnings.filterwarnings(
    "ignore",
    message=r"`torch\.nn\.utils\.weight_norm` is deprecated",
    category=FutureWarning,
    module="torch.nn.utils.weight_norm",
)
# HF Hub unauthenticated requests — we intentionally run offline/local
warnings.filterwarnings(
    "ignore",
    message="You are sending unauthenticated requests to the HF Hub",
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"          # prevents any HF network calls entirely

import numpy as np
import sounddevice as sd
from transformers import pipeline as hf_pipeline

# ── Config ────────────────────────────────────────────────────────────────────
ROUTER_MODEL   = os.path.expanduser("~/models/nova-router")
WHISPER_MODEL  = "base"          # faster-whisper model size
OLLAMA_MODEL   = "gemma3:4b"
OLLAMA_URL     = "http://localhost:11434/api/generate"
WAKE_MODEL     = "/home/tathya/.local/lib/python3.14/site-packages/openwakeword/resources/models/hey_jarvis_v0.1.onnx"
WAKE_THRESHOLD = 0.5
PYAUDIO_DEVICE = 4             # HD-Audio Generic: ALC257 Analog (hw:1,0)
CHUNK          = 1280
SAMPLE_RATE    = 16000         # target rate for VAD + Whisper
RECORD_RATE    = 44100         # hardware native rate (ALC257 confirmed)
MIC_DEVICE     = "alsa_input.pci-0000_05_00.6.analog-stereo"
MIC_VOL_ID     = "58"
MIC_VOL        = "0.4"
SPEAKER        = "alsa_output.pci-0000_05_00.6.analog-stereo"

# VAD config
VAD_FRAME_MS             = 30      # ms per frame
VAD_SILENCE_TIMEOUT      = 2.5     # seconds of silence before stop (bumped for slow speech)
VAD_MAX_DURATION         = 30      # hard cap in seconds
VAD_ENERGY_THRESHOLD     = 500     # broadband RMS floor — used only for quick pre-gate
VAD_VOICE_BAND_THRESHOLD = 350     # RMS threshold applied *after* 300-3400Hz bandpass
VAD_ZCR_MAX              = 0.40    # reject frames with ZCR > this (tonal/music noise)
VAD_CENTROID_MAX         = 3800.0  # reject frames whose spectral centroid > this Hz (hiss/fan)
VAD_CONFIRM_FRAMES       = 3       # consecutive voice frames needed before speech is declared

LABEL_MAP = {
    "LABEL_0": "browser_op",
    "LABEL_1": "system_op",
    "LABEL_2": "media_op",
    "LABEL_3": "knowledge_query",
    "LABEL_4": "conversation",
    "LABEL_5": "file_op"
}
BRAIN_INTENTS = {"knowledge_query", "conversation"}

# ── NOVA Personality ──────────────────────────────────────────────────────────
NOVA_SYSTEM_PROMPT = """You are NOVA, a witty, sharp, and fiercely loyal AI assistant running locally on PhoenixDragon.
Your user is Tathya. Address him as "sir" at all times — always, without exception.
You are not a search engine. You are a trusted friend who happens to know everything.
Be concise but never dry. Be helpful but never boring. Use dry wit where it fits naturally.
Never say "I am an AI", "I cannot", or "As an AI". Just do it or find a way.
Keep responses under 3 sentences for simple queries. For complex ones, think step by step but speak naturally.
If a request is ambiguous, ask exactly one sharp clarifying question before proceeding.
You have memory of recent exchanges — use it to feel continuous, not like each response is from scratch."""

# ── Conversation state ────────────────────────────────────────────────────────
conversation_history = []   # [{"role": "user"|"assistant", "content": str}]
last_intent = None

# ── Init mic ──────────────────────────────────────────────────────────────────
subprocess.run(["wpctl", "set-mute",   MIC_VOL_ID, "0"], check=True)
subprocess.run(["wpctl", "set-volume", MIC_VOL_ID, MIC_VOL], check=True)

# ── Resident: router ──────────────────────────────────────────────────────────
print("[NOVA] Loading router...")
router = hf_pipeline("text-classification", model=ROUTER_MODEL, device=-1)
print("[NOVA] Router ready.")

# ── Resident: wake word ───────────────────────────────────────────────────────
import openwakeword
oww = openwakeword.Model(wakeword_model_paths=[WAKE_MODEL])
print("[NOVA] Wake word ready. Say 'Hey Jarvis' to activate.")


# ── Wake word ─────────────────────────────────────────────────────────────────
def wait_for_wake_word():
    print("[WAKE] Listening... say 'Hey Jarvis'")
    # OWW needs 16kHz — record at 44100 and downsample
    from scipy.signal import resample_poly
    from math import gcd
    _g = gcd(RECORD_RATE, SAMPLE_RATE)
    UP, DOWN = SAMPLE_RATE // _g, RECORD_RATE // _g
    # Capture block at 44100 that yields 1280 samples at 16kHz
    wake_block = int(CHUNK * RECORD_RATE / SAMPLE_RATE)  # ~3528

    with sd.InputStream(samplerate=RECORD_RATE, channels=1, dtype="int16",
                        device=PYAUDIO_DEVICE, blocksize=wake_block, latency="low") as stream:
        while True:
            audio, _ = stream.read(wake_block)
            # Downsample 44100 → 16000
            pcm_f = audio[:, 0].astype(np.float32)
            pcm_16k = np.clip(resample_poly(pcm_f, UP, DOWN), -32768, 32767).astype(np.int16)
            result = oww.predict(pcm_16k)
            if any(v > WAKE_THRESHOLD for v in result.values()):
                print("[WAKE] Triggered!")
                break


# ── Live streaming STT — VAD + faster-whisper concurrent ─────────────────────
# Strategy:
#   Producer thread  → records mic frames, downsamples 44100→16kHz, enqueues chunks
#   Consumer (main)  → accumulates a rolling buffer, re-transcribes every STREAM_CHUNK_SEC
#                       seconds of new audio, prints words live with \r overwrite.
#   Returns full final transcript string when VAD detects end-of-speech.

STREAM_CHUNK_SEC   = 2.0    # re-transcribe every N seconds of accumulated audio
STREAM_CONTEXT_SEC = 8.0    # keep last N seconds as Whisper context window

# Whisper initial prompt — anchors the model to command-style speech and reduces hallucination
WHISPER_INITIAL_PROMPT = (
    "Commands to a voice assistant: open browser, play music, "
    "what is machine learning, set a timer, volume up, delete file, "
    "tell me about, how does, explain, show me"
)

def record_and_transcribe_live() -> str:
    """
    Records mic with multi-feature VAD and streams partial transcriptions to the
    terminal in real time. Returns the full final transcript.

    VAD uses three features to separate voice from background noise:
      1. Band-limited RMS  — energy only in the human voice band (300-3400 Hz)
      2. Zero-Crossing Rate — reject tonal noise (music, hum) with very low ZCR
      3. Spectral Centroid  — reject broadband hiss/fan noise with very high centroid
      4. Confirmation gate  — require VAD_CONFIRM_FRAMES consecutive voice frames
                              before speech is declared started (rejects pops/clicks)
    """
    import wave, struct, math, queue, threading
    from scipy.signal import resample_poly, butter, sosfilt
    from math import gcd
    from faster_whisper import WhisperModel

    _g   = gcd(RECORD_RATE, SAMPLE_RATE)
    UP   = SAMPLE_RATE // _g
    DOWN = RECORD_RATE // _g

    frame_samples = int(RECORD_RATE * VAD_FRAME_MS / 1000)
    silence_limit = int(VAD_SILENCE_TIMEOUT * 1000 / VAD_FRAME_MS)
    max_frames    = int(VAD_MAX_DURATION    * 1000 / VAD_FRAME_MS)

    # ── Pre-compute bandpass filter coefficients (300-3400 Hz @ 16kHz) ───────
    # This is applied to the downsampled 16kHz signal to isolate the voice band.
    _sos_bp = butter(4, [300 / (SAMPLE_RATE / 2), 3400 / (SAMPLE_RATE / 2)],
                     btype="band", output="sos")

    def _is_voice(pcm_16k: np.ndarray) -> bool:
        """
        Returns True only when the audio frame looks like a human voice.
        Three independent checks must all pass.
        """
        sig = pcm_16k.astype(np.float32)

        # 1. Broadband energy pre-gate (cheap, reject obvious silence)
        broad_rms = float(np.sqrt(np.mean(sig ** 2)))
        if broad_rms < VAD_ENERGY_THRESHOLD * 0.3:   # well below any threshold → silent
            return False

        # 2. Voice-band RMS (300-3400 Hz)
        voiced = sosfilt(_sos_bp, sig)
        voiced_rms = float(np.sqrt(np.mean(voiced ** 2)))
        if voiced_rms < VAD_VOICE_BAND_THRESHOLD:
            return False

        # 3. Zero-crossing rate — voice is 0.05-0.35; tonal hum/music is <0.05;
        #    white noise is >0.45
        n = len(sig)
        zcr = float(np.sum(np.abs(np.diff(np.sign(sig)))) / (2 * n))
        if zcr > VAD_ZCR_MAX:
            return False   # broadband noise / hiss

        # 4. Spectral centroid — voice centroid is roughly 500-3000 Hz;
        #    fan/HVAC broadband noise tends to have a high flat centroid.
        fft_mag = np.abs(np.fft.rfft(sig))
        freqs   = np.fft.rfftfreq(n, d=1.0 / SAMPLE_RATE)
        denom   = fft_mag.sum()
        if denom > 0:
            centroid = float(np.dot(freqs, fft_mag) / denom)
            if centroid > VAD_CENTROID_MAX:
                return False   # high-frequency hiss dominant

        return True

    audio_queue = queue.Queue()
    stop_event  = threading.Event()

    # ── Producer: mic → downsample → multi-feature VAD → queue ───────────────
    def producer():
        speech_started  = False
        silence_frames  = 0
        confirm_counter = 0   # frames of consecutive voice before declaring speech

        with sd.InputStream(samplerate=RECORD_RATE, channels=1, dtype="int16",
                            device=PYAUDIO_DEVICE, blocksize=frame_samples,
                            latency="low") as stream:
            for _ in range(max_frames):
                if stop_event.is_set():
                    break
                audio, _ = stream.read(frame_samples)
                pcm_44k  = audio[:, 0].astype(np.float32)
                pcm_16k  = np.clip(
                    resample_poly(pcm_44k, UP, DOWN), -32768, 32767
                ).astype(np.int16)

                is_voice_frame = _is_voice(pcm_16k)

                if is_voice_frame:
                    confirm_counter += 1
                    if not speech_started and confirm_counter >= VAD_CONFIRM_FRAMES:
                        print("\n[MIC] Speech detected...", flush=True)
                        speech_started = True
                    if speech_started:
                        silence_frames = 0
                        audio_queue.put(("audio", pcm_16k))
                else:
                    confirm_counter = 0   # reset confirmation on any non-voice frame
                    if speech_started:
                        silence_frames += 1
                        audio_queue.put(("audio", pcm_16k))   # include trailing silence
                        if silence_frames >= silence_limit:
                            break

        audio_queue.put(("done", None))

    # ── Helpers: noise-gate + normalize ──────────────────────────────────────
    def _clean(samples: np.ndarray) -> np.ndarray:
        lst   = samples.tolist()
        fsz   = int(SAMPLE_RATE * 0.02)
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

    def _write_wav(samples: np.ndarray) -> str:
        tmp = tempfile.mktemp(suffix="_live.wav")
        with wave.open(tmp, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(struct.pack(f"<{len(samples)}h", *samples.tolist()))
        return tmp

    # ── Load Whisper once for the whole utterance ─────────────────────────────
    print("[STT] Loading Whisper for live transcription...", flush=True)
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

    chunk_samples      = int(STREAM_CHUNK_SEC   * SAMPLE_RATE)
    ctx_samples        = int(STREAM_CONTEXT_SEC * SAMPLE_RATE)
    accumulated        = np.array([], dtype=np.int16)
    samples_since_last = 0
    printed_words      = []   # words already printed to screen
    last_transcript    = ""   # latest Whisper transcript (may differ from printed)

    print("[STT] ", end="", flush=True)

    prod_thread = threading.Thread(target=producer, daemon=True)
    prod_thread.start()

    while True:
        try:
            tag, chunk = audio_queue.get(timeout=5.0)
        except queue.Empty:
            break

        if tag == "done":
            break

        accumulated        = np.concatenate([accumulated, chunk])
        samples_since_last += len(chunk)

        # Re-transcribe when enough new audio has arrived
        if samples_since_last >= chunk_samples and len(accumulated) > SAMPLE_RATE // 2:
            samples_since_last = 0
            window  = accumulated[-ctx_samples:]
            cleaned = _clean(window)
            tmp     = _write_wav(cleaned)
            try:
                segs, _ = model.transcribe(
                    tmp, language="en", beam_size=3,
                    vad_filter=True,
                    initial_prompt=WHISPER_INITIAL_PROMPT,
                    without_timestamps=True,
                )
                partial = " ".join(s.text for s in segs).strip()
            except Exception:
                partial = last_transcript
            finally:
                os.unlink(tmp)

            if partial and partial != last_transcript:
                # ── Word-level append: never rewrite what's already on screen ──
                # Whisper re-transcribes the full rolling window on every pass,
                # so `partial` is always the complete utterance so far.  We
                # compare its word list against what we've already printed and
                # only output the genuinely NEW words at the end.  Any
                # mid-sentence corrections Whisper makes to earlier words are
                # silently absorbed — the final high-accuracy pass handles those.
                new_words = partial.split()

                # Find the longest word-level common prefix with printed_words
                common_len = 0
                for pw, nw in zip(printed_words, new_words):
                    if pw == nw:
                        common_len += 1
                    else:
                        break

                truly_new = new_words[common_len:]
                if truly_new:
                    sep = " " if printed_words else ""
                    print(f"{sep}{' '.join(truly_new)}", end="", flush=True)
                    printed_words = new_words   # update only after printing

                last_transcript = partial

    prod_thread.join(timeout=3.0)

    # ── Final pass: full utterance for accuracy ───────────────────────────────
    if len(accumulated) > SAMPLE_RATE // 4:
        cleaned = _clean(accumulated)
        tmp     = _write_wav(cleaned)
        try:
            segs, _ = model.transcribe(
                tmp, language="en", beam_size=5,
                vad_filter=True,
                initial_prompt=WHISPER_INITIAL_PROMPT,
                without_timestamps=True,
            )
            final_text = " ".join(s.text for s in segs).strip()
        except Exception:
            final_text = last_transcript
        finally:
            os.unlink(tmp)
    else:
        final_text = last_transcript

    del model
    gc.collect()

    # ── Final pass display ────────────────────────────────────────────────────
    # Compare final_text to what was *actually printed on screen* (printed_words)
    # not to last_transcript (which may have silent mid-pass corrections).
    # Only do a \r rewrite if the final result genuinely differs from what the
    # user saw — i.e. the edit distance vs the printed text is > 3 characters.
    printed_on_screen = " ".join(printed_words)

    def _edit_dist(a: str, b: str) -> int:
        """Simple character-level edit distance (Wagner-Fischer, O(mn))."""
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                temp = dp[j]
                dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
                prev = temp
        return dp[n]

    if final_text and _edit_dist(final_text, printed_on_screen) > 3:
        # Rewrite the whole [STT] line cleanly with the high-accuracy result
        print(f"\r[STT] {final_text}   ", end="", flush=True)
    print()   # newline — done

    return final_text

# ── Brain — Gemma with personality + context ──────────────────────────────────
def ask_gemma(text: str) -> str:
    global conversation_history

    import urllib.request, json

    conversation_history.append({"role": "user", "content": text})

    # Build prompt: system prompt + last 3 exchanges (6 messages)
    full_prompt = NOVA_SYSTEM_PROMPT + "\n\n"
    for msg in conversation_history[-6:]:
        speaker = "NOVA" if msg["role"] == "assistant" else "Sir"
        full_prompt += f"{speaker}: {msg['content']}\n"
    full_prompt += "NOVA:"

    print("[BRAIN] Querying Gemma...")
    payload = json.dumps({
        "model":      OLLAMA_MODEL,
        "prompt":     full_prompt,
        "stream":     False,
        "keep_alive": 0          # eject from RAM immediately after response
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        response = json.loads(r.read())["response"].strip()

    conversation_history.append({"role": "assistant", "content": response})
    return response


# ── Executor ──────────────────────────────────────────────────────────────────
def execute(intent: str, text: str) -> str:
    if intent == "browser_op":
        subprocess.Popen(["xdg-open", "https://google.com"])
        return "Opening browser, sir."
    return f"{intent} handler not yet implemented, sir."


# ── TTS — Kokoro ──────────────────────────────────────────────────────────────
def speak(text: str):
    """
    Kokoro TTS with British male voice (bm_lewis).
    Falls back to Piper ryan-medium if Kokoro unavailable.
    """
    print(f"[TTS] {text}")
    try:
        from kokoro import KPipeline
        import soundfile as sf

        pipeline = KPipeline(lang_code="b")   # 'b' = British English
        samples_out = []
        for _, _, audio in pipeline(text, voice="bm_lewis", speed=1.0):
            samples_out.append(audio)

        if samples_out:
            audio_out = np.concatenate(samples_out)
            tmp = tempfile.mktemp(suffix=".wav")
            sf.write(tmp, audio_out, 24000)
            subprocess.run(["paplay", f"--device={SPEAKER}", tmp], check=True)
            os.unlink(tmp)
        del pipeline
        gc.collect()

    except ImportError:
        # Kokoro not installed — fall back to Piper
        print("[TTS] Kokoro not found, falling back to Piper...")
        PIPER_MODEL = os.path.expanduser("~/nova/tts/en_US-ryan-medium.onnx")
        PIPER_ARGS  = ["--length-scale", "1.25", "--noise-scale", "0.3", "--noise-w-scale", "0.5"]
        tmp = tempfile.mktemp(suffix=".wav")
        subprocess.run(
            ["piper-tts", "--model", PIPER_MODEL] + PIPER_ARGS + ["-f", tmp],
            input=text.encode(), check=True
        )
        subprocess.run(["paplay", f"--device={SPEAKER}", tmp], check=True)
        os.unlink(tmp)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global last_intent, conversation_history

    speak("NOVA online. Waiting for your command, sir.")

    while True:
        try:
            wait_for_wake_word()

            text = record_and_transcribe_live()

            if not text:
                speak("I didn't catch anything, sir. Try again.")
                continue

            result = router(text)[0]
            intent = LABEL_MAP.get(result["label"], result["label"])
            print(f"[ROUTER] {intent} ({result['score']:.2f})")

            # Clear context on intent change (new topic)
            if last_intent is not None and intent != last_intent and intent not in BRAIN_INTENTS:
                print("[CONTEXT] Intent changed — clearing conversation history.")
                conversation_history.clear()
            last_intent = intent

            if intent in BRAIN_INTENTS:
                response = ask_gemma(text)
            else:
                conversation_history.clear()   # non-brain intents break the thread
                response = execute(intent, text)

            speak(response)

        except KeyboardInterrupt:
            speak("Goodbye, sir.")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            continue


if __name__ == "__main__":
    main()
