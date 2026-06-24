#!/usr/bin/env python3
"""
Real-time transcription with configurable language support and optional speaker diarization.

Usage:
  python3 transcribe.py --language Chinese English
  python3 transcribe.py --language Chinese English --diarize
  python3 transcribe.py --language Japanese --save
"""

import argparse
import itertools
import os
import sys
import threading
import time
import warnings
from datetime import datetime
from difflib import SequenceMatcher

# Suppress pkg_resources deprecation warning from webrtcvad (third-party issue)
warnings.filterwarnings("ignore", category=UserWarning, module="webrtcvad")
warnings.filterwarnings("ignore", message=".*pkg_resources.*")

# Suppress ctranslate2 float16→float32 warning (harmless on Apple Silicon)
os.environ.setdefault("CT2_VERBOSE", "0")

import ctranslate2
ctranslate2.set_log_level(40)  # ERROR — silences WARNING-level messages

from RealtimeSTT import AudioToTextRecorder
import numpy as np

YELLOW     = "\033[93m"
GREEN      = "\033[92m"
DIM        = "\033[2m"
RESET      = "\033[0m"
CLEAR_LINE = "\r\033[K"

# Colors cycled across speakers in diarization mode
_SPEAKER_COLORS = ["\033[96m", "\033[95m", "\033[94m", "\033[33m", "\033[36m"]
_SPEAKER_LABELS = list("ABCDEFGH")

# Map human-readable names → Whisper language codes
_NAME_TO_CODE = {
    "english":    "en",
    "chinese":    "zh",
    "mandarin":   "zh",
    "japanese":   "ja",
    "korean":     "ko",
    "french":     "fr",
    "spanish":    "es",
    "german":     "de",
    "portuguese": "pt",
    "russian":    "ru",
}

# Whisper initial prompts per language code — biases language detection
_PROMPTS = {
    "en": "The following is a conversation in English.",
    "zh": "以下是普通话的对话。",
    "ja": "以下は日本語の会話です。",
    "ko": "다음은 한국어로 된 대화입니다.",
    "fr": "Ce qui suit est une conversation en français.",
    "es": "Lo siguiente es una conversación en español.",
    "de": "Das Folgende ist ein Gespräch auf Deutsch.",
    "pt": "O seguinte é uma conversa em português.",
    "ru": "Следующее — разговор на русском языке.",
}

# Always-permitted Unicode ranges regardless of selected languages
_ALWAYS_ALLOWED: list[tuple[int, int]] = [
    (0x0000, 0x007F),  # Basic ASCII
    (0x2000, 0x206F),  # General Punctuation
    (0x3000, 0x303F),  # CJK Symbols and Punctuation
    (0xFE10, 0xFE4F),  # CJK Compatibility Forms
    (0xFF00, 0xFFEF),  # Halfwidth and Fullwidth Forms
]

# Unicode script ranges allowed per language code
_LANGUAGE_SCRIPTS: dict[str, list[tuple[int, int]]] = {
    "en": [(0x0080, 0x024F)],
    "zh": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF)],
    "ja": [(0x3040, 0x309F), (0x30A0, 0x30FF), (0x4E00, 0x9FFF), (0x3400, 0x4DBF)],
    "ko": [(0xAC00, 0xD7A3), (0x1100, 0x11FF), (0x3130, 0x318F),
           (0xA960, 0xA97F), (0xD7B0, 0xD7FF)],
    "ru": [(0x0400, 0x04FF), (0x0500, 0x052F)],
    "fr": [(0x0080, 0x024F)],
    "es": [(0x0080, 0x024F)],
    "de": [(0x0080, 0x024F)],
    "pt": [(0x0080, 0x024F)],
}

# Known Whisper hallucination substrings
_HALLUCINATION_PHRASES = frozenset([
    "请不吝点赞", "订阅转发", "打赏支持", "明镜和点点",
    "感谢您的观看", "请订阅我们的频道", "如果你喜欢这个视频", "别忘了点赞订阅",
])


# ---------------------------------------------------------------------------
# Wave animation
# ---------------------------------------------------------------------------

class _WaveAnimation:
    _FRAMES = [
        "▁▂▃▄▅▄▃▂", "▂▃▄▅▆▅▄▃", "▃▄▅▆▇▆▅▄",
        "▄▅▆▇█▇▆▅", "▅▆▇█▇▆▅▄", "▆▇█▇▆▅▄▃",
        "▇█▇▆▅▄▃▂", "█▇▆▅▄▃▂▁", "▇▆▅▄▃▂▁▂",
        "▆▅▄▃▂▁▂▃", "▅▄▃▂▁▂▃▄", "▄▃▂▁▂▃▄▅",
        "▃▂▁▂▃▄▅▆", "▂▁▂▃▄▅▆▇", "▁▂▃▄▅▆▇█",
    ]

    def __init__(self):
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self._stop.clear()
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.3)

    def _run(self):
        for frame in itertools.cycle(self._FRAMES):
            if self._stop.is_set():
                break
            elapsed = time.time() - self._t0
            print(f"{CLEAR_LINE}{YELLOW}● {frame}  {elapsed:.1f}s{RESET}", end="", flush=True)
            time.sleep(0.08)


# ---------------------------------------------------------------------------
# Speaker tracker
# ---------------------------------------------------------------------------

class SpeakerTracker:
    """
    Identifies speakers by comparing voice embeddings with cosine similarity.
    Slides a window across each segment to detect speaker changes within it.
    Assigns consistent labels (SPEAKER A, B, …) within a session.
    """

    SAMPLE_RATE = 16000
    MIN_SECONDS  = 0.5   # too brief for a reliable embedding
    WINDOW_SECS  = 1.5   # each analysis window
    HOP_SECS     = 0.75  # step between windows

    def __init__(self, threshold: float = 0.75):
        from resemblyzer import VoiceEncoder
        self._encoder = VoiceEncoder()
        # Each entry: (label, color, embedding)
        self._speakers: list[tuple[str, str, np.ndarray]] = []
        self._threshold = threshold

    def _match_or_register(self, embedding: np.ndarray) -> tuple[str, str]:
        if not self._speakers:
            return self._register(embedding)
        similarities = [
            float(np.dot(embedding, emb) /
                  (np.linalg.norm(embedding) * np.linalg.norm(emb) + 1e-9))
            for _, _, emb in self._speakers
        ]
        best = int(np.argmax(similarities))
        if similarities[best] >= self._threshold:
            label, color, _ = self._speakers[best]
            return (label, color)
        return self._register(embedding)

    def identify(self, audio_bytes: bytes) -> tuple[str, str]:
        """
        Return (label, color) for the dominant speaker in audio_bytes.
        If multiple speakers are detected within the segment, returns their
        labels joined by ' → ' in order of first appearance.
        Returns ("", "") if the audio is too short to embed reliably.
        """
        from resemblyzer import preprocess_wav

        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if len(audio) < self.SAMPLE_RATE * self.MIN_SECONDS:
            return ("", "")

        wav = preprocess_wav(audio, source_sr=self.SAMPLE_RATE)

        window = int(self.SAMPLE_RATE * self.WINDOW_SECS)
        hop    = int(self.SAMPLE_RATE * self.HOP_SECS)

        # Segment shorter than one window — embed it whole
        if len(wav) < window:
            return self._match_or_register(self._encoder.embed_utterance(wav))

        # Slide across the segment; collect per-window speaker assignments
        seen: list[tuple[str, str]] = []
        for start in range(0, len(wav) - window + 1, hop):
            result = self._match_or_register(
                self._encoder.embed_utterance(wav[start:start + window])
            )
            # Only record a new entry when the speaker changes (consecutive dedup)
            if not seen or seen[-1] != result:
                seen.append(result)

        if len(seen) == 1:
            return seen[0]

        # Multiple speakers — deduplicate globally (keep first-appearance order)
        unique: list[tuple[str, str]] = []
        for s in seen:
            if s not in unique:
                unique.append(s)
        label = " → ".join(lbl for lbl, _ in unique)
        return (label, unique[0][1])

    def _register(self, embedding: np.ndarray) -> tuple[str, str]:
        idx = len(self._speakers)
        label = f"SPEAKER {_SPEAKER_LABELS[idx % len(_SPEAKER_LABELS)]}"
        color = _SPEAKER_COLORS[idx % len(_SPEAKER_COLORS)]
        self._speakers.append((label, color, embedding))
        print(f"\n{color}[new speaker detected: {label}]{RESET}")
        return (label, color)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_languages(names: list[str]) -> list[str]:
    codes = []
    for name in names:
        key = name.lower()
        if key not in _NAME_TO_CODE:
            known = ", ".join(sorted({k.title() for k in _NAME_TO_CODE}))
            print(f"Unknown language '{name}'. Supported: {known}", file=sys.stderr)
            sys.exit(1)
        code = _NAME_TO_CODE[key]
        if code not in codes:
            codes.append(code)
    return codes


def _build_allowed_ranges(codes: list[str]) -> list[tuple[int, int]]:
    ranges = list(_ALWAYS_ALLOWED)
    for code in codes:
        ranges.extend(_LANGUAGE_SCRIPTS.get(code, []))
    return ranges


def _build_initial_prompt(codes: list[str]) -> str:
    # Single-language prompt prevents Whisper from mimicking bilingual output
    return _PROMPTS.get(codes[0], "") if codes else ""


# ---------------------------------------------------------------------------
# Transcriber
# ---------------------------------------------------------------------------

class Transcriber:
    def __init__(self, language_codes: list[str], save_to_file: bool = False,
                 diarize: bool = False):
        self.session_lines: list[tuple[str, str, str]] = []  # (ts, speaker, text)
        self.save_to_file = save_to_file
        self._last_text: str = ""
        self._wave = _WaveAnimation()
        self._allowed_ranges = _build_allowed_ranges(language_codes)
        self._audio_buffer: list[bytes] = []
        self._pending_audio: bytes = b""  # snapshotted at recording stop
        self._speaker_tracker = SpeakerTracker() if diarize else None
        self.output_path = (
            f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            if save_to_file else None
        )

        whisper_lang = language_codes[0] if len(language_codes) == 1 else None

        self.recorder = AudioToTextRecorder(
            model="large-v3",
            language=whisper_lang,
            spinner=False,
            initial_prompt=_build_initial_prompt(language_codes),
            on_recording_start=self._on_recording_start,
            on_recording_stop=self._on_recording_stop,
            on_recorded_chunk=self._on_chunk if diarize else None,
        )

    def _on_chunk(self, data: bytes):
        self._audio_buffer.append(data)

    def _on_recording_start(self):
        self._audio_buffer.clear()
        self._wave.start()

    def _on_recording_stop(self):
        self._wave.stop()
        # Snapshot the buffer now — _on_recording_start for the *next* utterance
        # may fire (clearing _audio_buffer) before _on_final is called for this one.
        self._pending_audio = b"".join(self._audio_buffer)
        self._audio_buffer.clear()
        print(f"{CLEAR_LINE}{DIM}Processing...{RESET}", end="", flush=True)

    def _allowed(self, text: str) -> bool:
        return all(
            any(lo <= ord(ch) <= hi for lo, hi in self._allowed_ranges)
            for ch in text
        )

    def _is_hallucination(self, text: str) -> bool:
        if any(phrase in text for phrase in _HALLUCINATION_PHRASES):
            return True
        if self._last_text and SequenceMatcher(None, text, self._last_text).ratio() > 0.85:
            return True
        return False

    def _on_final(self, text: str):
        if not self._allowed(text):
            print(f"{CLEAR_LINE}{DIM}[skipped: out-of-language script]{RESET}")
            return
        if self._is_hallucination(text):
            print(f"{CLEAR_LINE}{DIM}[skipped: hallucination]{RESET}")
            return

        self._last_text = text
        ts = datetime.now().strftime("%H:%M:%S")

        speaker_label = ""
        line_color = GREEN

        if self._speaker_tracker and self._pending_audio:
            label, color = self._speaker_tracker.identify(self._pending_audio)
            if label:
                speaker_label = f"{label}: "
                line_color = color

        print(f"{CLEAR_LINE}{line_color}[{ts}] {speaker_label}{text}{RESET}")
        self.session_lines.append((ts, speaker_label.strip(": "), text))

        if self.save_to_file:
            with open(self.output_path, "a", encoding="utf-8") as f:
                prefix = f"{speaker_label}" if speaker_label else ""
                f.write(f"[{ts}] {prefix}{text}\n")

    def run(self):
        print(f"{GREEN}Listening... (Ctrl+C to stop){RESET}\n")
        if self._speaker_tracker:
            print(f"{DIM}Speaker diarization enabled{RESET}\n")
        if self.output_path:
            print(f"Saving to: {self.output_path}\n")
        try:
            while True:
                self.recorder.text(self._on_final)
        except KeyboardInterrupt:
            self._stop()

    def _stop(self):
        self.recorder.stop()
        n = len(self.session_lines)
        print(f"\n\n{GREEN}Session ended. {n} segment{'s' if n != 1 else ''} transcribed.{RESET}")
        if self.output_path and self.session_lines:
            print(f"Transcript saved to: {self.output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time speech transcription")
    parser.add_argument(
        "--language", nargs="+", default=["Chinese", "English"],
        metavar="LANG",
        help="One or more languages to transcribe (default: Chinese English)",
    )
    parser.add_argument("--save", action="store_true", help="Save transcript to file")
    parser.add_argument("--diarize", action="store_true", help="Enable speaker diarization")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    codes = _resolve_languages(args.language)
    print(f"Languages: {', '.join(c.upper() for c in codes)}")
    Transcriber(language_codes=codes, save_to_file=args.save, diarize=args.diarize).run()
