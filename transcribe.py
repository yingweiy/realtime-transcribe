#!/usr/bin/env python3
"""
Real-time transcription with configurable language support.

Usage:
  python3 transcribe.py --language Chinese English
  python3 transcribe.py --language Japanese --save
"""

import argparse
import itertools
import sys
import threading
import time
from datetime import datetime
from difflib import SequenceMatcher
from RealtimeSTT import AudioToTextRecorder

YELLOW = "\033[93m"
GREEN  = "\033[92m"
DIM    = "\033[2m"
RESET  = "\033[0m"
CLEAR_LINE = "\r\033[K"

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

# Ranges allowed regardless of language: ASCII, punctuation, digits, CJK punctuation.
_ALWAYS_ALLOWED: list[tuple[int, int]] = [
    (0x0000, 0x007F),  # Basic ASCII (a-z, A-Z, 0-9, punctuation, space)
    (0x2000, 0x206F),  # General Punctuation
    (0x3000, 0x303F),  # CJK Symbols and Punctuation
    (0xFE10, 0xFE4F),  # CJK Compatibility Forms
    (0xFF00, 0xFFEF),  # Halfwidth and Fullwidth Forms
]

# Allowlist: Unicode ranges that are permitted per Whisper language code.
# Only characters in these ranges (or _ALWAYS_ALLOWED) are accepted.
_LANGUAGE_SCRIPTS: dict[str, list[tuple[int, int]]] = {
    "en": [(0x0080, 0x024F)],   # Latin Extended (accented chars, covers most European scripts too)
    "zh": [
        (0x4E00, 0x9FFF),        # CJK Unified Ideographs
        (0x3400, 0x4DBF),        # CJK Extension A
        (0xF900, 0xFAFF),        # CJK Compatibility Ideographs
    ],
    "ja": [
        (0x3040, 0x309F),        # Hiragana
        (0x30A0, 0x30FF),        # Katakana
        (0x4E00, 0x9FFF),        # CJK (shared with Chinese)
        (0x3400, 0x4DBF),        # CJK Extension A
    ],
    "ko": [
        (0xAC00, 0xD7A3),        # Hangul Syllables
        (0x1100, 0x11FF),        # Hangul Jamo
        (0x3130, 0x318F),        # Hangul Compatibility Jamo
        (0xA960, 0xA97F),        # Hangul Jamo Extended-A
        (0xD7B0, 0xD7FF),        # Hangul Jamo Extended-B
    ],
    "ru": [
        (0x0400, 0x04FF),        # Cyrillic
        (0x0500, 0x052F),        # Cyrillic Supplement
    ],
    "fr": [(0x0080, 0x024F)],
    "es": [(0x0080, 0x024F)],
    "de": [(0x0080, 0x024F)],
    "pt": [(0x0080, 0x024F)],
}

class _WaveAnimation:
    """Animates a scrolling wave bar with elapsed time while recording."""

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


# Known Whisper hallucination substrings (language-model training data leaking through)
_HALLUCINATION_PHRASES = frozenset([
    "请不吝点赞",
    "订阅转发",
    "打赏支持",
    "明镜和点点",
    "感谢您的观看",
    "请订阅我们的频道",
    "如果你喜欢这个视频",
    "别忘了点赞订阅",
])


def _resolve_languages(names: list[str]) -> list[str]:
    """Convert language names to Whisper codes, with clear error on unknown names."""
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


def _build_allowed_ranges(selected_codes: list[str]) -> list[tuple[int, int]]:
    """Return the union of always-allowed ranges and the scripts for selected languages."""
    ranges = list(_ALWAYS_ALLOWED)
    for code in selected_codes:
        ranges.extend(_LANGUAGE_SCRIPTS.get(code, []))
    return ranges


def _build_initial_prompt(codes: list[str]) -> str:
    # Use only the first language's prompt. A bilingual prompt causes Whisper to
    # mimic the pattern and append translations to every transcribed segment.
    return _PROMPTS.get(codes[0], "") if codes else ""


class Transcriber:
    def __init__(self, language_codes: list[str], save_to_file: bool = False):
        self.session_lines: list[tuple[str, str]] = []
        self.save_to_file = save_to_file
        self._last_text: str = ""
        self._wave = _WaveAnimation()
        self._allowed_ranges = _build_allowed_ranges(language_codes)
        self.output_path = (
            f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            if save_to_file else None
        )

        # Use exact language code when only one is selected; None triggers auto-detect
        whisper_lang = language_codes[0] if len(language_codes) == 1 else None

        self.recorder = AudioToTextRecorder(
            model="large-v3",
            language=whisper_lang,
            spinner=False,
            initial_prompt=_build_initial_prompt(language_codes),
            on_recording_start=self._on_recording_start,
            on_recording_stop=self._on_recording_stop,
        )

    def _on_recording_start(self):
        self._wave.start()

    def _on_recording_stop(self):
        self._wave.stop()
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
        print(f"{CLEAR_LINE}{GREEN}[{ts}] {text}{RESET}")
        self.session_lines.append((ts, text))
        if self.save_to_file:
            with open(self.output_path, "a", encoding="utf-8") as f:
                f.write(f"[{ts}] {text}\n")

    def run(self):
        print(f"{GREEN}Listening... (Ctrl+C to stop){RESET}\n")
        if self.output_path:
            print(f"Saving to: {self.output_path}\n")
        try:
            while True:
                self.recorder.text(self._on_final)
        except KeyboardInterrupt:
            self._stop()

    def _stop(self):
        self.recorder.stop()
        print(f"\n\n{GREEN}Session ended. {len(self.session_lines)} segments transcribed.{RESET}")
        if self.output_path and self.session_lines:
            print(f"Transcript saved to: {self.output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time speech transcription")
    parser.add_argument(
        "--language", nargs="+", default=["Chinese", "English"],
        metavar="LANG",
        help="One or more languages to transcribe (default: Chinese English)",
    )
    parser.add_argument("--save", action="store_true", help="Save transcript to file")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    codes = _resolve_languages(args.language)
    print(f"Languages: {', '.join(c.upper() for c in codes)}")
    Transcriber(language_codes=codes, save_to_file=args.save).run()
