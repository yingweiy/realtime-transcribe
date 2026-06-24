# realtime-transcribe

Real-time speech transcription using [Whisper](https://github.com/openai/whisper), optimized for multilingual use (English, Chinese, and more). Runs fully locally — no cloud API required.

## How it works

Uses a hybrid approach:

- **While speaking** — a wave animation shows recording is active
- **On pause** — `large-v3` transcribes the segment and prints the final text
- **Filtering** — out-of-language scripts and known Whisper hallucinations are silently dropped

## Requirements

- Python 3.10+
- [Homebrew](https://brew.sh) (macOS)
- Apple Silicon recommended (M1 or later) for fast local inference

## Installation

```bash
brew install portaudio

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Models are downloaded automatically from Hugging Face on first run (~3 GB total).

## Usage

```bash
source .venv/bin/activate

# Default: Chinese + English
python3 transcribe.py

# Single language (faster — skips auto-detection)
python3 transcribe.py --language Chinese
python3 transcribe.py --language English

# Multiple languages
python3 transcribe.py --language Chinese English
python3 transcribe.py --language Japanese English

# Save transcript to file
python3 transcribe.py --language Chinese English --save
```

Press `Ctrl+C` to stop. Saved transcripts are written to `transcript_YYYYMMDD_HHMMSS.txt`.

## Supported languages

| Name | Name | Name |
|------|------|------|
| Chinese (Mandarin) | English | Japanese |
| Korean | French | Spanish |
| German | Portuguese | Russian |

## Output

```
Languages: ZH, EN
Listening... (Ctrl+C to stop)

● ▄▅▆▇█▇▆▅  1.4s          ← wave animation while speaking
Processing...               ← Whisper running
[22:07:12] 不是,如果100块钱就很好说。   ← final transcript
[22:07:18] Yeah that makes sense.
```
