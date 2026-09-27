# Fish Speech Local Server Setup

This guide describes how to run a local neural text-to-speech service with [Fish Speech](https://github.com/fishaudio/fish-speech) for LaneTalk driving coaching audio.

## Architecture & Principles

- **Local-Only Inference**: Runs entirely on your local machine; does not use the hosted Fish Audio API and requires **no API keys**.
- **Isolated Dependencies**: Fish Speech, PyTorch, and its heavy dependencies run in their own dedicated Python environment and directory, keeping LaneTalk's laptop dependencies lightweight (`requirements.txt`).
- **Clean Repo**: Model weights are stored externally and never checked into the project repository.
- **Fail-Safe Fallback**: If the local Fish Speech server is stopped or unreachable, LaneTalk automatically falls back to native operating system speech (macOS `say`, Windows `SAPI.SpVoice`, Linux `spd-say`/`espeak`).

---

## 1. Prerequisites

- **Python**: 3.10 or 3.11 (recommended in a separate Conda or virtual environment).
- **Hardware Acceleration**:
  - NVIDIA GPU with CUDA 12+ (Linux/Windows recommended for lowest synthesis latency)
  - Apple Silicon Mac with MPS
  - CPU inference is supported as a fallback.
- **System Tools**: `git`, `ffmpeg`, and `libsndfile1` (Linux: `sudo apt install ffmpeg libsndfile1`).

---

## 2. Server Installation (External Directory)

Set up the Fish Speech server in an external directory outside this repository (for example, in your home folder `~/fish-speech`):

```bash
# 1. Clone Fish Speech to an external directory
cd ~
git clone https://github.com/fishaudio/fish-speech.git
cd fish-speech

# 2. Create and activate a separate virtual environment
python3.10 -m venv .venv
source .venv/bin/activate
# Or with conda:
# conda create -n fish-speech python=3.10 -y && conda activate fish-speech

# 3. Install PyTorch according to your platform (e.g. CUDA 12)
pip install torch torchvision torchaudio

# 4. Install Fish Speech requirements
pip install -e .
```

---

## 3. Download Model Checkpoints

Download the Fish Speech model weights into your external `~/fish-speech/checkpoints` folder (do **not** copy weights into the `shellhacks-hardware` repo):

```bash
cd ~/fish-speech
pip install huggingface_hub

# Download Fish Speech 1.5 weights
huggingface-cli download fishaudio/fish-speech-1.5 --local-dir checkpoints/fish-speech-1.5
```

---

## 4. Run the Local API Server

Launch the Fish Speech HTTP API server on port 8080:

```bash
cd ~/fish-speech
source .venv/bin/activate

python tools/api_server.py \
  --listen 127.0.0.1:8080 \
  --llama-checkpoint-path checkpoints/fish-speech-1.5 \
  --decoder-checkpoint-path checkpoints/fish-speech-1.5/codec.pth
```

To verify the server is responding, test it from another terminal:

```bash
curl -X POST http://127.0.0.1:8080/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "LaneTalk is online.", "format": "wav"}' \
  --output /tmp/test.wav
```

---

## 5. Pointing LaneTalk at the Local Server

LaneTalk's `speech.py` module reads the following environment variables:

| Variable | Default | Description |
|---|---|---|
| `FISH_SPEECH_URL` | `http://127.0.0.1:8080/v1/tts` | URL of the local Fish Speech server. |
| `FISH_SPEECH_TIMEOUT` | `5.0` | Timeout in seconds per synthesis request. |
| `FISH_SPEECH_REFERENCE_ID` | `""` | Optional reference voice ID for custom voice cloning. |
| `FISH_SPEECH_DISABLED` | `0` | Set to `1` or `true` to force local platform TTS fallback. |

### Running LaneTalk

To run the LaneTalk app pointing at your local Fish Speech server:

```bash
# In the shellhacks-hardware repo (laptop/ directory):
export FISH_SPEECH_URL="http://127.0.0.1:8080"

# Test speech output directly:
python -m lanetalk.speech "Drifting left. Steer gently back to center."

# Or run the full application:
python -m lanetalk.app
```

---

## 6. Automatic Local Speech Fallback

If the local Fish Speech service is offline or slow to respond:
- A brief warning is logged once to stdout (`[speech] Fish Speech unavailable...`).
- A 15-second retry cooldown activates so video frame processing is not delayed by repeated connection attempts.
- LaneTalk immediately falls back to your operating system's native voice:
  - **macOS**: Built-in `/usr/bin/say` command.
  - **Windows**: Built-in PowerShell `SAPI.SpVoice`.
  - **Linux**: `spd-say` or `espeak`.
  - **Terminal**: Always logs `LaneTalk: <message>` to stdout.
