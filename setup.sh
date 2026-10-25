#!/usr/bin/env bash
# Setup sekali per device: venv + dependency. Jalankan: ./setup.sh
set -euo pipefail
cd "$(dirname "$0")"

command -v ffmpeg >/dev/null || {
  echo "ffmpeg belum terpasang."
  echo "  macOS:  brew install ffmpeg"
  echo "  Ubuntu: sudo apt install ffmpeg"
  exit 1
}

command -v python3.11 >/dev/null || command -v uv >/dev/null || {
  echo "Python 3.11 atau uv belum ada. Install salah satunya dulu."
  exit 1
}

if [ ! -d .venv ]; then
  if command -v uv >/dev/null; then
    uv venv --python 3.11 .venv
  else
    python3.11 -m venv .venv
  fi
fi

if command -v uv >/dev/null; then
  uv pip install --python .venv/bin/python -r requirements.txt
else
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
fi

echo
echo "✅ Setup selesai. Langkah berikutnya:"
echo "   export HF_TOKEN=hf_xxx           # atau: .venv/bin/hf auth login"
echo "   .venv/bin/python transcribe.py \"audio/1 on 1 23oct.m4a\" medium 2"
