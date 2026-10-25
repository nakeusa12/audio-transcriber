"""Self-check: sintesis audio 2 suara beda via macOS `say`, lalu transkripsi + diarization.

Jalankan:  .venv/bin/python verify.py
HF_TOKEN di env → cek diarization juga; tanpa token → cek transkripsi saja.
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd):
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        sys.exit(f"❌ perintah gagal: {' '.join(cmd)}\n{e.stderr.decode()[:400]}")


def main():
    if not shutil.which("say"):
        sys.exit("❌ skrip ini butuh macOS (perintah `say` tidak ada)")
    if not shutil.which("ffmpeg"):
        sys.exit("❌ ffmpeg tidak ada — install: brew install ffmpeg")

    work = Path(tempfile.mkdtemp(prefix="verify_tx_"))
    a1, a2, wav = work / "a1.aiff", work / "a2.aiff", work / "both.wav"
    print("⏳ sintesis 2 suara (Daniel + Samantha)…")
    try:
        run(["say", "-v", "Daniel",
             "Hello there. This is the first speaker, talking about the weather today.",
             "-o", str(a1)])
        run(["say", "-v", "Samantha",
             "Hi! I am the second speaker, and this is a test of the transcription tool.",
             "-o", str(a2)])
    except SystemExit:
        sys.exit('❌ voice tidak tersedia. Lihat daftar: say -v "?"')
    run(["ffmpeg", "-y", "-i", str(a1), "-i", str(a2),
         "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1[out]", "-map", "[out]", str(wav)])

    import whisperx

    print("⏳ memuat model 'base' (download pertama ±150 MB)…")
    asr = whisperx.load_model("base", "cpu", compute_type="int8")
    print("⏳ transkripsi…")
    result = asr.transcribe(whisperx.load_audio(str(wav)), batch_size=16)
    text = " ".join(s["text"] for s in result["segments"]).strip()
    assert len(text) > 20, f"❌ transkrip kosong/terlalu pendek: {text!r}"
    print(f"✅ transkripsi ok — bahasa={result.get('language')} ({len(text)} chars): {text[:80]}…")

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("⚠️ HF_TOKEN tidak diset — cek diarization dilewati. Set: export HF_TOKEN=hf_xxx")
        return

    from whisperx.diarize import DiarizationPipeline

    print("⏳ diarization (download model pertama)…")
    turns = DiarizationPipeline(model_name="pyannote/speaker-diarization-3.1", token=token)(
        whisperx.load_audio(str(wav))
    )  # DataFrame start/end/speaker
    speakers = set(turns["speaker"].unique())
    assert len(speakers) >= 2, f"❌ deteksi {len(speakers)} pembicara, harusnya ≥ 2: {speakers}"
    print(f"✅ diarization ok — {len(speakers)} pembicara terdeteksi: {sorted(speakers)}")


if __name__ == "__main__":
    main()
