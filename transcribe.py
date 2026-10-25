"""Transkripsi CLI tanpa UI: .venv/bin/python transcribe.py <audio> [model] [num_speakers]

Default model 'small'. Bahasa dipaksa 'id'. Device otomatis: CUDA kalau tersedia, else CPU.
Hasil ke output/<nama>.txt/.srt/.json — sama dengan versi app.
"""
import json
import os
import sys
import time
from pathlib import Path

OUT_DIR = Path(__file__).parent / "output"


def fmt_ts(sec):
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def srt_ts(sec):
    ms = int(round(sec * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def pretty(name):
    try:
        return f"Speaker {int(str(name).split('_')[-1]) + 1}"
    except (ValueError, IndexError):
        return str(name) or "Speaker 1"


def main():
    src = Path(sys.argv[1])
    model_size = sys.argv[2] if len(sys.argv) > 2 else "small"
    num_speakers = int(sys.argv[3]) if len(sys.argv) > 3 else None
    token = os.environ.get("HF_TOKEN", "")
    if not token:  # kalau sudah `hf auth login`, pakai token dari cache
        try:
            from huggingface_hub import get_token
            token = get_token() or ""
        except Exception:
            pass
    t0 = time.time()

    import torch
    import whisperx

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"MULAI: {src.name} | model={model_size} | device={device} | lang=id | "
          f"speakers={num_speakers or 'auto'}", flush=True)
    audio = whisperx.load_audio(str(src))
    asr = whisperx.load_model(model_size, device, compute_type=compute_type)
    print(f"model siap ({time.time()-t0:.0f}s) — transkripsi…", flush=True)
    result = asr.transcribe(audio, batch_size=16, language="id")
    segs = result["segments"]
    print(f"transkripsi ok: {len(segs)} segmen ({time.time()-t0:.0f}s) — align…", flush=True)

    try:
        am, md = whisperx.load_align_model("id", device)
        result = whisperx.align(segs, am, md, audio, device, return_char_alignments=False)
        segs = result["segments"]
    except Exception as e:
        print(f"align dilewati: {type(e).__name__}: {e}", flush=True)

    # ponytail: token opsional — model pyannote yang sudah ter-cache bisa load offline (HF_HUB_OFFLINE=1)
    try:
        from whisperx.diarize import DiarizationPipeline, assign_word_speakers

        df = DiarizationPipeline(
            model_name="pyannote/speaker-diarization-3.1", token=token or None, device=device
        )(audio, num_speakers=num_speakers)
        if segs and "words" in segs[0]:
            result = assign_word_speakers(df, result)
            segs = result["segments"]
        else:  # fallback: label per segmen via overlap midpoint
            turns = list(zip(df["start"], df["end"], df["speaker"]))
            for s in segs:
                mid = (s["start"] + s["end"]) / 2
                s["speaker"] = next((sp for a, b, sp in turns if a <= mid <= b), "SPEAKER_00")
        print(f"diarization ok ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"diarization GAGAL: {type(e).__name__}: {e} — label Speaker 1 semua", flush=True)

    for s in segs:
        s["speaker"] = pretty(s.get("speaker", "SPEAKER_00"))

    dur = max((s["end"] for s in segs), default=0.0)
    stem = src.stem
    OUT_DIR.mkdir(exist_ok=True)
    txt = "\n".join(f"[{fmt_ts(s['start'])} – {fmt_ts(s['end'])}] {s['speaker']}: {s['text'].strip()}" for s in segs)
    (OUT_DIR / f"{stem}.txt").write_text(txt, encoding="utf-8")
    srt = "\n".join(
        f"{i}\n{srt_ts(s['start'])} --> {srt_ts(s['end'])}\n{s['speaker']}: {s['text'].strip()}\n"
        for i, s in enumerate(segs, 1)
    )
    (OUT_DIR / f"{stem}.srt").write_text(srt, encoding="utf-8")
    data = {
        "file": src.name, "language": "id", "duration_sec": round(dur, 2),
        "speakers": len({s["speaker"] for s in segs}), "segments": segs,
    }
    (OUT_DIR / f"{stem}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"HASIL TERSIMPAN: {OUT_DIR}/{stem}.txt/.srt/.json | {len(segs)} segmen, "
          f"{data['speakers']} pembicara, total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
