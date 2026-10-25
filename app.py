"""Transkripsi audio → teks lengkap + label pembicara. 100% lokal via WhisperX.

Pekerjaan berat jalan di background thread — selamat dari rerun halaman,
reconnect, atau Mac tidur. Hasil selalu tertulis ke output/.
"""
import json
import os
import shutil
import tempfile
import threading
import time
import traceback
from pathlib import Path

import streamlit as st

import jobstore

st.set_page_config(page_title="Transkripsi Audio", page_icon="🎙️", layout="wide")

MODEL_CHOICES = {
    "large-v3 — paling akurat (±3 GB, lambat di CPU)": "large-v3",
    "medium — seimbang": "medium",
    "small — cepat": "small",
    "base — tercepat (buat coba-coba)": "base",
}
LANG_CHOICES = {"Deteksi otomatis": None, "Bahasa Indonesia": "id", "English": "en"}
AUDIO_TYPES = ["mp3", "wav", "m4a", "ogg", "oga", "webm", "flac", "aac", "aiff"]
OUT_DIR = Path(__file__).parent / "output"


def fmt_ts(sec: float) -> str:
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def srt_ts(sec: float) -> str:
    ms = int(round(sec * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def pretty_speaker(name) -> str:
    # "SPEAKER_00" → "Speaker 1"
    try:
        return f"Speaker {int(str(name).split('_')[-1]) + 1}"
    except (ValueError, IndexError):
        return str(name) or "Speaker 1"


@st.cache_resource(show_spinner=False)
def load_asr(model_size: str):
    import whisperx

    # ponytail: CPU int8 di Mac — ganti ke device="cuda", compute_type="float16" kalau ada GPU NVIDIA
    return whisperx.load_model(model_size, "cpu", compute_type="int8")


@st.cache_resource(show_spinner=False)
def load_diarize(token: str):
    from whisperx.diarize import DiarizationPipeline

    # whisperx >=3.8: param "token"; pin 3.1 (default "community-1" butuh accept lisensi lain)
    return DiarizationPipeline(model_name="pyannote/speaker-diarization-3.1", token=token)


def assign_by_overlap(diarize_df, segments):
    # ponytail: fallback kalau word-align gagal — cukup overlap midpoint utk label per segmen
    turns = list(zip(diarize_df["start"], diarize_df["end"], diarize_df["speaker"]))
    for seg in segments:
        mid = (seg["start"] + seg["end"]) / 2
        seg["speaker"] = next((sp for s, e, sp in turns if s <= mid <= e), "SPEAKER_00")


def transcribe(audio: Path, model_size: str, lang_code, token: str):
    """Return (bahasa, segments, warnings). Tanpa st.* — dipanggil dari background thread."""
    import whisperx

    warns = []
    audio_arr = whisperx.load_audio(str(audio))  # ffmpeg → mono 16 kHz
    asr = load_asr(model_size)
    result = asr.transcribe(audio_arr, batch_size=16, language=lang_code)
    detected = result.get("language") or lang_code or "?"

    try:
        align_model, metadata = whisperx.load_align_model(detected, "cpu")
        result = whisperx.align(
            result["segments"], align_model, metadata, audio_arr, "cpu",
            return_char_alignments=False,
        )
    except Exception as e:  # model align utk bahasa ini gagal / tidak tersedia
        warns.append(f"Word alignment dilewati ({type(e).__name__}) — pakai timestamp level segmen.")

    if token:
        try:
            diarize_df = load_diarize(token)(audio_arr)
            if result["segments"] and "words" in result["segments"][0]:
                from whisperx.diarize import assign_word_speakers
                result = assign_word_speakers(diarize_df, result)
            else:
                assign_by_overlap(diarize_df, result["segments"])
        except Exception as e:
            warns.append(
                f"Deteksi pembicara gagal: {e}. Cek HF_TOKEN dan akses model pyannote (README). "
                "Semua segmen dilabel Speaker 1."
            )
    else:
        warns.append("Tanpa HF token — deteksi perubahan suara dilewati, semua dilabel Speaker 1.")

    for seg in result["segments"]:
        seg["speaker"] = pretty_speaker(seg.get("speaker", "SPEAKER_00"))
    return detected, result["segments"], warns


def build_txt(segments) -> str:
    return "\n".join(
        f"[{fmt_ts(s['start'])} – {fmt_ts(s['end'])}] {s['speaker']}: {s['text'].strip()}"
        for s in segments
    )


def build_srt(segments) -> str:
    blocks = [
        f"{i}\n{srt_ts(s['start'])} --> {srt_ts(s['end'])}\n{s['speaker']}: {s['text'].strip()}\n"
        for i, s in enumerate(segments, 1)
    ]
    return "\n".join(blocks)


def build_json(segments, detected, dur, fname) -> str:
    return json.dumps(
        {
            "file": fname,
            "language": detected,
            "duration_sec": round(dur, 2),
            "speakers": len({s["speaker"] for s in segments}),
            "segments": [
                {"start": round(s["start"], 2), "end": round(s["end"], 2),
                 "speaker": s["speaker"], "text": s["text"].strip()}
                for s in segments
            ],
        },
        ensure_ascii=False,
        indent=2,
    )


def save_outputs(fname, segments, detected):
    stem = Path(fname).stem
    dur = max((s["end"] for s in segments), default=0.0)
    OUT_DIR.mkdir(exist_ok=True)
    print(f"HASIL TERSIMPAN: {OUT_DIR / stem}.{{txt,srt,json}}", flush=True)  # sinyal sukses di log server
    (OUT_DIR / f"{stem}.txt").write_text(build_txt(segments), encoding="utf-8")
    (OUT_DIR / f"{stem}.srt").write_text(build_srt(segments), encoding="utf-8")
    (OUT_DIR / f"{stem}.json").write_text(build_json(segments, detected, dur, fname), encoding="utf-8")


def start_job(up, model_size: str, lang_code, token: str) -> dict:
    # satu job pada satu waktu — dobel-klik/klik saat ada job jalan diabaikan,
    # halaman tetap nunjukin status job yang lagi jalan
    if jobstore.JOB and not jobstore.JOB["done"]:
        return jobstore.JOB
    tmp = Path(tempfile.gettempdir()) / f"job_{int(time.time() * 1000)}_{up.name}"
    tmp.write_bytes(up.getvalue())
    job = {
        "tmp": str(tmp), "fname": up.name, "t0": time.time(),
        "stage": f"Memuat model `{model_size}` (download pertama bisa lama)…",
        "done": False, "error": None, "segments": None, "detected": None, "warnings": [],
    }

    def work():
        try:
            job["stage"] = "Transkripsi + deteksi pembicara…"
            job["detected"], job["segments"], job["warnings"] = transcribe(
                Path(job["tmp"]), model_size, lang_code, token
            )
            save_outputs(job["fname"], job["segments"], job["detected"])
        except Exception as e:
            job["error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc()  # ke log server — error thread nggak boleh silent
        finally:
            job["done"] = True
            Path(job["tmp"]).unlink(missing_ok=True)

    threading.Thread(target=work, daemon=True).start()
    return job


st.title("🎙️ Transkripsi Audio → Teks")
st.caption("Lokal 100% — WhisperX (transkripsi) + pyannote (deteksi perubahan suara). Proses jalan di background: halaman boleh di-refresh / Mac boleh tidur, hasil tetap aman di `output/`.")

with st.sidebar:
    st.header("Pengaturan")
    model_label = st.selectbox("Model", list(MODEL_CHOICES))
    lang_label = st.selectbox("Bahasa audio", list(LANG_CHOICES))
    token = st.text_input(
        "HuggingFace token (deteksi pembicara)",
        value=os.environ.get("HF_TOKEN", ""),
        type="password",
        help="Gratis: hf.co/settings/tokens. Tanpa token, deteksi pembicara dilewati.",
    )
    if not token:
        st.info("Tanpa HF token: transkripsi tetap jalan, semua teks dilabel Speaker 1.")
    if not shutil.which("ffmpeg"):
        st.error("ffmpeg tidak ditemukan — install dulu: `brew install ffmpeg`")

up = st.file_uploader("Upload audio", type=AUDIO_TYPES)
if up and st.button("🎙️ Transkripsi", type="primary", disabled=not shutil.which("ffmpeg")):
    jobstore.JOB = start_job(up, MODEL_CHOICES[model_label], LANG_CHOICES[lang_label], token)

job = jobstore.JOB  # global proses — hidup walau sesi browser mati / ganti tab
if job and not job["done"]:
    elapsed = int(time.time() - job["t0"])
    with st.status(f"⏳ {job['stage']} (sudah {elapsed}s)", expanded=True):
        st.write(f"File: **{job['fname']}**")
        st.write("Jalan di background — halaman boleh di-refresh, Mac boleh tidur. "
                 "Hasil otomatis tersimpan di `output/` walau tab ini ditutup.")
    time.sleep(3)
    st.rerun()

if job and job["done"]:
    for w in job["warnings"]:
        st.warning(w)
    if job["error"]:
        st.error(f"Gagal memproses: {job['error']}")
    elif not job["segments"]:
        st.warning("Tidak ada ucapan terdeteksi di audio ini.")
    else:
        segments, detected = job["segments"], job["detected"]
        dur = max(s["end"] for s in segments)
        speakers = sorted({s["speaker"] for s in segments})
        words = sum(len(s["text"].split()) for s in segments)
        elapsed = int(time.time() - job["t0"])

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Durasi audio", fmt_ts(dur))
        c2.metric("Pembicara", len(speakers))
        c3.metric("Kata", words)
        c4.metric("Segmen", len(segments))
        c5.metric("Waktu proses", f"{elapsed}s")
        st.caption(f"Bahasa terdeteksi: `{detected}`")

        st.subheader("Transkrip lengkap")
        for s in segments:
            st.markdown(f"`[{fmt_ts(s['start'])} – {fmt_ts(s['end'])}]` **{s['speaker']}**: {s['text'].strip()}")

        stem = Path(job["fname"]).stem
        txt = build_txt(segments)
        srt = build_srt(segments)
        js = build_json(segments, detected, dur, job["fname"])
        st.subheader("Unduh hasil")
        d1, d2, d3 = st.columns(3)
        d1.download_button("📄 TXT", txt, file_name=f"{stem}.txt")
        d2.download_button("🎬 SRT", srt, file_name=f"{stem}.srt")
        d3.download_button("🧾 JSON", js, file_name=f"{stem}.json", mime="application/json")
        st.caption(f"Juga tersimpan di `{OUT_DIR}/`")

if OUT_DIR.exists():
    files = sorted(OUT_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if files:
        with st.expander(f"📁 Hasil sebelumnya — {len(files)} file di `output/`"):
            for p in files[:30]:
                st.download_button(f"⬇ {p.name}", p.read_bytes(), file_name=p.name)
