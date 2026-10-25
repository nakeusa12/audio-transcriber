# 🎙️ Transkripsi Audio → Teks + Pemisahan Pembicara

Transkrip **lengkap dan verbatim** (tanpa diringkas) + label **siapa bicara kapan**.
100% lokal: [WhisperX](https://github.com/m-bain/whisperX) (faster-whisper + pyannote).
Tidak ada biaya API — audio tidak keluar dari komputer.

## Isi repo

| Path | Keterangan |
|---|---|
| `audio/` | Tempat menaruh rekaman sumber — **tidak disertakan** di repo (lihat "Menyiapkan audio") |
| `transcribe.py` | CLI — cara utama untuk transkrip |
| `app.py` | UI Streamlit (upload + antrean background) |
| `output/` | Hasil: `<nama>.txt`, `.srt`, `.json` — tidak di-commit |
| `setup.sh` | Setup venv + dependency (macOS/Linux) |
| `verify.py` | Cek instalasi cepat (audio uji 2 suara) |

Repo ini **hanya berisi kode**. Audio dan hasil transkrip sengaja di-`.gitignore`
karena berisi rekaman rapat.

## ⚠️ Yang harus di-install dulu di device baru

1. **Python 3.11** — `python3.11 --version`
2. **ffmpeg** — macOS: `brew install ffmpeg` · Ubuntu: `sudo apt install ffmpeg` · Windows: `winget install ffmpeg`
3. **HuggingFace token (gratis)** — wajib saat model di-download pertama kali:
   - Buat token tipe **Read**: <https://huggingface.co/settings/tokens>
   - Login/accept lisensi di ketiga halaman ini:
     - <https://huggingface.co/pyannote/segmentation-3.0>
     - <https://huggingface.co/pyannote/speaker-diarization-3.1>
     - <https://huggingface.co/pyannote/speaker-diarization-community-1>
4. **RAM** — lihat tabel model di bawah. `medium` butuh ±16 GB; `small` cukup 8 GB.
5. **(Opsional) GPU NVIDIA** — CUDA driver + PyTorch CUDA. Script mendeteksi CUDA otomatis.

Disk: ±6 GB (torch + model ASR + model diarization).

## Setup (sekali per device)

**macOS / Linux**
```bash
cd audio-transcriber
./setup.sh
```
Atau manual (kalau tidak mau pakai `setup.sh`):
```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
# tanpa uv:
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
py -3.11 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

## Token HuggingFace

```bash
export HF_TOKEN=hf_xxxxxxxx      # macOS/Linux
# Windows: $env:HF_TOKEN = "hf_xxxxxxxx"
```
atau simpan permanen sekali:
```bash
.venv/bin/hf auth login
```
Script otomatis memakai `HF_TOKEN`, lalu fallback ke token tersimpan dari `hf auth login`.

## Menyiapkan audio

Repo ini publik dan **tidak** memuat rekaman. Pindahkan file audio ke device baru
(AirDrop/USB/cloud pribadi), lalu taruh di folder `audio/`:

```bash
# contoh: dari Downloads
cp ~/Downloads/"1 on 1 16oct.m4a" audio/
cp ~/Downloads/"1 on 1 23oct.m4a" audio/
```

Atau langsung beri path file di mana pun tanpa menyalin ke `audio/`.

## Jalankan

**CLI (utama)**
```bash
.venv/bin/python transcribe.py "audio/1 on 1 23oct.m4a" medium 2
# format: transcribe.py <audio> [model] [jumlah_pembicara]
# jumlah_pembicara opsional — pakai 2 untuk rekaman 1-on-1
```

Contoh untuk dua rekaman 1-on-1 (setelah ditaruh di `audio/`):
```bash
.venv/bin/python transcribe.py "audio/1 on 1 16oct.m4a" medium 2
.venv/bin/python transcribe.py "audio/1 on 1 23oct.m4a" medium 2
```

**UI Streamlit**
```bash
HF_TOKEN=hf_xxx .venv/bin/streamlit run app.py
```

Hasil selalu ditulis ke `output/<nama-file>.txt`, `.srt`, `.json`.
Format TXT/SRT: `[mm:ss – mm:ss] Speaker N: teks`.

## Pilihan model

| Model | RAM | Kecepatan (CPU) | Kualitas |
|---|---|---|---|
| `base` | ±4 GB | ±0,3× durasi | rendah |
| `small` | ±8 GB | ±0,2× durasi | cukup (default CLI) |
| `medium` | ±16 GB | ±1–2× durasi | **disarankan** |
| `large-v3` | ±32 GB | ±2–4× durasi | terbaik |

Dengan GPU CUDA, semua model jauh lebih cepat (device terdeteksi otomatis; compute type `float16`).

> **Mac 8 GB:** hanya `small`/`base` yang aman. `medium` dan `large-v3` deadlock di
> ctranslate2 (`pthread_cond_wait`) setelah swap penuh — sudah terbukti 4 Okt & 25 Okt 2026.
> Kalau RAM lega (16 GB+), `medium` aman.

## Catatan penting

- **Diarization adalah bottleneck di CPU** — ±1,5–2 jam untuk audio 60 menit, terlepas dari model ASR.
  Transkripsi `small` sendiri hanya ±10 menit untuk audio 68 menit.
- **Jumlah pembicara**: tanpa arg ke-3, pyannote menebak sendiri dan kadang memecah 1 orang
  menjadi beberapa cluster (mis. 5 pembicara pada rekaman 1-on-1). Pakai arg `2` untuk memaksa 2 pembicara.
- **Hallucination**: Whisper bisa mengulang frasa saat hening/noise (`Terima kasih banyak`,
  `Bila kemana?`, `tenenenen…`). Itu artefak model, bukan isi rapat. Transkrip lama juga mengalaminya.
- **Mac tidur** saat job berjalan → proses pause, wall-clock molor (bukan hang). Jalankan
  `caffeinate -i` di terminal lain kalau ingin tetap bangun.
- Audio mono 16 kHz diproses via ffmpeg; format mp3/wav/m4a/ogg/webm/flac/aac/aiff didukung.

## Troubleshooting

| Gejala | Penyebab / solusi |
|---|---|
| `diarization GAGAL ... 401` | Token salah / lisensi pyannote belum di-accept. Ulangi langkah token. |
| Proses diam, CPU ~0%, swap penuh | Model terlalu besar untuk RAM. Pakai `small` atau device lain. |
| Hasil 5 pembicara di rekaman 1-on-1 | Jalankan ulang dengan arg ke-3 `2`. |
| CUDA tidak terpakai | Pastikan `torch.cuda.is_available()` True dan driver NVIDIA terpasang. |
| OOM saat model `medium` di GPU kecil | Turunkan `batch_size` di `transcribe.py` (default 16). |
| Diarization lambat | Normal di CPU. Pakai GPU atau jalankan sambil ditinggal. |

## Verifikasi instalasi

```bash
.venv/bin/python verify.py
```
Membuat audio uji 2 suara via `say` (macOS), lalu memastikan transkripsi + diarization jalan.
Di Linux/Windows, siapkan audio 2 pembicara sendiri dan jalankan `transcribe.py`.
