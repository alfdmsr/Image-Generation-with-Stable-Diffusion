# StudioAI — Image Generation with Stable Diffusion

Proyek ini adalah eksplorasi dan implementasi **image generation** menggunakan model **Stable Diffusion v1.5** (via HuggingFace `diffusers`). Proyek terdiri dari dua bagian utama:

1. **Pipeline Notebook** — eksperimen end-to-end mulai dari text-to-image, tuning hyperparameter, hingga image editing (inpainting & outpainting).
2. **Streamlit App ("StudioAI")** — aplikasi web interaktif untuk generate dan mengedit gambar langsung dari browser, di-deploy via Ngrok.

---

## Fitur

### 1. Text-to-Image
- Generate gambar dari prompt teks menggunakan `runwayml/stable-diffusion-v1-5`.
- Perbandingan hasil generate **sederhana** (default parameter) vs **advanced** (custom `guidance_scale` & `num_inference_steps`).
- Eksperimen pengaruh **Guidance Scale (CFG)** terhadap kesesuaian gambar dengan prompt.
- Eksperimen pengaruh **jumlah inference steps** terhadap detail & kehalusan gambar.
- **Batch inference** — generate beberapa gambar sekaligus dari satu prompt (ditampilkan grid 2×2).
- Ganti **scheduler/sampler** tanpa reload model: `Euler A`, `DPM++`, `DDIM`.

### 2. Image-to-Image & Refinement
- Teknik **two-stage generation (Base + Refiner)** — tahap awal 80% denoising, disempurnakan 20% sisanya lewat img2img.

### 3. Inpainting
- Model khusus: `runwayml/stable-diffusion-inpainting`.
- **Manual masking** — menentukan koordinat area edit secara hardcode.
- **Auto masking** — menggunakan model segmentasi `facebook/mask2former-swin-base-coco-panoptic` untuk mendeteksi objek (mis. `person`) dan membuat mask otomatis.

### 4. Outpainting
- **Ekspansi satu arah** (kiri/kanan/atas/bawah) untuk memperluas kanvas gambar.
- **Zoom Out iteratif** — memperluas gambar ke segala arah sekaligus dengan filler background blur.

### 5. Aplikasi Web (Streamlit)
- UI **StudioAI** dengan dua tab: `GENERATE` dan `EDIT`.
- Sidebar kontrol: jumlah steps, CFG scale, seed, pilihan scheduler, batch size.
- Canvas gambar interaktif (`streamlit-drawable-canvas`) untuk menggambar mask inpainting langsung di browser.
- Tombol **Flush RAM** untuk membersihkan cache GPU secara manual.
- Deploy publik via **Ngrok tunnel** (cocok dijalankan di Google Colab / Kaggle Notebook).

---

## Struktur Proyek

```
.
├── Pipeline_submission_BFGAI_Muhammad_Alif_Dhiyaumasrur.ipynb   # Notebook eksperimen (text2img, img2img, inpainting, outpainting)
├── Streamlit_submission_BFGAI_Muhammad_Alif_Dhiyaumasrur.ipynb  # Notebook yang generate app.py & logic.py, lalu deploy via Ngrok
├── requirements.txt
└── README.md
```

> Catatan: `app.py` dan `logic.py` tidak disertakan sebagai file terpisah — keduanya di-generate secara otomatis (`%%writefile`) saat menjalankan sel-sel di `Streamlit_submission_....ipynb`.

---

## Tech Stack

- **PyTorch** — deep learning framework
- **HuggingFace `diffusers`** — Stable Diffusion pipelines (Text2Img, Img2Img, Inpaint)
- **HuggingFace `transformers`** — text encoder & model segmentasi (Mask2Former)
- **Streamlit** — web UI
- **streamlit-drawable-canvas** — canvas gambar interaktif untuk masking manual
- **Pyngrok** — expose Streamlit local server ke public URL
- **Pillow / NumPy** — pemrosesan gambar

---

## Instalasi

Disarankan menjalankan proyek ini di lingkungan dengan **GPU** (Google Colab / Kaggle Notebook), karena Stable Diffusion cukup berat untuk dijalankan di CPU.

```bash
pip install -r requirements.txt
```

Isi `requirements.txt`:
```
torch
diffusers
transformers
accelerate
streamlit
streamlit-drawable-canvas==0.8.0
pyngrok
Pillow
numpy
```

---

## Cara Menjalankan

### A. Pipeline Notebook (Eksperimen)
1. Buka `stable_diffusion_pipeline.ipynb` di Colab/Kaggle (aktifkan GPU runtime).
2. Jalankan sel secara berurutan dari atas ke bawah.
3. Notebook ini mencakup: load model, generate text-to-image, eksperimen hyperparameter, refiner, inpainting (manual & auto mask), serta outpainting/zoom-out.

### B. Streamlit App (StudioAI)
1. Buka `studioai_streamlit_deploy.ipynb` di Colab/Kaggle (aktifkan GPU runtime).
2. Jalankan sel secara berurutan — ini akan otomatis membuat file `logic.py` dan `app.py`.
3. Daftar akun gratis di [ngrok.com](https://ngrok.com), ambil **Authtoken** dari dashboard.
4. Masukkan token tersebut ke variabel `auth_token` pada sel konfigurasi Ngrok.
5. Jalankan sel untuk memulai server Streamlit (`subprocess.Popen(["streamlit", "run", "app.py"])`).
6. Jalankan sel `ngrok.connect(8501)` untuk mendapatkan public URL, lalu buka URL tersebut di browser.
7. Setelah selesai, jalankan `ngrok.kill()` untuk menutup semua tunnel aktif.

---

## Parameter Utama

| Parameter | Deskripsi |
|---|---|
| `prompt` / `negative_prompt` | Deskripsi gambar yang diinginkan / dihindari |
| `seed` | Angka untuk hasil yang reproducible |
| `guidance_scale` (CFG) | Seberapa ketat model mengikuti prompt (rendah = lebih kreatif, tinggi = lebih patuh prompt) |
| `num_inference_steps` | Jumlah langkah denoising (lebih tinggi = lebih detail, lebih lambat) |
| `scheduler` | Algoritma sampling: `Euler A`, `DPM++`, `DDIM` |
| `strength` | (khusus inpainting) seberapa kuat area mask diubah |

---

## Model yang Digunakan

- [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5) — text-to-image & img2img
- [`runwayml/stable-diffusion-inpainting`](https://huggingface.co/runwayml/stable-diffusion-inpainting) — inpainting & outpainting
- [`facebook/mask2former-swin-base-coco-panoptic`](https://huggingface.co/facebook/mask2former-swin-base-coco-panoptic) — segmentasi otomatis untuk auto-masking

---
