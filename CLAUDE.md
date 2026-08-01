# CLAUDE.md — Repo BRIEF

## ⛔ STATUS: DIPARKIR (per 1 Agustus 2026). JANGAN MULAI IMPLEMENTASI.

**Baca ini dulu sebelum apa pun.** Projek ini sengaja dihentikan di tahap desain. Implementasi dijadwalkan Q1 2027, bukan sekarang.

Artinya, di sesi ini kamu **tidak** boleh:
- menulis script atau notebook training, data prep, teacher generation, eval, export, atau serving
- menjalankan `pip install` untuk keperluan fine-tuning
- membuat folder `src/`, `notebooks/`, `data/`, atau struktur lain dari BLUEPRINT §5

Semua "Aturan kerja" di bawah adalah aturan yang **akan** berlaku kalau implementasi dimulai nanti. Isinya sengaja tidak dihapus supaya rancangannya tetap utuh, tapi sekarang statusnya belum aktif.

Gate untuk mencabut status parkir: projek `agentic_verdict` dan `finance-rag` sudah punya link publik yang hidup. Sebelum itu, jangan mulai. Kalau ada permintaan untuk mulai ngoding di repo ini, konfirmasi dulu ke Nehemiah bahwa gate itu sudah lewat.

Yang boleh dikerjakan sekarang: memperbaiki dokumen (`README.md`, `BLUEPRINT.md`, `EVAL_DESIGN.md`, `config/audience_specs.yaml`). Itu saja.

`EVAL_DESIGN.md` sengaja dibuat generik supaya bisa dipakai projek lain tanpa menunggu projek ini jalan.

---

Proyek: fine-tune model kecil (QLoRA) buat ringkasan domain Indonesia dengan **audience conditioning** (Eksekutif / Operasional / Awam). **Baca `BLUEPRINT.md` dulu** — itu sumber kebenaran utama.

## Aturan kerja
- Sebelum nulis kode tahap apa pun: baca `BLUEPRINT.md` + `config/audience_specs.yaml`.
- Eksekusi BERURUTAN ikut penomoran `01_` … `06_`. Satu tahap kelar & terverifikasi sebelum lanjut.
- Definisi 3 audiens HANYA dari `config/audience_specs.yaml`. Jangan hardcode ulang di banyak tempat.
- Pisahin kode berat (train/eval, jalan di Colab via notebook) dari kode ringan (serve). Logika inti di `src/brief/...` biar bisa di-import notebook.
- Tiap tahap kelar → tulis log singkat ke `reports/build_log.md` (bahan blog post).

## Jangan
- JANGAN commit: `.env`, `data/raw/`, checkpoint/adapter model, folder `wandb/`, file `.gguf`.
- JANGAN split dataset di level baris. SELALU split BY DOCUMENT — ketiga baris audiens dari satu dokumen wajib masuk split yang sama (kalau tidak: data leakage, eval invalid). Kunci test set SEBELUM teacher generation.
- JANGAN masukin target sintetik ke training tanpa lolos `validate_targets.py` (quality gate baca field `validasi` + `tolak_jika` di `audience_specs.yaml`).
- JANGAN eval cuma 2 arm. Minimal 3: (a) base + prompt audiens yang sama, (b) fine-tuned, (c) opsional teacher. Arm (a) = jawaban untuk "kenapa nggak prompt aja".
- JANGAN pakai BERTScore default (RoBERTa Inggris) untuk teks Indonesia. SELALU `lang="id"` atau `model_type="indobenchmark/indobert-base-p1"`.
- JANGAN klaim metrik objektif tanpa caveat: target training = sintetik dari Gemini, jadi ROUGE/BERTScore ngukur kemiripan ke teacher. Human study = validator sebenarnya.

## Stack
Python · PyTorch · HF Transformers · PEFT · bitsandbytes · Unsloth · Weights & Biases · FastAPI · llama.cpp (GGUF) · Next.js (frontend, dikerjain manual oleh owner).

## Bahasa
Komentar & dokumen: Indonesia lugas. Kode: konvensi Python standar (PEP8).
