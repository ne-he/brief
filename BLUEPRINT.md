# BRIEF: Technical Blueprint

> Fine-tune model kecil (QLoRA) buat domain Indonesia. **Satu model, banyak lensa audiens.**
> Input dokumen sama → **Brief Eksekutif / Brief Operasional / Brief Awam**.
> Bukti bahwa author adalah *model trainer*, bukan *API caller*.

**Owner:** Nehemiah · **Status:** Spec (pre-build) · **Target durasi:** 6–8 minggu
**Eksekusi:** Claude Code di Antigravity IDE (baca file ini sebagai sumber kebenaran)

---

## 0. North Star (jangan lupa kenapa)

Yang bikin proyek ini *standout* BUKAN "berhasil fine-tune". Itu komoditas.
Yang standout = **dua hal yang digabung dengan benar:**

1. **Audience conditioning**: satu model fine-tuned bisa nulis 3 gaya brief berbeda dari input yang sama. Ini produk + skill komunikasi, bukan cuma ML.
2. **Eval rigor gabungan**: lexical (ROUGE) + semantic (BERTScore) + **human preference study**. Plus jujur soal limitasi metrik (lihat §4). Kejujuran metodologis = sinyal senioritas.

Kalau di akhir cuma punya "model yang bisa nyummarize" tanpa 3 lensa yang **jelas beda** dan tanpa angka base-vs-tuned + human study, proyek ini gagal jadi flagship. Jaga dua hal itu di atas segalanya.

---

## 1. Keputusan Arsitektur Kunci (baca sebelum nulis kode)

### 1.1 Dari mana datang label "Eksekutif / Operasional / Awam"?
**Ini masalah paling penting dan paling sering di-skip.** Dataset summarization Indonesia (IndoSum, Liputan6) cuma punya `(artikel → 1 ringkasan referensi)`. Nggak ada ground-truth per-audiens.

**Solusi (yang dipakai): Synthetic distillation dari teacher LLM.**
- Pakai **Gemini** (Nehemiah udah punya akses) sebagai *teacher*. Untuk tiap dokumen, generate 3 target brief sesuai rubrik audiens yang ketat (`audience_specs.yaml`).
- Fine-tune model kecil di `(dokumen + tag audiens → brief audiens)`.
- **Narasi ML-nya kuat & jujur:** "Gue *distill* kemampuan audience-conditioning dari frontier model ke model 2B lokal yang murah." Itu cerita engineering beneran.

**Konsekuensi penting untuk eval (JANGAN diabaikan):**
Kalau target = output Gemini, maka ROUGE/BERTScore vs target itu ngukur **"seberapa mirip gue sama teacher"**, BUKAN "seberapa bagus secara objektif". Makanya:
- Simpan juga **ringkasan referensi asli** dari dataset → buat ngukur *faithfulness/coverage* (anti-halu), bukan cuma imitasi.
- **Human preference study** = validator kualitas sebenarnya. Wajib ada.
- Boleh tambah **LLM-as-judge** (Gemini nge-rate audience-fit) sebagai proxy skalabel, tapi **akui bias** (teacher = judge). Tulis caveat ini di laporan. Recruiter ML yang ngerti bakal respect lo justru karena sadar limitasi ini.

### 1.2 Korpus: Berita dulu, Legal sebagai stretch
| Pilihan | Pro | Kontra |
|---|---|---|
| **Berita (IndoSum / Liputan6)** ✅ primary | Bersih, ada di HF, dokumen pendek (muat di T4), pipeline ke-de-risk | "Initiative signal" lebih rendah |
| **Legal (putusan MA)** stretch | Sinyal inisiatif tinggi, lebih "wow" | Scraping + cleaning makan 1–2 minggu, dokumen super panjang (>8k token, susah di T4) |

**Keputusan:** Mulai dari **IndoSum** (paling bersih, ~19k, splits rapi) untuk **menjamin pipeline jalan**. Setelah M2 sukses, kalau waktu cukup, masukin **subset kecil putusan MA** sebagai demo *domain transfer* di M3/stretch. **Jangan biarin data-cleaning legal nyandera timeline.** Pipeline jalan dulu > data keren tapi nggak kelar.

### 1.3 Model dasar
- **Primary: `google/gemma-2-2b-it`**, Indonesia lumayan, lisensi oke, didukung Unsloth, muat 4-bit di T4.
- **Fallback/pembanding: `Qwen/Qwen2.5-1.5B-Instruct`**, lebih kenceng, bagus buat ablation kecil di §4.
- (Opsional ambisius, kalau VRAM mepet jangan): base ber-pretraining Indonesia kayak SEA-LION / SahabatAI, lebih jago Indonesia tapi rata-rata lebih gede dari budget T4.

**Audience tagging:** pakai **instruksi natural language** di dalam prompt (`[AUDIENS: Eksekutif] ...`), **bukan** special token. Alasan: data kecil + nggak perlu operasi tokenizer + lebih generalizable.

### 1.4 Realita serving (gotcha deploy)
**Model 2B TIDAK akan jalan di Vercel serverless.** Pisahkan:
- **Frontend (Next.js)** → Vercel.
- **Model API (FastAPI)** → host Python: **HF Spaces (CPU gratis)** pakai **GGUF + llama-cpp-python**, atau lokal + ngrok buat demo day, atau Render free.
- Makanya **export GGUF (quantized)** bukan sekadar "nice to have", itu yang bikin demo bisa hidup di CPU gratis.

**Guardrail serving (wajib, biar demo publik nggak mati di CPU gratis):**
- Batas panjang input (mis. ~3000 token) → dokumen kepanjangan = error rapi "dokumen terlalu panjang", bukan timeout.
- `max_new_tokens` per audiens diturunkan dari `panjang_target` di `audience_specs.yaml` (jangan biarin model ngoceh 1000 token di CPU).
- Timeout per request + proses 1 request at a time (CPU kecil, antri lebih baik daripada crash).
- **Cache respons untuk dokumen preset/contoh** → demo pengunjung terasa instan walau model jalan di CPU.
- Rate limit ringan di endpoint publik.

### 1.5 Higiene data (jebakan halus yang bikin angka eval BOHONG)
1. **Split BY DOCUMENT, bukan by row.** Satu dokumen menghasilkan 3 baris training (eksekutif/operasional/awam). Ketiganya WAJIB masuk split yang sama. Kalau split di level baris, dokumen yang sama bocor ke train dan test → semua angka eval invalid, dan nggak akan ketahuan kalau nggak dicek dari awal.
2. **Kunci test set SEBELUM teacher generation.** Split dulu, baru generate target. Dedup near-duplicate artikel antar split (judul/teks mirip).
3. **Quality gate target sintetik** (`validate_targets.py`, jalan sebelum training): tiap brief dicek otomatis terhadap `audience_specs.yaml`: panjang dalam rentang, struktur sesuai (eksekutif = bullet ≤5, operasional = list bernomor), bahasa Indonesia, bukan refusal/kosong, tidak bocorin tag `[AUDIENS: ...]` di body. Gagal → regenerate 1x → masih gagal → buang barisnya. Laporkan % lolos di `reports/build_log.md`. **Data-centric > model-centric: kualitas target sintetik = plafon kualitas model lo.**

---

## 2. Definisi 3 Lensa Audiens (SUMBER KEBENARAN)

Ini "rahasia dapur" sekaligus edge komunikasi. **Definisi ini dipakai 3 tempat:** prompt teacher (Gemini), instruksi inferensi, dan rubrik human eval. Satu sumber → `audience_specs.yaml`.

**Brief Eksekutif**: buat decision-maker sibuk.
- *Bottom line up front*: kesimpulan/keputusan di kalimat pertama.
- Fokus "so what": dampak, risiko, 1 angka kunci.
- Maks ~5 bullet. Tanpa jargon. Tanpa langkah detail.

**Brief Operasional**: buat orang yang ngeksekusi.
- Actionable: siapa-ngapain, langkah, tenggat, dependensi, angka spesifik.
- Boleh lebih panjang & detail. Pakai list bernomor.

**Brief Awam**: buat publik umum.
- Bahasa polos, nol jargon (kalau ada, dijelasin).
- "Kenapa ini penting buat gue", boleh pakai analogi.
- Nada netral, tidak menggurui.

> **Tes keberhasilan conditioning:** ketiga output dari input yang sama harus **jelas beda** (panjang, struktur, diksi). Ukur kuantitatif: ROUGE antar-audiens harus **rendah** + distinct-n tinggi (lihat §4.3).

---

## 3. Arsitektur Pipeline

```
                 ┌─────────────────────────────────────────────┐
                 │  audience_specs.yaml  (sumber kebenaran)     │
                 └───────────────┬─────────────────────────────┘
                                 │ (dipakai 3 tempat)
   Dataset ID (IndoSum)          ▼
        │            ┌──────────────────────┐
        ▼            │  TEACHER: Gemini      │  generate 3 brief/ dokumen
  [01 prep] ───────▶ │  (synthetic targets) │ ──► dataset_audience.jsonl
   bersih + split    └──────────────────────┘     (train/val/test)
        │
        ▼
  [02 baseline eval]  model base (gemma-2-2b-it) di test set
        │             ROUGE + BERTScore(ID!) ► angka pembanding (W&B)
        ▼
  [03 QLoRA train]    Unsloth + PEFT + bitsandbytes 4-bit @ Colab T4
        │             log loss/lr + sample generations ► W&B
        │             simpan adapter checkpoints
        ▼
  [04 eval]           base vs fine-tuned (ROUGE/BERTScore/distinct-n)
        │             + mini human study (form) + LLM-as-judge (opsional)
        ▼
  [05 merge+export]   merge adapter → fp16 → GGUF q4_k_m (llama.cpp)
        │
        ▼
  [06 serve]          FastAPI /generate ◄── Next.js demo UI (Vercel)
                      backend di HF Spaces (GGUF, CPU)
```

---

## 4. Eval Harness (jantung "rigor")

### 4.1 Metrik wajib
- **ROUGE-1 / ROUGE-2 / ROUGE-L** (`evaluate` / `rouge_score`). Tokenisasi level kata cukup untuk Indonesia.
- **BERTScore**: ⚠️ **GOTCHA FATAL:** default pakai RoBERTa Inggris → skor sampah untuk teks Indonesia. **WAJIB** set `lang="id"` atau `model_type="indobenchmark/indobert-base-p1"` (atau mBERT). Tulis ini di kode + komentar.
- **3 kondisi yang dibandingkan, BUKAN 2** (dipecah per-audiens):
  - **(a) Base + prompt audiens yang sama** (prompt engineering murni di model base, boleh few-shot), ini baseline yang FAIR.
  - **(b) Fine-tuned** (QLoRA).
  - **(c)** opsional: **teacher Gemini** sebagai plafon atas.
  - Kenapa: ini jawaban buat **pertanyaan pembunuh** dari reviewer/recruiter: *"kenapa repot fine-tune, kenapa nggak prompt aja?"* Kalau (b) nggak ngalahin (a), tesis proyek runtuh, dan lo harus tau itu dari data, bukan asumsi. Kalau (b) menang, itu justru bukti terkuat lo.
- **Bootstrap confidence interval** (resampling ≥1000x) untuk delta ROUGE/BERTScore antara (a) dan (b), biar klaim "fine-tuned lebih baik" punya errorbar, bukan angka tunggal. Murah diimplement, sinyal senioritas gede.

### 4.2 Anti-halu / faithfulness
- Skor coverage vs **ringkasan referensi asli** (bukan target Gemini), atau cek entailment ringan. Tunjukin model nggak ngarang fakta.

### 4.3 Apakah conditioning beneran kerja? (metrik khas proyek ini)
- **Pairwise ROUGE antar 3 output** dari input sama → harus **rendah** (kalau tinggi = conditioning gagal, ketiganya mirip).
- **Distinct-1/2** + rata-rata panjang per audiens → buktiin gaya beda terukur.

### 4.4 Human preference study (5–10 orang)
- **Blind A/B**: prompted-base vs fine-tuned (mana lebih bagus?) + **audience-fit Likert 1–5** (apakah "Brief Eksekutif" beneran kerasa eksekutif?) + faithfulness.
- Form simpel (Google Form / halaman mini). Acak urutan, sembunyiin label model.
- **Desain minimum yang bikin hasilnya bisa dipercaya:** ~10–15 dokumen × 3 audiens, tiap item dirating ≥2 orang → laporkan juga **persentase agreement antar-rater** (atau Krippendorff's α kalau niat). Tulis instruksi rating 1 halaman SEBELUM ngumpulin rater, jangan improvisasi.
- **Lapor jujur:** n kecil → hasil *directional*, sertakan caveat statistik. Kerendahan hati = sinyal senior.

### 4.5 Output eval
- Tabel ringkas (markdown + CSV) → masuk slide deck & dashboard frontend.
- Semua angka tercatat di **W&B** (run base vs tuned bisa dibandingin langsung).

### 4.6 Tabel banding wajib (struktur dikunci sekarang, isi nanti)

Ini bukti yang harus kelihatan untuk klaim skill fine-tuning. Strukturnya ditulis sekarang selagi alasannya masih segar. Isinya diisi waktu implementasi, dan sampai itu terjadi tabel ini memang sengaja dibiarkan kosong.

| Model | Kualitas (metrik) | Latensi p95 | Biaya per 1000 request |
|---|---|---|---|
| Gemini (teacher, baseline) | (isi nanti) | (isi nanti) | (isi nanti) |
| Gemma-2-2b-it + QLoRA | (isi nanti) | (isi nanti) | (isi nanti) |

Kolom kualitas diisi **ROUGE-L** terhadap target audiens, dipilih karena murah, deterministik, dan bisa dihitung ulang persis oleh siapa pun yang mau memverifikasi. Tapi ROUGE-L saja tidak cukup untuk mendukung klaim apa pun di sini: targetnya sintetik dari Gemini, jadi angka itu mengukur seberapa mirip output dengan teacher, bukan seberapa bagus outputnya. Karena itu kolom ini harus selalu dibaca bareng BERTScore berkonfigurasi Indonesia (§4.1), skor faithfulness terhadap ringkasan referensi asli (§4.2), dan hasil human preference study (§4.4). Uraian lengkap soal kenapa satu metrik tidak pernah cukup ada di `EVAL_DESIGN.md`.

---

## 5. Struktur Repo (Claude Code: ikutin ini)

```
brief/
├── BLUEPRINT.md              # file ini, sumber kebenaran
├── CLAUDE.md                 # guardrail singkat buat Claude Code (lihat §9)
├── README.md                 # ringkasan + cara jalanin
├── requirements.txt          # / pyproject.toml
├── .env.example              # GEMINI_API_KEY, WANDB_API_KEY, dst (JANGAN commit .env)
│
├── config/
│   ├── audience_specs.yaml   # definisi 3 lensa (SUMBER KEBENARAN §2)
│   ├── train_config.yaml     # hyperparam QLoRA
│   └── prompts.py            # template prompt teacher & inferensi
│
├── data/
│   ├── raw/                  # dataset mentah (gitignored)
│   ├── interim/
│   └── processed/            # dataset_audience.jsonl (train/val/test)
│
├── notebooks/
│   ├── 01_data_prep.ipynb        # ke-Colab juga
│   └── 03_qlora_train.ipynb      # NOTEBOOK UTAMA Colab T4 (Unsloth)
│
├── src/brief/
│   ├── data/
│   │   ├── load_indosum.py       # + split BY DOCUMENT di sini (kunci test set duluan)
│   │   ├── teacher_generate.py   # Gemini → 3 brief/dokumen (+ retry, cache, RESUMABLE per-N-dokumen)
│   │   ├── validate_targets.py   # quality gate target sintetik (§1.5), jalan SEBELUM training
│   │   └── build_dataset.py      # format instruksi + jsonl
│   ├── train/
│   │   └── train_qlora.py        # Unsloth/PEFT loop + W&B
│   ├── eval/
│   │   ├── metrics.py            # ROUGE, BERTScore(ID), distinct-n, pairwise
│   │   ├── run_eval.py           # base vs tuned → tabel/CSV
│   │   └── llm_judge.py          # opsional, Gemini-as-judge + caveat bias
│   ├── export/
│   │   └── merge_and_gguf.py     # merge adapter → fp16 → GGUF quantize
│   └── serve/
│       ├── app.py                # FastAPI: /generate, /health, /examples
│       └── inference.py          # loader GGUF (llama-cpp) / transformers
│
├── human_study/
│   ├── samples.jsonl             # item buat dirate (urutan teracak, label disembunyiin)
│   └── results_template.csv
│
├── reports/
│   ├── eval_table.md / .csv      # hasil prompted-base vs tuned (+ CI)
│   └── build_log.md              # catatan proses (buat blog post)
│
├── MODEL_CARD.md                 # model card gaya HF (M4), ikut ke-publish di HF Hub
│
├── deck/                         # slide deck (M4)
└── frontend/                     # Next.js (Nehemiah bikin sendiri, §7)
```

**Penomoran tahap** (`01_` … `06_`) bikin Claude Code bisa eksekusi berurutan & lo gampang resume.

---

## 6. Tech Stack (terkunci)

| Layer | Pilihan | Catatan |
|---|---|---|
| Base model | `gemma-2-2b-it` (primary), `Qwen2.5-1.5B-Instruct` (fallback) | |
| Training | **Unsloth** + PEFT + bitsandbytes (QLoRA 4-bit NF4) | 2x cepet, hemat VRAM, gratis di Colab T4 |
| Teacher (label) | **Gemini** (`gemini-2.0-flash`) | generate target audiens |
| Tracking | **Weights & Biases** | loss, lr, sample table, base-vs-tuned |
| Metrik | `evaluate` (ROUGE), `bert-score` (set lang `id`!), custom distinct-n | |
| Export | llama.cpp → **GGUF q4_k_m** | biar jalan di CPU gratis |
| Serve | **FastAPI** + `llama-cpp-python` (atau transformers) | host HF Spaces / lokal+ngrok |
| Frontend | **Next.js** (App Router) | Vercel |
| Container | Docker (stretch) | reproducibility |

**1 hal beneran baru yang lo pelajarin:** PEFT/QLoRA training loop. Fokus paham itu dalam-dalam (kenapa LoRA, kenapa 4-bit, apa itu rank/alpha), itu yang ditanya recruiter.

**Hyperparam awal QLoRA** (`train_config.yaml`):
```yaml
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
load_in_4bit: true
bnb_4bit_quant_type: nf4
max_seq_length: 2048      # naikin hati-hati kalau dokumen panjang (legal)
gradient_checkpointing: true
lr: 2.0e-4
epochs: 2                 # awas overfitting di data sintetik kecil
warmup_ratio: 0.03
batch_size: 2
grad_accum: 4
```

---

## 7. FITUR FRONTEND (cuma daftar, desain & implementasi: Nehemiah)

> Backend ngasih: `POST /generate {document, audience}` → `{brief}` (streaming kalau bisa), `GET /examples`, `GET /metrics`, `GET /health`. Frontend bebas dikreasiin, asal nutupin fitur ini:

**Inti (wajib):**
1. **Input dokumen**: textarea besar + upload file (`.txt`, `.pdf`) + counter token/karakter.
2. **Pemilih audiens**: toggle 3 arah: Eksekutif / Operasional / Awam.
3. **"Generate Ketiganya Sekaligus"**: tampilkan 3 brief berdampingan dari 1 input. **Ini money-shot demo**, paling jelas nunjukin audience conditioning.
4. **Panel output**: streaming text per audiens, tombol copy, badge model+quant (mis. "gemma-2-2b · Q4_K_M").

**Pembeda (yang bikin portfolio = bukti, bukan klaim):**
5. **Mode Banding: Base vs Fine-tuned**, output dua model sisi-sisi dari input sama. Bukti training-nya kerja.
6. **Halaman Eval / Metodologi**: tabel ROUGE + BERTScore + win-rate human study + contoh sampel. Ubah portfolio jadi *evidence*.
7. **Dokumen contoh / preset "Coba Ini"**: biar pengunjung nggak perlu bawa dokumen sendiri.
8. **Panel transparansi "Behind the scenes"**: tampilkan prompt/tag audiens yang dipakai. Transparansi = trust.

**Polish (kalau sempat):**
9. Indikator latensi/loading, **ekspor brief** (copy/download), tombol bagikan.
10. Bagian "Metodologi/Build log" yang nyambung ke blog post (§8 M4).
11. Responsif mobile + dark mode (gampang nyambung ke identitas Saturn nanti).

---

## 8. Milestones (4, sistem Nehemiah): mulai 2026-07-14

> Reminder per milestone: **48 jam · 24 jam · 1 jam** sebelum target. Kalau slip → langsung pecah jadi 2 tugas lebih pendek di sesi yang sama.

```
📌 BRIEF
├── M1: Data Prep + Baseline      Target: Sen, 27 Jul 2026 (~2 mgg)
│     • load IndoSum, bersihin, SPLIT BY DOCUMENT (kunci test set duluan, §1.5)
│     • audience_specs.yaml + prompts teacher
│     • teacher_generate.py: Gemini bikin 3 brief/dokumen (mulai 300–500 dokumen, resumable)
│     • validate_targets.py: quality gate sintetik + % lolos di build_log
│     • build dataset_audience.jsonl (train/val/test)
│     • eval BASELINE: base + prompt audiens (arm (a) §4.1) → angka pembanding di W&B
│     ✅ Selesai = jsonl valid (lolos gate) + tabel baseline prompted-base
│
├── M2: QLoRA Train + Checkpoint  Target: Sen, 10 Agt 2026 (~2 mgg)
│     • notebook Colab T4 + Unsloth jalan end-to-end
│     • W&B logging (loss + sample generations)
│     • checkpoint fine-tuned pertama, sanity-check 3 lensa beda
│     ✅ Selesai = adapter tersimpan + 3 brief jelas beda dari 1 input
│
├── M3: Eval Rigor + Human Study  Target: Sen, 24 Agt 2026 (~2 mgg)
│     • metrics.py (ROUGE, BERTScore-ID, distinct-n, pairwise, bootstrap CI)
│     • run_eval 3 arm: prompted-base vs tuned (vs teacher) → reports/eval_table
│     • human study: form + 5–10 perater + win-rate + agreement antar-rater
│     • (stretch) 1 ablation kecil (mis. lora_r 8 vs 16, ATAU 1 vs 2 epoch) → bukti paham hyperparam
│     • (stretch) subset legal putusan MA buat domain-transfer
│     ✅ Selesai = tabel 3-arm dengan CI + hasil human study
│
└── M4: Serve + Demo + Deck       Target: Sen, 7 Sep 2026 (~2 mgg)
      • merge + export GGUF q4_k_m
      • publish adapter + GGUF ke HF Hub + MODEL_CARD.md (bukti publik gratis!)
      • FastAPI (+ guardrail §1.4) + deploy backend (HF Spaces) + frontend (Vercel)
      • slide deck + build-log post
      ✅ Selesai = demo LIVE publik + model di HF Hub + deck + post = SHIPPED
```

**Rencana eskalasi:** kalau teacher-generation (M1) lelet/mahal → batasi 300 dokumen dulu, scale belakangan. Kalau T4 OOM (M2) → turunin `max_seq_length` ke 1024 atau pindah ke Qwen-1.5B.

---

## 9. Guardrail buat Claude Code (ringkas, taruh juga di CLAUDE.md repo)
- Baca `BLUEPRINT.md` + `config/audience_specs.yaml` sebelum nulis kode tahap apa pun.
- **Jangan** hardcode definisi audiens di banyak tempat, selalu dari `audience_specs.yaml`.
- **Jangan** commit `.env`, `data/raw/`, checkpoint model, atau `wandb/`.
- BERTScore Indonesia: **selalu** set `lang="id"`/model ID. Jangan default Inggris.
- Pisahkan kode "berat" (train/eval di Colab) dari "ringan" (serve). Train via notebook, logic via `src/brief/...` yang bisa di-import notebook.
- Tiap tahap: tulis log singkat ke `reports/build_log.md` (bahan blog post M4).
- Eksekusi berurutan ikut penomoran `01_`…`06_`. Satu tahap kelar & terverifikasi sebelum lanjut.

---

## 10. Risiko & Kejujuran (baca ini)
1. **Target sintetik → metrik bias.** Sudah diakui di §1.1/§4. Human study + faithfulness = penyeimbang. Tulis caveat di laporan.
2. **Model 2B kecil → output bisa medioker.** "Wow" datang dari **beda 3 lensa yang jelas**, bukan dari kualitas prosa. Investasi di desain data & rubrik > ukuran model.
3. **Deploy 2B ≠ Vercel.** Frontend Vercel, model di HF Spaces (GGUF/CPU). Rencanain dari awal.
4. **Opportunity cost.** Ini proyek aktif ke-4. Komit: BRIEF di-SHIP (demo publik) sebelum mulai eksperimen baru lagi. Flagship yang nganggur = nol nilai buat recruiter.
5. **ToS teacher (Gemini).** Pakai output LLM komersial buat training model lain itu area abu-abu di ToS (klausul "no competing model"). Untuk proyek portfolio summarizer 2B risikonya praktis kecil, tapi AKUI eksplisit di laporan/model card ("targets distilled from Gemini"). Mau 100% aman? Ganti teacher ke model open-weight besar, tapi jangan biarin keputusan ini nyandera timeline.
6. **Kuota free tier teacher.** Gemini flash free tier punya limit RPM/harian → generate 500 dok × 3 brief bisa makan berjam-jam/beberapa hari. Makanya `teacher_generate.py` WAJIB resumable (checkpoint tiap N dokumen) + cache: sekali jalan putus, jangan mulai dari nol.
```
