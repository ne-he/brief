# BRIEF: rancangan melatih satu model kecil supaya bisa meringkas dokumen yang sama dengan tiga gaya, sesuai siapa pembacanya

| Sekilas | Keterangan |
|---|---|
| Apa | Spesifikasi teknis dan rancangan evaluasi untuk fine-tuning QLoRA model bahasa kecil yang menghasilkan tiga gaya ringkasan (Eksekutif, Operasional, Awam) dari satu input |
| Untuk siapa | Pembaca dokumen berbahasa Indonesia yang kebutuhannya beda: pengambil keputusan, pelaksana, dan publik umum |
| Status | Fase desain. Nol baris kode training. Implementasi dijadwalkan Q1 2027 menurut README |
| Link live | Belum ada, tahap desain |
| Repo | github.com/ne-he/brief (publik) |
| Teknologi yang direncanakan | Gemma 2 2B (cadangan Qwen2.5 1.5B), Unsloth, PEFT, bitsandbytes, Gemini sebagai teacher, Weights & Biases, llama.cpp/GGUF, FastAPI, Next.js di Vercel |
| Terakhir diperbarui | Commit terakhir 28 Sep 2026. Panduan ini dibuat November 2026 tanpa mengubah file lain |

Satu hal dulu sebelum baca lebih jauh: **projek ini belum diimplementasikan.** Tidak ada model yang sudah dilatih, tidak ada angka hasil, tidak ada demo. Semua yang ditulis di bawah adalah rencana, kecuali bagian yang jelas menyebut "yang sudah ada di repo".

## 1. Projek ini tentang apa

Bayangkan satu berita tentang kenaikan tarif listrik. Direktur perusahaan cuma mau tahu satu hal: ini berdampak ke biaya kita atau tidak, dan berapa. Staf keuangan mau tahu langkahnya: kapan berlaku, tagihan mana yang berubah, siapa yang harus menyesuaikan anggaran. Ibu rumah tangga mau tahu kenapa tagihan bulan depan beda dan apa artinya buat dia.

Beritanya satu. Pembacanya tiga. Ringkasan yang pas untuk satu orang bakal terasa salah untuk dua orang lainnya.

BRIEF adalah rancangan untuk melatih satu model AI kecil supaya bisa menulis ketiga versi itu. Caranya dengan menempelkan semacam "label pembaca" di awal perintah, misalnya `[AUDIENS: Eksekutif]`. Modelnya sama, dokumennya sama, labelnya yang menentukan gaya tulisannya.

Anggap saja juru bicara yang terlatih bicara ke direksi dan ke warga dengan bahasa berbeda. Bedanya, juru bicara ini model 2 miliar parameter yang cukup kecil untuk jalan di komputer gratisan.

## 2. Masalah yang mau diselesaikan

Ada dua masalah, satu soal produk dan satu soal teknis.

Masalah produknya: ringkasan otomatis yang umum dipakai sekarang menghasilkan satu versi untuk semua orang. Akibatnya ringkasan itu nanggung. Terlalu panjang untuk eksekutif, terlalu dangkal untuk pelaksana, dan masih penuh istilah untuk orang awam.

Masalah teknisnya lebih menarik dan jadi inti projek: **tidak ada data latih untuk ini.** Dataset ringkasan bahasa Indonesia seperti IndoSum hanya punya pasangan "artikel dan satu ringkasan". Tidak ada yang menulis versi eksekutif, versi operasional, dan versi awam untuk tiap artikel. README menyebutnya terang-terangan: ground truth per audiens tidak ada dan tidak akan pernah ada.

Pertanyaan sebenarnya: labelnya dari mana, dan kalau labelnya buatan mesin, gimana menilai hasilnya dengan jujur. Sebagian besar isi repo membahas itu.

## 3. Cara kerjanya nanti, langkah demi langkah

Sekali lagi: ini **RENCANA** dari `BLUEPRINT.md` bagian 3 dan 5. Belum ada satu pun tahap yang dijalankan.

```
config/audience_specs.yaml  (definisi 3 lensa, dipakai di 3 tempat)
        |
        v
[01 prep]   IndoSum -> bersihkan -> split per dokumen -> kunci test set
        |
        v
[teacher]   Gemini menulis 3 brief per dokumen -> quality gate -> dataset_audience.jsonl
        |
        v
[02 baseline]  model dasar + prompt audiens yang sama -> angka pembanding
        |
        v
[03 train]  QLoRA 4-bit di Colab T4 (Unsloth + PEFT) -> adapter
        |
        v
[04 eval]   base vs fine-tuned (vs teacher) -> ROUGE, BERTScore, distinct-n, human study
        |
        v
[05 export] gabung adapter -> fp16 -> GGUF q4_k_m
        |
        v
[06 serve]  FastAPI di CPU gratis (HF Spaces) <- frontend Next.js di Vercel
```

Penjelasan per langkah:

1. **Siapkan data.** Ambil IndoSum. BLUEPRINT menyebut ukurannya sekitar 19 ribu dokumen dengan split yang rapi (angka ini belum dicek ulang [?]). Bersihkan, buang artikel yang hampir kembar antar split, lalu bagi jadi train, validasi, dan test **per dokumen**. Test set dikunci sebelum langkah berikutnya.
2. **Buat label lewat teacher.** Gemini (BLUEPRINT menyebut `gemini-2.0-flash`, tapi model itu dimatikan Google Oktober 2026, jadi saat mulai nanti pakai model Flash yang masih hidup, mis. `gemini-2.5-flash`) diminta menulis tiga brief untuk tiap dokumen, mengikuti aturan di `audience_specs.yaml`. Rencananya mulai dari 300 sampai 500 dokumen dulu, dengan skrip yang bisa dilanjutkan kalau terputus karena kuota gratis.
3. **Saring label.** Skrip `validate_targets.py` (belum ada) mengecek tiap brief: panjangnya masuk rentang, strukturnya benar, bahasanya Indonesia, bukan penolakan, dan tidak membocorkan tag audiens. Yang gagal ditulis ulang sekali. Masih gagal, dibuang. Persentase lolosnya dicatat.
4. **Ukur baseline.** Sebelum melatih apa pun, model dasar diberi prompt audiens yang sama persis. Ini pembanding yang adil.
5. **Latih dengan QLoRA.** Model dasar dimuat dalam 4-bit, lalu hanya lapisan adapter kecil yang dilatih. Semua ini harus muat di GPU T4 gratis dari Google Colab.
6. **Evaluasi.** Bandingkan tiga "arm": base yang di-prompt, hasil fine-tune, dan opsional Gemini sebagai plafon atas. Detailnya di bagian 6.
7. **Ekspor dan sajikan.** Adapter digabung ke model, dikompres ke format GGUF, lalu dijalankan di CPU gratis lewat FastAPI. Frontend Next.js di Vercel memanggil API itu.

Contoh di bawah adalah **ilustrasi tulisan tangan untuk menunjukkan bentuk target**, bukan output model, dan sengaja dipendekkan. Misalkan dokumennya berita fiktif: pemerintah kota menutup satu ruas jalan selama dua minggu untuk perbaikan saluran air.

```
[AUDIENS: Eksekutif]
- Ruas jalan ditutup 2 minggu, rute distribusi kita terdampak.
- Risiko: keterlambatan pengiriman pagi.
- Keputusan yang dibutuhkan: setujui rute alternatif minggu ini.

[AUDIENS: Operasional]
1. Tim logistik memetakan rute alternatif sebelum hari Senin.
2. Admin mengabari pelanggan area terdampak soal jadwal baru.
3. Supervisor mengecek ulang kondisi jalan di akhir minggu pertama.

[AUDIENS: Awam]
Mulai minggu depan satu jalan di kota ditutup sekitar dua minggu
karena saluran airnya diperbaiki. Kalau biasa lewat sana, siapkan
waktu ekstra atau cari jalan lain. Setelah selesai, genangan saat
hujan diharapkan berkurang.
```

Perhatikan bentuknya: eksekutif pakai bullet dan kesimpulan di depan, operasional pakai daftar bernomor yang menyebut siapa mengerjakan apa, awam pakai paragraf biasa. Itu persis aturan di `audience_specs.yaml`.

## 4. Peta isi repo: file penting

Yang benar-benar ada sekarang:

| Path | Isinya apa |
|---|---|
| `README.md` | Pintu depan. Banner status desain, masalah, pendekatan, asal label, ringkasan rancangan evaluasi, dan alasan projek diparkir |
| `BLUEPRINT.md` | Sumber kebenaran teknis: keputusan arsitektur, diagram pipeline, metrik, struktur repo yang direncanakan, tech stack, hyperparameter QLoRA awal, fitur frontend, milestone, dan daftar risiko |
| `EVAL_DESIGN.md` | Rancangan evaluasi yang sengaja dibuat lepas dari kasus QLoRA. Membahas metrik lexical dan semantic, masalah teacher sebagai judge, human study, faithfulness, negative set, dan checklist sebelum melaporkan angka |
| `config/audience_specs.yaml` | Definisi tiga lensa audiens: label, tag, aturan tulisan, rentang panjang, aturan validasi, rubrik human eval, ambang cek conditioning, dan daftar alasan tolak |
| `.gitignore` | Daftar yang tidak boleh masuk repo: `.env`, data mentah, cache Python, dan sejenisnya |
| `CLAUDE.md` | Catatan untuk coding agent. Hanya ada di laptop, sudah dikeluarkan dari repo publik sejak 14 Agt 2026 |
| `docs/PANDUAN_PROJEK.md` | Dokumen ini |

Struktur yang **direncanakan** (folder `src/brief/`, `notebooks/`, `data/`, `human_study/`, `reports/`, `frontend/`, dan lain-lain) ada di BLUEPRINT bagian 5. Semuanya belum dibuat.

## 5. Teknologi yang direncanakan dan kenapa

| Lapisan | Pilihan | Alasan menurut BLUEPRINT |
|---|---|---|
| Model dasar | `google/gemma-2-2b-it` | Bahasa Indonesianya lumayan, lisensinya oke, didukung Unsloth, muat 4-bit di T4 |
| Cadangan | `Qwen/Qwen2.5-1.5B-Instruct` | Lebih cepat, berguna untuk ablation kecil atau kalau T4 kehabisan memori |
| Training | Unsloth + PEFT + bitsandbytes, QLoRA 4-bit NF4 | Hemat VRAM, gratis di Colab T4 |
| Teacher | Gemini (BLUEPRINT menulis `gemini-2.0-flash`, sudah pensiun Oktober 2026, ganti ke Flash yang masih aktif) | Sudah ada akses, dipakai untuk membuat target sintetik |
| Tracking | Weights & Biases | Loss, learning rate, contoh generasi, perbandingan base dan tuned |
| Metrik | `evaluate`/`rouge_score`, `bert-score` (wajib diset bahasa Indonesia), distinct-n buatan sendiri | Lihat bagian 6 |
| Ekspor | llama.cpp ke GGUF q4_k_m | Supaya model bisa jalan di CPU gratis |
| Serving | FastAPI + `llama-cpp-python`, host di HF Spaces CPU | Model 2B tidak bisa jalan di Vercel serverless |
| Frontend | Next.js App Router di Vercel | Dikerjakan sendiri oleh Nehemiah |

Hyperparameter awal yang dicatat di BLUEPRINT: `lora_r` 16, `lora_alpha` 32, dropout 0.05, adapter dipasang di semua proyeksi attention dan MLP, `max_seq_length` 2048, learning rate 2e-4, 2 epoch, batch 2 dengan gradient accumulation 4. Semua ini titik awal, belum diuji.

Benang merah pemilihan stack ini adalah budget nol rupiah. Colab T4 gratis untuk training, HF Spaces CPU gratis untuk model, Vercel gratis untuk frontend. Itu juga alasan GGUF bukan pelengkap tapi syarat: tanpa kompresi, model 2B tidak akan hidup di CPU gratisan.

BLUEPRINT juga menulis guardrail serving supaya demo di CPU gratis tidak mati: batas panjang input sekitar 3000 token, batas output per audiens diambil dari YAML, dan cache untuk dokumen contoh.

## 6. Rancangan evaluasi

Ini bagian paling matang dari repo, dan README sendiri bilang ini satu-satunya bagian yang sudah berguna sebelum ada implementasi. **Belum ada satu angka hasil pun.** Tabel banding di BLUEPRINT bagian 4.6 memang sengaja dibiarkan kosong dengan isi "(isi nanti)".

Yang direncanakan diukur:

- **ROUGE-1, ROUGE-2, ROUGE-L.** Menghitung tumpang tindih kata antara output dan target. Murah dan deterministik, tapi menghukum parafrase yang benar dan buta terhadap fakta yang salah.
- **BERTScore yang diset ke bahasa Indonesia.** Mengukur kemiripan makna lewat embedding. Jebakan utamanya: default BERTScore memakai model bahasa Inggris dan tetap mengeluarkan angka yang kelihatan wajar untuk teks Indonesia. Rencananya pakai `lang="id"` atau `indobenchmark/indobert-base-p1`.
- **Tiga arm.** (a) model dasar dengan prompt audiens yang sama, (b) hasil fine-tune, (c) opsional Gemini sebagai plafon. Arm (a) adalah jawaban untuk pertanyaan "kenapa nggak prompt aja". Kalau (b) tidak mengalahkan (a), tesis projeknya gugur, dan itu harus kelihatan dari data.
- **Bootstrap confidence interval.** Resample minimal 1000 kali untuk selisih antar arm. Selisih kecil yang selangnya melewati nol dianggap noise.
- **Faithfulness.** Diukur terhadap ringkasan referensi asli dari dataset atau terhadap dokumen sumber, bukan terhadap target Gemini. Ini satu-satunya jalur otomatis yang tidak lewat teacher.
- **Cek conditioning.** ROUGE-L antar tiga output dari input yang sama harus **rendah**. YAML menaruh ambang awal 0.55 dan menyebutnya heuristik yang harus dikalibrasi ulang. Ditambah distinct-1/2 dan panjang rata-rata per audiens.
- **Human preference study.** Blind A/B antara base yang di-prompt dan hasil fine-tune, skala Likert 1 sampai 5 untuk kecocokan audiens, pertanyaan soal fakta yang dikarang. Sekitar 10 sampai 15 dokumen dikali tiga audiens, tiap item dinilai minimal dua orang supaya agreement antar penilai bisa dilaporkan. Hasilnya disebut directional karena sampelnya kecil.
- **LLM-as-judge (opsional).** Boleh dipakai sebagai proksi, tapi bias "teacher jadi judge" wajib ditulis. EVAL_DESIGN menyarankan judge dari keluarga model lain dan kalibrasi ke label manusia.

Inti filosofinya ada di satu kalimat EVAL_DESIGN: setiap metrik otomatis adalah proksi. Karena target latihnya buatan Gemini, ROUGE dan BERTScore terhadap target itu cuma mengukur seberapa mirip model dengan Gemini. Human study yang jadi validator kualitas sebenarnya.

## 7. Keputusan desain penting

**Label dari distilasi sintetik.** Karena data per audiens tidak ada, Gemini yang membuatnya. Untungnya, projek bisa jalan. Ruginya, semua metrik kemiripan jadi bias ke gaya Gemini. Ada juga area abu-abu di syarat layanan soal memakai output model komersial untuk melatih model lain. BLUEPRINT menilai risikonya kecil untuk projek portofolio tapi mewajibkan pengakuan eksplisit di model card.

**Tag audiens berupa teks biasa, bukan token khusus.** `[AUDIENS: Eksekutif]` ditulis sebagai instruksi biasa. Alasannya, datanya kecil dan cara ini tidak perlu mengutak-atik tokenizer. Gantinya, model harus belajar membaca instruksi itu dari contoh saja.

**IndoSum dulu, putusan MA belakangan.** Berita itu bersih dan pendek, jadi pipeline bisa dibuktikan jalan. Putusan Mahkamah Agung lebih mengesankan tapi butuh scraping dan pembersihan berminggu-minggu, dan dokumennya bisa lebih dari 8 ribu token yang susah dimuat di T4. Statusnya stretch goal yang sengaja tidak boleh menyandera jadwal.

**Satu file definisi untuk tiga kegunaan.** `audience_specs.yaml` dipakai untuk prompt teacher, instruksi saat inferensi, dan rubrik human eval. Kalau ketiganya ditulis terpisah, yang dilatih dan yang dinilai bisa diam-diam beda definisi. Konsekuensinya, mengubah YAML berarti mengubah ketiganya sekaligus, jadi harus hati-hati.

**Split per dokumen, test set dikunci duluan.** Satu dokumen menghasilkan tiga baris latih. Kalau dibagi per baris, dokumen yang sama bisa muncul di train dan test, dan semua angka evaluasi jadi tidak sah tanpa gejala.

**Model kecil di CPU, bukan model besar di GPU.** Gratis dan bisa dipamerkan publik, tapi outputnya lebih lambat dan kualitas prosanya mungkin biasa saja. BLUEPRINT bilang nilai "wow"-nya ada di perbedaan tiga gaya yang jelas, bukan di keindahan kalimat.

**Diparkir dan dipublikasikan apa adanya.** README menjelaskan bahwa nilai jual "bisa fine-tune" melemah di 2026 karena prompting makin murah dan context window makin besar. Menghabiskan enam sampai delapan minggu untuk itu dianggap kurang sepadan selama masih ada projek lain yang tinggal dirilis. Jadi spesifikasinya diterbitkan dengan banner yang jelas bahwa ini belum jadi.

## 8. Update terbaru

- **1 Agt 2026** (`6509e65`): spec awal masuk repo. README, BLUEPRINT, EVAL_DESIGN, `audience_specs.yaml`, `.gitignore`, dan CLAUDE.md.
- **14 Agt 2026** (`e179a88`): CLAUDE.md dikeluarkan dari repo publik dan ditambahkan ke `.gitignore`.
- **28 Sep 2026** (`dbf5cd3`): pembersihan em dash di BLUEPRINT dan `audience_specs.yaml`. Ini commit terakhir.
- **11 Nov 2026**: sesi kerja tanpa perubahan kode.
- **November 2026**: panduan ini ditambahkan di `docs/`, belum di-commit.

## 9. Batasan jujur dan yang belum dikerjakan

- Nol kode. Tidak ada skrip data, training, evaluasi, ekspor, maupun serving. Tidak ada `requirements.txt`.
- Tidak ada angka. Klaim apa pun soal kualitas model belum bisa dibuat.
- Tanggal milestone di BLUEPRINT bagian 8 (27 Jul sampai 7 Sep 2026) sudah kedaluwarsa. README menyatakan jadwal yang berlaku adalah Q1 2027.
- Ambang conditioning 0.55 dan batas input sekitar 3000 token masih tebakan awal.
- Ukuran IndoSum sekitar 19 ribu dokumen hanya dikutip dari BLUEPRINT [?].
- Pohon file di README masih mencantumkan CLAUDE.md, padahal file itu tidak ada di repo publik.
- Commit pembersihan em dash belum tuntas: BLUEPRINT masih memuat tanda pisah rentang (en dash) di beberapa rentang angka, misalnya durasi minggu.
- Risiko terbesar secara ide: bisa jadi base model yang di-prompt dengan baik sudah cukup, sehingga fine-tune tidak menang. Rancangan evaluasinya memang dibuat untuk mendeteksi itu, tapi belum diuji.
- Kuota gratis Gemini bisa membuat pembuatan label untuk ratusan dokumen makan waktu berhari-hari.
- Catatan lokal di CLAUDE.md menyebut syarat buka parkir: projek `agentic_verdict` dan `finance-rag` harus punya link publik yang hidup dulu.
