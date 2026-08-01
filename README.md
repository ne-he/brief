# BRIEF

> **Status: DESIGN PHASE. Blueprint selesai, implementasi dijadwalkan Q1 2027.**
> **Repo ini berisi spesifikasi, bukan implementasi. Nol baris kode training.**

Kalau kamu ke sini mencari model yang bisa dicoba atau demo yang bisa diklik, belum ada. Yang ada di sini adalah rancangan teknis lengkap sebelum satu baris kode ditulis, dan rancangan evaluasi yang sudah bisa dipakai projek lain hari ini.

---

## Masalahnya apa

Satu dokumen dibaca oleh orang-orang yang butuhnya beda-beda. Decision-maker butuh kesimpulan di kalimat pertama. Orang yang mengeksekusi butuh langkah, tenggat, dan angka spesifik. Publik umum butuh bahasa tanpa jargon. Satu ringkasan generik tidak melayani ketiganya.

## Pendekatannya apa

Fine-tune satu model kecil dengan QLoRA supaya bisa menghasilkan tiga gaya brief berbeda dari input yang sama, dikendalikan lewat tag audiens di dalam prompt.

- Base model: `google/gemma-2-2b-it`, dengan `Qwen/Qwen2.5-1.5B-Instruct` sebagai fallback.
- Training: Unsloth + PEFT + bitsandbytes, QLoRA 4-bit NF4, muat di Colab T4 gratis.
- Korpus: IndoSum lebih dulu karena bersih dan splitnya rapi. Subset putusan MA cuma stretch goal, sengaja tidak dibiarkan menyandera timeline.
- Serving: model 2B tidak jalan di Vercel serverless, jadi frontend Next.js di Vercel dan model diekspor ke GGUF supaya bisa hidup di CPU gratis lewat FastAPI.

Tag audiens dipakai lewat instruksi bahasa natural (`[AUDIENS: Eksekutif] ...`), bukan special token, karena datanya kecil dan cara ini tidak butuh operasi tokenizer.

## Kenapa audience conditioning

Karena "berhasil fine-tune" saja sudah komoditas. Yang menarik bukan bahwa modelnya bisa meringkas, tapi bahwa satu model bisa menggeser struktur, panjang, dan diksi sesuai pembacanya, dan pergeseran itu bisa diukur.

Tiga lensa yang didefinisikan, sumber kebenarannya di `config/audience_specs.yaml`:

| Lensa | Untuk siapa | Ciri |
|---|---|---|
| Eksekutif | decision-maker sibuk | kesimpulan di kalimat pertama, maksimal sekitar 5 bullet, tanpa jargon |
| Operasional | yang mengeksekusi | siapa mengerjakan apa, langkah bernomor, tenggat, angka spesifik |
| Awam | publik umum | bahasa polos, jargon dijelaskan, nada netral |

Satu definisi ini dipakai di tiga tempat sekaligus: prompt teacher, instruksi inferensi, dan rubrik human eval. Kalau ketiganya tidak berasal dari satu file yang sama, yang dilatih dan yang dinilai bisa diam-diam berbeda.

## Labelnya dari mana

Ini masalah yang paling sering dilewat, dan jawabannya menentukan arti semua angka evaluasi nanti.

Dataset summarization Indonesia yang tersedia cuma punya pasangan `(artikel, satu ringkasan referensi)`. Tidak ada ground truth per audiens, dan tidak akan pernah ada.

Solusinya distilasi sintetik: Gemini dipakai sebagai teacher untuk menghasilkan tiga target brief per dokumen mengikuti rubrik di `audience_specs.yaml`, lalu model kecil dilatih pada pasangan `(dokumen + tag audiens, brief audiens)`.

Konsekuensinya disebut terus terang, bukan disembunyikan: **kalau targetnya buatan Gemini, maka ROUGE dan BERTScore terhadap target itu mengukur kemiripan dengan teacher, bukan kualitas objektif.** Itu sebabnya rancangan evaluasinya tidak berhenti di metrik otomatis.

Higiene data yang sudah dikunci di blueprint karena kalau salah semua angka jadi bohong tanpa gejala:

- Split **by document**, bukan by row. Tiga baris audiens dari satu dokumen wajib masuk split yang sama.
- Test set dikunci **sebelum** teacher generation dijalankan.
- Target sintetik lewat quality gate otomatis dulu terhadap `audience_specs.yaml`. Yang gagal diregenerate sekali, masih gagal berarti dibuang, dan persentase lolosnya dilaporkan.

## Rancangan evaluasinya

Bagian yang paling matang dari repo ini, dan satu-satunya bagian yang sudah berguna sebelum ada implementasi apa pun.

- **Lexical**: ROUGE-1, ROUGE-2, ROUGE-L.
- **Semantic**: BERTScore, wajib dikonfigurasi untuk bahasa Indonesia. Default BERTScore memakai model bahasa Inggris dan tetap mengeluarkan angka yang kelihatan masuk akal untuk teks Indonesia, dan itu yang membuatnya berbahaya.
- **Tiga arm, bukan dua**: base dengan prompt audiens yang sama, fine-tuned, dan opsional teacher sebagai plafon atas. Arm pertama itu jawaban untuk pertanyaan "kenapa repot fine-tune, kenapa tidak prompt saja". Kalau fine-tuned tidak mengalahkannya, tesis projeknya runtuh, dan itu harus ketahuan dari data.
- **Bootstrap confidence interval** untuk delta antar arm, supaya klaim perbaikan punya errorbar.
- **Faithfulness** diukur terhadap ringkasan referensi asli dari dataset, bukan terhadap target Gemini, supaya ada satu jalur pengukuran yang tidak lewat teacher.
- **Apakah conditioning benar-benar bekerja**: ROUGE antar tiga output dari input yang sama harus **rendah**, ditambah distinct-n dan panjang rata-rata per audiens.
- **Human preference study**: blind A/B ditambah Likert audience-fit, tiap item dinilai minimal dua orang supaya agreement antar-rater bisa dilaporkan, dan ukuran sampel yang kecil disebut apa adanya.

Rancangan ini sudah diekstrak jadi dokumen berdiri sendiri yang tidak lagi terikat kasus QLoRA: lihat **[`EVAL_DESIGN.md`](EVAL_DESIGN.md)**.

## Isi repo

```
BRIEF/
├── README.md                  # file ini
├── BLUEPRINT.md               # spesifikasi teknis lengkap, sumber kebenaran
├── EVAL_DESIGN.md             # rancangan evaluasi generik, bisa dipakai projek lain
├── CLAUDE.md                  # guardrail untuk sesi coding agent
└── config/
    └── audience_specs.yaml    # definisi 3 lensa audiens
```

Struktur repo lengkap yang direncanakan ada di `BLUEPRINT.md` §5. Yang di atas adalah yang benar-benar ada sekarang.

## Kenapa diparkir

Nilai jual "bisa fine-tune, bukan cuma memanggil API" menurun di 2026. Context window besar dan prompting yang makin murah membuat pendekatan tanpa training jadi pembanding yang jauh lebih kuat untuk kasus seperti ini, dan blueprint ini sudah memasukkan pembanding itu sebagai arm evaluasi wajib justru karena alasan tersebut.

Menghabiskan enam sampai delapan minggu untuk membuktikan sesuatu yang argumennya sedang melemah adalah pertukaran yang buruk ketika masih ada projek lain yang sudah selesai dikoding tapi belum dirilis. Jadi implementasinya ditunda, dan rancangannya dipublikasikan apa adanya.

Spec yang bagus itu artefak yang sah. Yang tidak sah adalah memajangnya seolah sudah jadi. Banner di paling atas ada supaya perbedaan itu jelas.

> Catatan: tanggal milestone di `BLUEPRINT.md` §8 berasal dari rencana Juli 2026 dan sudah tidak berlaku. Jadwal yang berlaku adalah yang di banner.
