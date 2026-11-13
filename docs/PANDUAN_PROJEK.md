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
