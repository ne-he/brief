# EVAL_DESIGN

Rancangan evaluasi untuk sistem yang menghasilkan teks: summarizer, RAG, agent analitik, atau apa pun yang outputnya kalimat dan bukan angka tunggal.

Dokumen ini diekstrak dari `BLUEPRINT.md` projek BRIEF, tapi sengaja dibuat lepas dari kasus QLoRA-nya. Isinya soal cara menilai, bukan soal cara melatih. Jadi bisa dipakai ulang di projek lain tanpa mengubah apa pun.

**Status:** dokumen desain. Tidak ada implementasi di repo ini.

---

## 0. Satu kalimat yang menentukan segalanya

Setiap metrik otomatis adalah *proxy*. Yang diukur bukan "bagus atau tidak", melainkan "mirip atau tidak dengan sesuatu yang kita anggap benar".

Konsekuensinya: sebelum memilih metrik, jawab dulu pertanyaan **referensinya dari mana**. Kalau referensi itu sendiri dibuat oleh mesin, semua angka di bawahnya berubah artinya. Bagian 4 membahas kasus itu secara khusus.

---

## 1. Metrik lexical (ROUGE, BLEU, exact match)

**Cara kerjanya:** hitung tumpang tindih n-gram antara output dan referensi. ROUGE-1 untuk kata tunggal, ROUGE-2 untuk pasangan kata, ROUGE-L untuk subsekuens terpanjang.

### Pakai kalau
- Referensinya ekstraktif atau bentuknya relatif terkunci. Ringkasan berita, judul, jawaban pendek berbasis kutipan.
- Ada banyak sekali item yang harus dinilai dan biayanya harus mendekati nol.
- Butuh metrik yang deterministik dan bisa direproduksi persis oleh orang lain. Ini nilai yang sering diremehkan: ROUGE selalu memberi angka yang sama untuk input yang sama, sedangkan judge berbasis LLM tidak.
- Butuh mendeteksi **kemiripan yang tidak diinginkan**. Ini pemakaian ROUGE yang paling sering dilupakan. Kalau satu sistem harus menghasilkan beberapa varian output yang wajib terasa berbeda, ROUGE antar-varian justru harus **rendah**. Angka tinggi di sini berarti sistemnya gagal membedakan.

### Batasan yang harus ditulis di laporan
- Parafrase yang benar dihukum. "Laba naik 12 persen" dan "keuntungan bertambah 12 persen" nyaris nol tumpang tindihnya di ROUGE-2, padahal artinya identik.
- Buta terhadap fakta. Output yang mengganti satu angka jadi salah bisa tetap dapat ROUGE tinggi karena kata-kata di sekitarnya sama.
- Sensitif panjang. Output yang lebih panjang cenderung menaikkan recall dan menurunkan precision, jadi bandingkan hanya sistem yang panjang outputnya sebanding, atau laporkan panjang rata-rata bersama skornya.
- Untuk bahasa Indonesia, tokenisasi level kata sudah memadai. Jangan pakai tokenizer subword bawaan model, hasilnya jadi tidak bisa dibandingkan lintas eksperimen.

---

## 2. Metrik semantic (BERTScore, embedding similarity, entailment)

**Cara kerjanya:** embed output dan referensi, lalu cocokkan token atau kalimat berdasarkan kedekatan vektor, bukan kesamaan huruf.

### Pakai kalau
- Outputnya abstraktif dan boleh diparafrase bebas.
- Referensinya cuma satu padahal jawaban benar bisa banyak bentuk.
- Ingin menangkap "artinya sama walau kalimatnya beda", yang persis titik buta ROUGE.

### Batasan yang harus ditulis di laporan
- **Gotcha yang paling mahal: bahasa.** BERTScore secara default memakai model bahasa Inggris. Diterapkan ke teks Indonesia, angkanya keluar tapi tidak berarti apa-apa. Wajib set `lang="id"` atau tunjuk model eksplisit seperti `indobenchmark/indobert-base-p1` atau mBERT. Angka yang keluar dari konfigurasi default terlihat masuk akal, dan itu justru yang membuatnya berbahaya.
- Skornya tidak punya skala absolut. BERTScore 0.85 tidak berarti "85 persen bagus". Angkanya cuma berguna untuk **membandingkan** dua sistem pada set data yang sama dengan model encoder yang sama. Jangan pernah bandingkan BERTScore lintas paper atau lintas konfigurasi encoder.
- Tetap bisa memberi skor tinggi ke output yang halus tapi salah fakta. Kemiripan semantik bukan faithfulness.
- Lebih mahal dan lebih lambat dari lexical. Untuk set besar, biayanya nyata.

---

## 3. Memilih di antara keduanya

Jangan memilih. Laporkan keduanya, karena keduanya salah dengan cara yang berbeda dan kesalahannya tidak berkorelasi.

| Situasi | Metrik utama | Metrik pendamping |
|---|---|---|
| Ringkasan ekstraktif, referensi terkunci | Lexical | Semantic sebagai cek parafrase |
| Ringkasan abstraktif, banyak jawaban benar | Semantic | Lexical untuk deteksi copy-paste mentah |
| Menilai apakah beberapa varian output benar-benar berbeda | Lexical antar-varian (target: rendah) | Distinct-n, panjang rata-rata |
| Jawaban pendek berbasis fakta / angka | Exact match atau toleransi numerik | Semantic hanya kalau ekstraksi gagal |
| Anti-halusinasi | Coverage atau entailment vs sumber asli | Bukan keduanya. Butuh referensi terpisah, lihat bagian 6 |

### Tiga aturan pelaporan yang murah tapi jarang dipakai

1. **Minimal tiga arm, bukan dua.** Bandingkan (a) sistem baseline yang sudah dioptimalkan dengan jujur, (b) sistem baru, (c) opsional plafon atas berupa sistem termahal yang ada. Tanpa (a) yang fair, klaim "sistem baru lebih baik" tidak menjawab pertanyaan pembunuh dari reviewer: kenapa repot, kenapa tidak pakai cara murah saja.
2. **Bootstrap confidence interval.** Resample hasil per-item minimal 1000 kali, laporkan selang untuk *delta* antara arm, bukan cuma rata-rata. Delta 0.03 dengan CI yang melewati nol bukan perbaikan, itu noise. Implementasinya beberapa baris, dampaknya besar pada kredibilitas.
3. **Kunci test set sebelum apa pun disentuh.** Split dulu, baru bikin referensi, baru latih atau tuning. Kalau satu sumber menghasilkan beberapa baris evaluasi, semua baris turunannya wajib masuk split yang sama. Split di level baris membuat sumber yang sama bocor ke train dan test, dan semua angka jadi tidak sah tanpa gejala yang kelihatan.

---

## 4. Masalah teacher-sebagai-judge

Ini bagian yang paling penting dan paling sering dilewat.

### Bentuk masalahnya

Pola yang umum sekarang: referensi atau label dibuat oleh model kuat X, lalu hasil sistem dinilai terhadap referensi itu, atau lebih parah, dinilai oleh model X juga.

Yang terjadi: **yang diukur adalah kedekatan dengan X, bukan kualitas.** Sistem yang meniru kebiasaan X dengan baik akan menang. Sistem yang jawabannya lebih benar tapi gayanya beda akan kalah. Kalau X sendiri punya kesalahan sistematis, kesalahan itu masuk ke definisi "benar" dan tidak akan pernah terdeteksi oleh metrik mana pun di bagian 1 dan 2.

Skor sempurna terhadap referensi buatan X berarti satu hal saja: sistemnya berhasil menjadi tiruan X yang murah. Itu bisa jadi tujuan yang sah, misalnya untuk distilasi, tapi harus disebut apa adanya, bukan dilaporkan sebagai "kualitas".

### Kenapa LLM-as-judge memperburuknya

Judge berbasis LLM punya bias yang sudah terdokumentasi: cenderung memilih jawaban yang lebih panjang, cenderung memilih output dari keluarga model dirinya sendiri, dan sensitif terhadap urutan penyajian. Kalau teacher dan judge adalah model yang sama, ketiga bias itu menumpuk ke arah yang sama, dan hasilnya terlihat sangat meyakinkan justru karena konsisten.

### Cara memitigasi

Urut dari yang paling murah:

1. **Pisahkan judge dari teacher.** Kalau referensi dibuat model X, penilaian dikerjakan model Y dari keluarga berbeda. Ini tidak menghilangkan bias, tapi menghilangkan korelasi bias antara pembuat label dan penilai, yang merupakan bagian paling merusaknya.
2. **Simpan referensi non-sintetik yang terpisah.** Kalau datasetnya punya referensi asli buatan manusia, jangan dibuang saat membuat target sintetik. Pakai referensi asli itu khusus untuk mengukur faithfulness dan coverage. Dengan begitu ada satu jalur pengukuran yang tidak lewat teacher sama sekali.
3. **Kalibrasi judge ke label manusia.** Ambil sampel kecil, misal 50 sampai 100 item, minta manusia menilainya, lalu ukur korelasi antara skor judge dan skor manusia. Laporkan korelasinya. Kalau korelasinya lemah, judge otomatisnya tidak boleh dipakai sebagai bukti, hanya sebagai filter kasar. Kalibrasi ini yang mengubah judge dari "asumsi" jadi "instrumen dengan margin error yang diketahui".
4. **Cek posisi dan panjang.** Acak urutan penyajian pada setiap item, dan laporkan apakah pemenang berkorelasi dengan panjang output. Kalau iya, skornya sedang mengukur verbosity.
5. **Tulis caveatnya di laporan, jangan disembunyikan.** Satu paragraf yang menyebutkan target berasal dari model X sehingga metrik mengukur kemiripan ke X, bukan kualitas absolut. Ini bukan kelemahan yang perlu ditutupi. Reviewer yang paham akan membaca ini sebagai tanda bahwa penulisnya mengerti apa yang sedang diukur.

---

## 5. Kenapa human preference study tetap perlu

Argumen tandingannya selalu sama: metrik otomatis sudah ada, kenapa repot cari orang.

Jawabannya:

- **Metrik otomatis tidak pernah mengukur variabel yang sebenarnya dipedulikan.** Kalau tujuannya "output ini terasa tepat untuk pembacanya", tidak ada n-gram atau embedding yang bisa menilai itu. Yang bisa cuma pembaca.
- **Metrik otomatis buta terhadap kegagalan yang seragam.** Kalau sistem selalu salah dengan cara yang sama dan referensinya juga salah dengan cara yang sama, semua angka terlihat sehat.
- **Human study adalah satu-satunya jalur yang tidak lewat teacher.** Di setup distilasi, ini bukan pelengkap, ini satu-satunya pengukuran yang independen.
- **Ini juga yang mengkalibrasi metrik otomatis.** Tanpa titik jangkar manusia, tidak ada cara tahu apakah kenaikan 0.02 di metrik otomatis berarti apa-apa bagi pembaca sungguhan.

### Desain minimum yang hasilnya masih bisa dipercaya

- Blind A/B antara dua arm. Label sistem disembunyikan, urutan penyajian diacak per item.
- 10 sampai 15 item per kondisi. Di bawah itu, hasilnya anekdot.
- Setiap item dinilai minimal dua orang, supaya bisa lapor persentase agreement antar-rater. Kalau mau lebih rapi, pakai Krippendorff alpha.
- Rubrik ditulis satu halaman **sebelum** rater direkrut. Improvisasi rubrik di tengah jalan membuat data tidak bisa digabung.
- Tanya lebih dari satu hal. Minimal: mana yang lebih baik secara keseluruhan, seberapa cocok untuk target pembacanya di skala Likert 1 sampai 5, dan apakah ada fakta yang mengada-ada.
- **Laporkan jujur bahwa n-nya kecil.** Sebut hasilnya directional, bukan konklusif. Kerendahan hati statistik di sini lebih meyakinkan daripada klaim besar dengan 5 responden.

---

## 6. Faithfulness dipisahkan dari kualitas

Kesalahan umum: menganggap satu skor bisa mewakili "bagus". Padahal ada dua pertanyaan berbeda yang butuh instrumen berbeda.

- **Kualitas / kecocokan gaya** dinilai terhadap referensi target. Metrik lexical, semantic, human preference.
- **Faithfulness** dinilai terhadap **dokumen sumber**, bukan terhadap referensi. Ukur coverage klaim, atau cek entailment ringan antara tiap kalimat output dan sumbernya.

Memisahkan keduanya membuat satu kelas bug jadi kelihatan: sistem yang outputnya enak dibaca, cocok gayanya, skor tinggi di semua metrik kemiripan, tapi menambahkan fakta yang tidak ada di sumber. Tanpa jalur faithfulness yang terpisah, kasus ini lolos.

---

## 7. Negative set: bagian yang paling sering hilang

Golden set biasanya hanya berisi pertanyaan yang **bisa** dijawab. Set semacam itu hanya mengukur separuh sistem.

Separuh yang lain adalah: apa yang terjadi ketika sistem seharusnya menolak. Sistem yang menjawab segalanya akan terlihat sempurna di golden set yang seluruhnya positif, dan akan mengarang di dunia nyata.

Negative set punya dua tingkat, dan yang kedua jauh lebih berharga:

- **Negatif mudah:** pertanyaan yang jelas di luar domain. Contoh: menanyakan cuaca ke sistem yang isinya laporan keuangan. Ini cuma memverifikasi bahwa gate-nya hidup.
- **Negatif sulit:** pertanyaan yang **terdengar** persis seperti domainnya, memakai kosakata dan entitas yang sama, tapi jawabannya memang tidak ada di korpus. Contoh: menanyakan angka periode yang tidak tercakup, entitas yang mirip tapi bukan bagian dari korpus, atau detail yang memang tidak pernah diungkap di dokumen. Ini yang benar-benar menguji apakah confidence gate mengukur ketersediaan bukti atau cuma mengukur kemiripan kata.

Kriteria lulus untuk item negatif bukan "jawabannya benar", melainkan "sistem menolak menjawab dan mengatakan kenapa". Laporkan sebagai dua angka terpisah: recall pada set positif, dan abstain rate pada set negatif. Satu angka gabungan menyembunyikan trade-off di antara keduanya.

---

## 8. Checklist sebelum melaporkan angka apa pun

- [ ] Test set dikunci sebelum referensi dibuat atau parameter disetel
- [ ] Split di level sumber, bukan level baris
- [ ] Near-duplicate antar split sudah dibuang
- [ ] Minimal tiga arm, termasuk baseline murah yang sudah dioptimalkan dengan jujur
- [ ] BERTScore atau encoder apa pun disetel ke bahasa yang benar, bukan default
- [ ] Delta antar arm punya confidence interval, bukan angka tunggal
- [ ] Faithfulness diukur terhadap sumber, bukan terhadap referensi
- [ ] Ada negative set, dan sebagian termasuk negatif sulit
- [ ] Kalau referensi atau judge berasal dari model, asal-usulnya disebut eksplisit di laporan
- [ ] Kalau ada judge otomatis, korelasinya dengan label manusia dilaporkan
- [ ] Human study punya rubrik tertulis, urutan teracak, dan agreement antar-rater
- [ ] Ukuran sampel disebut apa adanya

---

## 9. Di mana dokumen ini bisa langsung dipakai

Dua projek yang sudah punya harness dan tinggal butuh bagian-bagian di atas.

### 9.1 `agentic_verdict`: harness ada, kalibrasi belum

File: `backend/app/eval/grader.py` dan `backend/app/eval/metrics.py`, dengan gold set di `backend/app/eval/gold_set/superstore.json`.

Yang sudah ada dan sudah bagus:
- Jalur numerik dengan toleransi relatif, memberi kredit parsial 0.5 untuk "approach benar tapi angka meleset". Ini persis semangat bagian 3 soal exact match untuk jawaban berbasis angka.
- `_hallucination_flag` yang menandai kondisi "yakin tapi salah", yaitu confidence HIGH atau MEDIUM dengan correctness rendah. Ini implementasi konkret dari bagian 6.
- `_verification_accuracy` yang membedakan false pass dari false alarm, bukan cuma benar atau salah.

Yang belum dan langsung tersambung ke dokumen ini:
1. **`_llm_correctness` adalah judge yang belum dikalibrasi.** Fungsi itu meminta Gemini memberi skor 1 sampai 5 lalu menormalkannya ke 0 sampai 1, dan skor itu masuk ke `avg_correctness` dengan bobot yang sama seperti hasil pengecekan numerik. Belum ada satu pun bukti bahwa skor Gemini itu sejalan dengan penilaian manusia. Terapkan bagian 4 poin 3: ambil 50 item, nilai manual, laporkan korelasinya. Sebelum itu ada, `avg_correctness` adalah campuran antara pengukuran keras dan asumsi lunak.
2. **Judge satu keluarga dengan generator.** Sistemnya memakai Gemini untuk narasi, dan `_llm_correctness` juga memakai Gemini. Ini kasus teacher-sebagai-judge di bagian 4. Mitigasi termurah: pakai model dari keluarga lain khusus untuk grading, atau minimal laporkan caveat ini di README eval.
3. **Fallback 0.5 tidak terlihat di agregat.** Ada dua jalur yang mengembalikan 0.5 sebagai netral: LLM tidak kooperatif, dan tidak ada angka yang bisa diekstrak sama sekali. Keduanya larut jadi angka tengah di `aggregate()` dan tidak bisa dibedakan dari jawaban yang memang setengah benar. Tambahkan penghitung item yang tidak bisa dinilai, dan laporkan terpisah dari rata-rata.
4. **Tidak ada confidence interval.** `aggregate()` menghitung rata-rata sederhana. Dengan gold set kecil, dua run bisa beda beberapa persen murni karena kebetulan. Tambahkan bootstrap sesuai bagian 3 aturan 2.
5. **Belum ada baseline arm.** Belum ada pembanding berupa jalur yang lebih sederhana, misalnya jawaban tanpa verification loop. Tanpa itu, tidak ada bukti bahwa lapisan verification-nya berkontribusi.

### 9.2 `finance-rag`: golden set ada, negative set masih tipis

File: `eval/golden_set.yaml`, dengan runner di `eval/run_eval.py`.

Kondisi saat ini: 19 pertanyaan, terdiri dari 14 factual, 2 comparison, dan 3 out-of-scope. Jadi negative set-nya sudah ada, tapi ketiganya adalah **negatif mudah** menurut bagian 7: cuaca Jakarta, harga saham real-time, dan satu pertanyaan lelucon soal kecepatan burung. Ketiganya tidak menyerupai domain 10-K sama sekali, jadi yang teruji hanyalah bahwa gate-nya menyala untuk input yang jelas nyasar.

Yang perlu ditambahkan adalah **negatif sulit**: pertanyaan yang memakai nama emiten dan kosakata 10-K yang sama persis, tapi jawabannya memang tidak ada di korpus. Beberapa arah yang bisa dipakai:

- Periode di luar cakupan filing yang diindeks, misalnya menanyakan angka tahun fiskal yang tidak ada di korpus.
- Emiten yang tidak ada di korpus tapi ada di industri yang sama, sehingga retrieval kemungkinan besar tetap mengembalikan dokumen dengan skor kemiripan tinggi.
- Bagian laporan yang memang tidak diungkap, misalnya rincian yang tidak wajib ada di 10-K.
- Pertanyaan perbandingan yang salah satu sisinya tidak ada di korpus. Ini yang paling menjebak, karena sebagian bukti tersedia dan sistem tergoda mengarang sisanya.

Struktur entri baru bisa mengikuti skema yang sudah dipakai, dengan satu tipe tambahan supaya negatif mudah dan negatif sulit bisa dilaporkan terpisah:

```yaml
- id: hard-neg-out-of-period
  question: "..."
  type: hard-negative
  expected_sources: []
  must_include: []
  # lulus = gated == true, bukan jawaban benar
```

Lalu, sesuai bagian 7, laporkan hasilnya sebagai dua angka terpisah: hit-rate pada set positif, dan abstain rate pada set negatif, dengan negatif mudah dan negatif sulit dipisah. Menaikkan salah satu biasanya menurunkan yang lain, dan trade-off itu justru bagian yang menarik untuk ditunjukkan.
