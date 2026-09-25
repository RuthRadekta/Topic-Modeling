# Topic Modeling Skripsi Menggunakan BERTopic dan IndoBERT

Repositori ini berisi implementasi *Topic Modeling* pada ratusan dokumen abstrak skripsi akademik. Pendekatan utama menggunakan model **BERTopic** yang dikombinasikan dengan *embedding* **IndoBERT** untuk menangkap semantik bahasa Indonesia. Selain melakukan topic modeling, model juga dikomparasikan dengan dua model lain, yaitu Latent Dirichlet Allocation dan Semantic Signal Separation.

## 📌 Deskripsi Proyek
Proyek ini menguji stabilitas dan kualitas topik dengan melakukan 10 iterasi *random seed* (10 hingga 100) pada proses reduksi dimensi (UMAP). Evaluasi kinerja model diukur menggunakan metrik:
- Coherence Score (C_v)
- Coherence Score (C_NPMI)
- Topic Diversity
- Topic Quality

## 🛠️ Teknologi & Library
- **Python 3.9+**
- **Pemrosesan Teks:** Sastrawi, Scikit-Learn
- **Embedding:** `indobenchmark/indobert-base-p2` (Sentence-Transformers)
- **Pemodelan Topik:** BERTopic, UMAP, HDBSCAN, Gensim
- **Deployment:** Streamlit

## Penjelasan Terkait Proyek
* Problem: Proyek ini dibuat karena pertambahan dokumen akademik seperti research paper di scopus terus meningkat setiap tahunnya. Hal ini membuat pentingnya melakukan pemetaan topik dan distribusi pertahunnya. Tujuannya agar menemukan tren topik pada suatu data dalam kurun waktu tertentu dan untuk menghindari redudansi topik.
* Dataset: Proyek ini dibuat untuk keperluan tugas akhir (skripsi) sehingga dataset yang digunakan adalah skripsi mahasiswa 2020-2025 dan diambil dari repositori online universitas dengan web scrapping (telah melalui izin resmi).
* Embedding: Indobert, dipilih karena sesuai dengan dataset yang digunakan, yaitu berbahasa Indonesia.
* Model: BERTopic, Latent Dirichlet Allocation, Semantic Signal Separation. Pemilihan ketiga model sekaligus bertujuan untuk membandingkan pipeline ketiga model.
* Metric Evaluation: C_v, NPMI, Topic Diversity, Topic Quality. Pemilihan metrik bertujuan untuk menentukan model terbaik berdasarkan topik yang dihasilkan, kedekatan antar topik, keberagaman topik, dan kualitas topik.
* Scenario: Setiap model melakukan topic modelling dan dievaluasi menggunakan empat metrik tersebut selama 10 kali iterasi dengan random seeds [10, 20, 30, ... 100]
* Topic Interpretation: Dilakukan topic interpretasi sehingga kualitas model tidak hanya ditentukan dari hasil kuantitatif, tetapi juga dari hasil kualitatif, seperti kata dari setiap topik yang dihasilkan apakah berbobot/bernilai?