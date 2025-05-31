# Laporan Proyek Machine Learning - Tika Putri Marsanti
# Sistem Rekomendasi Buku Menggunakan Machine Learning

## Project Overview

Di tengah era digital yang dipenuhi oleh informasi, tantangan dalam dunia literasi adalah menyajikan bacaan yang sesuai dengan preferensi masing-masing individu. Meskipun kemajuan toko buku daring dan layanan digital telah meningkatkan akses terhadap buku, banyaknya pilihan yang tersedia justru menyulitkan pembaca untuk menemukan bacaan yang sesuai minatnya. Akibatnya, proses pencarian menjadi tidak efisien dan membingungkan, yang bisa mengurangi minat baca seseorang.

Untuk mengatasi permasalahan ini, dibutuhkan sebuah sistem rekomendasi buku yang mampu memberikan saran bacaan secara personal dan akurat sesuai dengan kesukaan pembaca. Sistem ini diharapkan tidak hanya membantu pembaca menemukan buku yang relevan, tetapi juga memberikan peluang lebih luas bagi penulis dan penerbit dalam menjangkau audiens. Dalam jurnal Machine Learning Techniques for Book Recommendation: An Overview oleh Khalid Anwar et al, dijelaskan bahwa sistem rekomendasi buku dapat mendukung pustakawan dalam mengelola katalog lebih efisien serta membantu pembaca menemukan buku yang paling sesuai dengan preferensi mereka. Dari sisi bisnis, sistem ini juga dapat membantu pelaku industri buku dalam mengatur inventaris serta meningkatkan penjualan. Oleh karena itu, pengembangan sistem rekomendasi buku menjadi krusial untuk memperkaya pengalaman membaca serta mendukung kemajuan industri literasi secara menyeluruh.

Untuk pengembangan model, proyek ini menggunakan data dari Book Recommendation Dataset. Dataset ini mencakup informasi pengguna anonim beserta data demografis serta penilaian mereka (baik eksplisit maupun implisit) terhadap berbagai buku. Data ini dikumpulkan selama empat minggu oleh komunitas Book-Crossing pada tahun 2004. Dengan memanfaatkan data tersebut, proyek ini bertujuan membangun sistem rekomendasi buku yang inovatif dan bermanfaat bagi pembaca, pelaku industri, serta mampu meningkatkan ekosistem literasi.

## Referensi
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3356349#
https://www.researchgate.net/profile/Dhiman-Sarma/publication/348968927_Personalized_Book_Recommendation_System_using_Machine_Learning_Algorithm/links/606402eba6fdccbfea1a621e/Personalized-Book-Recommendation-System-using-Machine-Learning-Algorithm.pdf

## Business Understanding

Pada bagian ini dijelaskan proses klarifikasi terhadap permasalahan bisnis yang ingin diselesaikan dengan bantuan teknologi sistem rekomendasi berbasis data. Proses ini mencakup identifikasi masalah, tujuan proyek, serta pendekatan solusi yang akan diimplementasikan.

### Problem Statements
Bayangkan sebuah perusahaan penjual buku yang telah beroperasi selama beberapa waktu dan telah mengumpulkan berbagai data terkait pelanggannya, daftar buku dari berbagai penerbit, serta penilaian dari pelanggan atas buku yang telah mereka beli. Semua data tersebut tercantum dalam Book Recommendation Dataset.

Seorang Data Scientist di perusahaan tersebut ingin memanfaatkan data ini untuk meningkatkan penjualan. Maka dikembangkanlah sistem rekomendasi buku guna menjawab pertanyaan berikut:

1. Bagaimana cara membangun sistem rekomendasi yang dapat memberikan saran buku secara personal kepada pengguna berdasarkan teknik content-based filtering?
2. Bagaimana memanfaatkan data rating untuk merekomendasikan buku lain yang berpotensi disukai pengguna, meskipun belum pernah mereka baca atau kunjungi sebelumnya?

### Goals

Untuk menjawab pertanyaan di atas, perusahaan menetapkan tujuan sebagai berikut:

- Menghasilkan rekomendasi buku yang relevan dan bersifat personal bagi pengguna dengan menggunakan pendekatan content-based filtering.
- Memberikan saran buku yang sesuai dengan preferensi pengguna serta belum pernah mereka baca, dengan memanfaatkan pendekatan collaborative filtering.

## Solution Statements
Untuk mencapai tujuan tersebut, proyek ini akan mengimplementasikan dua pendekatan sistem rekomendasi berikut:
1. Pendekatan Content-Based Filtering
Sistem ini akan merekomendasikan buku berdasarkan fitur konten dari buku-buku yang pernah disukai atau diberi rating tinggi oleh pengguna, seperti nama penulis dan sinopsis buku. Pendekatan ini menggunakan metode seperti TF-IDF Vectorizer untuk mengekstraksi fitur dan cosine similarity untuk mengukur kemiripan antar buku.

2. Pendekatan Collaborative Filtering
Sistem ini akan merekomendasikan buku berdasarkan pola kesamaan rating antar pengguna. Teknik yang digunakan adalah User-Item Collaborative Filtering, di mana prediksi rating dilakukan dengan menganalisis kesamaan antara pengguna dan item. Evaluasi dilakukan menggunakan metrik RMSE (Root Mean Squared Error).

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah Book Recommendation Dataset yang diperoleh dari platform Kaggle. Dataset ini dikumpulkan oleh Cai-Nicolas Ziegler selama periode empat minggu pada tahun 2004 melalui komunitas Book-Crossing. Dataset ini mencakup 278.858 pengguna anonim yang memberikan 1.149.780 penilaian (eksplisit maupun implisit) terhadap 271.379 buku.

Struktur Dataset
Book Recommendation Dataset terdiri dari tiga file dalam format CSV, yaitu:

├── book-dataset/
    ├── books.csv     <- Informasi mengenai buku
    ├── ratings.csv   <- Penilaian buku dari pengguna
    └── users.csv     <- Informasi demografi pengguna

Penjelasan masing-masing file adalah sebagai berikut:
- **`books.csv`**: Berisi 271.360 entri buku dengan 8 kolom: ISBN, Book-Title, Book-Author, Year-Of-Publication, Publisher, Image-URL-S, Image-URL-M, dan Image-URL-L.
- **`ratings.csv`**: Berisi 1.149.780 entri rating dari pengguna dengan 3 kolom: User-ID, ISBN, dan Book-Rating.
- **`users.csv`**: Berisi 278.858 entri pengguna dengan 3 kolom: User-ID, Location, dan Age.

### Variabel pada Dataset
**Books**:
- `ISBN`: Nomor unik identifikasi buku.
- `Book-Title`: Judul buku.
- `Book-Author`: Nama penulis buku.
- `Year-Of-Publication`: Tahun buku diterbitkan.
- `Publisher`: Nama penerbit.
- `Image-URL-S/M/L`: Link gambar buku dalam ukuran kecil, sedang, dan besar.

**Ratings**:
- `User-ID`: ID pengguna.
- `ISBN`: Nomor ISBN dari buku.
- `Book-Rating`: Penilaian dari pengguna (0–10).

**Users**:
- `User-ID`: ID pengguna.
- `Location`: Lokasi pengguna.
- `Age`: Usia pengguna.

### Tahapan untuk memahami data:
- Data Loading
- Univariate Exploratory Data Analysis
- Data Preprocessing

### Data Loading
# Dataset Books 
<img width="1317" alt="Screenshot 2025-05-31 at 17 36 28" src="https://github.com/user-attachments/assets/941dbb29-4953-45b0-8c89-7a2f947b8092" />
# Dataset Ratings
<img width="275" alt="Screenshot 2025-05-31 at 17 36 33" src="https://github.com/user-attachments/assets/dcaac9bd-86bc-44c5-80fe-da26611468bc" />
# Dataset Users
<img width="346" alt="Screenshot 2025-05-31 at 17 36 37" src="https://github.com/user-attachments/assets/46edd881-d631-4be6-9bb8-05a14ca7a84b" />
# Analisis 
Berdasarkan output yang ditampilkan, diperoleh informasi struktur dari tiga variabel utama dalam dataset:

### 1. `books` (271.360 data, 8 kolom)
Berisi informasi detail tentang buku:
- `ISBN`: Nomor identitas unik untuk setiap buku.
- `Book-Title`: Judul buku.
- `Book-Author`: Nama penulis buku.
- `Year-Of-Publication`: Tahun terbit buku.
- `Publisher`: Nama penerbit buku.
- `Image-URL-S`: URL gambar buku ukuran kecil.
- `Image-URL-M`: URL gambar buku ukuran sedang.
- `Image-URL-L`: URL gambar buku ukuran besar.
### 2. `ratings` (340.556 data, 3 kolom)
Mencatat penilaian pengguna terhadap buku:
- `User-ID`: Kode unik pengguna anonim.
- `ISBN`: Nomor identitas buku.
- `Book-Rating`: Nilai rating yang diberikan pengguna.
### 3. `users` (278.858 data, 3 kolom)
Menyimpan informasi pengguna:
- `User-ID`: Kode unik pengguna anonim.
- `Location`: Lokasi tempat tinggal pengguna.
- `Age`: Usia pengguna.

### Univariate Exploratory Data Analysis
Pada tahap ini, dilakukan analisis dan eksplorasi terhadap masing-masing variabel dalam dataset guna memahami distribusi dan karakteristik masing-masing variabel. Pemahaman ini bertujuan untuk menentukan pendekatan atau algoritma yang paling sesuai diterapkan. Adapun variabel-variabel dalam Book Recommendation Dataset adalah sebagai berikut:
- books : berisi informasi terkait buku.
- ratings : mencerminkan penilaian yang diberikan oleh pengguna terhadap buku.
- users : menyimpan data pengguna termasuk informasi demografis.
**Books Variabel**
Dengan menggunakan fungsi info(), diketahui bahwa dataset books (dari file books.csv) memiliki 271.360 entri dan 8 kolom, yaitu: ISBN, Book-Title, Book-Author, Year-Of-Publication, Publisher, Image-URL-S, Image-URL-M, dan Image-URL-L. Ditemukan bahwa kolom Year-Of-Publication memiliki tipe data object, padahal seharusnya bertipe integer. Upaya untuk mengubah tipe data menggunakan books['Year-Of-Publication'].astype('int') menghasilkan error ValueError: invalid literal for int() with base 10: 'DK Publishing Inc'. Ini menunjukkan adanya kesalahan input berupa nilai non-numerik.

Setelah ditelusuri, terdapat dua nilai teks yang salah pada kolom tersebut yaitu 'DK Publishing Inc' dan 'Gallimard'. Keduanya kemudian dihapus agar tidak menghambat proses konversi tipe data ke integer. Setelah penghapusan, kolom Year-Of-Publication berhasil diubah ke tipe data integer.

Langkah berikutnya adalah membersihkan data dengan menghapus kolom yang tidak relevan dengan proses pengembangan model. Karena model yang akan dikembangkan adalah sistem rekomendasi berbasis konten (content-based filtering), maka atribut terkait gambar (Image-URL-S, Image-URL-M, Image-URL-L) tidak dibutuhkan dan dihapus. Hasil akhir dari dataset books yang telah dibersihkan ditampilkan pada Tabel 4.
# Tampilan dataset books setelah pembersihan
<img width="871" alt="Screenshot 2025-05-31 at 17 41 21" src="https://github.com/user-attachments/assets/82e55272-e1c8-4878-a74a-fccd858e767a" />
# Analisis
Setelah dilakukan pembersihan, dataset books memiliki lima kolom utama. Statistik terkait jumlah entri unik pada masing-masing kolom adalah:
- Jumlah ISBN buku: 271.357
- Jumlah judul buku: 242.132
- Jumlah penulis: 102.022
- Jumlah tahun publikasi: 116
- Jumlah penerbit: 16.805
Terlihat bahwa jumlah ISBN lebih banyak daripada jumlah judul buku, mengindikasikan bahwa terdapat beberapa entri dengan ISBN tidak valid atau duplikat. Mengingat setiap buku seharusnya memiliki satu ISBN yang unik, maka dataset akan difilter agar hanya menyertakan entri dengan ISBN valid.

### Variabel `ratings`

Selanjutnya, dilakukan eksplorasi terhadap variabel `ratings` yang mencerminkan nilai atau peringkat buku dari pembaca. Dengan fungsi `info()`, diketahui bahwa terdapat 1.149.780 entri dan tiga kolom: `User-ID`, `ISBN`, dan `Book-Rating`. Informasi penting yang dapat diambil dari variabel ini adalah:

- Total pengguna: 105.283
- Jumlah ISBN buku yang dirating: 340.556
- Rentang nilai rating: 0–10 (dengan 0 sebagai nilai terendah dan 10 tertinggi)

Karena jumlah data sangat besar (lebih dari 1 juta baris), hanya sebagian data yang akan digunakan untuk proses pelatihan model collaborative filtering. Data yang diambil dibatasi hingga 5000 baris pertama. Dataset hasil subset ini kemudian disimpan dalam variabel `df_rating`.

### Variabel `users`

Variabel terakhir adalah `users`, yang berisi informasi pengguna dan demografinya. Hasil eksplorasi menunjukkan:

- Total entri: 278.858
- Kolom: `User-ID`, `Location`, dan `Age`
- Terdapat entri dengan nilai usia yang tidak diketahui

Variabel `users` berguna untuk sistem rekomendasi berbasis demografi. Namun, pada studi kasus ini, data `users` tidak akan digunakan dalam pengembangan model. Model hanya akan menggunakan data `books` dan `ratings`.
"""

### Data Preprocessing
Berdasarkan hasil eksplorasi awal (data understanding), diketahui bahwa Book Recommendation Dataset terdiri dari tiga file terpisah, yaitu books, ratings, dan users. Pada tahap ini, ketiga file tersebut digabungkan menjadi satu dataset agar dapat digunakan untuk proses pengembangan model. Setelah proses penggabungan dilakukan, terbentuk dataset dengan 7 variabel dan total 1.149.780 baris data. Tabel 5 menampilkan cuplikan dari dataset hasil gabungan antara ratings dan books, yang nantinya akan digunakan sebagai dasar dalam pembuatan sistem rekomendasi.
# Dataset Gabungan ratings dan books
<img width="1051" alt="Screenshot 2025-05-31 at 17 46 35" src="https://github.com/user-attachments/assets/f28f218b-efe4-4c23-a5a0-68721ad3b030" />

## Data Preparation
Karena sistem rekomendasi akan dikembangkan menggunakan dua pendekatan, yaitu content-based filtering dan collaborative filtering, maka tahap persiapan data akan dibedakan berdasarkan masing-masing teknik.

# Persiapan Data untuk Model Content-Based Filtering
Dalam tahap ini, dilakukan sejumlah proses untuk menyiapkan data, di antaranya:
- Menghapus data yang memiliki missing value
- Menyeragamkan jenis buku berdasarkan ISBN
Pada metode content-based filtering, setiap nomor ISBN merepresentasikan satu judul buku, sehingga ISBN harus bersifat unik untuk mencegah terjadinya duplikasi dan bias data. Oleh karena itu, data perlu dipastikan sudah bersih dan siap digunakan dalam proses pelatihan model.
# Penanganan Missing Value
Langkah pertama adalah memeriksa keberadaan missing value menggunakan perintah books.isnull().sum(). Hasilnya menunjukkan bahwa hanya fitur User-ID, ISBN, dan Book-Rating yang tidak memiliki missing value. Sebaliknya, fitur seperti Publisher memiliki missing value terbanyak, yaitu 118.650. Jumlah ini dinilai masih dapat ditoleransi (sekitar 10,3% dari total data), sehingga keputusan diambil untuk menghapus baris-baris yang memiliki missing value, dan menyimpan hasilnya dalam variabel all_books_clean. Setelah pembersihan, jumlah baris data menyusut menjadi 1.031.129.
# Penyamaan Judul Buku Berdasarkan ISBN
Setelah menghapus missing value, langkah berikutnya adalah memastikan tidak ada satu ISBN yang digunakan oleh lebih dari satu judul buku. Hal ini penting agar model tidak salah mengenali buku yang sebenarnya berbeda namun memiliki ISBN yang sama. Setelah dilakukan pemeriksaan, ditemukan bahwa terdapat duplikasi ISBN pada beberapa judul buku. Untuk mengatasi hal ini, data dideduplikasi berdasarkan kolom ISBN, sehingga setiap ISBN hanya mewakili satu entri buku. Setelah proses ini, data tersisa sebanyak 270.145 baris.

Langkah berikutnya adalah menyusun dictionary yang terdiri dari pasangan key-value untuk setiap ISBN, judul buku, penulis, tahun terbit, dan penerbit. Hasil akhir disimpan dalam variabel books_new. 
<img width="933" alt="Screenshot 2025-05-31 at 17 49 02" src="https://github.com/user-attachments/assets/ca6c73d0-8412-40b7-82e9-a3e5f710f800" />

# Persiapan Data untuk Model Collaborative Filtering
Pada pendekatan collaborative filtering, data perlu dikonversi ke dalam bentuk matriks numerik agar dapat digunakan dalam pelatihan model. Namun sebelum proses pembagian data menjadi training dan validation set, dilakukan beberapa tahap persiapan data sebagai berikut:
- Melakukan encoding terhadap fitur 'User-ID' dan 'ISBN' ke dalam format indeks numerik (integer).
- Memetakan hasil encoding ke dataframe terkait.
- Mengecek kembali jumlah unik pengguna dan buku.
- Mengubah tipe data Book-Rating menjadi float agar kompatibel dengan proses pelatihan model.
Setelah semua langkah di atas selesai, data siap untuk dibagi menjadi data pelatihan dan data validasi sebagai bagian dari proses training model collaborative filtering.

## Modeling
# Model Development dengan Collaborative Filtering
### Konsep Dasar

Content-Based Filtering merupakan pendekatan sistem rekomendasi yang memanfaatkan informasi konten dari item atau pengguna untuk menghasilkan rekomendasi. Sistem ini bekerja dengan mencocokkan preferensi pengguna dengan karakteristik item yang pernah disukai sebelumnya.

**Contoh implementasi**: Jika pengguna menyukai buku _"Introduction to Machine Learning"_ karya **Alex Smola**, sistem akan mencari dan merekomendasikan buku-buku lain dengan penulis yang sama atau karakteristik serupa.

### Keunggulan Content-Based Filtering

- **Personalisasi tinggi**: Memberikan rekomendasi yang disesuaikan dengan preferensi unik setiap pengguna  
- **Solusi cold-start**: Efektif mengatasi masalah ketika data pengguna masih terbatas  
- **Independensi data**: Tidak bergantung pada informasi pengguna lain  
- **Cocok untuk item terstruktur**: Bekerja optimal dengan item yang memiliki atribut jelas seperti genre, penulis, atau kategori  

### Keterbatasan Content-Based Filtering

- **Kurang eksplorasi**: Hanya merekomendasikan item serupa tanpa memberikan kejutan atau variasi  
- **Kompleksitas preferensi**: Kesulitan menangkap preferensi pengguna yang dinamis dan kompleks  
- **Filter bubble**: Risiko terjebak dalam "gelembung filter" yang membatasi diversitas rekomendasi  

### Implementasi Teknis

#### TF-IDF Vectorization

Sistem menggunakan **TF-IDF (Term Frequency - Inverse Document Frequency)** Vectorizer untuk mengekstrak fitur penting dari setiap judul buku:

- **Term Frequency (TF)**: Mengukur frekuensi kemunculan kata dalam dokumen  
- **Inverse Document Frequency (IDF)**: Mengukur keunikan kata dalam seluruh koleksi dokumen  

**Proses implementasi**:

1. Import fungsi `TfidfVectorizer()` dari `sklearn`  
2. Melakukan mapping array dari indeks integer ke nama fitur menggunakan `get_feature_names_out()`  
3. Transformasi data ke bentuk matriks berukuran (20.000, 8.746)  
4. Konversi ke matriks tf-idf menggunakan fungsi `todense()`
# Dataframe dari matriks tf-idf

<img width="1295" alt="Screenshot 2025-05-31 at 18 01 03" src="https://github.com/user-attachments/assets/753ccc3a-b540-43b2-8ed5-32abf2af601a" />

#### Perhitungan Kesamaan dengan Cosine Similarity

Cosine Similarity digunakan untuk mengukur tingkat kesamaan antar judul buku dengan menghitung sudut kosinus antara dua vektor dalam ruang multidimensi.

**Implementasi**:

- Menggunakan fungsi `cosine_similarity()` dari `sklearn`  
- Menghasilkan matriks kesamaan berukuran (20.000 x 20.000)  
- Semakin kecil sudut antar vektor, semakin tinggi tingkat kesamaan  
# Dataframe hasil perhitungan cosine similarity

<img width="1321" alt="Screenshot 2025-05-31 at 18 01 12" src="https://github.com/user-attachments/assets/7067a39a-eaac-401b-999b-99b9abf514cb" />

#### Fungsi Rekomendasi

Fungsi `book_recommendations()` dikembangkan dengan parameter:

- `book_title`: Judul buku referensi  
- `similarity_data`: DataFrame similarity yang telah dihitung  
- `items`: Fitur untuk mendefinisikan kemiripan (`'book_title'` dan `'book_author'`)  
- `k`: Jumlah rekomendasi (default: 5)  

**Contoh**:  
Untuk buku _"Entering the Silence: Becoming a Monk and a Writer"_ karya **Thomas Merton**, sistem menghasilkan 5 rekomendasi buku lain dengan penulis yang sama.
<img width="439" alt="Screenshot 2025-05-31 at 18 02 44" src="https://github.com/user-attachments/assets/bc1b8807-132f-4a5e-bfcd-1cd1d1098c5b" />

### Model Development dengan Collaborative Filtering
### Konsep Dasar

Collaborative Filtering memprediksi preferensi pengguna berdasarkan informasi dari pengguna lain yang memiliki pola preferensi serupa. Pendekatan ini menggunakan prinsip bahwa pengguna dengan preferensi serupa di masa lalu akan memiliki preferensi serupa di masa depan.

### Keunggulan Collaborative Filtering

- **Personalisasi mendalam**: Memanfaatkan preferensi dan perilaku pengguna secara langsung  
- **Tidak memerlukan pengetahuan item**: Fokus pada pola preferensi tanpa analisis konten mendalam  
- **Adaptif terhadap item baru**: Dapat menangani item tanpa riwayat penggunaan  

### Keterbatasan Collaborative Filtering

- **Masalah cold-start**: Kesulitan memberikan rekomendasi tanpa data historis  
- **Data sparse**: Performa menurun dengan data yang jarang (sebagian besar pengguna hanya memberi rating sedikit item)  
- **Skalabilitas**: Kompleksitas perhitungan meningkat dengan bertambahnya pengguna dan item  

### Implementasi Teknis

#### Arsitektur Model

Model menggunakan **Neural Collaborative Filtering** dengan teknik embedding:

- **Embedding Layer**: Untuk representasi pengguna dan buku  
- **Dot Product**: Operasi perkalian antara embedding pengguna dan buku  
- **Bias Addition**: Penambahan bias untuk setiap pengguna dan buku  
- **Sigmoid Activation**: Normalisasi skor kecocokan ke skala [0,1]  

#### Konfigurasi Training

- **Loss Function**: Binary Crossentropy  
- **Optimizer**: Adam (Adaptive Moment Estimation)  
- **Metrics**: Root Mean Squared Error (RMSE)  
- **Data Split**: 90% training, 10% validasi  

#### Hasil Evaluasi

Model mencapai konvergensi pada epoch ke-50 dengan performa:

- **RMSE Training**: 0.2939  
- **RMSE Validasi**: 0.3353  

Nilai RMSE ini menunjukkan performa yang baik untuk sistem rekomendasi.

### Proses Rekomendasi

Langkah-langkah:

1. Pemilihan user secara acak  
2. Identifikasi buku yang belum dibaca (`book_not_readed`)  
3. Prediksi menggunakan `model.predict()`  
4. Menghasilkan top-N recommendation  

**Contoh**:  
Untuk user ID **276798**, sistem menghasilkan 10 rekomendasi buku teratas beserta informasi penulis yang disesuaikan dengan pola rating historis pengguna.
<img width="492" alt="Screenshot 2025-05-31 at 18 04 15" src="https://github.com/user-attachments/assets/aa8e2b1b-f9d1-468a-88cb-1ada982112bf" />

## Evaluation

### Evaluasi Model Content-Based Filtering

Dalam proyek ini, evaluasi model content-based filtering dilakukan menggunakan tiga metrik utama, yaitu **Precision**, **Recall**, dan **F1-Score**. Ketiga metrik ini lazim digunakan untuk menilai performa model klasifikasi, termasuk sistem rekomendasi.  
- **Precision** mengukur proporsi item relevan yang benar-benar direkomendasikan oleh model dibandingkan dengan seluruh item yang direkomendasikan.  
- **Recall** mengukur proporsi item relevan yang berhasil direkomendasikan dari total keseluruhan item yang seharusnya direkomendasikan.  
- **F1-Score** merupakan harmoni rata-rata dari Precision dan Recall, yang menunjukkan keseimbangan antara keduanya.

Berikut rumus matematisnya:
<img width="416" alt="Screenshot 2025-05-31 at 18 08 41" src="https://github.com/user-attachments/assets/41aff0fd-f12c-49ae-bbe1-29e7f4ee8a83" />

Sebelum perhitungan metrik evaluasi dilakukan, diperlukan data acuan berupa **ground truth**, yang berfungsi sebagai label kebenaran untuk mengevaluasi prediksi model. Ground truth ini dibentuk berdasarkan nilai **cosine similarity** antar buku, di mana setiap pasangan buku diberi label 1 jika tingkat kemiripannya ≥ 0.5 (threshold) dan label 0 jika di bawahnya. Matriks ground truth kemudian dibentuk menggunakan fungsi `np.where()` dari NumPy dan diubah ke dalam bentuk DataFrame dengan judul buku sebagai indeks.

Setelah ground truth tersedia, dilakukan evaluasi menggunakan fungsi `precision_recall_fscore_support` dari pustaka [Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html). Karena keterbatasan memori, hanya 10.000 sampel dari matriks cosine similarity dan ground truth yang digunakan dalam evaluasi untuk efisiensi proses. Matriks dikonversi ke array satu dimensi agar perhitungan metrik lebih mudah dilakukan.

Fungsi evaluasi dijalankan dengan parameter `average='binary'` untuk klasifikasi biner, serta `zero_division=1` agar pembagian dengan nol dapat dihindari. Hasil evaluasi yang diperoleh:

- **Precision**: 1.0  
- **Recall**: 1.0  
- **F1-score**: 1.0

Nilai-nilai ini menunjukkan bahwa model mampu memberikan rekomendasi yang sangat tepat. Tidak ada prediksi yang salah (false positive) dan semua item relevan berhasil dikenali. F1-score yang sempurna mencerminkan keseimbangan yang ideal antara presisi dan jangkauan rekomendasi. Secara keseluruhan, model content-based filtering menunjukkan performa yang sangat tinggi.

---

### Evaluasi Model Collaborative Filtering

Untuk model collaborative filtering, metrik evaluasi yang digunakan adalah **Root Mean Squared Error (RMSE)**. Metrik ini umum digunakan dalam prediksi nilai kontinu dan sangat cocok untuk mengukur akurasi model dalam memprediksi rating pengguna terhadap item. Rumus RMSE sebagai berikut:
<img width="289" alt="Screenshot 2025-05-31 at 18 09 41" src="https://github.com/user-attachments/assets/05afa919-162d-47fc-9a4b-e6d37090cec2" />


Keterangan:
- **N**: Jumlah prediksi  
- **yᵢ**: Nilai sebenarnya  
- **ŷᵢ**: Nilai yang diprediksi oleh model

Berdasarkan proses pelatihan model yang telah dilakukan, performa model diukur menggunakan nilai RMSE pada data pelatihan dan validasi. Untuk memvisualisasikan proses ini, dibuat grafik evaluasi menggunakan pustaka **matplotlib**, seperti ditunjukkan pada :
**Visualisasi metrik evaluasi RMSE**
<img width="495" alt="Screenshot 2025-05-31 at 18 09 53" src="https://github.com/user-attachments/assets/0e261c6b-0225-4534-b49d-be2507a356e2" />

Dari grafik tersebut, terlihat bahwa model mencapai titik konvergensi sekitar **epoch ke-50**, dengan tren nilai error yang menurun secara konsisten. Hasil akhir dari proses pelatihan menunjukkan bahwa:
- **RMSE pada data latih**: 0.2939  
- **RMSE pada data validasi**: 0.3353

Nilai RMSE yang rendah ini menunjukkan bahwa model memiliki tingkat kesalahan prediksi yang kecil, yang berarti prediksi model terhadap preferensi pengguna cukup akurat. Semakin rendah nilai RMSE, semakin baik performa model dalam memberikan rekomendasi yang sesuai dengan preferensi pengguna. Oleh karena itu, dapat disimpulkan bahwa pendekatan collaborative filtering juga memberikan hasil yang baik dalam sistem rekomendasi ini.

### Keterkaitan Hasil Evaluasi dengan Business Understanding

Setelah dilakukan evaluasi terhadap kedua pendekatan model, yakni **Content-Based Filtering** dan **Collaborative Filtering**, bagian ini akan menjelaskan sejauh mana hasil tersebut telah menjawab problem statement, memenuhi tujuan (goals), dan mendukung solution statement yang telah ditetapkan dalam tahap *business understanding*.

#### Problem Statement 1:
> Bagaimana cara membangun sistem rekomendasi yang dapat memberikan saran buku secara personal kepada pengguna berdasarkan teknik content-based filtering?

Model **Content-Based Filtering** yang telah dikembangkan berhasil menghasilkan sistem rekomendasi buku yang sangat personal dengan tingkat presisi dan akurasi yang tinggi. Hal ini dibuktikan dari hasil evaluasi dengan metrik **Precision**, **Recall**, dan **F1-Score** yang semuanya mencapai angka **1.0**. Ini berarti:
- Setiap rekomendasi yang diberikan benar-benar relevan bagi pengguna.
- Model berhasil memahami preferensi pengguna berdasarkan fitur konten buku, seperti judul dan deskripsi.

Dengan demikian, sistem ini sangat efektif dalam menjawab problem statement pertama karena berhasil membangun sistem rekomendasi berbasis konten yang akurat dan relevan secara personal.

#### Problem Statement 2:
> Bagaimana memanfaatkan data rating untuk merekomendasikan buku lain yang berpotensi disukai pengguna, meskipun belum pernah mereka baca atau kunjungi sebelumnya?

Model **Collaborative Filtering** dirancang khusus untuk menjawab permasalahan ini. Melalui pendekatan pembelajaran dari pola rating antar pengguna, model ini mampu merekomendasikan buku baru yang berpotensi disukai meskipun belum pernah diakses sebelumnya oleh pengguna tertentu.  
Hasil evaluasi menggunakan metrik **RMSE** menunjukkan nilai error yang sangat rendah (RMSE Train: **0.2939**, RMSE Validation: **0.3353**), yang menandakan bahwa model memiliki akurasi prediksi yang baik.

Evaluasi menyeluruh terhadap kedua model membuktikan bahwa sistem rekomendasi yang dibangun telah menjawab setiap pernyataan masalah (*problem statement*), memenuhi tujuan proyek (*goals*), dan solusi yang dirancang terbukti memberikan dampak positif secara langsung terhadap pengembangan sistem rekomendasi buku yang efektif. Oleh karena itu, dari sisi **business understanding**, sistem ini sangat relevan untuk diimplementasikan sebagai bagian dari strategi peningkatan kualitas layanan dan pengalaman pengguna di platform buku digital.




