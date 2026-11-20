# Tanya Jawab Sidang: Optuna vs Grid Search

## 1. Bagaimana learning rate, max depth, dan parameter lain ditentukan?

Jawaban:

“Rentang ditentukan dari fungsi parameter pada XGBoost, baseline model, ukuran data, trade-off bias–variance, dan budget 5-fold CV. `learning_rate` diuji pada skala log karena efek relatifnya, `max_depth` dibatasi 3–8 untuk tree dangkal sampai sedang, dan `n_estimators` 100–400 untuk mengimbangi learning rate kecil. Parameter regularisasi, row sampling, dan column sampling juga dimasukkan karena mengendalikan overfitting. Dasarnya dirinci di `TUNING_JUSTIFICATION.md` dan dokumentasi resmi XGBoost.”[1]

## 2. Apakah Optuna dan Grid Search awal mempunyai kesempatan pencarian yang sama?

Jawaban:

“Belum. Metric, data, fold, preprocessing, dan seed sama, tetapi search space serta budget berbeda. Optuna menjalankan 30 trial pada delapan parameter dengan domain luas. Grid hanya 16 kombinasi; `subsample`, `colsample_bytree`, `reg_lambda`, dan `reg_alpha` masing-masing hanya mempunyai satu nilai. Jadi Grid awal tidak benar-benar mengoptimasi empat parameter tersebut.”

## 3. Bukankah nama hyperparameter pada keduanya sama?

Jawaban:

“Nama parameternya sama, tetapi kesempatan pencariannya tidak sama. Parameter dengan satu nilai adalah parameter yang dikontrol tetap, bukan parameter yang dioptimasi. Fairness harus dilihat dari domain dan distribusi, bukan hanya daftar nama.”

## 4. Mengapa Grid Search hanya 16 kombinasi?

Jawaban:

“Empat parameter memiliki dua nilai dan empat parameter hanya satu nilai, sehingga jumlahnya 2×2×2×2×1×1×1×1 = 16. GridSearchCV mengevaluasi semua kombinasi yang diberikan.[2]”

## 5. Berapa total model fit pada eksperimen awal?

Jawaban:

“Optuna: 30 trial × 5 fold = 150 fit selama pencarian. Grid: 16 kombinasi × 5 fold = 80 CV fit, kemudian GridSearchCV melakukan satu refit estimator terbaik karena `refit='pr_auc'`, sehingga 81 fit dalam objek search. Fit tambahan untuk baseline dan tabel pelaporan saya pisahkan dari budget pencarian.”

## 6. Apakah objective keduanya sama?

Jawaban:

“Ya. Keduanya memaksimalkan mean PR-AUC atau average precision pada development CV. Recall dan F2 hanya dicatat sebagai metric pendukung. Locked test tidak digunakan untuk memilih hyperparameter.”[5]

## 7. Mengapa menggunakan PR-AUC?

Jawaban:

“Target tidak seimbang, dengan positive rate sekitar 19.15% pada development set. PR-AUC berfokus pada hubungan precision–recall untuk kelas positif. Metric ini ditetapkan konsisten sebelum membandingkan metode.”

## 8. Bagaimana jika Grid diberi konfigurasi lain dan performanya lebih baik?

Jawaban:

“Itu mungkin, dan audit saya membuktikannya. Best Optuna awal memakai beberapa nilai yang tidak tersedia pada Grid awal. Dalam eksperimen tambahan dengan domain diskret dan budget yang disamakan, Grid justru memperoleh PR-AUC terbaik 0.441084 dibanding Optuna-TPE 0.438238. Jadi hasil bergantung pada search space dan budget.”

## 9. Apakah eksperimen tambahan sepenuhnya adil?

Jawaban:

“Lebih terkontrol, tetapi saya tidak menyebutnya adil secara universal. Dataset, split, preprocessing, metric, fold, seed, domain, jumlah evaluasi, dan jumlah fit disamakan. Namun domain sengaja diperkecil agar exhaustive Grid feasible, dua regularization parameter dibuat tetap, hanya satu seed digunakan, dan Optuna mengulang beberapa konfigurasi.”

## 10. Mengapa Optuna hanya mendapatkan 17 konfigurasi unik dari 64 trial?

Jawaban:

“TPESampler memilih kandidat secara adaptif berdasarkan model probabilistik trial baik dan buruk.[6] Pada domain kategorikal kecil, ia dapat mengulang konfigurasi yang dianggap menjanjikan. Grid exhaustive mengunjungi seluruh 64 konfigurasi satu kali. Ini menunjukkan bahwa jumlah trial sama tidak otomatis berarti cakupan konfigurasi unik sama.”

## 11. Bukankah itu membuat eksperimen terkontrol menguntungkan Grid?

Jawaban:

“Pada domain kecil yang seluruhnya dapat dienumerasi, exhaustive Grid memang mempunyai keuntungan coverage. Karena itu hasil tambahan tidak dipakai untuk menyatakan Grid selalu lebih baik. Eksperimen tersebut hanya menunjukkan bahwa klaim Optuna selalu lebih unggul juga tidak dapat dipertahankan.”

## 12. Apa perbedaan fair berdasarkan jumlah evaluasi dan fair berdasarkan waktu?

Jawaban:

“Budget evaluasi sama berarti jumlah kandidat atau model fit sama. Budget waktu sama berarti wall-clock sama. Keduanya tidak identik karena konfigurasi 400 trees lebih mahal daripada 100 trees. Penelitian ini melaporkan jumlah evaluasi, jumlah fit, dan durasi agar pembaca dapat melihat keduanya.”

## 13. Apakah selisih hasil awal besar?

Jawaban:

“Tidak. Optuna 0.442337 dan Grid 0.441603, selisih 0.000734 PR-AUC. Grid bahkan mempunyai recall dan F2 sedikit lebih tinggi. Selisih PR-AUC tersebut kecil dibanding variasi antar-fold, sehingga tidak layak disebut kemenangan kuat.”

## 14. Jadi apakah Optuna lebih baik?

Jawaban pendek:

“Tidak dapat disimpulkan secara umum. Optuna terbaik secara numerik pada eksperimen asli, tetapi memperoleh ruang dan budget yang lebih luas. Pada eksperimen terkontrol, Grid terbaik. Kesimpulan yang valid adalah performa sensitif terhadap desain search space.”

## 15. Klaim apa yang aman ditulis di skripsi?

“Pada konfigurasi eksperimen asli, Optuna menghasilkan mean PR-AUC tertinggi secara numerik. Namun, karena ruang pencarian dan jumlah evaluasi berbeda, hasil ini diperlakukan sebagai hasil konfigurasi spesifik, bukan bukti superioritas universal Optuna atas Grid Search.”

## 16. Bukti apa yang ditunjukkan saat sidang?

Urutan yang disarankan:

1. Buka `final_outputs/11_tuning_comparison/SEARCH_SPACE_COMPARISON.csv`; tunjukkan empat parameter Grid yang hanya memiliki satu nilai.
2. Tunjukkan perhitungan 30×5=150 vs 16×5=80 fit.
3. Buka `outputs/app/train/sklearn_search_tuning_comparison.csv`; tunjukkan selisih PR-AUC hanya 0.000734 dan Grid unggul pada recall/F2.
4. Buka `final_outputs/11_tuning_comparison/FAIR_COMPARISON_RESULTS.csv`; filter per `method`, urutkan `mean_pr_auc`, dan tunjukkan 64 evaluasi/320 fit per metode.
5. Tunjukkan kolom `is_unique_configuration`: Grid 64 unik, Optuna 17 unik.
6. Jika diminta reproduksi, tunjukkan `scripts/run_optuna_grid_fair_comparison.py`.

## 17. Apakah hasil asli diubah agar sesuai dengan audit?

Jawaban:

“Tidak. Artefak asli tetap dipertahankan. Eksperimen tambahan disimpan terpisah di `FAIR_COMPARISON_RESULTS.csv`, sehingga audit tidak mengganti atau memaksakan pemenang.”

## Kalimat penutup yang disarankan

“Tujuan tuning dalam penelitian ini adalah mencari konfigurasi yang baik di bawah domain dan budget yang didefinisikan, bukan membuktikan satu optimizer selalu superior. Setelah fairness diaudit, saya membatasi klaim pada eksperimen yang benar-benar dilakukan.”

## Sources

[1] https://xgboost.readthedocs.io/en/stable/parameter.html
[2] https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
[5] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
[6] https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.TPESampler.html
