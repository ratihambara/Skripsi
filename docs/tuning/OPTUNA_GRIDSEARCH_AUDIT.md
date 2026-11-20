# Audit Perbandingan Optuna vs Grid Search

Tanggal audit: 17 November 2026

## Ringkasan eksekutif

Implementasi asli belum memberi Optuna dan Grid Search kesempatan pencarian yang setara. Keduanya memakai dataset, locked split, preprocessing, 5-fold StratifiedKFold, random seed, model, class weighting, dan objective PR-AUC yang sama. Namun, Optuna memperoleh 30 evaluasi pada ruang delapan hyperparameter yang luas, sedangkan Grid Search hanya mengevaluasi 16 kombinasi dan empat dari delapan parameternya dibuat konstan. Karena itu, selisih PR-AUC asli 0.4423 vs 0.4416 tidak cukup untuk menyimpulkan bahwa Optuna secara umum lebih unggul.

Eksperimen tambahan terkontrol dilakukan tanpa mengubah hasil asli. Pada domain diskret yang sama, budget yang sama (64 evaluasi), evaluator yang sama, dan 320 model fit per metode, Grid Search memperoleh PR-AUC terbaik 0.441084 dan Optuna-TPE 0.438238. Hasil ini menunjukkan bahwa pemenang bergantung pada domain dan budget; bukan bukti bahwa Grid Search selalu lebih baik. Optuna hanya mengevaluasi 17 konfigurasi unik dari 64 trial karena TPE berulang kali memilih konfigurasi yang sama pada domain kategorikal kecil.

## Sumber implementasi yang diaudit

- `practicefusion/pipelines/train.py:315-326`: development/test split 70/30, stratified, random_state 42.
- `practicefusion/pipelines/train.py:384-412`: preprocessing di dalam pipeline.
- `practicefusion/pipelines/train.py:750-767`: baseline XGBoost dan seed model.
- `practicefusion/pipelines/train.py:964-973`: StratifiedKFold dengan shuffle dan seed.
- `practicefusion/pipelines/train.py:3252-3300`: ruang Optuna.
- `practicefusion/pipelines/train.py:3303-3325`: ruang Grid/Random Search XGBoost.
- `practicefusion/pipelines/train.py:3436-3701`: GridSearchCV/RandomizedSearchCV.
- `practicefusion/pipelines/train.py:3704-3884`: Optuna-TPE.

## Konfigurasi eksperimen asli

| Aspek | Optuna asli | Grid Search asli | Status fairness |
|---|---:|---:|---|
| Dataset | `final_dataset.csv` | sama | setara |
| Development/test | 70/30 stratified | sama | setara |
| Development rows | 6,941 | 6,941 | setara |
| Locked test rows | 2,975 | 2,975 | setara; tidak dipakai memilih hyperparameter |
| Feature set | `clinical_core_extreme` (8 fitur konseptual) | sama | setara |
| Model | XGBoost | XGBoost | setara |
| Class balancing | `scale_pos_weight` dari rasio kelas development/fold | sama | setara |
| CV | StratifiedKFold(5, shuffle=True, random_state=42) | sama | setara |
| Objective pemilihan | mean PR-AUC | mean PR-AUC (`average_precision`, refit) | setara |
| Sampler/search | TPE | enumerasi Cartesian | berbeda secara alami |
| Seed | 42 untuk split, CV, model, dan TPESampler | 42 untuk split, CV, model | setara |
| Konfigurasi dievaluasi | 30 trial | 16 kombinasi | tidak setara |
| Fit selama pencarian | 30 × 5 = 150 | 16 × 5 = 80, ditambah 1 refit GridSearchCV | tidak setara |
| Fit evaluasi pascapencarian dalam fungsi | baseline 5 + best 5 | baseline 5 + Optuna existing 5 + Grid best 5 + Random best 5 | bukan bagian budget pencarian; pelaporan berbeda |
| Durasi | tidak direkam | tidak direkam | tidak dapat dibandingkan dari artefak asli |

GridSearchCV melakukan exhaustive search atas nilai yang diberikan dan, ketika `refit` diaktifkan, memasang ulang estimator terbaik pada seluruh data yang diberikan.[2] StratifiedKFold mempertahankan proporsi kelas pada setiap fold; implementasi repository juga mengaktifkan shuffle dan seed 42.[4]

## Hyperparameter asli

Definisi dan arah regularisasi parameter XGBoost pada bagian ini mengikuti dokumentasi resmi XGBoost.[1]

| Hyperparameter | Optuna | Grid Search | Temuan |
|---|---|---|---|
| `n_estimators` | integer 100–400, step 25 | 100, 200 | Optuna jauh lebih luas; best Optuna 400 tidak mungkin dicapai Grid |
| `max_depth` | integer 3–8 | 3, 4 | Optuna lebih luas |
| `learning_rate` | 0.01–0.20, log sampling | 0.03, 0.08 | overlap ada, tetapi resolusi/distribusi berbeda |
| `subsample` | 0.6–1.0 kontinu | 0.8 saja | Grid tidak benar-benar mengoptimasi parameter ini |
| `colsample_bytree` | 0.6–1.0 kontinu | 0.8 saja | Grid tidak benar-benar mengoptimasi parameter ini |
| `min_child_weight` | integer 1–10 | 1, 5 | Grid lebih sempit |
| `reg_lambda` | 0.001–10, log sampling | 1.0 saja | Grid tidak mengoptimasi regularisasi L2 |
| `reg_alpha` | 0.0001–5, log sampling | 0.0 saja | Grid tidak mengoptimasi regularisasi L1 |

Jumlah kombinasi Grid asli = 2 × 2 × 2 × 2 × 1 × 1 × 1 × 1 = 16.

Best Optuna asli (`n_estimators=400`, `subsample≈0.604`, `colsample_bytree≈0.675`, `reg_lambda≈9.94`, `reg_alpha≈0.198`) berada di luar nilai yang dapat dipilih Grid asli pada lima parameter. Ini adalah bukti langsung bahwa hasil dapat dipengaruhi oleh perbedaan search space.

## Hasil asli

| Metode | Mean PR-AUC | SD PR-AUC | Mean recall | Mean F2 | Evaluasi pencarian |
|---|---:|---:|---:|---:|---:|
| Optuna existing | 0.442337 | 0.025790 | 0.729891 | 0.596279 | 30 |
| Grid Search | 0.441603 | 0.022513 | 0.742687 | 0.599331 | 16 |
| Baseline | 0.439340 | 0.031244 | 0.719364 | 0.588566 | 1 konfigurasi |

Selisih Optuna–Grid hanya 0.000734 PR-AUC. Grid justru memiliki recall dan F2 sedikit lebih tinggi. Selisih ini kecil dibanding variasi antar-fold dan belum disertai repeated CV atau uji statistik berpasangan. Oleh karena itu, istilah “unggul” tidak didukung secara kuat.

## Evaluasi fairness

1. **Apakah parameter yang dioptimasi sama?** Nama delapan parameter dicantumkan pada kedua metode, tetapi secara substantif tidak sama: empat parameter Grid hanya memiliki satu nilai dan tidak dioptimasi.
2. **Apakah ruang pencariannya sebanding?** Tidak. Optuna lebih luas dan kontinu/log-kontinu pada beberapa dimensi; Grid sempit dan sebagian besar konstan.
3. **Apakah objective metric sama?** Ya. Keduanya memilih berdasarkan mean PR-AUC/average precision pada development CV.[5]
4. **Apakah jumlah trial dan model fit sebanding?** Tidak. Optuna 30 konfigurasi/150 fit; Grid 16 konfigurasi/80 CV fit (+1 refit).
5. **Mungkinkah perbedaan hasil disebabkan search space?** Ya, sangat mungkin. Best Optuna mengandung nilai yang tidak tersedia bagi Grid.
6. **Apakah audit asli cukup untuk klaim Optuna lebih unggul?** Tidak. Hasil asli hanya menunjukkan Optuna memperoleh PR-AUC terbaik secara numerik pada konfigurasi eksperimen tersebut.

## Kesetaraan konfigurasi vs kesetaraan waktu

- **Kesetaraan jumlah konfigurasi** berarti kedua metode memperoleh jumlah evaluasi kandidat yang sama. Ini mengendalikan budget evaluasi, tetapi biaya setiap kandidat dapat berbeda; misalnya 400 trees lebih mahal daripada 100 trees.
- **Kesetaraan jumlah model fit** juga harus memperhitungkan fold CV: 64 konfigurasi × 5 fold = 320 fit.
- **Kesetaraan waktu** berarti kedua metode dihentikan setelah wall-clock budget yang sama. Ini merefleksikan efisiensi praktis, tetapi jumlah kandidat dapat berbeda karena konfigurasi mahal memperlambat salah satu metode.
- Keduanya menjawab pertanyaan berbeda. Audit metodologis ideal melaporkan hasil berbasis jumlah evaluasi dan hasil berbasis waktu, bukan menganggap keduanya identik.

## Eksperimen terkontrol tambahan

Eksperimen tambahan menggunakan:

- dataset, development split, preprocessing, class weighting, dan 5 fold yang sama;
- objective mean PR-AUC yang sama;
- seed 42;
- domain umum 64 konfigurasi: `n_estimators` {100,400}, `max_depth` {3,6}, `learning_rate` {0.02,0.08}, `subsample` {0.65,1.0}, `colsample_bytree` {0.65,1.0}, `min_child_weight` {1,8};
- `reg_lambda=1.0` dan `reg_alpha=0.0` dikontrol tetap untuk kedua metode;
- Grid mengevaluasi semua 64 kombinasi; Optuna-TPE memperoleh 64 trial pada domain kategorikal yang sama;
- 320 fit per metode; tidak ada refit tambahan dan locked test tidak digunakan.

| Metode | Best PR-AUC | SD | Recall | F2 | ROC-AUC | Evaluasi | Konfigurasi unik | Fit | Durasi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Grid controlled | 0.441084 | 0.026921 | 0.726895 | 0.593625 | 0.774625 | 64 | 64 | 320 | 46.35 s |
| Optuna-TPE controlled | 0.438238 | 0.024778 | 0.723124 | 0.589251 | 0.774672 | 64 | 17 | 320 | 47.80 s |

TPESampler membangun model probabilistik dari trial yang dianggap baik dan buruk untuk memilih kandidat berikutnya; ia bukan mekanisme enumerasi tanpa pengulangan.[6] Pada domain diskret kecil ini, eksploitasi menyebabkan banyak pengulangan. Ini bukan kegagalan Optuna secara umum, tetapi batasan desain eksperimen ini dan bukti bahwa “64 trial” tidak selalu berarti “64 konfigurasi unik”.

### Batasan eksperimen tambahan

- Domain terkontrol sengaja diperkecil agar exhaustive Grid feasible; hasil tidak menggeneralisasi ke ruang kontinu asli.
- `reg_alpha` dan `reg_lambda` dikontrol, bukan dioptimasi.
- Hanya satu seed dan satu 5-fold split; belum repeated nested CV.
- Durasi adalah satu run pada mesin ini dan dapat berubah menurut hardware/load.
- Tidak ada klaim signifikansi statistik atau superioritas universal.

## Jawaban eksplisit

### Apakah saya boleh mengklaim Optuna lebih baik daripada Grid Search?

Tidak sebagai klaim umum. Klaim yang aman: **“Pada eksperimen asli, Optuna menghasilkan mean PR-AUC tertinggi secara numerik (0.4423 vs 0.4416), tetapi search space dan budget tidak setara sehingga hasil tersebut tidak membuktikan superioritas metode.”**

### Jika tidak, klaim apa yang paling tepat?

**“Kinerja kedua metode sangat berdekatan dan sensitif terhadap desain search space. Optuna terbaik pada ruang asli yang lebih luas; pada eksperimen diskret dengan domain dan jumlah fit yang disamakan, exhaustive Grid memperoleh skor terbaik. Karena itu, penelitian ini memilih konfigurasi berdasarkan hasil eksperimen spesifik, bukan menyimpulkan satu algoritme tuning selalu lebih baik.”**

### Apa jawaban terbaik saat dosen mempertanyakan penentuan parameter?

“Rentang ditetapkan dari fungsi tiap parameter, dokumentasi resmi XGBoost, baseline repository, dan keterbatasan komputasi. Audit menunjukkan Grid awal terlalu sempit. Karena itu saya menambahkan eksperimen terkontrol dengan domain, metric, fold, seed, preprocessing, dan jumlah fit yang sama. Hasilnya menunjukkan kesimpulan bergantung pada ruang pencarian, sehingga saya tidak mengklaim superioritas universal Optuna.”

### Bukti eksperimen yang dapat ditunjukkan saat sidang

1. `final_outputs/11_tuning_comparison/SEARCH_SPACE_COMPARISON.csv` untuk membuktikan ketidakseimbangan ruang asli.
2. `outputs/app/train/optuna_trials.csv` dan `outputs/app/train/sklearn_search_tuning_results.csv` sebagai jejak kandidat asli.
3. `outputs/app/train/sklearn_search_tuning_comparison.csv` untuk hasil CV asli.
4. `final_outputs/11_tuning_comparison/FAIR_COMPARISON_RESULTS.csv` untuk 128 baris evaluasi terkontrol, budget, fit, seed, domain, dan durasi.
5. `scripts/run_optuna_grid_fair_comparison.py` untuk reproduksibilitas eksperimen tambahan.

## Sources

[1] https://xgboost.readthedocs.io/en/stable/parameter.html
[2] https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
[4] https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
[5] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
[6] https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.TPESampler.html
