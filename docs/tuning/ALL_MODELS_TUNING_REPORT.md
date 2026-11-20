# Perbandingan Seluruh Model: Optuna vs Grid Search

- Feature set: `clinical_core_extreme`
- CV: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Budget: `24` evaluasi atau `120` model fit per model-metode
- Objective primer: `mean PR-AUC`
- Locked test tidak digunakan untuk pemilihan.

## Ranking Kandidat Tuning

1. `XGBoost + grid_search`: PR-AUC `0.438386`, Recall `0.753976`, F2 `0.605817`, unik `24/24`, waktu `17.89` detik.
2. `XGBoost + optuna_tpe`: PR-AUC `0.438202`, Recall `0.733656`, F2 `0.596187`, unik `11/24`, waktu `18.01` detik.
3. `Gradient Boosting + grid_search`: PR-AUC `0.435168`, Recall `0.735157`, F2 `0.598084`, unik `24/24`, waktu `189.75` detik.
4. `Gradient Boosting + optuna_tpe`: PR-AUC `0.435168`, Recall `0.735157`, F2 `0.598084`, unik `10/24`, waktu `177.66` detik.
5. `LightGBM + grid_search`: PR-AUC `0.433705`, Recall `0.710333`, F2 `0.584595`, unik `24/24`, waktu `32.55` detik.
6. `LightGBM + optuna_tpe`: PR-AUC `0.433705`, Recall `0.710333`, F2 `0.584595`, unik `12/24`, waktu `28.77` detik.
7. `SVM + grid_search`: PR-AUC `0.430443`, Recall `0.776550`, F2 `0.611609`, unik `24/24`, waktu `128.44` detik.
8. `SVM + optuna_tpe`: PR-AUC `0.427640`, Recall `0.778056`, F2 `0.612512`, unik `14/24`, waktu `133.02` detik.
9. `Logistic Regression + grid_search`: PR-AUC `0.417525`, Recall `0.710322`, F2 `0.588127`, unik `24/24`, waktu `15.32` detik.
10. `Logistic Regression + optuna_tpe`: PR-AUC `0.417525`, Recall `0.710322`, F2 `0.588127`, unik `11/24`, waktu `23.51` detik.
11. `KNN + grid_search`: PR-AUC `0.404933`, Recall `0.115889`, F2 `0.137278`, unik `24/24`, waktu `12.38` detik.
12. `KNN + optuna_tpe`: PR-AUC `0.404933`, Recall `0.115889`, F2 `0.137278`, unik `13/24`, waktu `13.73` detik.

## Interpretasi

- Perbandingan Optuna vs Grid harus dibaca di dalam model yang sama.
- Ranking lintas model menjawab kombinasi model+tuning terbaik pada development CV.
- Selisih kecil dibanding variasi fold tidak dianggap bukti superioritas universal.
- Kandidat peringkat pertama baru boleh dievaluasi satu kali pada locked test setelah keputusan dikunci.

## Jawaban atas contoh SVM Grid vs XGBoost Optuna

- Berdasarkan objective primer PR-AUC, `SVM + Grid Search` (`0.430443`) tidak mengalahkan `XGBoost + Optuna` (`0.438202`).
- Namun, `SVM + Grid Search` mempunyai recall lebih tinggi (`0.776550` vs `0.733656`) dan F2 lebih tinggi (`0.611609` vs `0.596187`).
- Karena metric pemilihan sudah ditetapkan sebagai PR-AUC, kandidat terbaik tetap `XGBoost + Grid Search` dengan PR-AUC `0.438386`.
- Jika tujuan klinis mengutamakan sensitivitas, ranking berdasarkan recall/F2 dapat dibahas sebagai analisis sekunder, tetapi objective tidak boleh diganti setelah melihat hasil.

## Kesimpulan Optuna vs Grid pada Enam Model

| Model | Grid PR-AUC | Optuna PR-AUC | Hasil |
|---|---:|---:|---|
| Logistic Regression | 0.417525 | 0.417525 | seri |
| SVM | 0.430443 | 0.427640 | Grid lebih tinggi |
| KNN | 0.404933 | 0.404933 | seri |
| Gradient Boosting | 0.435168 | 0.435168 | seri |
| XGBoost | 0.438386 | 0.438202 | Grid sedikit lebih tinggi |
| LightGBM | 0.433705 | 0.433705 | seri |

Grid mengevaluasi seluruh 24 konfigurasi unik per model. Optuna memperoleh 24 trial tetapi hanya 10–14 konfigurasi unik, tergantung model. Oleh sebab itu, hasil ini menunjukkan keuntungan coverage Grid pada domain diskret kecil; hasil ini bukan bukti bahwa Grid selalu lebih baik pada ruang pencarian kontinu atau budget yang lebih besar.

## Batasan

- Hanya satu seed (`42`) dan satu skema 5-fold CV yang digunakan.
- Search domain dibuat diskret dan berukuran 24 konfigurasi per model agar budget model-metode sama.
- Jenis hyperparameter berbeda antar-model karena arsitektur model berbeda.
- Tidak dilakukan uji pada locked test dalam eksperimen pemilihan ini.
- Belum dilakukan repeated nested CV atau uji statistik berpasangan.
