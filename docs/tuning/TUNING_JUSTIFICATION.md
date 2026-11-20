# Justifikasi Hyperparameter XGBoost

Dokumen ini menjelaskan parameter yang benar-benar digunakan pada audit Optuna dan Grid Search. Definisi perilaku parameter mengikuti dokumentasi resmi XGBoost.[1]

## Prinsip penetapan rentang

Rentang bukan angka yang menjamin hasil terbaik. Rentang adalah hipotesis eksperimen yang harus:

1. mencakup baseline yang masuk akal;
2. cukup lebar untuk menangkap trade-off bias–variance;
3. tidak terlalu lebar hingga budget kecil menjadi sangat jarang;
4. konsisten dengan ukuran data dan biaya 5-fold CV;
5. diaudit melalui boundary check: bila nilai terbaik terus berada di batas, rentang perlu diperluas pada eksperimen berikutnya.

Dataset mempunyai 6,941 baris development dan 8 fitur konseptual pada feature set utama. Karena data relatif tabular dan dimensi fitur kecil, rentang pohon dangkal hingga sedang diprioritaskan.

## Parameter dan alasan

| Parameter | Fungsi | Rentang asli Optuna | Grid asli | Pengaruh kompleksitas dan risiko | Justifikasi batas |
|---|---|---|---|---|---|
| `learning_rate` / eta | Mengecilkan kontribusi setiap boosting step | 0.01–0.20, log | 0.03, 0.08 | Terlalu tinggi: update agresif, model mudah overfit/tidak stabil. Terlalu rendah: underfit bila jumlah tree tidak cukup dan biaya naik karena memerlukan lebih banyak tree. | 0.01 memberi langkah konservatif; 0.20 membatasi update agar tidak terlalu agresif. Sampling log tepat karena perbedaan relatif lebih bermakna daripada selisih linear. |
| `max_depth` | Kedalaman maksimum setiap tree | 3–8 | 3, 4 | Lebih tinggi meningkatkan interaksi dan kapasitas, tetapi meningkatkan overfitting serta biaya. Terlalu rendah dapat gagal menangkap interaksi nonlinear. | Batas 3 mewakili tree dangkal; 8 adalah kompleksitas sedang-tinggi untuk data ini. Best asli pada 3 menunjukkan tree dangkal memadai, bukan bukti bahwa 8 salah. |
| `n_estimators` | Jumlah boosting rounds/trees pada wrapper sklearn | 100–400, step 25 | 100, 200 | Terlalu sedikit menyebabkan underfit; terlalu banyak dapat overfit dan menambah waktu, khususnya bila learning rate tinggi. | 100 adalah baseline repository; 400 memberi ruang bagi learning rate kecil. Best asli 400 berada di upper boundary, sehingga eksperimen lanjutan dapat menguji >400 bersama early stopping, bukan otomatis menaikkan tanpa kontrol. |
| `subsample` | Proporsi baris training yang dipakai per boosting iteration | 0.6–1.0 | 0.8 tetap | Nilai <1 menambah stochastic regularization dan dapat mengurangi overfit; terlalu rendah meningkatkan variance/kehilangan sinyal. Nilai 1 memakai seluruh baris. | 0.6 dipilih sebagai lower bound moderat agar mayoritas data tetap dipakai; 1.0 mencakup boosting deterministik tanpa row subsampling. |
| `colsample_bytree` | Proporsi fitur yang dipakai ketika membangun setiap tree | 0.6–1.0 | 0.8 tetap | Nilai rendah memberi regularisasi/diversitas tetapi berisiko membuang fitur penting; 1 memakai semua fitur. | Karena hanya ada sekitar 8 fitur, lower bound tidak dibuat sangat kecil; 0.6 tetap memberi mayoritas fitur pada setiap tree. |
| `min_child_weight` | Minimum total Hessian/weight yang diperlukan pada child | 1–10 | 1, 5 | Nilai tinggi membuat split lebih konservatif dan dapat underfit; nilai rendah memungkinkan leaf kecil dan dapat overfit noise. | 1 adalah default umum/baseline longgar; 10 menguji regularisasi struktur yang lebih kuat. Best asli 8 mendukung perlunya nilai konservatif dalam ruang Optuna. |
| `reg_lambda` | Regularisasi L2 pada bobot leaf | 0.001–10, log | 1.0 tetap | Nilai tinggi mengecilkan bobot dan mengurangi variance, tetapi terlalu tinggi dapat underfit. Nilai mendekati nol memberi regularisasi lemah. | Rentang empat orde magnitudo menguji regularisasi dari sangat lemah hingga kuat; log sampling mencegah dominasi angka besar. Grid awal tidak menguji efek ini. |
| `reg_alpha` | Regularisasi L1 pada bobot leaf | 0.0001–5, log | 0.0 tetap | Nilai tinggi mendorong bobot/split lebih sparse dan dapat mengurangi overfit; terlalu tinggi dapat meniadakan sinyal. | Lower bound mendekati tanpa L1 namun tetap kompatibel dengan log sampling; upper bound 5 menguji regularisasi kuat. Grid memakai 0 agar baseline tanpa L1 tetap tersedia. |
| `scale_pos_weight` | Menyeimbangkan gradien kelas positif pada data imbalance | dihitung sebagai negatif/positif | sama | Nilai terlalu besar dapat meningkatkan recall tetapi menurunkan precision dan kalibrasi; nilai 1 mengabaikan imbalance. | Bukan objek tuning pada audit; dihitung dari training fold agar tidak memakai informasi validation/test dan agar perlakuan kedua metode sama. |
| `tree_method` | Algoritme konstruksi tree | `hist` tetap | sama | Mempengaruhi efisiensi, bukan search fairness bila tetap sama. | Dikontrol agar runtime dan implementasi estimator konsisten. |
| `random_state` | Seed stochastic components | 42 | 42 | Bukan kontrol kompleksitas, tetapi penting untuk reproduksibilitas. Satu seed belum mengukur sensitivitas seed. | Disamakan pada split, CV, model, dan sampler Optuna. |

XGBoost menjelaskan bahwa `eta` mengecilkan feature weights setelah setiap boosting step, `max_depth` yang besar membuat model lebih kompleks dan cenderung overfit, `min_child_weight` yang besar membuat algoritme lebih konservatif, `subsample` dan `colsample_bytree` mengatur subsampling, serta `reg_alpha`/`reg_lambda` adalah regularisasi L1/L2.[1]

## Hubungan antarkomponen

Parameter tidak boleh dibenarkan secara terpisah saja:

- `learning_rate` kecil biasanya membutuhkan `n_estimators` lebih besar.
- `max_depth` besar dapat dikompensasi oleh `min_child_weight`, `subsample`, `colsample_bytree`, dan regularisasi yang lebih kuat.
- Grid Cartesian membesar secara eksponensial. Delapan parameter dengan banyak nilai cepat menjadi tidak feasible, sehingga coarse-to-fine grid atau random/Bayesian search biasanya lebih praktis.
- Optuna dapat menjelajah kontinu, tetapi hasil tetap dibatasi oleh domain, trial budget, sampler, dan seed.

## Audit terhadap batas yang dipilih

- Best Optuna asli memakai `n_estimators=400`, tepat di upper boundary. Ini sinyal untuk sensitivity analysis >400 atau early stopping; bukan alasan untuk mengubah hasil utama secara retroaktif.
- Best memakai `max_depth=3`, lower boundary, sehingga pohon dangkal tampak memadai dalam eksperimen ini.
- Best `subsample≈0.604` dekat lower boundary dan `reg_lambda≈9.94` dekat upper boundary. Ini menunjukkan regularisasi kuat membantu, sekaligus menunjukkan Grid awal yang menetapkan 0.8 dan 1.0 tidak mendapat kesempatan setara.
- Karena beberapa optimum berada dekat batas, klaim “rentang sudah optimal” tidak boleh dibuat. Klaim yang tepat adalah “rentang merupakan domain eksperimen yang terdokumentasi dan diuji, dengan boundary limitation yang dilaporkan.”

## Prosedur yang disarankan untuk penelitian lanjutan

1. Tetapkan metric primer sebelum tuning.
2. Pisahkan locked test sebelum tuning.
3. Tentukan domain dari dokumentasi, baseline, ukuran data, dan pilot run.
4. Gunakan evaluator CV yang identik.
5. Laporkan domain, distribusi, seed, trial, fit, dan wall-clock.
6. Periksa apakah optimum berada di boundary.
7. Lakukan repeated/nested CV bila ingin membuat klaim metode tuning, bukan sekadar memilih konfigurasi model.
8. Jangan memilih domain setelah melihat locked-test performance.

## Sources

[1] https://xgboost.readthedocs.io/en/stable/parameter.html
