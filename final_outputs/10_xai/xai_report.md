# XAI Report

- Created at: `2026-09-06T08:57:11+07:00`
- Dataset: `D:\Ratih\PracticeFusion\outputs\app\prepare_data\final_dataset.csv`
- Feature set: `clinical_core_extreme` (Set B - Clinical Core + Extreme)
- Model: `XGBoost`
- Threshold klasifikasi: `0.35`
- Development/Test split: `70/30` dengan random_state `42`
- Random state: `42`
- Weighted model: `True`
- Optuna params dipakai bila tersedia: `True`

## Ringkasan Sesuai Proposal

- SHAP digunakan pada model final untuk membaca kontribusi fitur secara global dan lokal.
- Output yang disiapkan mengikuti arah proposal: `summary plot`, `dependence plot`, dan `force plot` untuk pasien individual.
- Interpretasi SHAP dibaca sebagai kontribusi model, bukan bukti hubungan sebab-akibat.

## Fitur Paling Berpengaruh

- `Age`: mean |SHAP| = `0.8103`
- `BMI_Max`: mean |SHAP| = `0.2950`
- `DiastolicBP_Mean`: mean |SHAP| = `0.2509`
- `BMI_Mean`: mean |SHAP| = `0.2488`
- `SystolicBP_Max`: mean |SHAP| = `0.2281`
- `Gender_F`: mean |SHAP| = `0.1385`
- `SystolicBP_Mean`: mean |SHAP| = `0.0963`
- `DiastolicBP_Max`: mean |SHAP| = `0.0650`
- `Gender_M`: mean |SHAP| = `0.0347`

## Contoh Interpretasi Lokal

Ringkasan kategori contoh (TP, FN, FP, dan TN) dipakai selama analisis internal. Artefak individual yang dapat dikaitkan ke baris pasien tidak disertakan dalam repository publik untuk menjaga privasi data.

## Artifacts

- Summary plot: `final_outputs/10_xai/plots/shap_summary_plot.png`
- Summary bar plot: `final_outputs/10_xai/plots/shap_summary_bar.png`
- Dependence plots: `final_outputs/10_xai/plots/shap_dependence_*.png`
- Top feature CSV: `final_outputs/10_xai/xai_top_features.csv`