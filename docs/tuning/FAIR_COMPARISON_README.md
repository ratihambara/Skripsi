# Catatan Eksperimen Terkontrol

`final_outputs/11_tuning_comparison/FAIR_COMPARISON_RESULTS.csv` berisi satu baris per evaluasi kandidat:

- 64 baris `grid_search_controlled`;
- 64 baris `optuna_tpe_controlled`;
- 5-fold CV, sehingga 320 model fit per metode;
- seed 42;
- locked test tidak digunakan untuk pemilihan;
- durasi total Grid 46.35 detik dan Optuna 47.80 detik pada run ini.

Eksperimen dapat direproduksi dengan:

`PYTHONPATH=. /path/to/python scripts/run_optuna_grid_fair_comparison.py`

File ini hanya penjelas tambahan dan bukan pengganti laporan audit utama.
