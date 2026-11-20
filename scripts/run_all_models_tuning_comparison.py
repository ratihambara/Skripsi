from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

from practicefusion.pipelines.train import (
    build_feature_sets,
    collect_cv_results,
    make_development_test_split,
    summarize_cv_results,
)
from practicefusion.utils.io import load_csv


DEFAULT_DATASET = Path("outputs/app/prepare_data/final_dataset.csv")
DEFAULT_OUTPUT_DIR = Path("final_outputs/11_tuning_comparison")
DEFAULT_DETAILS = DEFAULT_OUTPUT_DIR / "ALL_MODELS_TUNING_DETAILS.csv"
DEFAULT_SUMMARY = DEFAULT_OUTPUT_DIR / "ALL_MODELS_TUNING_COMPARISON.csv"
DEFAULT_REPORT = Path("docs/tuning/ALL_MODELS_TUNING_REPORT.md")
FEATURE_SET = "clinical_core_extreme"
RANDOM_STATE = 42
N_SPLITS = 5
BUDGET_PER_METHOD = 24

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "Logistic Regression": {
        "kind": "logistic",
        "tunable_parameters": ["C"],
        "C": np.logspace(-3, np.log10(20.0), BUDGET_PER_METHOD).tolist(),
    },
    "SVM": {
        "kind": "svm",
        "tunable_parameters": ["C", "kernel", "gamma"],
        "C": [0.001, 0.005, 0.02, 0.1, 0.5, 2.0, 8.0, 20.0],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],
    },
    "KNN": {
        "kind": "cartesian",
        "tunable_parameters": ["n_neighbors", "weights", "p"],
        "domain": {
            "n_neighbors": [5, 7, 11, 15, 21, 31],
            "weights": ["uniform", "distance"],
            "p": [1, 2],
        },
    },
    "Gradient Boosting": {
        "kind": "cartesian",
        "tunable_parameters": ["n_estimators", "learning_rate", "min_samples_leaf", "subsample"],
        "domain": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.03, 0.10],
            "min_samples_leaf": [1, 10],
            "subsample": [0.75, 1.0],
        },
    },
    "XGBoost": {
        "kind": "cartesian",
        "tunable_parameters": ["n_estimators", "max_depth", "learning_rate", "min_child_weight"],
        "domain": {
            "n_estimators": [100, 250, 400],
            "max_depth": [3, 6],
            "learning_rate": [0.02, 0.08],
            "min_child_weight": [1, 8],
        },
    },
    "LightGBM": {
        "kind": "cartesian",
        "tunable_parameters": ["n_estimators", "learning_rate", "num_leaves", "min_child_samples"],
        "domain": {
            "n_estimators": [100, 250, 400],
            "learning_rate": [0.02, 0.08],
            "num_leaves": [15, 63],
            "min_child_samples": [10, 40],
        },
    },
}


def build_grid_configurations(spec: dict[str, Any]) -> list[dict[str, Any]]:
    kind = str(spec["kind"])
    if kind == "logistic":
        return [{"C": float(value)} for value in spec["C"]]
    if kind == "svm":
        configurations: list[dict[str, Any]] = []
        for c_value in spec["C"]:
            configurations.append({"C": c_value, "kernel": "linear", "gamma": "scale"})
            for gamma in spec["gamma"]:
                configurations.append({"C": c_value, "kernel": "rbf", "gamma": gamma})
        return configurations
    domain = spec["domain"]
    keys = list(domain)
    return [dict(zip(keys, values)) for values in product(*(domain[key] for key in keys))]


def sample_optuna_configuration(trial: optuna.Trial, spec: dict[str, Any]) -> dict[str, Any]:
    kind = str(spec["kind"])
    if kind == "logistic":
        return {"C": trial.suggest_categorical("C", spec["C"])}
    if kind == "svm":
        kernel = trial.suggest_categorical("kernel", spec["kernel"])
        params: dict[str, Any] = {
            "C": trial.suggest_categorical("C", spec["C"]),
            "kernel": kernel,
            "gamma": "scale",
        }
        if kernel == "rbf":
            params["gamma"] = trial.suggest_categorical("gamma", spec["gamma"])
        return params
    return {name: trial.suggest_categorical(name, values) for name, values in spec["domain"].items()}


def evaluate_configuration(
    df: pd.DataFrame,
    feature_cols: list[str],
    development_idx: pd.Index,
    model_name: str,
    params: dict[str, Any] | None = None,
) -> dict[str, float]:
    overrides = {model_name: params} if params else None
    summary = summarize_cv_results(
        collect_cv_results(
            df=df,
            feature_sets={FEATURE_SET: feature_cols},
            model_names=[model_name],
            development_idx=development_idx,
            n_splits=N_SPLITS,
            random_state=RANDOM_STATE,
            use_class_balancing=True,
            use_smote=False,
            overrides=overrides,
        )
    ).iloc[0]
    metric_names = [
        "mean_accuracy",
        "mean_precision",
        "mean_recall",
        "mean_specificity",
        "mean_f1",
        "mean_f2",
        "mean_roc_auc",
        "mean_pr_auc",
        "std_pr_auc",
        "mean_brier_score",
    ]
    return {name: float(summary[name]) for name in metric_names}


def run_model_comparison(
    df: pd.DataFrame,
    feature_cols: list[str],
    development_idx: pd.Index,
    model_name: str,
    spec: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grid_configs = build_grid_configurations(spec)
    if len(grid_configs) != BUDGET_PER_METHOD:
        raise ValueError(f"{model_name} has {len(grid_configs)} grid configurations; expected {BUDGET_PER_METHOD}.")

    details: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    baseline_started = time.perf_counter()
    baseline_metrics = evaluate_configuration(df, feature_cols, development_idx, model_name)
    baseline_seconds = time.perf_counter() - baseline_started
    summaries.append(
        {
            "model": model_name,
            "tuning_method": "baseline",
            "best_params": "{}",
            "evaluations": 1,
            "unique_configurations": 1,
            "cv_folds": N_SPLITS,
            "model_fits": N_SPLITS,
            "total_seconds": baseline_seconds,
            **baseline_metrics,
        }
    )

    grid_started = time.perf_counter()
    grid_rows: list[dict[str, Any]] = []
    for number, params in enumerate(grid_configs, start=1):
        started = time.perf_counter()
        metrics = evaluate_configuration(df, feature_cols, development_idx, model_name, params)
        grid_rows.append(
            {
                "model": model_name,
                "tuning_method": "grid_search",
                "evaluation_number": number,
                "is_unique_configuration": True,
                "params": json.dumps(params, sort_keys=True),
                "evaluation_seconds": time.perf_counter() - started,
                **metrics,
            }
        )
    grid_seconds = time.perf_counter() - grid_started
    details.extend(grid_rows)
    best_grid = max(grid_rows, key=lambda row: row["mean_pr_auc"])
    summaries.append(
        {
            "model": model_name,
            "tuning_method": "grid_search",
            "best_params": best_grid["params"],
            "evaluations": BUDGET_PER_METHOD,
            "unique_configurations": BUDGET_PER_METHOD,
            "cv_folds": N_SPLITS,
            "model_fits": BUDGET_PER_METHOD * N_SPLITS,
            "total_seconds": grid_seconds,
            **{key: best_grid[key] for key in baseline_metrics},
        }
    )

    optuna_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    optuna_started = time.perf_counter()

    def objective(trial: optuna.Trial) -> float:
        params = sample_optuna_configuration(trial, spec)
        config_key = json.dumps(params, sort_keys=True)
        started = time.perf_counter()
        metrics = evaluate_configuration(df, feature_cols, development_idx, model_name, params)
        optuna_rows.append(
            {
                "model": model_name,
                "tuning_method": "optuna_tpe",
                "evaluation_number": int(trial.number) + 1,
                "is_unique_configuration": config_key not in seen,
                "params": config_key,
                "evaluation_seconds": time.perf_counter() - started,
                **metrics,
            }
        )
        seen.add(config_key)
        return metrics["mean_pr_auc"]

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=f"all_models__{model_name}")
    study.optimize(objective, n_trials=BUDGET_PER_METHOD, show_progress_bar=False)
    optuna_seconds = time.perf_counter() - optuna_started
    details.extend(optuna_rows)
    best_optuna = max(optuna_rows, key=lambda row: row["mean_pr_auc"])
    summaries.append(
        {
            "model": model_name,
            "tuning_method": "optuna_tpe",
            "best_params": best_optuna["params"],
            "evaluations": BUDGET_PER_METHOD,
            "unique_configurations": int(sum(row["is_unique_configuration"] for row in optuna_rows)),
            "cv_folds": N_SPLITS,
            "model_fits": BUDGET_PER_METHOD * N_SPLITS,
            "total_seconds": optuna_seconds,
            **{key: best_optuna[key] for key in baseline_metrics},
        }
    )
    return details, summaries


def write_report(summary: pd.DataFrame, path: Path) -> None:
    ranked = summary[summary["tuning_method"] != "baseline"].sort_values(
        ["mean_pr_auc", "mean_recall", "mean_f2"], ascending=False
    )
    candidate_lookup = ranked.set_index(["model", "tuning_method"])
    svm_grid = candidate_lookup.loc[("SVM", "grid_search")]
    xgb_optuna = candidate_lookup.loc[("XGBoost", "optuna_tpe")]
    best_candidate = ranked.iloc[0]
    lines = [
        "# Perbandingan Seluruh Model: Optuna vs Grid Search",
        "",
        f"- Feature set: `{FEATURE_SET}`",
        f"- CV: `StratifiedKFold(n_splits={N_SPLITS}, shuffle=True, random_state={RANDOM_STATE})`",
        f"- Budget: `{BUDGET_PER_METHOD}` evaluasi atau `{BUDGET_PER_METHOD * N_SPLITS}` model fit per model-metode",
        "- Objective primer: `mean PR-AUC`",
        "- Locked test tidak digunakan untuk pemilihan.",
        "",
        "## Ranking Kandidat Tuning",
        "",
    ]
    for rank, row in enumerate(ranked.to_dict(orient="records"), start=1):
        lines.append(
            f"{rank}. `{row['model']} + {row['tuning_method']}`: PR-AUC `{row['mean_pr_auc']:.6f}`, "
            f"Recall `{row['mean_recall']:.6f}`, F2 `{row['mean_f2']:.6f}`, "
            f"unik `{int(row['unique_configurations'])}/{int(row['evaluations'])}`, waktu `{row['total_seconds']:.2f}` detik."
        )
    lines.extend(
        [
            "",
            "## Interpretasi",
            "",
            "- Perbandingan Optuna vs Grid harus dibaca di dalam model yang sama.",
            "- Ranking lintas model menjawab kombinasi model+tuning terbaik pada development CV.",
            "- Selisih kecil dibanding variasi fold tidak dianggap bukti superioritas universal.",
            "- Kandidat peringkat pertama baru boleh dievaluasi satu kali pada locked test setelah keputusan dikunci.",
            "",
            "## Jawaban atas contoh SVM Grid vs XGBoost Optuna",
            "",
            f"- SVM + Grid PR-AUC `{svm_grid['mean_pr_auc']:.6f}`; XGBoost + Optuna `{xgb_optuna['mean_pr_auc']:.6f}`.",
            f"- SVM + Grid recall `{svm_grid['mean_recall']:.6f}` dan F2 `{svm_grid['mean_f2']:.6f}`; "
            f"XGBoost + Optuna recall `{xgb_optuna['mean_recall']:.6f}` dan F2 `{xgb_optuna['mean_f2']:.6f}`.",
            f"- Berdasarkan objective primer, kandidat terbaik adalah `{best_candidate['model']} + {best_candidate['tuning_method']}` "
            f"dengan PR-AUC `{best_candidate['mean_pr_auc']:.6f}`.",
            "- Recall/F2 tetap dapat dibahas sebagai analisis sekunder, tetapi objective tidak diganti setelah melihat hasil.",
            "",
            "## Batasan",
            "",
            "- Domain diskret berisi 24 konfigurasi per model-metode agar budget sama.",
            "- Grid mengunjungi seluruh konfigurasi; Optuna dapat mengulang konfigurasi.",
            "- Hanya satu seed dan satu 5-fold CV; belum repeated nested CV.",
            "- Locked test tidak dipakai pada proses pemilihan ini.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--details", type=Path, default=DEFAULT_DETAILS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)

    args = parser.parse_args()

    df = load_csv(args.dataset)
    feature_sets, _ = build_feature_sets(df)
    feature_cols = feature_sets[FEATURE_SET]
    development_idx, _ = make_development_test_split(df, test_size=0.3, random_state=RANDOM_STATE)

    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for model_name, spec in MODEL_SPECS.items():
        print(f"Running {model_name}...", flush=True)
        model_details, model_summaries = run_model_comparison(df, feature_cols, development_idx, model_name, spec)
        detail_rows.extend(model_details)
        summary_rows.extend(model_summaries)

    details_df = pd.DataFrame(detail_rows)
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["mean_pr_auc", "mean_recall", "mean_f2"], ascending=False
    ).reset_index(drop=True)
    summary_df.insert(0, "overall_rank", range(1, len(summary_df) + 1))
    for frame in (details_df, summary_df):
        frame["feature_set"] = FEATURE_SET
        frame["objective_metric"] = "mean_pr_auc (average_precision)"
        frame["random_state"] = RANDOM_STATE
        frame["development_rows"] = len(development_idx)
        frame["locked_test_used_for_selection"] = False

    args.details.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    details_df.to_csv(args.details, index=False)
    summary_df.to_csv(args.summary, index=False)
    write_report(summary_df, args.report)

    print(summary_df[["overall_rank", "model", "tuning_method", "mean_pr_auc", "mean_recall", "mean_f2"]].to_string(index=False))
    print("details", args.details.resolve())
    print("summary", args.summary.resolve())
    print("report", args.report.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
