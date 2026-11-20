from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path
from typing import Any

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
DEFAULT_OUTPUT = Path("final_outputs/11_tuning_comparison/FAIR_COMPARISON_RESULTS.csv")
FEATURE_SET = "clinical_core_extreme"
MODEL = "XGBoost"
RANDOM_STATE = 42
N_SPLITS = 5

# A deliberately compact common domain: 2^6 = 64 possible configurations.
# reg_alpha and reg_lambda are controlled (fixed), rather than optimized, so both
# methods receive exactly the same six-dimensional search problem.
COMMON_DOMAIN: dict[str, list[Any]] = {
    "n_estimators": [100, 400],
    "max_depth": [3, 6],
    "learning_rate": [0.02, 0.08],
    "subsample": [0.65, 1.0],
    "colsample_bytree": [0.65, 1.0],
    "min_child_weight": [1, 8],
}
FIXED_PARAMS = {"reg_lambda": 1.0, "reg_alpha": 0.0}


def evaluate(
    df: pd.DataFrame,
    development_idx: pd.Index,
    params: dict[str, Any],
) -> dict[str, float]:
    summary = summarize_cv_results(
        collect_cv_results(
            df=df,
            feature_sets={FEATURE_SET: build_feature_sets(df)[0][FEATURE_SET]},
            model_names=[MODEL],
            development_idx=development_idx,
            n_splits=N_SPLITS,
            random_state=RANDOM_STATE,
            use_class_balancing=True,
            use_smote=False,
            overrides={MODEL: {**params, **FIXED_PARAMS}},
        )
    ).iloc[0]
    return {
        "mean_pr_auc": float(summary["mean_pr_auc"]),
        "std_pr_auc": float(summary["std_pr_auc"]),
        "mean_recall": float(summary["mean_recall"]),
        "mean_f2": float(summary["mean_f2"]),
        "mean_roc_auc": float(summary["mean_roc_auc"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    df = load_csv(args.dataset)
    development_idx, _ = make_development_test_split(df, test_size=0.3, random_state=RANDOM_STATE)
    keys = list(COMMON_DOMAIN)
    configurations = [dict(zip(keys, values)) for values in product(*(COMMON_DOMAIN[key] for key in keys))]
    budget = len(configurations)
    rows: list[dict[str, Any]] = []

    grid_started = time.perf_counter()
    for evaluation_number, params in enumerate(configurations, start=1):
        started = time.perf_counter()
        metrics = evaluate(df, development_idx, params)
        rows.append(
            {
                "method": "grid_search_controlled",
                "evaluation_number": evaluation_number,
                "is_unique_configuration": True,
                "evaluation_seconds": time.perf_counter() - started,
                **params,
                **FIXED_PARAMS,
                **metrics,
            }
        )
    grid_seconds = time.perf_counter() - grid_started

    seen: set[str] = set()
    optuna_started = time.perf_counter()

    def objective(trial: optuna.Trial) -> float:
        params = {key: trial.suggest_categorical(key, values) for key, values in COMMON_DOMAIN.items()}
        config_key = json.dumps(params, sort_keys=True)
        started = time.perf_counter()
        metrics = evaluate(df, development_idx, params)
        elapsed = time.perf_counter() - started
        unique = config_key not in seen
        seen.add(config_key)
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("elapsed", elapsed)
        trial.set_user_attr("unique", unique)
        return metrics["mean_pr_auc"]

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="controlled_optuna_vs_grid")
    study.optimize(objective, n_trials=budget, show_progress_bar=False)
    optuna_seconds = time.perf_counter() - optuna_started

    for trial in study.trials:
        metrics = trial.user_attrs["metrics"]
        rows.append(
            {
                "method": "optuna_tpe_controlled",
                "evaluation_number": int(trial.number) + 1,
                "is_unique_configuration": bool(trial.user_attrs["unique"]),
                "evaluation_seconds": float(trial.user_attrs["elapsed"]),
                **trial.params,
                **FIXED_PARAMS,
                **metrics,
            }
        )

    results = pd.DataFrame(rows)
    results["method_total_seconds"] = results["method"].map(
        {
            "grid_search_controlled": grid_seconds,
            "optuna_tpe_controlled": optuna_seconds,
        }
    )
    results["search_budget_configurations"] = budget
    results["cv_folds"] = N_SPLITS
    results["model_fits_per_method"] = budget * N_SPLITS
    results["random_state"] = RANDOM_STATE
    results["feature_set"] = FEATURE_SET
    results["model"] = MODEL
    results["objective_metric"] = "mean_pr_auc (average_precision)"
    results["development_rows"] = len(development_idx)
    results["locked_test_used_for_selection"] = False
    results["common_domain_json"] = json.dumps(COMMON_DOMAIN, sort_keys=True)
    results["fixed_params_json"] = json.dumps(FIXED_PARAMS, sort_keys=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)

    summary = (
        results.sort_values(["method", "mean_pr_auc"], ascending=[True, False])
        .groupby("method", as_index=False)
        .first()[
            [
                "method",
                "mean_pr_auc",
                "std_pr_auc",
                "mean_recall",
                "mean_f2",
                "mean_roc_auc",
                "method_total_seconds",
                "search_budget_configurations",
                "model_fits_per_method",
            ]
        ]
    )
    unique_counts = results.groupby("method")["is_unique_configuration"].sum().to_dict()
    print(summary.to_string(index=False))
    print("unique_configurations", unique_counts)
    print("output", args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
