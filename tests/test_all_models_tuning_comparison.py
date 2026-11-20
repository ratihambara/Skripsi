from __future__ import annotations

import unittest

from scripts.run_all_models_tuning_comparison import MODEL_SPECS, build_grid_configurations


class AllModelsTuningComparisonTests(unittest.TestCase):
    def test_every_model_has_exactly_24_grid_configurations(self) -> None:
        expected_models = {
            "Logistic Regression",
            "SVM",
            "KNN",
            "Gradient Boosting",
            "XGBoost",
            "LightGBM",
        }
        self.assertEqual(set(MODEL_SPECS), expected_models)
        for model_name, spec in MODEL_SPECS.items():
            with self.subTest(model=model_name):
                configurations = build_grid_configurations(spec)
                self.assertEqual(len(configurations), 24)
                self.assertEqual(len({repr(sorted(config.items())) for config in configurations}), 24)

    def test_optuna_and_grid_expose_same_tunable_parameters_per_model(self) -> None:
        for model_name, spec in MODEL_SPECS.items():
            with self.subTest(model=model_name):
                grid_keys = set().union(*(config.keys() for config in build_grid_configurations(spec)))
                self.assertEqual(grid_keys, set(spec["tunable_parameters"]))


if __name__ == "__main__":
    unittest.main()
