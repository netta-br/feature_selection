"""End-to-end integration tests for the feature-selection pipeline.

Generates synthetic data, builds a pipeline config programmatically,
runs the full pipeline, and verifies that all expected outputs exist
and contain valid content.

Runnable directly:  ``python -m tests.test_integration``
Also pytest-compatible:  ``pytest tests/test_integration.py -v``
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no GUI windows

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so imports resolve
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from feature_selection.src.pipeline.config_schema import (
    PipelineConfig,
    DataConfig,
    PreprocessingConfig,
    TargetConfig,
    BaselineConfig,
    SelectorConfig,
    ComparisonConfig,
    OutputConfig,
    ScoreMatrixConfig,
    validate_config,
    save_config,
)
from feature_selection.src.pipeline.runner import Pipeline
from feature_selection.src.results import SelectionResult, WrapperSelectionResult
from feature_selection.src.pipeline.cli import main as cli_main

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def create_synthetic_data(
    n_samples: int = 100,
    n_features: int = 50,
    n_informative: int = 5,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create synthetic features CSV and labels CSV for testing."""
    rng = np.random.RandomState(random_seed)

    # Generate feature matrix
    X = rng.randn(n_samples, n_features)
    feature_names = [f"feat_{i}" for i in range(n_features)]

    # Make first n_informative features correlate with target
    weights = rng.randn(n_informative)
    linear_comb = X[:, :n_informative] @ weights

    # Classification target: threshold the linear combination + noise
    noise = rng.randn(n_samples) * 0.5
    clf_target = (linear_comb + noise > 0).astype(int)

    # Regression target: linear combination + noise
    reg_target = linear_comb + noise

    # Sample IDs
    sample_ids = [f"sample_{i}" for i in range(n_samples)]

    # Features DataFrame: samples × features
    features_df = pd.DataFrame(X, columns=feature_names)
    features_df.insert(0, "samplename", sample_ids)

    # Labels DataFrame
    labels_df = pd.DataFrame({
        "samplename": sample_ids,
        "clf_target": clf_target,
        "reg_target": reg_target,
    })

    return features_df, labels_df


# ---------------------------------------------------------------------------
# Helper: build pipeline config
# ---------------------------------------------------------------------------

def build_test_config(
    features_path: str,
    labels_path: str,
    output_dir: str,
) -> PipelineConfig:
    """Build a PipelineConfig for end-to-end testing."""
    config = PipelineConfig(
        data=DataConfig(
            features_path=features_path,
            labels_path=labels_path,
            sample_id_column="samplename",
            auto_transpose=True,
            fillna_value=0,
        ),
        preprocessing=PreprocessingConfig(
            random_seed=42,
            train_pct=0.7,
            val_pct=0.1,
            test_pct=0.2,
        ),
        targets=[
            TargetConfig(name="clf_target", task_type="classification"),
            TargetConfig(name="reg_target", task_type="regression"),
        ],
        baseline=BaselineConfig(
            enabled=True,
            num_runs=5,
            eval_feature_counts="from_selectors",
        ),
        selectors=[
            SelectorConfig(
                label="mRmR-Pearson",
                type="mrmr",
                enabled=True,
                params={
                    "relevance_method": "pearson",
                    "mrmr_score_method": "difference",
                    "n_features_to_select": 10,
                    "eval_every_k": 2,
                },
                targets=["clf_target", "reg_target"],
            ),
            SelectorConfig(
                label="IWSS-MB",
                type="wrapper",
                enabled=True,
                params={
                    "use_su_ranking": True,
                    "use_mb_pruning": True,
                    "mb_threshold": 0.0,
                    "n_features_to_select": 8,
                    "test_eval_every_k": 2,
                    "cv_folds": 3,
                    "cv_min_folds": 2,
                },
                targets=["clf_target"],
            ),
        ],
        comparisons=[
            ComparisonConfig(
                label="clf_comparison",
                target="clf_target",
                selectors=["mRmR-Pearson", "IWSS-MB"],
                include_baseline=True,
                type="compare_results",
            ),
        ],
        score_matrix=ScoreMatrixConfig(),
        output=OutputConfig(
            base_dir=output_dir,
            save_results_json=True,
            generate_dashboard=True,
        ),
    )
    return config


# ---------------------------------------------------------------------------
# Test 1: End-to-end pipeline run
# ---------------------------------------------------------------------------

def test_end_to_end_pipeline() -> None:
    """Run the full pipeline on synthetic data and verify all outputs."""
    print("\n" + "=" * 70)
    print("TEST: End-to-end pipeline run")
    print("=" * 70)

    tmp_dir = tempfile.mkdtemp(prefix="integration_test_")
    try:
        # ---- Generate synthetic data ----
        features_df, labels_df = create_synthetic_data()
        features_path = os.path.join(tmp_dir, "features.csv")
        labels_path = os.path.join(tmp_dir, "labels.csv")
        features_df.to_csv(features_path, index=False)
        labels_df.to_csv(labels_path, index=False)
        print(f"  ✓ Synthetic data written to {tmp_dir}")

        output_dir = os.path.join(tmp_dir, "output")
        config = build_test_config(features_path, labels_path, output_dir)

        # ---- Validate config ----
        errors = validate_config(config)
        assert not errors, f"Config validation failed: {errors}"
        print("  ✓ Config validation passed")

        # ---- Run pipeline ----
        pipeline = Pipeline(config)
        run_output_dir = pipeline.run()
        print(f"  ✓ Pipeline completed, output: {run_output_dir}")

        # ---- Verify output directory ----
        assert os.path.isdir(run_output_dir), (
            f"Output directory does not exist: {run_output_dir}"
        )
        print("  ✓ Output directory exists")

        # ---- Verify pipeline_config.json copy ----
        config_copy = os.path.join(run_output_dir, "pipeline_config.json")
        assert os.path.isfile(config_copy), (
            f"pipeline_config.json not found in {run_output_dir}"
        )
        print("  ✓ pipeline_config.json copy exists")

        # ---- Verify baseline JSON files ----
        for target_name in ["clf_target", "reg_target"]:
            baseline_file = os.path.join(
                run_output_dir, f"common__baseline__{target_name}.json"
            )
            assert os.path.isfile(baseline_file), (
                f"Baseline file not found: {baseline_file}"
            )
            print(f"  ✓ Baseline JSON exists for {target_name}")

        # ---- Verify result JSON files ----
        expected_results = [
            "mRmR-Pearson__clf_target.json",
            "mRmR-Pearson__reg_target.json",
            "IWSS-MB__clf_target.json",
        ]
        for result_file in expected_results:
            result_path = os.path.join(run_output_dir, result_file)
            assert os.path.isfile(result_path), (
                f"Result file not found: {result_path}"
            )
            print(f"  ✓ Result JSON exists: {result_file}")

        # ---- Verify comparison JSON ----
        comparison_file = os.path.join(
            run_output_dir, "comparison__clf_comparison.json"
        )
        assert os.path.isfile(comparison_file), (
            f"Comparison file not found: {comparison_file}"
        )
        print("  ✓ Comparison JSON exists")

        # ---- Verify dashboard.html (skip gracefully if panel not installed) ----
        dashboard_path = os.path.join(run_output_dir, "dashboard.html")
        try:
            import panel  # noqa: F401
            assert os.path.isfile(dashboard_path), (
                f"dashboard.html not found: {dashboard_path}"
            )
            print("  ✓ dashboard.html exists")
        except ImportError:
            print("  ⊘ Dashboard check skipped (panel not installed)")

        # ==== Verify result content ====
        print("\n  --- Verifying result content ---")

        # mRmR-Pearson classification
        mrmr_clf = SelectionResult.load_from_json(
            os.path.join(run_output_dir, "mRmR-Pearson__clf_target.json")
        )
        assert len(mrmr_clf.selected_features) > 0, "mRmR clf: no features selected"
        assert len(mrmr_clf.performance_history) > 0, "mRmR clf: empty performance_history"
        assert mrmr_clf.n_steps > 0, "mRmR clf: n_steps must be > 0"
        assert mrmr_clf.task_type == "classification", (
            f"mRmR clf: wrong task_type '{mrmr_clf.task_type}'"
        )
        assert isinstance(mrmr_clf, SelectionResult)
        print(f"  ✓ mRmR-Pearson__clf_target: {mrmr_clf.n_steps} features, "
              f"task_type={mrmr_clf.task_type}")

        # mRmR-Pearson regression
        mrmr_reg = SelectionResult.load_from_json(
            os.path.join(run_output_dir, "mRmR-Pearson__reg_target.json")
        )
        assert len(mrmr_reg.selected_features) > 0, "mRmR reg: no features selected"
        assert len(mrmr_reg.performance_history) > 0, "mRmR reg: empty performance_history"
        assert mrmr_reg.n_steps > 0, "mRmR reg: n_steps must be > 0"
        assert mrmr_reg.task_type == "regression", (
            f"mRmR reg: wrong task_type '{mrmr_reg.task_type}'"
        )
        print(f"  ✓ mRmR-Pearson__reg_target: {mrmr_reg.n_steps} features, "
              f"task_type={mrmr_reg.task_type}")

        # IWSS-MB classification (wrapper result)
        iwss_clf = SelectionResult.load_from_json(
            os.path.join(run_output_dir, "IWSS-MB__clf_target.json")
        )
        assert len(iwss_clf.selected_features) > 0, "IWSS-MB clf: no features selected"
        assert len(iwss_clf.performance_history) > 0, "IWSS-MB clf: empty performance_history"
        assert iwss_clf.n_steps > 0, "IWSS-MB clf: n_steps must be > 0"
        assert isinstance(iwss_clf, WrapperSelectionResult), (
            f"IWSS-MB should be WrapperSelectionResult, got {type(iwss_clf).__name__}"
        )
        print(f"  ✓ IWSS-MB__clf_target: {iwss_clf.n_steps} features, "
              f"type={type(iwss_clf).__name__}")

        print("\n  ✅ test_end_to_end_pipeline PASSED")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"  Cleanup: removed {tmp_dir}")


# ---------------------------------------------------------------------------
# Test 2: CLI validate subcommand
# ---------------------------------------------------------------------------

def test_cli_validate() -> None:
    """Save a config to a temp file and validate it via the CLI."""
    print("\n" + "=" * 70)
    print("TEST: CLI validate subcommand")
    print("=" * 70)

    tmp_dir = tempfile.mkdtemp(prefix="integration_cli_validate_")
    try:
        features_path = os.path.join(tmp_dir, "features.csv")
        labels_path = os.path.join(tmp_dir, "labels.csv")

        # Write dummy CSVs (validation only checks config, not file existence)
        features_df, labels_df = create_synthetic_data()
        features_df.to_csv(features_path, index=False)
        labels_df.to_csv(labels_path, index=False)

        output_dir = os.path.join(tmp_dir, "output")
        config = build_test_config(features_path, labels_path, output_dir)

        config_filepath = os.path.join(tmp_dir, "test_config.json")
        save_config(config, config_filepath)
        print(f"  ✓ Config saved to {config_filepath}")

        # CLI validate should exit(0) for valid config
        try:
            cli_main(["validate", "--config", config_filepath])
        except SystemExit as e:
            assert e.code == 0, f"CLI validate exited with code {e.code} (expected 0)"
        print("  ✓ CLI validate exited with code 0")

        print("\n  ✅ test_cli_validate PASSED")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"  Cleanup: removed {tmp_dir}")


# ---------------------------------------------------------------------------
# Test 3: CLI quick-run mode
# ---------------------------------------------------------------------------

def test_cli_quick_run() -> None:
    """Run the pipeline via CLI quick-run flags."""
    print("\n" + "=" * 70)
    print("TEST: CLI quick-run mode")
    print("=" * 70)

    tmp_dir = tempfile.mkdtemp(prefix="integration_cli_quickrun_")
    try:
        features_df, labels_df = create_synthetic_data()
        features_path = os.path.join(tmp_dir, "features.csv")
        labels_path = os.path.join(tmp_dir, "labels.csv")
        features_df.to_csv(features_path, index=False)
        labels_df.to_csv(labels_path, index=False)
        print(f"  ✓ Synthetic data written to {tmp_dir}")

        output_dir = os.path.join(tmp_dir, "output")

        cli_main([
            "run",
            "--data", features_path,
            "--labels", labels_path,
            "--target", "clf_target",
            "--task", "classification",
            "--mRmR-P",
            "--no-dashboard",
            "--no-baseline",
            "--output", output_dir,
            "--seed", "42",
        ])
        print("  ✓ CLI quick-run completed")

        # Verify the output directory was created (it'll be a timestamped subdir)
        assert os.path.isdir(output_dir), f"Output dir not found: {output_dir}"
        run_dirs = [
            d for d in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("run_")
        ]
        assert len(run_dirs) >= 1, f"No run_* subdirectory found in {output_dir}"
        run_dir = os.path.join(output_dir, sorted(run_dirs)[-1])
        print(f"  ✓ Run output directory: {run_dir}")

        # Verify mRmR-Pearson result exists
        result_file = os.path.join(run_dir, "mRmR-Pearson__clf_target.json")
        assert os.path.isfile(result_file), f"Result not found: {result_file}"
        print(f"  ✓ Result JSON exists: mRmR-Pearson__clf_target.json")

        # Verify content
        result = SelectionResult.load_from_json(result_file)
        assert len(result.selected_features) > 0, "No features selected"
        assert result.n_steps > 0, "n_steps must be > 0"
        print(f"  ✓ Result content valid: {result.n_steps} features selected")

        print("\n  ✅ test_cli_quick_run PASSED")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"  Cleanup: removed {tmp_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_end_to_end_pipeline()
    test_cli_validate()
    test_cli_quick_run()
    print("\n" + "=" * 70)
    print("✅ All integration tests passed!")
    print("=" * 70)
