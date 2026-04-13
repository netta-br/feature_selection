"""Pipeline orchestrator — loads config, data, runs selectors & comparisons."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

from .config_schema import PipelineConfig, SelectorConfig, config_to_dict
from ..data_loader import DataLoader, DataBundle, TargetData
from ..results import SelectionResult, WrapperSelectionResult

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_label(label: str) -> str:
    """Convert a label to a safe filename component."""
    return label.replace(" ", "_").replace("/", "_").replace("\\", "_")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    """Orchestrates the complete feature selection experiment."""

    def __init__(self, config: PipelineConfig) -> None:
        """Store config, initialize empty result caches."""
        self.config = config
        self.results: dict[str, SelectionResult] = {}   # "{label}__{target}" -> result
        self.baselines: dict[str, pd.DataFrame] = {}     # target_name -> baseline summary
        self._data: DataBundle | None = None
        self._output_dir: str | None = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> str:
        """Execute the complete pipeline. Returns path to output directory."""
        log.info("Pipeline starting")

        # 1. Create output directory
        self._output_dir = self._create_output_dir()

        # 2. Save config copy
        self._save_config_copy()

        # 3. Load data
        self._data = self._load_data()
        log.info(
            "Data loaded: %d samples, %d features",
            self._data.n_samples,
            self._data.n_features,
        )

        # 4. Run baselines (per target)
        if self.config.baseline.enabled:
            self._run_baselines()

        # 5. Run selectors (per selector × per target)
        self._run_selectors()

        # 6. Run comparisons
        self._run_comparisons()

        # 7. Generate dashboard (if enabled)
        if self.config.output.generate_dashboard:
            try:
                from .dashboard import generate_dashboard  # type: ignore[import-not-found]

                generate_dashboard(
                    self._output_dir, self.results, self.baselines, self.config
                )
            except ImportError:
                log.warning(
                    "Dashboard generation skipped: panel/hvplot/bokeh not installed"
                )
            except Exception as e:
                log.error("Dashboard generation failed: %s", e)

        log.info("Pipeline complete. Output: %s", self._output_dir)
        return self._output_dir

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------

    def _create_output_dir(self) -> str:
        """Create timestamped output dir: ``{base_dir}/run_YYYYMMDD_HHMMSS/``."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.config.output.base_dir, f"run_{timestamp}")
        os.makedirs(path, exist_ok=True)
        log.info("Output directory: %s", path)
        return path

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self) -> DataBundle:
        """Use :class:`DataLoader` to load and split data."""
        loader = DataLoader(
            features_path=self.config.data.features_path,
            labels_path=self.config.data.labels_path,
            sample_id_column=self.config.data.sample_id_column,
            auto_transpose=self.config.data.auto_transpose,
            fillna_value=self.config.data.fillna_value,
        )
        targets = [
            {"name": t.name, "task_type": t.task_type} for t in self.config.targets
        ]
        return loader.load_and_split(
            targets=targets,
            random_seed=self.config.preprocessing.random_seed,
            train_pct=self.config.preprocessing.train_pct,
            val_pct=self.config.preprocessing.val_pct,
        )

    # ------------------------------------------------------------------
    # Baseline helpers
    # ------------------------------------------------------------------

    def _compute_eval_feature_counts(self, target_name: str) -> list[int]:
        """Compute baseline eval feature counts from selector configs or explicit list.

        If ``config.baseline.eval_feature_counts == "from_selectors"``, the
        checkpoint lists of every enabled selector targeting *target_name* are
        unioned and sorted.  Otherwise the explicit list is returned directly.
        """
        efc = self.config.baseline.eval_feature_counts

        if isinstance(efc, list):
            return list(efc)

        # "from_selectors" — derive from selector params
        all_checkpoints: set[int] = set()

        for sel in self.config.selectors:
            if not sel.enabled:
                continue
            if target_name not in sel.targets:
                continue

            params = sel.params
            if sel.type == "mrmr":
                n = params.get("n_features_to_select", 30)
                k = params.get("eval_every_k", 1)
            elif sel.type == "wrapper":
                n = params.get("n_features_to_select", 20)
                k = params.get("test_eval_every_k", 1)
            else:
                continue

            checkpoints = list(range(k, n + 1, k))
            if not checkpoints or checkpoints[-1] != n:
                checkpoints.append(n)
            all_checkpoints.update(checkpoints)

        return sorted(all_checkpoints)

    def _run_baselines(self) -> None:
        """Run classification/regression baselines for each target."""
        for target in self.config.targets:
            target_data = self._data.targets[target.name]
            N_values = self._compute_eval_feature_counts(target.name)

            if not N_values:
                log.warning(
                    "No eval feature counts for target '%s', skipping baseline",
                    target.name,
                )
                continue

            # Cap N_values at actual feature count
            N_values = [n for n in N_values if n <= self._data.n_features]

            if not N_values:
                log.warning(
                    "All eval feature counts exceed n_features=%d for target '%s', "
                    "skipping baseline",
                    self._data.n_features,
                    target.name,
                )
                continue

            log.info(
                "Running baseline for target '%s' (%s), N_values=%s",
                target.name,
                target.task_type,
                N_values,
            )

            if target.task_type == "classification":
                from feature_selection.src.baseline.logistic_regression_random_baseline import (
                    plot_performance_with_stats,
                )

                baseline_summary = plot_performance_with_stats(
                    self._data.X_train,
                    target_data.y_train,
                    self._data.X_val,
                    target_data.y_val,
                    N_values=N_values,
                    random_seed=self.config.preprocessing.random_seed,
                    num_runs=self.config.baseline.num_runs,
                    return_summary=True,
                )
            elif target.task_type == "regression":
                from feature_selection.src.baseline.linear_regression_random_baseline import (
                    plot_performance_with_stats as reg_baseline,
                )

                baseline_summary = reg_baseline(
                    self._data.X_train,
                    target_data.y_train,
                    self._data.X_val,
                    target_data.y_val,
                    N_values=N_values,
                    random_seed=self.config.preprocessing.random_seed,
                    num_runs=self.config.baseline.num_runs,
                    return_summary=True,
                )
            else:
                log.error("Unknown task_type '%s' for target '%s'", target.task_type, target.name)
                continue

            self.baselines[target.name] = baseline_summary

            # Save baseline as JSON
            if self.config.output.save_results_json:
                baseline_path = os.path.join(
                    self._output_dir, f"common__baseline__{target.name}.json"
                )
                baseline_dict = baseline_summary.to_dict(orient="records")
                with open(baseline_path, "w", encoding="utf-8") as f:
                    json.dump(baseline_dict, f, indent=2, default=str)
                log.info("Saved baseline: %s", baseline_path)

    # ------------------------------------------------------------------
    # Selector dispatch
    # ------------------------------------------------------------------

    def _run_selectors(self) -> None:
        """Run each enabled selector for each of its target variables."""
        for selector in self.config.selectors:
            if not selector.enabled:
                continue

            for target_name in selector.targets:
                target_data = self._data.targets[target_name]
                result_key = f"{selector.label}__{target_name}"
                log.info("Running selector: %s", result_key)

                if selector.type == "mrmr":
                    result = self._run_mrmr_selector(selector, target_name, target_data)
                elif selector.type == "wrapper":
                    result = self._run_wrapper_selector(selector, target_name, target_data)
                else:
                    log.error("Unknown selector type: %s", selector.type)
                    continue

                result.label = selector.label
                self.results[result_key] = result

                # Save result JSON
                if self.config.output.save_results_json:
                    result_path = os.path.join(self._output_dir, f"{result_key}.json")
                    result.save_as_json(result_path)
                    log.info("Saved result: %s", result_path)

    def _run_mrmr_selector(
        self,
        selector: SelectorConfig,
        target_name: str,
        target_data: TargetData,
    ) -> SelectionResult:
        """Instantiate and run an mRmR selector."""
        from feature_selection.src.filter.mRmR import mRmRSelector

        params = selector.params
        n_features = params.get("n_features_to_select", 30)
        eval_every_k = params.get("eval_every_k", 1)

        sel = mRmRSelector(
            X_train=self._data.X_train,
            y_train=target_data.y_train,
            relevance_method=params.get("relevance_method", "pearson"),
            mrmr_score_method=params.get("mrmr_score_method", "difference"),
            redundancy_method=params.get("redundancy_method", "pearson"),
            redundancy_agg=params.get("redundancy_agg", "mean"),
            correlation_filepath=params.get("correlation_filepath"),
            relevance_scores_filepath=params.get("relevance_scores_filepath"),
            random_seed=self.config.preprocessing.random_seed,
            X_val=self._data.X_val,
            y_val=target_data.y_val,
        )

        return sel.forward_selection(
            n_features_to_select=n_features,
            eval_every_k=eval_every_k,
        )

    def _run_wrapper_selector(
        self,
        selector: SelectorConfig,
        target_name: str,
        target_data: TargetData,
    ) -> WrapperSelectionResult:
        """Instantiate and run a wrapper selector."""
        from feature_selection.src.wrapper.MarkovBlanketWrapper import WrapperSelector

        params = selector.params
        n_features = params.get("n_features_to_select", 20)
        test_eval_every_k = params.get("test_eval_every_k", 1)

        sel = WrapperSelector(
            X_train=self._data.X_train,
            y_train=target_data.y_train,
            X_val=self._data.X_val,
            y_val=target_data.y_val,
            use_su_ranking=params.get("use_su_ranking", True),
            use_mb_pruning=params.get("use_mb_pruning", True),
            mb_threshold=params.get("mb_threshold", 0.0),
            cv_folds=params.get("cv_folds", 5),
            cv_min_folds=params.get("cv_min_folds", 3),
            patience=params.get("patience"),
            su_filepath=params.get("su_filepath"),
            random_seed=self.config.preprocessing.random_seed,
        )

        return sel.run(
            n_features_to_select=n_features,
            test_eval_every_k=test_eval_every_k,
        )

    # ------------------------------------------------------------------
    # Comparisons
    # ------------------------------------------------------------------

    def _run_comparisons(self) -> None:
        """Execute comparison operations defined in config."""
        for comp in self.config.comparisons:
            log.info("Running comparison: %s", comp.label)
            try:
                if comp.type == "compare_results":
                    self._run_compare_results(comp)
                elif comp.type == "compare_with":
                    self._run_compare_with(comp)
                elif comp.type == "summary_report":
                    self._run_summary_report(comp)
                else:
                    log.error("Unknown comparison type: %s", comp.type)
            except Exception as e:
                log.error("Comparison '%s' failed: %s", comp.label, e)

    def _run_compare_results(self, comp) -> None:
        """Run a ``compare_results`` comparison (multi-result plot)."""
        results_list: list[SelectionResult] = []
        labels_list: list[str] = []

        for sel_label in comp.selectors:
            key = f"{sel_label}__{comp.target}"
            if key not in self.results:
                log.warning(
                    "compare_results '%s': result '%s' not found, skipping",
                    comp.label,
                    key,
                )
                continue
            results_list.append(self.results[key])
            labels_list.append(sel_label)

        if not results_list:
            log.warning("compare_results '%s': no results to compare", comp.label)
            return

        baseline = None
        if comp.include_baseline and comp.target in self.baselines:
            baseline = self.baselines[comp.target]

        SelectionResult.compare_results(
            results_list, labels_list, baseline, title_suffix=f" — {comp.label}"
        )

        # Save comparison info to JSON
        if self.config.output.save_results_json:
            safe = _safe_label(comp.label)
            comp_path = os.path.join(self._output_dir, f"comparison__{safe}.json")
            comp_dict = {
                "type": "compare_results",
                "label": comp.label,
                "target": comp.target,
                "selectors": labels_list,
                "include_baseline": comp.include_baseline,
            }
            with open(comp_path, "w", encoding="utf-8") as f:
                json.dump(comp_dict, f, indent=2)
            log.info("Saved comparison: %s", comp_path)

    def _run_compare_with(self, comp) -> None:
        """Run a ``compare_with`` comparison (two-result rank diff)."""
        if len(comp.selectors) != 2:
            log.error(
                "compare_with '%s' requires exactly 2 selectors, got %d",
                comp.label,
                len(comp.selectors),
            )
            return

        keys = [f"{s}__{comp.target}" for s in comp.selectors]
        for key in keys:
            if key not in self.results:
                log.warning(
                    "compare_with '%s': result '%s' not found, skipping",
                    comp.label,
                    key,
                )
                return

        result1 = self.results[keys[0]]
        result2 = self.results[keys[1]]
        labels = comp.selectors

        result1.compare_with(result2, self_label=labels[0], other_label=labels[1])

        # Save comparison info to JSON
        if self.config.output.save_results_json:
            safe = _safe_label(comp.label)
            comp_path = os.path.join(self._output_dir, f"comparison__{safe}.json")
            comp_dict = {
                "type": "compare_with",
                "label": comp.label,
                "target": comp.target,
                "selectors": list(labels),
            }
            with open(comp_path, "w", encoding="utf-8") as f:
                json.dump(comp_dict, f, indent=2)
            log.info("Saved comparison: %s", comp_path)

    def _run_summary_report(self, comp) -> None:
        """Run a ``summary_report`` comparison (per-feature stats)."""
        results_list: list[SelectionResult] = []
        labels_list: list[str] = []

        for sel_label in comp.selectors:
            key = f"{sel_label}__{comp.target}"
            if key not in self.results:
                log.warning(
                    "summary_report '%s': result '%s' not found, skipping",
                    comp.label,
                    key,
                )
                continue
            results_list.append(self.results[key])
            labels_list.append(sel_label)

        if not results_list:
            log.warning("summary_report '%s': no results to summarise", comp.label)
            return

        df = SelectionResult.summary_report(results_list, labels_list)

        # Save the summary DataFrame to JSON
        if self.config.output.save_results_json:
            safe = _safe_label(comp.label)
            summary_path = os.path.join(self._output_dir, f"summary__{safe}.json")
            summary_records = df.to_dict(orient="records")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_records, f, indent=2, default=str)
            log.info("Saved summary: %s", summary_path)

    # ------------------------------------------------------------------
    # Config copy
    # ------------------------------------------------------------------

    def _save_config_copy(self) -> None:
        """Save a copy of the config to the output directory."""
        config_path = os.path.join(self._output_dir, "pipeline_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_to_dict(self.config), f, indent=2)
        log.info("Saved config copy: %s", config_path)
