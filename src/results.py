from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


# ---------------------------------------------------------------------------
# SelectionResult dataclass
# ---------------------------------------------------------------------------
@dataclass
class SelectionResult:
    """Holds the output of a completed ``mRmRSelector.forward_selection`` run.

    Each entry in ``performance_history`` contains a ``'step'`` key (1-based)
    equal to the number of features selected at that evaluation point.

    Parameters
    ----------
    task_type : str
        ``'classification'`` or ``'regression'``.  Controls which metrics are
        rendered by :py:meth:`__str__` and which panels are drawn by
        :py:meth:`plot_vs_random_baseline` / :py:meth:`compare_results`.
    """

    selected_features: list[str]
    performance_history: list[dict]        # one dict per step where evaluation ran; each has 'step' key
    stopping_reason: str                   # "max_features_reached" | "metric_threshold_reached" | "no_features_remaining"
    n_steps: int
    final_metrics: dict | None             # last entry of performance_history, or None
    selection_time_seconds: float          # lazy on-demand compute time only
    evaluation_time_seconds: float | None  # None if evaluation not performed
    task_type: str = "classification"      # "classification" | "regression"

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------
    def __str__(self) -> str:
        lines = [
            "SelectionResult:",
            f"  features selected : {self.n_steps}",
            f"  task type         : {self.task_type}",
            f"  stopping reason   : {self.stopping_reason}",
            f"  selection time    : {self.selection_time_seconds:.3f}s",
        ]
        if self.evaluation_time_seconds is not None:
            lines.append(f"  evaluation time   : {self.evaluation_time_seconds:.3f}s")

        if self.final_metrics is not None:
            if self.task_type == "classification":
                acc = self.final_metrics.get("accuracy", "n/a")
                f1 = (
                    self.final_metrics.get("macro avg", {}).get("f1-score", "n/a")
                    if isinstance(self.final_metrics.get("macro avg"), dict)
                    else "n/a"
                )
                lines.append(f"  final accuracy    : {acc}")
                lines.append(f"  final macro F1    : {f1}")
            else:  # regression
                r2  = self.final_metrics.get("r2",  "n/a")
                mse = self.final_metrics.get("mse", "n/a")
                mae = self.final_metrics.get("mae", "n/a")
                lines.append(f"  final R²          : {r2}")
                lines.append(f"  final MSE         : {mse}")
                lines.append(f"  final MAE         : {mae}")

        lines.append(f"  features          : {self.selected_features}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # plot_vs_random_baseline (instance method)
    # ------------------------------------------------------------------
    def plot_vs_random_baseline(
        self,
        baseline_summary: pd.DataFrame,
        title_suffix: str = "",
    ) -> None:
        """Plot this result's performance against a random baseline.

        Parameters
        ----------
        baseline_summary : pd.DataFrame
            Wide-format summary returned by
            ``LR_random_baseline.plot_performance_with_stats(return_summary=True)``
            (classification) or
            ``LR_regression_baseline.plot_performance_with_stats(return_summary=True)``
            (regression).
        title_suffix : str
            Optional suffix appended to the figure title.
        """
        if self.task_type == "classification":
            self._plot_vs_baseline_clf(baseline_summary, title_suffix)
        else:
            self._plot_vs_baseline_reg(baseline_summary, title_suffix)

    def _plot_vs_baseline_clf(
        self,
        baseline_summary: pd.DataFrame,
        title_suffix: str,
    ) -> None:
        """Classification branch of plot_vs_random_baseline."""
        mrmr_records = [
            {
                "n_features": entry["step"],
                "accuracy":   entry.get("accuracy", float("nan")),
                "macro_f1":   (
                    entry.get("macro avg", {}).get("f1-score", float("nan"))
                    if isinstance(entry.get("macro avg"), dict)
                    else float("nan")
                ),
            }
            for entry in self.performance_history
        ]
        mrmr_df = pd.DataFrame(mrmr_records)

        # Print summary table
        print("\nSummary Statistics — Random Baseline (Mean ± Std):")
        try:
            display(baseline_summary)  # type: ignore[name-defined]  # noqa: F821
        except NameError:
            print(baseline_summary.to_string(index=False))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        panels = [
            ("accuracy", "acc_mean", "acc_std", "Accuracy"),
            ("macro_f1", "f1_mean",  "f1_std",  "Macro F1"),
        ]

        for ax, (mrmr_col, bl_mean, bl_std, title) in zip(axes, panels):
            ax.plot(
                mrmr_df["n_features"], mrmr_df[mrmr_col],
                marker="o", linewidth=2, color="steelblue", label="mRmR",
            )
            bx = baseline_summary["features_num"]
            ax.plot(
                bx, baseline_summary[bl_mean],
                marker="s", linewidth=2, color="darkorange",
                linestyle="--", label="Random baseline (mean)",
            )
            ax.fill_between(
                bx,
                baseline_summary[bl_mean] - baseline_summary[bl_std],
                baseline_summary[bl_mean] + baseline_summary[bl_std],
                alpha=0.25, color="darkorange", label="±1 std (random)",
            )
            ax.set_title(title, fontsize=13)
            ax.set_xlabel("Number of features")
            ax.set_ylabel(title)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.4)

        fig.suptitle(
            f"Classification: mRmR vs LR Random Baseline{title_suffix}",
            fontsize=13, fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        plt.show()

    def _plot_vs_baseline_reg(
        self,
        baseline_summary: pd.DataFrame,
        title_suffix: str,
    ) -> None:
        """Regression branch of plot_vs_random_baseline."""
        mrmr_records = [
            {
                "n_features": entry["step"],
                "r2":  entry.get("r2",  float("nan")),
                "mse": entry.get("mse", float("nan")),
                "mae": entry.get("mae", float("nan")),
            }
            for entry in self.performance_history
        ]
        mrmr_df = pd.DataFrame(mrmr_records)

        # Print summary table
        print("\nSummary Statistics — LinReg Random Baseline (Mean ± Std):")
        try:
            display(baseline_summary)  # type: ignore[name-defined]  # noqa: F821
        except NameError:
            print(baseline_summary.to_string(index=False))

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        panels = [
            ("r2",  "r2_mean",  "r2_std",  "R²"),
            ("mse", "mse_mean", "mse_std", "MSE"),
            ("mae", "mae_mean", "mae_std", "MAE"),
        ]

        for ax, (mrmr_col, bl_mean, bl_std, title) in zip(axes, panels):
            ax.plot(
                mrmr_df["n_features"], mrmr_df[mrmr_col],
                marker="o", linewidth=2, color="steelblue", label="mRmR",
            )
            bx = baseline_summary["features_num"]
            ax.plot(
                bx, baseline_summary[bl_mean],
                marker="s", linewidth=2, color="darkorange",
                linestyle="--", label="Random baseline (mean)",
            )
            ax.fill_between(
                bx,
                baseline_summary[bl_mean] - baseline_summary[bl_std],
                baseline_summary[bl_mean] + baseline_summary[bl_std],
                alpha=0.25, color="darkorange", label="±1 std (random)",
            )
            ax.set_title(title, fontsize=13)
            ax.set_xlabel("Number of features")
            ax.set_ylabel(title)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.4)

        fig.suptitle(
            f"Regression: mRmR vs LinReg Random Baseline{title_suffix}",
            fontsize=13, fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # compare_results (static method)
    # ------------------------------------------------------------------
    @staticmethod
    def compare_results(
        results: list[SelectionResult],
        labels: list[str],
        baseline_summary: pd.DataFrame | None = None,
        title_suffix: str = "",
    ) -> None:
        """Plot multiple ``SelectionResult`` objects on shared axes for comparison.

        All results must share the same ``task_type``.

        Parameters
        ----------
        results : list[SelectionResult]
            Results to compare.  Must be non-empty.
        labels : list[str]
            One label per result (used in the legend).  Must match ``len(results)``.
        baseline_summary : pd.DataFrame | None
            Optional wide-format baseline summary (same format as accepted by
            :py:meth:`plot_vs_random_baseline`).  When provided, an orange mean
            line and ±1 std band are drawn on each panel.
        title_suffix : str
            Optional suffix appended to the figure title.
        """
        if len(results) != len(labels):
            raise ValueError(
                f"len(results)={len(results)} must equal len(labels)={len(labels)}"
            )
        if not results:
            raise ValueError("results must be non-empty")

        task_types = {r.task_type for r in results}
        if len(task_types) > 1:
            raise ValueError(
                f"All results must share the same task_type; found: {task_types}"
            )
        task_type = next(iter(task_types))

        if task_type == "classification":
            SelectionResult._compare_clf(results, labels, baseline_summary, title_suffix)
        else:
            SelectionResult._compare_reg(results, labels, baseline_summary, title_suffix)

    @staticmethod
    def _compare_clf(
        results: list[SelectionResult],
        labels: list[str],
        baseline_summary: pd.DataFrame | None,
        title_suffix: str,
    ) -> None:
        """Classification comparison — 2 panels: Accuracy and Macro F1."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        panels = [
            ("accuracy", "acc_mean", "acc_std", "Accuracy"),
            ("macro_f1", "f1_mean",  "f1_std",  "Macro F1"),
        ]

        for ax, (mrmr_col, bl_mean, bl_std, panel_title) in zip(axes, panels):
            for i, (res, label) in enumerate(zip(results, labels)):
                records = [
                    {
                        "n_features": entry["step"],
                        "accuracy":   entry.get("accuracy", float("nan")),
                        "macro_f1":   (
                            entry.get("macro avg", {}).get("f1-score", float("nan"))
                            if isinstance(entry.get("macro avg"), dict)
                            else float("nan")
                        ),
                    }
                    for entry in res.performance_history
                ]
                df = pd.DataFrame(records)
                color = prop_cycle[i % len(prop_cycle)]
                ax.plot(
                    df["n_features"], df[mrmr_col],
                    marker="o", linewidth=2, color=color, label=label,
                )

            if baseline_summary is not None:
                bx = baseline_summary["features_num"]
                ax.plot(
                    bx, baseline_summary[bl_mean],
                    marker="s", linewidth=2, color="darkorange",
                    linestyle="--", label="Random baseline (mean)",
                )
                ax.fill_between(
                    bx,
                    baseline_summary[bl_mean] - baseline_summary[bl_std],
                    baseline_summary[bl_mean] + baseline_summary[bl_std],
                    alpha=0.25, color="darkorange", label="±1 std (random)",
                )

            ax.set_title(panel_title, fontsize=13)
            ax.set_xlabel("Number of features")
            ax.set_ylabel(panel_title)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.4)

        fig.suptitle(
            f"Classification — Method Comparison{title_suffix}",
            fontsize=13, fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _compare_reg(
        results: list[SelectionResult],
        labels: list[str],
        baseline_summary: pd.DataFrame | None,
        title_suffix: str,
    ) -> None:
        """Regression comparison — 3 panels: R², MSE, MAE."""
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        panels = [
            ("r2",  "r2_mean",  "r2_std",  "R²"),
            ("mse", "mse_mean", "mse_std", "MSE"),
            ("mae", "mae_mean", "mae_std", "MAE"),
        ]

        for ax, (metric_key, bl_mean, bl_std, panel_title) in zip(axes, panels):
            for i, (res, label) in enumerate(zip(results, labels)):
                records = [
                    {
                        "n_features": entry["step"],
                        metric_key:   entry.get(metric_key, float("nan")),
                    }
                    for entry in res.performance_history
                ]
                df = pd.DataFrame(records)
                color = prop_cycle[i % len(prop_cycle)]
                ax.plot(
                    df["n_features"], df[metric_key],
                    marker="o", linewidth=2, color=color, label=label,
                )

            if baseline_summary is not None:
                bx = baseline_summary["features_num"]
                ax.plot(
                    bx, baseline_summary[bl_mean],
                    marker="s", linewidth=2, color="darkorange",
                    linestyle="--", label="Random baseline (mean)",
                )
                ax.fill_between(
                    bx,
                    baseline_summary[bl_mean] - baseline_summary[bl_std],
                    baseline_summary[bl_mean] + baseline_summary[bl_std],
                    alpha=0.25, color="darkorange", label="±1 std (random)",
                )

            ax.set_title(panel_title, fontsize=13)
            ax.set_xlabel("Number of features")
            ax.set_ylabel(panel_title)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.4)

        fig.suptitle(
            f"Regression — Method Comparison{title_suffix}",
            fontsize=13, fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        plt.show()
