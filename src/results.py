from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# ANSI color helpers (fall back to plain text when colors are not supported)
# ---------------------------------------------------------------------------
def _is_color_supported() -> bool:
    """Return True when the output environment renders ANSI escape codes."""
    if sys.stdout.isatty():
        return True
    try:
        from IPython import get_ipython          # type: ignore[import-untyped]
        return get_ipython() is not None          # Jupyter / IPython kernel
    except ImportError:
        return False


def _ansi(code: str, text: str) -> str:
    """Wrap *text* in an ANSI escape sequence when the environment supports it."""
    if _is_color_supported():
        return f"\033[{code}m{text}\033[0m"
    return text


_GREEN  = "32"
_YELLOW = "33"
_RED    = "31"
_GREY   = "90"
_BOLD   = "1"


def _green(t: str)  -> str: return _ansi(_GREEN,  t)
def _yellow(t: str) -> str: return _ansi(_YELLOW, t)
def _red(t: str)    -> str: return _ansi(_RED,    t)
def _grey(t: str)   -> str: return _ansi(_GREY,   t)
def _bold(t: str)   -> str: return _ansi(_BOLD,   t)


# ---------------------------------------------------------------------------
# Numpy → native-Python recursive converter
# ---------------------------------------------------------------------------
def _to_native(obj):
    """Recursively convert numpy types to Python native types."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


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
    label : str
        Short human-readable abbreviation of the selector / configuration,
        e.g. ``"IWSS-MB"`` or ``"mRmR-P"``.  Defaults to ``""``; set by the
        selector when constructing the result, or by the caller afterwards.
    """

    selected_features: list[str]
    performance_history: list[dict]        # one dict per step where evaluation ran; each has 'step' key
    stopping_reason: str                   # "max_features_reached" | "metric_threshold_reached" | "no_features_remaining"
    n_steps: int
    final_metrics: dict | None             # last entry of performance_history, or None
    selection_time_seconds: float          # precompute + mRmR scoring loop time
    evaluation_time_seconds: float | None  # None if evaluation not performed
    task_type: str = "classification"      # "classification" | "regression"
    label: str = ""                        # short method abbreviation; "" = unset

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
        if self.label:
            lines.insert(1, f"  label             : {self.label}")
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
    # JSON serialization / deserialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Convert to a JSON-safe dictionary with a ``_type`` discriminator."""
        return _to_native({
            "_type": "SelectionResult",
            "selected_features": self.selected_features,
            "performance_history": self.performance_history,
            "stopping_reason": self.stopping_reason,
            "n_steps": self.n_steps,
            "final_metrics": self.final_metrics,
            "selection_time_seconds": self.selection_time_seconds,
            "evaluation_time_seconds": self.evaluation_time_seconds,
            "task_type": self.task_type,
            "label": self.label,
        })

    @staticmethod
    def from_dict(d: dict) -> SelectionResult:
        """Construct a ``SelectionResult`` or ``WrapperSelectionResult`` from a dict.

        Polymorphic dispatch is based on the ``_type`` discriminator field.
        """
        d = dict(d)  # shallow copy so we don't mutate the caller's dict
        type_tag = d.pop("_type", "SelectionResult")

        if type_tag == "WrapperSelectionResult":
            return WrapperSelectionResult(
                selected_features=d.get("selected_features", []),
                performance_history=d.get("performance_history", []),
                stopping_reason=d.get("stopping_reason", ""),
                n_steps=d.get("n_steps", 0),
                final_metrics=d.get("final_metrics"),
                selection_time_seconds=d.get("selection_time_seconds", 0.0),
                evaluation_time_seconds=d.get("evaluation_time_seconds"),
                task_type=d.get("task_type", "classification"),
                label=d.get("label", ""),
                filter_time_seconds=d.get("filter_time_seconds", 0.0),
                classifier_time_seconds=d.get("classifier_time_seconds", 0.0),
                test_evaluation_time_seconds=d.get("test_evaluation_time_seconds", 0.0),
                n_wrapper_evaluations=d.get("n_wrapper_evaluations", 0),
                n_candidates_pruned=d.get("n_candidates_pruned", 0),
                n_evaluations_skipped=d.get("n_evaluations_skipped", 0),
            )

        return SelectionResult(
            selected_features=d.get("selected_features", []),
            performance_history=d.get("performance_history", []),
            stopping_reason=d.get("stopping_reason", ""),
            n_steps=d.get("n_steps", 0),
            final_metrics=d.get("final_metrics"),
            selection_time_seconds=d.get("selection_time_seconds", 0.0),
            evaluation_time_seconds=d.get("evaluation_time_seconds"),
            task_type=d.get("task_type", "classification"),
            label=d.get("label", ""),
        )

    def save_as_json(self, filepath: str) -> None:
        """Serialize this result to a JSON file.

        Parent directories are created automatically if they don't exist.
        """
        parent = os.path.dirname(filepath)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)

    @staticmethod
    def load_from_json(filepath: str) -> SelectionResult:
        """Deserialize a ``SelectionResult`` (or subclass) from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        return SelectionResult.from_dict(d)

    # ------------------------------------------------------------------
    # compare_with — two-column rank-aligned diff table
    # ------------------------------------------------------------------
    def compare_with(
        self,
        other: SelectionResult,
        self_label: str | None = None,
        other_label: str | None = None,
    ) -> None:
        """Print a rank-aligned two-column comparison table for two results.

        "Rank" is **selection order** — rank 1 = first feature selected,
        rank 2 = second, etc.  A result with N features has exactly N rows;
        rows beyond N show ``—``.

        Color coding
        ------------
        - **Green**  — same feature at the same rank in both results.
        - **Yellow** — feature is present in both results but at different ranks.
        - **Red**    — feature appears in only one result (other cell shows ``—``).
        - No marker  — cell is ``—`` because the result has fewer features.

        Parameters
        ----------
        other : SelectionResult
            The result to compare against.
        self_label : str | None
            Display name for this result.  Falls back to ``self.label`` or ``"A"``.
        other_label : str | None
            Display name for *other*.  Falls back to ``other.label`` or ``"B"``.
        """
        lbl_a = self_label  or self.label  or "A"
        lbl_b = other_label or other.label or "B"

        feats_a: list[str] = self.selected_features
        feats_b: list[str] = other.selected_features

        set_a   = set(feats_a)
        set_b   = set(feats_b)
        both    = set_a & set_b        # intersection — shared features
        n_rows  = max(len(feats_a), len(feats_b))

        # Column widths — pad to longest feature name or label
        max_feat = max(
            (len(f) for f in feats_a + feats_b),
            default=8,
        )
        col_w = max(max_feat, len(lbl_a), len(lbl_b), 8)

        # Header
        rank_w = max(4, len(str(n_rows)))
        sep    = "─" * (rank_w + 2) + "┼" + "─" * (col_w + 2) + "┼" + "─" * (col_w + 2)
        header = (
            f"{'rank':>{rank_w}} │ {lbl_a:<{col_w}} │ {lbl_b:<{col_w}}"
        )
        print(f"\nComparing {lbl_a} (N={len(feats_a)}) vs {lbl_b} (N={len(feats_b)})")
        print(sep)
        print(header)
        print(sep)

        for r in range(1, n_rows + 1):
            cell_a = feats_a[r - 1] if r <= len(feats_a) else None
            cell_b = feats_b[r - 1] if r <= len(feats_b) else None

            def _fmt(feat: str | None, other_set: set[str]) -> str:
                if feat is None:
                    return _grey("—".ljust(col_w))
                if cell_a == cell_b:
                    # same feature at same rank → green
                    return _green(feat.ljust(col_w))
                if feat in both:
                    # in both results but at a different rank → yellow
                    return _yellow(feat.ljust(col_w))
                # only in one result → red
                return _red(feat.ljust(col_w))

            fa = _fmt(cell_a, set_b)
            fb = _fmt(cell_b, set_a)
            print(f"{r:>{rank_w}} │ {fa} │ {fb}")

        print(sep)
        # Legend
        print(
            _green("■") + " same rank  "
            + _yellow("■") + " diff rank  "
            + _red("■") + " one result only  "
            + _grey("—") + " beyond result length"
        )

    # ------------------------------------------------------------------
    # summary_report — multi-result per-feature stats
    # ------------------------------------------------------------------
    @staticmethod
    def summary_report(
        results: list[SelectionResult],
        labels: list[str] | None = None,
        top_n: int | None = None,
        max_labels_display: int = 5,
    ) -> pd.DataFrame:
        """Print and return a per-feature summary across multiple results.

        For every unique feature that appears in at least one result, compute:

        - ``count``      — number of results that include the feature.
        - ``pct``        — ``count / len(results) * 100``.
        - ``mean_rank``  — mean selection-order rank (1-based) across results
                           that include the feature.
        - ``min_rank``   — lowest (best) rank across those results.
        - ``max_rank``   — highest (worst) rank across those results.
        - ``selected_by``— up to ``max_labels_display`` result labels, ordered
                           by rank ascending then alphabetically, truncated
                           with ``…`` if more exist.

        Rows are sorted by ``count`` descending, then ``mean_rank`` ascending.

        Parameters
        ----------
        results : list[SelectionResult]
            Results to summarise.  Must be non-empty.
        labels : list[str] | None
            One label per result.  Falls back to ``result.label`` for each
            entry; entries still empty become ``"R1"``, ``"R2"``, … .
        top_n : int | None
            If set, return only the top-N rows (after sorting).
        max_labels_display : int
            Maximum number of result labels to show in the ``selected_by``
            column.  Default ``5``.

        Returns
        -------
        pd.DataFrame
            Columns: ``feature``, ``count``, ``pct``, ``mean_rank``,
            ``min_rank``, ``max_rank``, ``selected_by``.
        """
        if not results:
            raise ValueError("results must be non-empty")

        n = len(results)

        # Resolve labels
        resolved: list[str] = []
        for i, res in enumerate(results):
            if labels is not None and i < len(labels) and labels[i]:
                resolved.append(labels[i])
            elif res.label:
                resolved.append(res.label)
            else:
                resolved.append(f"R{i + 1}")

        # Build {feature: [(rank, label), ...]} mapping
        feat_info: dict[str, list[tuple[int, str]]] = {}
        for res, lbl in zip(results, resolved):
            for rank, feat in enumerate(res.selected_features, start=1):
                feat_info.setdefault(feat, []).append((rank, lbl))

        # Aggregate
        rows = []
        for feat, entries in feat_info.items():
            ranks = [r for r, _ in entries]
            count = len(ranks)
            pct   = count / n * 100.0
            mean_r = float(np.mean(ranks))
            min_r  = int(min(ranks))
            max_r  = int(max(ranks))

            # selected_by: sort entries by rank asc, then label asc; truncate
            sorted_entries = sorted(entries, key=lambda x: (x[0], x[1]))
            lbls = [lbl for _, lbl in sorted_entries]
            if len(lbls) <= max_labels_display:
                selected_by = ", ".join(lbls)
            else:
                selected_by = ", ".join(lbls[:max_labels_display]) + ", …"

            rows.append({
                "feature":     feat,
                "count":       count,
                "pct":         round(pct, 1),
                "mean_rank":   round(mean_r, 2),
                "min_rank":    min_r,
                "max_rank":    max_r,
                "selected_by": selected_by,
            })

        df = (
            pd.DataFrame(rows)
            .sort_values(["count", "mean_rank"], ascending=[False, True])
            .reset_index(drop=True)
        )

        if top_n is not None:
            df = df.head(top_n)

        # Print
        total_features = df["feature"].nunique()
        print(f"\nFeature summary across {n} results ({total_features} unique features):")
        print(f"Labels: {', '.join(resolved)}\n")
        try:
            display(df)          # type: ignore[name-defined]  # noqa: F821
        except NameError:
            print(df.to_string(index=False))

        return df

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


# ---------------------------------------------------------------------------
# WrapperSelectionResult dataclass
# ---------------------------------------------------------------------------
@dataclass
class WrapperSelectionResult(SelectionResult):
    """
    Extends :class:`SelectionResult` with wrapper/MB-specific timing and
    counters.

    Timing breakdown
    ----------------
    ``selection_time_seconds``
        Total selection time = ``filter_time_seconds`` + selection loop time.
        (Inherited from ``SelectionResult``; redefined to include filter step.)
    ``filter_time_seconds``
        Time for the SU init step: discretise all columns, fill target column,
        sort candidates descending by SU(f, C).  Set at ``__init__`` time.
    ``classifier_time_seconds``
        Accumulated time of all ``_cv_evaluate()`` calls made inside the
        selection loop.  Sub-component of ``selection_time_seconds``.
    ``test_evaluation_time_seconds``
        Accumulated time of all ``_test_evaluate()`` calls on the validation
        set.  These are made **outside** the timed selection loop.

    Counters
    --------
    ``n_wrapper_evaluations``
        Total ``_cv_evaluate()`` (cross-validation) calls made during
        selection.
    ``n_candidates_pruned``
        Features removed from the candidate list by Markov Blanket pruning.
        Zero if ``use_mb_pruning=False``.
    ``n_evaluations_skipped``
        Wrapper evaluations avoided because the candidate was pruned before
        its turn.  Zero if ``use_mb_pruning=False``.
    """

    filter_time_seconds: float = 0.0
    classifier_time_seconds: float = 0.0
    test_evaluation_time_seconds: float = 0.0
    n_wrapper_evaluations: int = 0
    n_candidates_pruned: int = 0
    n_evaluations_skipped: int = 0

    def __str__(self) -> str:
        base = super().__str__()
        lines = [
            base,
            f"  filter time         : {self.filter_time_seconds:.3f}s",
            f"  classifier time     : {self.classifier_time_seconds:.3f}s",
            f"  test eval time      : {self.test_evaluation_time_seconds:.3f}s",
            f"  wrapper evaluations : {self.n_wrapper_evaluations}",
            f"  candidates pruned   : {self.n_candidates_pruned}",
            f"  evaluations skipped : {self.n_evaluations_skipped}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to a JSON-safe dictionary with ``_type`` = ``WrapperSelectionResult``."""
        d = super().to_dict()
        d["_type"] = "WrapperSelectionResult"
        d["filter_time_seconds"] = _to_native(self.filter_time_seconds)
        d["classifier_time_seconds"] = _to_native(self.classifier_time_seconds)
        d["test_evaluation_time_seconds"] = _to_native(self.test_evaluation_time_seconds)
        d["n_wrapper_evaluations"] = _to_native(self.n_wrapper_evaluations)
        d["n_candidates_pruned"] = _to_native(self.n_candidates_pruned)
        d["n_evaluations_skipped"] = _to_native(self.n_evaluations_skipped)
        return d
