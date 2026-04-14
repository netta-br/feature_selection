"""Dashboard generator — standalone interactive HTML from pipeline results.

Uses Panel with a Bokeh backend.  If ``panel`` / ``bokeh`` are not installed the
public :func:`generate_dashboard` function logs a warning and returns ``""``.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..results import SelectionResult, WrapperSelectionResult
    from .config_schema import PipelineConfig

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guard
# ---------------------------------------------------------------------------
try:
    import panel as pn
    from bokeh.models import Band, ColumnDataSource, HoverTool, Legend, LegendItem
    from bokeh.palettes import Category10_10
    from bokeh.plotting import figure

    HAS_PANEL = True
except ImportError:  # pragma: no cover
    HAS_PANEL = False

# ---------------------------------------------------------------------------
# Metric definitions per task type
# ---------------------------------------------------------------------------
_CLF_METRICS = [
    ("accuracy", "Accuracy"),
    ("f1", "F1"),
]

_REG_METRICS = [
    ("r2", "R²"),
    ("mse", "MSE"),
    ("mae", "MAE"),
]

_BASELINE_CLF = {
    "accuracy": ("accuracy_mean", "accuracy_std"),
    "f1": ("f1_mean", "f1_std"),
}

_BASELINE_REG = {
    "r2": ("r2_mean", "r2_std"),
    "mse": ("mse_mean", "mse_std"),
    "mae": ("mae_mean", "mae_std"),
}


# ---------------------------------------------------------------------------
# Color helper
# ---------------------------------------------------------------------------

def _get_color_map(labels: list[str]) -> dict[str, str]:
    """Assign colors from a categorical palette to selector labels."""
    palette = list(Category10_10)
    return {label: palette[i % len(palette)] for i, label in enumerate(labels)}


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def _build_header(config: "PipelineConfig", output_dir: str) -> "pn.pane.HTML":
    """Build the dashboard header with run metadata."""
    # Try to extract timestamp from output_dir name (run_YYYYMMDD_HHMMSS)
    dir_name = os.path.basename(output_dir)
    timestamp_str = dir_name
    if dir_name.startswith("run_"):
        try:
            ts = datetime.strptime(dir_name[4:], "%Y%m%d_%H%M%S")
            timestamp_str = ts.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass

    target_names = [t.name for t in config.targets]
    target_tasks = [f"{t.name} ({t.task_type})" for t in config.targets]
    n_selectors = sum(1 for s in config.selectors if s.enabled)

    html = f"""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                color: #e0e0e0; padding: 20px 30px; border-radius: 8px;
                margin-bottom: 15px; font-family: 'Segoe UI', sans-serif;">
        <h1 style="margin: 0 0 10px 0; color: #00d4ff; font-size: 1.8em;">
            🔬 Feature Selection Dashboard
        </h1>
        <table style="border-collapse: collapse; color: #c0c0c0; font-size: 1.05em;">
            <tr><td style="padding: 3px 15px 3px 0; font-weight: bold;">Run</td>
                <td>{timestamp_str}</td></tr>
            <tr><td style="padding: 3px 15px 3px 0; font-weight: bold;">Output</td>
                <td><code>{output_dir}</code></td></tr>
            <tr><td style="padding: 3px 15px 3px 0; font-weight: bold;">Targets</td>
                <td>{', '.join(target_tasks)}</td></tr>
            <tr><td style="padding: 3px 15px 3px 0; font-weight: bold;">Selectors</td>
                <td>{n_selectors} enabled</td></tr>
            <tr><td style="padding: 3px 15px 3px 0; font-weight: bold;">Data</td>
                <td>features: <code>{config.data.features_path}</code>,
                    labels: <code>{config.data.labels_path}</code></td></tr>
            <tr><td style="padding: 3px 15px 3px 0; font-weight: bold;">Eval Data</td>
                <td>{config.preprocessing.eval_data_source}</td></tr>
        </table>
    </div>
    """
    return pn.pane.HTML(html, sizing_mode="stretch_width")


# ---------------------------------------------------------------------------
# Performance plots (Bokeh figures)
# ---------------------------------------------------------------------------

def _build_performance_plots(
    target_results: dict[str, "SelectionResult"],
    task_type: str,
    baseline: pd.DataFrame | None,
) -> "pn.Row":
    """Build performance line plots for a target."""
    metrics = _CLF_METRICS if task_type == "classification" else _REG_METRICS
    baseline_map = _BASELINE_CLF if task_type == "classification" else _BASELINE_REG

    labels = list(target_results.keys())
    color_map = _get_color_map(labels)

    figures: list = []

    for metric_key, metric_label in metrics:
        p = figure(
            title=f"{metric_label} vs Feature Count",
            x_axis_label="Number of Features",
            y_axis_label=metric_label,
            width=420,
            height=320,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            sizing_mode="stretch_width",
        )

        legend_items: list[LegendItem] = []

        # Plot each selector's performance line
        for label, result in target_results.items():
            steps = []
            values = []
            for entry in result.performance_history:
                step = entry.get("step")
                val = entry.get(metric_key)
                if step is not None and val is not None:
                    steps.append(step)
                    values.append(val)

            if not steps:
                continue

            source = ColumnDataSource(data=dict(
                x=steps,
                y=values,
                label=[label] * len(steps),
            ))

            line = p.line(
                "x", "y", source=source,
                line_width=2, color=color_map[label],
            )
            circle = p.scatter(
                "x", "y", source=source,
                size=6, color=color_map[label],
            )
            legend_items.append(LegendItem(label=label, renderers=[line, circle]))

            # Hover tooltip
            hover = HoverTool(
                renderers=[circle],
                tooltips=[
                    ("Selector", "@label"),
                    ("Features", "@x"),
                    (metric_label, "@y{0.0000}"),
                ],
            )
            p.add_tools(hover)

        # Baseline overlay
        if baseline is not None and metric_key in baseline_map:
            mean_col, std_col = baseline_map[metric_key]
            if mean_col in baseline.columns and std_col in baseline.columns:
                bl_x = baseline["features_num"].tolist()
                bl_mean = baseline[mean_col].tolist()
                bl_std = baseline[std_col].tolist()
                bl_upper = [m + s for m, s in zip(bl_mean, bl_std)]
                bl_lower = [m - s for m, s in zip(bl_mean, bl_std)]

                bl_source = ColumnDataSource(data=dict(
                    x=bl_x,
                    y=bl_mean,
                    upper=bl_upper,
                    lower=bl_lower,
                ))

                bl_line = p.line(
                    "x", "y", source=bl_source,
                    line_width=2, color="darkorange", line_dash="dashed",
                )
                band = Band(
                    base="x", upper="upper", lower="lower",
                    source=bl_source, level="underlay",
                    fill_alpha=0.2, fill_color="darkorange",
                    line_width=0,
                )
                p.add_layout(band)
                legend_items.append(
                    LegendItem(label="Baseline (mean ± std)", renderers=[bl_line])
                )

        # Add legend — scrollable via CSS so all selectors are visible
        if legend_items:
            legend = Legend(items=legend_items, location="top_left")
            legend.click_policy = "hide"
            legend.label_text_font_size = "11pt"
            legend.stylesheets.append("""
:host {
    max-height: 280px;
    overflow-y: auto;
}
""")
            p.add_layout(legend, "right")

        # Font sizes for Bokeh figures
        p.title.text_font_size = "13pt"
        p.xaxis.axis_label_text_font_size = "12pt"
        p.yaxis.axis_label_text_font_size = "12pt"
        p.xaxis.major_label_text_font_size = "11pt"
        p.yaxis.major_label_text_font_size = "11pt"

        p.xaxis.minor_tick_line_color = None
        p.grid.grid_line_alpha = 0.4
        figures.append(p)

    bokeh_panes = [pn.pane.Bokeh(fig, sizing_mode="stretch_width") for fig in figures]
    return pn.Column(*bokeh_panes, sizing_mode="stretch_width")


# ---------------------------------------------------------------------------
# Metrics summary table
# ---------------------------------------------------------------------------

def _build_metrics_table(
    target_results: dict[str, "SelectionResult"],
    task_type: str,
) -> "pn.pane.HTML":
    """Build the metrics summary table with a k-features toggle.

    A dropdown lets the user pick a feature count k; the table then shows each
    selector's metrics from the last ``performance_history`` entry where
    ``step <= k``.  The "Max" option (default) shows each selector's final
    evaluation regardless of step count.

    Panel embeds HTML pane content via innerHTML inside a shadow DOM, which means
    <script> tags never execute.  All interactivity uses inline onchange/onclick
    attributes.  Per-selector history is stored in a hidden ``<div>``'s
    ``data-history`` attribute as JSON so the onchange handler can read it
    without needing a ``<script>`` block.
    """
    import json as _json
    import uuid
    metrics = _CLF_METRICS if task_type == "classification" else _REG_METRICS
    table_id = f"metrics_tbl_{uuid.uuid4().hex[:8]}"
    hist_id  = f"hist_{table_id}"
    sel_id   = f"k_sel_{table_id}"
    desc_id  = f"k_desc_{table_id}"

    # ------------------------------------------------------------------
    # Collect all step values to build dropdown options
    # ------------------------------------------------------------------
    all_steps: list[int] = sorted({
        int(entry["step"])
        for result in target_results.values()
        for entry in result.performance_history
        if entry.get("step") is not None
    })

    # ------------------------------------------------------------------
    # Serialize performance_history for all selectors
    # Only keep keys needed for the toggle: step + metric keys
    # ------------------------------------------------------------------
    metric_keys = [mk for mk, _ in metrics]
    history_data: dict[str, list[dict]] = {}
    for label, result in target_results.items():
        history_data[label] = [
            {k: v for k, v in entry.items() if k in ("step", *metric_keys)}
            for entry in result.performance_history
            if entry.get("step") is not None
        ]

    # Encode history as Base64 so it survives Panel's html.escape() →
    # DOMParser.textContent → innerHTML decode chain without any quote
    # or entity conflicts.  JS reads it back with atob() + JSON.parse().
    import base64 as _b64
    history_json = _json.dumps(history_data, separators=(",", ":"))
    history_b64 = _b64.b64encode(history_json.encode("utf-8")).decode("ascii")

    # ------------------------------------------------------------------
    # Build initial rows (Max = last entry per selector)
    # ------------------------------------------------------------------
    def _row_html(label: str, entry: dict, stopping_reason: str) -> str:
        step = entry.get("step", "—")
        cells = (
            f'<td data-col="0" data-val="{label}">{label}</td>'
            f'<td data-col="1" data-val="{step}">{step}</td>'
            f'<td data-col="2" data-val="{stopping_reason}">{stopping_reason}</td>'
        )
        for i, (mk, _) in enumerate(metrics):
            val = entry.get(mk)
            if val is not None:
                cells += f'<td data-col="{3+i}" data-val="{val}">{val:.4f}</td>'
            else:
                cells += f'<td data-col="{3+i}" data-val="-1">—</td>'
        return f'<tr data-selector="{label}">{cells}</tr>'

    rows_html: list[str] = []
    for label, result in target_results.items():
        last = result.performance_history[-1] if result.performance_history else {}
        rows_html.append(_row_html(label, last, result.stopping_reason))

    # ------------------------------------------------------------------
    # Inline sort JS (reusable snippet — stored as a Python string and
    # referenced in both th onclick and the onchange re-sort call)
    # ------------------------------------------------------------------
    _th_style = "padding: 6px 10px; text-align: left; cursor: pointer; user-select: none;"

    # JS helper: sort the tbody rows by column col, direction asc.
    # Used both in _th() onclick and reproduced after each k-change.
    _sort_body_js = (
        "var tbody=tbl.querySelector('tbody');"
        "var rows=Array.from(tbody.querySelectorAll('tr'));"
        "rows.sort(function(a,b){"
        "  var ca=a.querySelectorAll('td')[sortCol];"
        "  var cb=b.querySelectorAll('td')[sortCol];"
        "  var va=ca?ca.getAttribute('data-val'):'';"
        "  var vb=cb?cb.getAttribute('data-val'):'';"
        "  var na=parseFloat(va),nb=parseFloat(vb);"
        "  if(!isNaN(na)&&!isNaN(nb))return sortAsc?na-nb:nb-na;"
        "  return sortAsc?va.localeCompare(vb):vb.localeCompare(va);"
        "});"
        "rows.forEach(function(r){tbody.appendChild(r);});"
    )

    def _th(col: int, label: str) -> str:
        js = (
            "(function(th){"
            f"var tblId='{table_id}';"
            "var root=th.getRootNode();"
            "var tbl=root.getElementById?root.getElementById(tblId):root.querySelector('#'+tblId);"
            "if(!tbl)return;"
            "var sortCol=parseInt(th.getAttribute('data-col'));"
            "var cur=th.getAttribute('data-sort');"
            "var sortAsc=cur!=='asc';"
            # store active sort state on the table element for the k-toggle to reuse
            "tbl._kSortCol=sortCol;tbl._kSortAsc=sortAsc;"
            "tbl.querySelectorAll('thead th[data-col]').forEach(function(h){"
            "  h.setAttribute('data-sort','');"
            "  var base=h.getAttribute('data-label');"
            "  h.textContent=base+' \u2195';"
            "});"
            "th.setAttribute('data-sort',sortAsc?'asc':'desc');"
            "th.textContent=th.getAttribute('data-label')+(sortAsc?' \u2191':' \u2193');"
            + _sort_body_js +
            "})(this)"
        )
        return (
            f'<th data-col="{col}" data-label="{label}" data-sort="" '
            f'style="{_th_style}" onclick="{js}">{label} \u2195</th>'
        )

    metric_headers = "".join(_th(3 + i, ml) for i, (_, ml) in enumerate(metrics))

    # ------------------------------------------------------------------
    # Dropdown options HTML
    # ------------------------------------------------------------------
    options_html = '<option value="max" selected>Max</option>'
    for s in all_steps:
        options_html += f'<option value="{s}">{s}</option>'

    # ------------------------------------------------------------------
    # onchange JS for the k-features dropdown
    # Reads history JSON from the hidden div, finds the last entry with
    # step <= k for each selector row, re-renders cells, then re-sorts.
    # ------------------------------------------------------------------
    onchange_js = (
        "(function(sel){"
        f"var tblId='{table_id}';"
        f"var histId='{hist_id}';"
        f"var descId='{desc_id}';"
        "var root=sel.getRootNode();"
        "var tbl=root.getElementById?root.getElementById(tblId):root.querySelector('#'+tblId);"
        "var histEl=root.getElementById?root.getElementById(histId):root.querySelector('#'+histId);"
        "var descEl=root.getElementById?root.getElementById(descId):root.querySelector('#'+descId);"
        "if(!tbl||!histEl)return;"
        "var k=sel.value==='max'?Infinity:parseInt(sel.value);"
        "if(descEl){"
        "  descEl.textContent=sel.value==='max'?"
        "    'Showing final metrics per selector':"
        "    ('Showing metrics at \u2264 '+k+' features');"
        "}"
        # decode Base64 history from data attribute, then JSON.parse
        "var hist=JSON.parse(atob(histEl.getAttribute('data-history')));"
        # iterate over rows
        "var rows=tbl.querySelectorAll('tbody tr');"
        "rows.forEach(function(row){"
        "  var label=row.getAttribute('data-selector');"
        "  var entries=hist[label]||[];"
        # find last entry with step <= k (or last entry if k=Infinity)
        "  var chosen=null;"
        "  for(var i=0;i<entries.length;i++){"
        "    if(entries[i].step<=k)chosen=entries[i];"
        "  }"
        "  var tds=row.querySelectorAll('td');"
        # col 1 = N Features
        "  if(tds[1]){"
        "    var step=chosen?chosen.step:'—';"
        "    tds[1].textContent=step;"
        "    tds[1].setAttribute('data-val',chosen?chosen.step:-1);"
        "  }"
        # cols 3..3+n_metrics-1 = metric values — use single-quoted JS array
        # to avoid double-quotes breaking the onchange="..." HTML attribute
        "  var metricKeys=[" + ",".join(f"'{k}'" for k in metric_keys) + "];"
        "  for(var m=0;m<metricKeys.length;m++){"
        "    var td=tds[3+m];"
        "    if(!td)continue;"
        "    var v=chosen?chosen[metricKeys[m]]:null;"
        "    if(v!=null&&v!==undefined){"
        "      td.textContent=v.toFixed(4);"
        "      td.setAttribute('data-val',v);"
        "    } else {"
        "      td.textContent='\u2014';"
        "      td.setAttribute('data-val',-1);"
        "    }"
        "  }"
        "});"
        # re-apply active sort if any
        "var sortCol=tbl._kSortCol;"
        "var sortAsc=tbl._kSortAsc;"
        "if(sortCol!==undefined&&sortCol!==null){"
        + _sort_body_js +
        "}"
        "})(this)"
    )

    # ------------------------------------------------------------------
    # Assemble final HTML
    # ------------------------------------------------------------------
    html = f"""
    <div style="margin: 10px 0;">
        <h3 style="margin-bottom: 8px;">&#x1F4CA; Final Metrics Summary</h3>
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;
                    background:#f7f7f7; border-radius:20px; padding:7px 16px;
                    border:1px solid #ddd; width:fit-content;">
            <label for="{sel_id}"
                   style="font-weight:bold; font-size:0.95em; white-space:nowrap;">
                &#x1F522; Compare at k features:
            </label>
            <select id="{sel_id}"
                    onchange="{onchange_js}"
                    style="padding:3px 8px; border:1px solid #bbb; border-radius:6px;
                           font-size:0.95em; background:#fff; cursor:pointer;">
                {options_html}
            </select>
            <span id="{desc_id}"
                  style="color:#888; font-size:0.88em; white-space:nowrap;">
                Showing final metrics per selector
            </span>
        </div>
        <div id="{hist_id}" data-history="{history_b64}" style="display:none;"></div>
        <table id="{table_id}" style="border-collapse: collapse; width: 100%; font-size: 1.0em;">
            <thead>
                <tr style="background: #f0f0f0; border-bottom: 2px solid #ccc;">
                    {_th(0, "Selector")}
                    {_th(1, "N Features")}
                    {_th(2, "Stopping Reason")}
                    {metric_headers}
                </tr>
            </thead>
            <tbody>
                {''.join(rows_html)}
            </tbody>
        </table>
    </div>
    """
    # Apply consistent cell styling
    html = html.replace("<td ", '<td style="padding: 5px 10px; border-bottom: 1px solid #eee;" ')
    return pn.pane.HTML(html, sizing_mode="stretch_width")


# ---------------------------------------------------------------------------
# Selector details accordion
# ---------------------------------------------------------------------------

def _build_selector_details(
    target_results: dict[str, "SelectionResult"],
    config: "PipelineConfig",
) -> "pn.Accordion":
    """Build accordion with per-selector detail panels."""
    from ..results import WrapperSelectionResult

    panels: list[tuple[str, pn.pane.HTML]] = []

    for label, result in target_results.items():
        # Selected features list
        features_list = "".join(
            f"<li>{feat}</li>" for feat in result.selected_features
        )

        # Configuration section (from pipeline config)
        config_section = ""
        sel_cfg = next((s for s in config.selectors if s.label == label), None)
        if sel_cfg:
            params_rows = "".join(
                f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in sel_cfg.params.items()
            )
            config_section = f"""
            <h4 style="margin: 10px 0 5px 0;">⚙ Configuration</h4>
            <table style="border-collapse: collapse; font-size: 1.0em;">
                <tr><td>Type</td><td>{sel_cfg.type}</td></tr>
                {params_rows}
            </table>
            """

        timing_rows = f"""
        <tr><td>Selection time</td><td>{result.selection_time_seconds:.3f}s</td></tr>
        """
        if result.evaluation_time_seconds is not None:
            timing_rows += f"""
            <tr><td>Evaluation time</td><td>{result.evaluation_time_seconds:.3f}s</td></tr>
            """

        wrapper_section = ""
        if isinstance(result, WrapperSelectionResult):
            timing_rows += f"""
            <tr><td>Filter time</td><td>{result.filter_time_seconds:.3f}s</td></tr>
            <tr><td>Classifier time</td><td>{result.classifier_time_seconds:.3f}s</td></tr>
            <tr><td>Test evaluation time</td><td>{result.test_evaluation_time_seconds:.3f}s</td></tr>
            """
            wrapper_section = f"""
            <h4 style="margin: 10px 0 5px 0;">🔧 Wrapper Statistics</h4>
            <table style="border-collapse: collapse; font-size: 1.0em;">
                <tr><td style="padding: 3px 10px;">Wrapper evaluations</td>
                    <td style="padding: 3px 10px;">{result.n_wrapper_evaluations}</td></tr>
                <tr><td style="padding: 3px 10px;">Candidates pruned</td>
                    <td style="padding: 3px 10px;">{result.n_candidates_pruned}</td></tr>
                <tr><td style="padding: 3px 10px;">Evaluations skipped</td>
                    <td style="padding: 3px 10px;">{result.n_evaluations_skipped}</td></tr>
            </table>
            """

        html = f"""
        <div style="padding: 10px;">
            <h4 style="margin: 0 0 5px 0;">📋 Selected Features ({len(result.selected_features)})</h4>
            <ol style="margin: 0; padding-left: 25px; font-size: 1.0em; columns: 2;">
                {features_list}
            </ol>
            {config_section}
            <h4 style="margin: 10px 0 5px 0;">⏱ Timing Breakdown</h4>
            <table style="border-collapse: collapse; font-size: 1.0em;">
                {timing_rows}
            </table>
            {wrapper_section}
        </div>
        """
        html = html.replace(
            "<td>", '<td style="padding: 3px 10px; border-bottom: 1px solid #eee;">'
        )
        panels.append((f"🏷 {label}", pn.pane.HTML(html, sizing_mode="stretch_width")))

    accordion = pn.Accordion(*panels, sizing_mode="stretch_width")
    return accordion


# ---------------------------------------------------------------------------
# Per-target tab
# ---------------------------------------------------------------------------

def _build_target_tab(
    target_name: str,
    task_type: str,
    target_results: dict[str, "SelectionResult"],
    baseline: pd.DataFrame | None,
    config: "PipelineConfig",
) -> "pn.Column":
    """Build a complete tab for one target variable."""
    perf_plots = _build_performance_plots(target_results, task_type, baseline)
    metrics_table = _build_metrics_table(target_results, task_type)
    selector_details = _build_selector_details(target_results, config)

    return pn.Column(
        pn.pane.HTML(
            f"<h2 style='margin: 5px 0;'>Target: {target_name} "
            f"<span style='font-size: 0.7em; color: #888;'>({task_type})</span></h2>",
            sizing_mode="stretch_width",
        ),
        pn.pane.HTML("<h3>📈 Performance Curves</h3>", sizing_mode="stretch_width"),
        perf_plots,
        metrics_table,
        pn.pane.HTML("<h3>🔍 Selector Details</h3>", sizing_mode="stretch_width"),
        selector_details,
        sizing_mode="stretch_width",
    )


# ---------------------------------------------------------------------------
# Comparisons section
# ---------------------------------------------------------------------------

def _build_comparisons_section(
    config: "PipelineConfig",
    results: dict[str, "SelectionResult"],
    baselines: dict[str, pd.DataFrame],
) -> "pn.Column":
    """Build the comparisons section from config."""
    from ..results import SelectionResult as SR

    if not config.comparisons:
        return pn.Column(sizing_mode="stretch_width")

    components: list = [
        pn.pane.HTML(
            "<h2 style='margin: 15px 0 10px 0;'>🔀 Comparisons</h2>",
            sizing_mode="stretch_width",
        ),
    ]

    for comp in config.comparisons:
        comp_header = pn.pane.HTML(
            f"<h3 style='margin: 10px 0 5px 0;'>{comp.label} "
            f"<span style='font-size: 0.7em; color: #888;'>({comp.type})</span></h3>",
            sizing_mode="stretch_width",
        )
        components.append(comp_header)

        if comp.type == "compare_results":
            # Multi-line performance plot for specified selectors on a target
            comp_results: dict[str, "SelectionResult"] = {}
            task_type = "classification"
            for sel_label in comp.selectors:
                key = f"{sel_label}__{comp.target}"
                if key in results:
                    comp_results[sel_label] = results[key]
                    task_type = results[key].task_type

            if comp_results:
                baseline = baselines.get(comp.target) if comp.include_baseline else None
                plots = _build_performance_plots(comp_results, task_type, baseline)
                components.append(plots)
            else:
                components.append(
                    pn.pane.HTML("<p><em>No matching results found.</em></p>")
                )

        elif comp.type == "compare_with":
            # Feature diff table (two selectors)
            if len(comp.selectors) == 2:
                key_a = f"{comp.selectors[0]}__{comp.target}"
                key_b = f"{comp.selectors[1]}__{comp.target}"
                if key_a in results and key_b in results:
                    res_a = results[key_a]
                    res_b = results[key_b]
                    diff_html = _build_feature_diff_table(
                        res_a, res_b, comp.selectors[0], comp.selectors[1]
                    )
                    components.append(diff_html)
                else:
                    components.append(
                        pn.pane.HTML("<p><em>One or both results not found.</em></p>")
                    )
            else:
                components.append(
                    pn.pane.HTML(
                        "<p><em>compare_with requires exactly 2 selectors.</em></p>"
                    )
                )

        elif comp.type == "summary_report":
            # Feature frequency table
            comp_results_list = []
            comp_labels_list = []
            for sel_label in comp.selectors:
                key = f"{sel_label}__{comp.target}"
                if key in results:
                    comp_results_list.append(results[key])
                    comp_labels_list.append(sel_label)

            if comp_results_list:
                summary_df = SR.summary_report(comp_results_list, comp_labels_list)
                summary_html = _dataframe_to_html(summary_df, "Feature Summary Report")
                components.append(summary_html)
            else:
                components.append(
                    pn.pane.HTML("<p><em>No matching results found.</em></p>")
                )

    return pn.Column(*components, sizing_mode="stretch_width")


def _build_feature_diff_table(
    result_a: "SelectionResult",
    result_b: "SelectionResult",
    label_a: str,
    label_b: str,
) -> "pn.pane.HTML":
    """Build an HTML feature diff table between two selectors."""
    feats_a = result_a.selected_features
    feats_b = result_b.selected_features
    set_a = set(feats_a)
    set_b = set(feats_b)
    both = set_a & set_b
    n_rows = max(len(feats_a), len(feats_b))

    rows_html: list[str] = []
    for r in range(n_rows):
        cell_a = feats_a[r] if r < len(feats_a) else None
        cell_b = feats_b[r] if r < len(feats_b) else None

        def _style(feat: str | None, other_set: set[str]) -> str:
            if feat is None:
                return '<td style="padding: 4px 8px; color: #999;">—</td>'
            if cell_a == cell_b:
                return f'<td style="padding: 4px 8px; background: #d4edda;">{feat}</td>'
            if feat in both:
                return f'<td style="padding: 4px 8px; background: #fff3cd;">{feat}</td>'
            return f'<td style="padding: 4px 8px; background: #f8d7da;">{feat}</td>'

        rows_html.append(
            f"<tr><td style='padding: 4px 8px; font-weight: bold;'>{r + 1}</td>"
            f"{_style(cell_a, set_b)}{_style(cell_b, set_a)}</tr>"
        )

    only_a = set_a - set_b
    only_b = set_b - set_a

    html = f"""
    <div style="margin: 10px 0;">
        <p style="font-size: 1.0em; margin-bottom: 5px;">
            Shared: <strong>{len(both)}</strong> &nbsp;|&nbsp;
            Only in {label_a}: <strong>{len(only_a)}</strong> &nbsp;|&nbsp;
            Only in {label_b}: <strong>{len(only_b)}</strong>
        </p>
        <table style="border-collapse: collapse; font-size: 0.95em;">
            <thead>
                <tr style="background: #f0f0f0; border-bottom: 2px solid #ccc;">
                    <th style="padding: 5px 8px;">Rank</th>
                    <th style="padding: 5px 8px;">{label_a} (N={len(feats_a)})</th>
                    <th style="padding: 5px 8px;">{label_b} (N={len(feats_b)})</th>
                </tr>
            </thead>
            <tbody>{''.join(rows_html)}</tbody>
        </table>
        <p style="font-size: 0.9em; margin-top: 5px; color: #666;">
            🟩 same rank &nbsp; 🟨 different rank &nbsp; 🟥 one result only
        </p>
    </div>
    """
    return pn.pane.HTML(html, sizing_mode="stretch_width")


def _dataframe_to_html(df: pd.DataFrame, title: str) -> "pn.pane.HTML":
    """Render a DataFrame as a styled HTML table pane."""
    table_html = df.to_html(index=False, border=0, classes="summary-table")
    html = f"""
    <div style="margin: 10px 0;">
        <style>
            .summary-table {{
                border-collapse: collapse; width: 100%; font-size: 0.95em;
            }}
            .summary-table th {{
                background: #f0f0f0; padding: 5px 10px; text-align: left;
                border-bottom: 2px solid #ccc;
            }}
            .summary-table td {{
                padding: 4px 10px; border-bottom: 1px solid #eee;
            }}
        </style>
        {table_html}
    </div>
    """
    return pn.pane.HTML(html, sizing_mode="stretch_width")


# ---------------------------------------------------------------------------
# Timing overview
# ---------------------------------------------------------------------------

def _build_timing_overview(
    results: dict[str, "SelectionResult"],
) -> "pn.pane.Bokeh":
    """Build a horizontal bar chart ranking selectors by time per feature."""
    if not results:
        return pn.pane.Bokeh(figure(title="No results"), sizing_mode="stretch_width")

    # Collect timing data
    entries: list[tuple[str, float, float, int, str]] = []
    for key, result in results.items():
        selector_type = "wrapper" if hasattr(result, "filter_time_seconds") and result.filter_time_seconds > 0 else "mrmr"
        n_features = result.n_steps if result.n_steps > 0 else 1  # avoid division by zero
        time_per_feat = result.selection_time_seconds / n_features
        entries.append((key, time_per_feat, result.selection_time_seconds, result.n_steps, selector_type))

    # Sort by time-per-feature ascending (longest at top for horizontal bar)
    entries.sort(key=lambda x: x[1])

    labels = [e[0] for e in entries]
    times_per_feat = [e[1] for e in entries]
    total_times = [e[2] for e in entries]
    n_features = [e[3] for e in entries]
    types = [e[4] for e in entries]
    colors = ["#e74c3c" if t == "wrapper" else "#3498db" for t in types]

    source = ColumnDataSource(data=dict(
        labels=labels,
        times_per_feat=times_per_feat,
        total_times=total_times,
        n_features=n_features,
        types=types,
        colors=colors,
    ))

    p = figure(
        title="⏱ Timing Overview — Selection Time per Feature (seconds)",
        y_range=labels,
        x_axis_label="Time per Feature (seconds)",
        height=max(200, 30 * len(labels) + 80),
        width=700,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        sizing_mode="stretch_width",
    )

    p.hbar(
        y="labels",
        right="times_per_feat",
        height=0.6,
        color="colors",
        source=source,
        alpha=0.85,
    )

    hover = HoverTool(tooltips=[
        ("Selector", "@labels"),
        ("Time/Feature", "@times_per_feat{0.000}s"),
        ("Total Time", "@total_times{0.000}s"),
        ("N Features", "@n_features"),
        ("Type", "@types"),
    ])
    p.add_tools(hover)

    # Font sizes for Bokeh figures
    p.title.text_font_size = "13pt"
    p.xaxis.axis_label_text_font_size = "12pt"
    p.yaxis.axis_label_text_font_size = "12pt"
    p.xaxis.major_label_text_font_size = "11pt"
    p.yaxis.major_label_text_font_size = "11pt"
    p.grid.grid_line_alpha = 0.3

    # Add legend manually with colored patches
    legend_html = """
    <div style="font-size: 0.95em; margin: 5px 0;">
        <span style="display: inline-block; width: 12px; height: 12px;
              background: #3498db; margin-right: 4px; vertical-align: middle;"></span>
        mRmR &nbsp;&nbsp;
        <span style="display: inline-block; width: 12px; height: 12px;
              background: #e74c3c; margin-right: 4px; vertical-align: middle;"></span>
        Wrapper
    </div>
    """

    return pn.Column(
        pn.pane.Bokeh(p, sizing_mode="stretch_width"),
        pn.pane.HTML(legend_html, sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
    )


# ---------------------------------------------------------------------------
# Wrapper evaluations chart
# ---------------------------------------------------------------------------

def _build_wrapper_evaluations_chart(
    results: dict[str, "SelectionResult"],
) -> "pn.Column":
    """Build a horizontal bar chart showing wrapper evaluator calls vs skipped."""
    from ..results import WrapperSelectionResult

    # Filter to wrapper results only
    wrapper_results: dict[str, WrapperSelectionResult] = {}
    for key, result in results.items():
        if isinstance(result, WrapperSelectionResult):
            wrapper_results[key] = result

    if not wrapper_results:
        return pn.Column(sizing_mode="stretch_width")

    labels = list(wrapper_results.keys())
    evaluations = [r.n_wrapper_evaluations for r in wrapper_results.values()]
    skipped = [r.n_evaluations_skipped for r in wrapper_results.values()]
    pruned = [r.n_candidates_pruned for r in wrapper_results.values()]

    source_eval = ColumnDataSource(data=dict(
        labels=labels,
        evaluations=evaluations,
        skipped=skipped,
        pruned=pruned,
    ))

    source_skip = ColumnDataSource(data=dict(
        labels=labels,
        evaluations=evaluations,
        skipped=skipped,
        pruned=pruned,
    ))

    p = figure(
        title="🔢 Wrapper Evaluator Calls",
        y_range=labels,
        x_axis_label="Count",
        height=max(200, 40 * len(labels) + 80),
        width=700,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        sizing_mode="stretch_width",
    )

    # Evaluations performed (blue bars)
    eval_bar = p.hbar(
        y="labels",
        right="evaluations",
        height=0.35,
        color="#3498db",
        source=source_eval,
        alpha=0.85,
        legend_label="Evaluations Performed",
    )

    # Evaluations skipped (red bars) — offset vertically
    from bokeh.transform import dodge
    skip_bar = p.hbar(
        y=dodge("labels", 0.35, range=p.y_range),
        right="skipped",
        height=0.35,
        color="#e74c3c",
        source=source_skip,
        alpha=0.85,
        legend_label="Evaluations Skipped (pruned)",
    )

    hover_eval = HoverTool(
        renderers=[eval_bar],
        tooltips=[
            ("Selector", "@labels"),
            ("Evaluations", "@evaluations"),
            ("Skipped", "@skipped"),
            ("Pruned Candidates", "@pruned"),
        ],
    )
    hover_skip = HoverTool(
        renderers=[skip_bar],
        tooltips=[
            ("Selector", "@labels"),
            ("Evaluations", "@evaluations"),
            ("Skipped", "@skipped"),
            ("Pruned Candidates", "@pruned"),
        ],
    )
    p.add_tools(hover_eval)
    p.add_tools(hover_skip)

    p.legend.location = "top_right"
    p.legend.click_policy = "hide"
    p.legend.label_text_font_size = "11pt"

    # Font sizes for Bokeh figures
    p.title.text_font_size = "13pt"
    p.xaxis.axis_label_text_font_size = "12pt"
    p.yaxis.axis_label_text_font_size = "12pt"
    p.xaxis.major_label_text_font_size = "11pt"
    p.yaxis.major_label_text_font_size = "11pt"
    p.grid.grid_line_alpha = 0.3

    return pn.Column(
        pn.pane.Bokeh(p, sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_dashboard(
    output_dir: str,
    results: dict[str, "SelectionResult"],
    baselines: dict[str, pd.DataFrame],
    config: "PipelineConfig",
) -> str:
    """Generate a standalone interactive HTML dashboard.

    Parameters
    ----------
    output_dir : str
        Directory to write ``dashboard.html`` into.
    results : dict[str, SelectionResult]
        Keyed ``"{label}__{target}"``.
    baselines : dict[str, pd.DataFrame]
        Keyed by target name.
    config : PipelineConfig
        The pipeline configuration used for the run.

    Returns
    -------
    str
        Path to the saved ``dashboard.html``, or ``""`` if generation was
        skipped.
    """
    if not HAS_PANEL:
        log.warning(
            "Dashboard generation skipped: panel/bokeh not installed. "
            "Install with: pip install panel bokeh"
        )
        return ""

    log.info("Generating interactive dashboard …")

    pn.extension(sizing_mode="stretch_width")

    # ---- Header ----------------------------------------------------------
    header = _build_header(config, output_dir)

    # ---- Per-target tabs -------------------------------------------------
    tabs_list: list[tuple[str, pn.Column]] = []

    for target_cfg in config.targets:
        target_name = target_cfg.name
        task_type = target_cfg.task_type

        # Gather results for this target
        target_results: dict[str, "SelectionResult"] = {}
        for key, result in results.items():
            # key format: "{label}__{target}"
            parts = key.rsplit("__", 1)
            if len(parts) == 2 and parts[1] == target_name:
                target_results[parts[0]] = result

        if not target_results:
            log.warning("No results for target '%s', skipping tab", target_name)
            continue

        baseline = baselines.get(target_name)
        tab_content = _build_target_tab(target_name, task_type, target_results, baseline, config)
        tabs_list.append((f"🎯 {target_name}", tab_content))

    target_tabs = pn.Tabs(*tabs_list, sizing_mode="stretch_width") if tabs_list else pn.Column()

    # ---- Comparisons section ---------------------------------------------
    comparisons_section = _build_comparisons_section(config, results, baselines)

    # ---- Timing overview -------------------------------------------------
    timing_header = pn.pane.HTML(
        "<h2 style='margin: 15px 0 10px 0;'>⏱ Timing Overview</h2>",
        sizing_mode="stretch_width",
    )
    timing_overview = _build_timing_overview(results)

    # ---- Wrapper evaluations chart ---------------------------------------
    wrapper_evals = _build_wrapper_evaluations_chart(results)

    # ---- Assemble layout -------------------------------------------------
    layout = pn.Column(
        header,
        target_tabs,
        comparisons_section,
        timing_header,
        timing_overview,
        wrapper_evals,
        sizing_mode="stretch_width",
    )

    # ---- Save ------------------------------------------------------------
    filepath = os.path.join(output_dir, "dashboard.html")
    os.makedirs(output_dir, exist_ok=True)
    layout.save(filepath, embed=True, title="Feature Selection Dashboard")
    log.info("Dashboard saved: %s", filepath)
    return filepath
