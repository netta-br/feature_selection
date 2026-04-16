# Feature Selection Pipeline

A fully configurable, reproducible pipeline that runs multiple feature-selection
experiments on arbitrary datasets, compares results, and produces an interactive
standalone HTML dashboard — all driven by a single JSON configuration file.

## Features

- **Multiple selection methods**: mRmR filter (Pearson, MI, RF, F-statistic) and IWSS/SFS wrapper methods
- **Generic interface**: Works with any samples × features tabular dataset
- **Auto-transpose**: Handles both samples×features and features×samples CSV formats
- **Configurable**: Single JSON file describes the entire experiment
- **Reproducible**: Timestamped output directories with config copies
- **Interactive dashboard**: Standalone HTML with performance plots and selector comparisons
- **CLI & API**: Run from command line or import as Python library

## Quick Start

### Installation

```bash
uv sync
```

### Run with config file

```bash
uv run python -m feature_selection.src.pipeline.cli run --config feature_selection/config/pipeline_config.json
```

### Quick run (no config file)

```bash
uv run python -m feature_selection.src.pipeline.cli run \
  --data ./data/features.csv \
  --labels ./data/labels.csv \
  --target my_target \
  --task classification \
  --mRmR-P --IWSS-MB
```

### Validate a config

```bash
uv run python -m feature_selection.src.pipeline.cli validate --config feature_selection/config/pipeline_config.json
```

## Project Structure

```
feature_selection/
├── config/                        # Example and template configurations
├── data/                          # Dataset directory (user-provided)
├── docs/                          # Additional documentation
├── examples/                      # Jupyter notebooks showing API usage
├── src/
│   ├── baseline/                  # Random-feature baselines
│   ├── evaluation/                # Post-selection evaluators
│   │   └── evaluator.py           # Generic Evaluator class
│   ├── filter/                    # mRmR filter methods
│   ├── pipeline/                  # Experiment orchestration and CLI
│   │   ├── cache.py               # Selector result hash-based caching
│   │   ├── dashboard.py           # Interactive HTML dashboard generator
│   │   └── runner.py              # Pipeline orchestrator
│   ├── wrapper/                   # Wrapper methods
│   └── results.py                 # Result format classes
tests/                             # Integration and unit tests
tools/
└── config_editor.html             # Standalone HTML config editor
pyproject.toml                     # Project metadata and dependencies
```

## Configuration

The pipeline is driven by a single JSON file. See `feature_selection/config/pipeline_config.json` for a
complete example. The top-level sections are:

| Section          | Description |
|------------------|-------------|
| `data`           | Paths to features and labels CSVs, sample ID column, auto-transpose flag, fill-NA value |
| `preprocessing`  | Random seed, train/val/test split percentages, eval data source |
| `targets`        | List of target variables, each with a name and task type (`classification` or `regression`) |
| `baseline`       | Whether to run random-feature baselines, number of runs, and eval feature counts |
| `selectors`      | List of selector configs — each specifies a label, type (`mrmr` or `wrapper`), params, and target variables |
| `evaluators`     | List of evaluators (e.g., `logistic_regression`) that run to grade the selected feature sets |
| `comparisons`    | Post-run comparison operations: `compare_results`, `compare_with`, `summary_report` |
| `score_matrix`   | Hyperparameters for score computation (random forest, mutual information) |
| `output`         | Output base directory, dashboard creation, result JSON saves, and caching toggle (`fetch_past_results`) |

### Config editor

Open `tools/config_editor.html` in a browser for a guided, form-based config
editor that produces valid JSON you can save directly as your pipeline config.

## CLI Reference

### `run` subcommand

```
uv run python -m feature_selection.src.pipeline.cli run [OPTIONS]
```

| Flag | Description |
|------|-------------|
| `--config FILE` | Path to pipeline config JSON (overrides all other flags except `--output`) |
| `--data FILE` | Path to features CSV (quick-run mode) |
| `--labels FILE` | Path to labels CSV (quick-run mode) |
| `--target NAME` | Target column name (quick-run mode) |
| `--task TYPE` | `classification` or `regression` (default: `classification`) |
| `--sample-id COL` | Sample ID column name (default: `samplename`) |
| `--seed INT` | Random seed (default: `2`) |
| `--output DIR` | Output base directory (default: `./output`) |
| `--no-baseline` | Disable baseline evaluation |
| `--no-dashboard` | Disable dashboard generation |

### Shorthand selector flags (quick-run mode)

| Flag | Selector | Details |
|------|----------|---------|
| `--mRmR-P` | mRmR-Pearson | Pearson relevance, difference scoring |
| `--mRmR-MI` | mRmR-MI | Mutual information relevance, ratio scoring |
| `--mRmR-RF` | mRmR-RF | Random forest relevance, ratio scoring |
| `--mRmR-FS` | mRmR-FS | F-statistic relevance, ratio scoring |
| `--IWSS` | IWSS | Wrapper with SU ranking, no MB pruning |
| `--IWSS-MB` | IWSS-MB | Wrapper with SU ranking and MB pruning |
| `--SFS` | SFS | Wrapper without SU ranking, no MB pruning |
| `--SFS-MB` | SFS-MB | Wrapper without SU ranking, with MB pruning |

If no selector flags are provided in quick-run mode, `--mRmR-P` is used by default.

### `validate` subcommand

```
uv run python -m feature_selection.src.pipeline.cli validate --config FILE
```

Validates the JSON config file and reports any errors. Exits with code 0 on
success, 1 on failure.

## Selection Methods

### mRmR Filter

Minimum Redundancy Maximum Relevance (mRmR) is a greedy forward-selection
filter that balances feature relevance to the target against redundancy among
already-selected features.

**Relevance methods:**

| Method | Key |
|--------|-----|
| Pearson correlation | `pearson` |
| Kendall correlation | `kendall` |
| Spearman correlation | `spearman` |
| Mutual information | `mutual_information` |
| Random forest importance | `random_forest` |
| F-statistic | `f_statistic` |

**Score combination methods:** `difference` (relevance − redundancy) or
`ratio` (relevance / redundancy).

**Redundancy methods:** Pearson, Kendall, or Spearman correlation between
candidate and selected features, aggregated via `mean` or `median`.

### Wrapper (IWSS / SFS)

Wrapper methods evaluate feature subsets using cross-validated model
performance.

| Variant | Description |
|---------|-------------|
| **IWSS** | Incremental Wrapper Subset Selection — ranks candidates by Symmetrical Uncertainty (SU), then adds features in SU order with CV evaluation |
| **SFS** | Sequential Forward Selection — evaluates all remaining candidates at each step, picks the one that improves CV score the most |
| **MB pruning** | Markov Blanket pruning — after each addition, tests whether any previously selected feature has become conditionally redundant and removes it |

Both IWSS and SFS support an optional `patience` parameter to stop early if CV
performance does not improve for a given number of steps.

## Output & Result Caching

Each pipeline run creates a timestamped directory under the configured base
directory. If `fetch_past_results` is true, the pipeline scans previous runs
for identical algorithm configurations to reuse cached results, tracking them via `selector_hashes.json`.

```
output/
└── run_20260413_140000/
    ├── pipeline_config.json           # Copy of the config used
    ├── selector_hashes.json           # Hash map of label-independent configs for caching
    ├── mRmR-Pearson__is_lumA.json     # Per-selector result files
    ├── common__baseline__is_lumA.json # Baseline result
    ├── perf_history__*.json           # Post-selection evaluator metrics
    └── dashboard.html                 # Interactive HTML dashboard
```

**Result JSON files** contain the ordered list of selected features, intrinsic wrapper evaluation
metrics at each checkpoint, and timing information.

**Dashboard** (`dashboard.html`) is a self-contained HTML (powered by Panel/Bokeh) featuring:
- **Interactive Performance Plots:** Multi-selector metric curves
- **Evaluator Dropdown:** Toggle between independent evaluator models for post-selection grading
- **K-features Toggle:** View dynamic metrics in the summary table by adjusting the current feature subset size limit
- **Selector Details Accordion:** Drill down into selected features and multi-evaluator comparison curves per selector

## Python API

### Run the full pipeline

```python
from feature_selection.src.pipeline.config_schema import load_config
from feature_selection.src.pipeline.runner import Pipeline

config = load_config("feature_selection/config/pipeline_config.json")
pipeline = Pipeline(config)
output_dir = pipeline.run()
```

### Use individual components

```python
from feature_selection.src.filter.mRmR import mRmRSelector
from feature_selection.src.wrapper.MarkovBlanketWrapper import WrapperSelector
from feature_selection.src.evaluation.evaluator import Evaluator
from sklearn.linear_model import LogisticRegression
```

**mRmR example:**

```python
selector = mRmRSelector(
    X_train=X_train,
    y_train=y_train,
    relevance_method="pearson",
    mrmr_score_method="difference",
    X_val=X_val,
    y_val=y_val,
)
result = selector.forward_selection(n_features_to_select=30, eval_every_k=2)
print(result.selected_features)
```

**Wrapper example:**

```python
selector = WrapperSelector(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    use_su_ranking=True,
    use_mb_pruning=True,
    mb_threshold=0.01,
)
result = selector.run(n_features_to_select=20, test_eval_every_k=2)
print(result.selected_features)
```

**Evaluator example:**

```python
evaluator = Evaluator(
    model=LogisticRegression(),
    task_type="classification",
    label="logistic_regression"
)
history = Evaluator.calculate_performance_history(
    X_train, y_train, X_test, y_test,
    selected_features=result.selected_features,
    evaluator=evaluator, eval_every_k=2
)
print(history)
```

## Config Editor

A standalone HTML config editor is available at `tools/config_editor.html`.
Open it directly in a browser — no server required. It provides a form-based
interface for building pipeline configurations with validation, and can export
the resulting JSON for use with the CLI.

## Testing

Run the integration test suite:

```bash
uv run python -m tests.test_integration
```

## Dependencies

| Package | Role |
|---------|------|
| `pandas` | Data loading, manipulation, and result storage |
| `numpy` | Numerical computations |
| `scikit-learn` | Model evaluation, CV, metrics, mutual information |
| `scipy` | Statistical tests, Symmetrical Uncertainty |
| `matplotlib` / `seaborn` | Static plots and baselines |
| `bokeh` / `hvplot` / `panel` | Interactive dashboard generation |
| `pyarrow` | Efficient CSV/Parquet I/O |
| `tqdm` | Progress bars for long-running selectors |
| `jupyter` | Notebook support for exploratory analysis |
