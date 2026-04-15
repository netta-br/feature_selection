"""Pipeline configuration schema, validation, and serialization utilities."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any


# ---------------------------------------------------------------------------
# Dataclass definitions
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    features_path: str = "./data/features.csv"
    labels_path: str = "./data/labels.csv"
    sample_id_column: str = "samplename"
    auto_transpose: bool = True
    fillna_value: float | int = 0


@dataclass
class PreprocessingConfig:
    random_seed: int = 2
    train_pct: float = 0.7
    val_pct: float = 0.1
    test_pct: float = 0.2
    eval_data_source: str = "validation"  # "validation" or "test"


@dataclass
class TargetConfig:
    name: str = ""
    task_type: str = "classification"  # "classification" or "regression"


@dataclass
class BaselineConfig:
    enabled: bool = True
    num_runs: int = 10
    eval_feature_counts: list[int] | str = "from_selectors"  # "from_selectors" or explicit list


@dataclass
class SelectorConfig:
    label: str = ""
    type: str = "mrmr"  # "mrmr" or "wrapper"
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)
    targets: list[str] = field(default_factory=list)


@dataclass
class ComparisonConfig:
    label: str = ""
    target: str = ""
    selectors: list[str] = field(default_factory=list)
    include_baseline: bool = False
    type: str = "compare_results"  # "compare_results", "compare_with", "summary_report"


@dataclass
class EvaluatorConfig:
    label: str = ""
    type: str = "logistic_regression"
    # valid: "logistic_regression" | "linear_regression" | "knn" | "naive_bayes"
    task_type: str = "classification"
    # valid: "classification" | "regression"
    params: dict[str, Any] = field(default_factory=dict)
    eval_every_k: int = 1
    # Step size for performance history: evaluate at k, 2k, 3k, … features.
    # Independent of the selector's own eval_every_k — allows backfilling
    # evaluations at different granularities without re-running selection.


@dataclass
class ScoreMatrixConfig:
    random_forest: dict[str, Any] = field(
        default_factory=lambda: {"n_estimators": 100, "max_depth": None}
    )
    mutual_information: dict[str, Any] = field(
        default_factory=lambda: {"n_neighbors": 3}
    )


@dataclass
class OutputConfig:
    base_dir: str = "./output"
    save_results_json: bool = True
    generate_dashboard: bool = True
    fetch_past_results: bool = False


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    targets: list[TargetConfig] = field(default_factory=list)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    selectors: list[SelectorConfig] = field(default_factory=list)
    comparisons: list[ComparisonConfig] = field(default_factory=list)
    score_matrix: ScoreMatrixConfig = field(default_factory=ScoreMatrixConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    evaluators: list[EvaluatorConfig] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_config(config: PipelineConfig) -> list[str]:
    """Return a list of validation error messages. Empty list means valid."""
    errors: list[str] = []

    # --- data paths ---
    if not config.data.features_path:
        errors.append("data.features_path must be a non-empty string.")
    if not config.data.labels_path:
        errors.append("data.labels_path must be a non-empty string.")

    # --- preprocessing percentages ---
    pct_sum = (
        config.preprocessing.train_pct
        + config.preprocessing.val_pct
        + config.preprocessing.test_pct
    )
    if abs(pct_sum - 1.0) > 0.01:
        errors.append(
            f"preprocessing percentages must sum to 1.0 (got {pct_sum:.4f})."
        )

    if config.preprocessing.eval_data_source not in ("validation", "test"):
        errors.append(
            f"preprocessing.eval_data_source must be 'validation' or 'test' "
            f"(got '{config.preprocessing.eval_data_source}')."
        )

    # --- targets ---
    if not config.targets:
        errors.append("At least one target must be defined.")
    valid_target_names: set[str] = set()
    for i, t in enumerate(config.targets):
        if not t.name:
            errors.append(f"targets[{i}].name must be non-empty.")
        else:
            valid_target_names.add(t.name)
        if t.task_type not in ("classification", "regression"):
            errors.append(
                f"targets[{i}].task_type must be 'classification' or 'regression' "
                f"(got '{t.task_type}')."
            )

    # --- baseline ---
    efc = config.baseline.eval_feature_counts
    if isinstance(efc, str):
        if efc != "from_selectors":
            errors.append(
                "baseline.eval_feature_counts string value must be 'from_selectors' "
                f"(got '{efc}')."
            )
    elif isinstance(efc, list):
        if not efc:
            errors.append(
                "baseline.eval_feature_counts list must be non-empty."
            )
        else:
            for v in efc:
                if not isinstance(v, int) or v <= 0:
                    errors.append(
                        "baseline.eval_feature_counts list must contain only "
                        f"positive integers (got {v!r})."
                    )
                    break
    else:
        errors.append(
            "baseline.eval_feature_counts must be 'from_selectors' or a list of "
            "positive integers."
        )

    # --- selectors ---
    enabled_selectors = [s for s in config.selectors if s.enabled]
    if not enabled_selectors:
        errors.append("At least one selector must be defined and enabled.")

    seen_labels: set[str] = set()
    valid_selector_labels: set[str] = set()
    for i, s in enumerate(config.selectors):
        if not s.label:
            errors.append(f"selectors[{i}].label must be non-empty.")
        else:
            if s.label in seen_labels:
                errors.append(f"selectors[{i}].label '{s.label}' is duplicated.")
            seen_labels.add(s.label)
            valid_selector_labels.add(s.label)

        if s.type not in ("mrmr", "wrapper"):
            errors.append(
                f"selectors[{i}].type must be 'mrmr' or 'wrapper' "
                f"(got '{s.type}')."
            )

        if not s.targets:
            errors.append(f"selectors[{i}].targets must be non-empty.")
        else:
            for tname in s.targets:
                if tname not in valid_target_names:
                    errors.append(
                        f"selectors[{i}].targets references unknown target "
                        f"'{tname}'."
                    )

    # --- comparisons ---
    valid_comparison_types = {"compare_results", "compare_with", "summary_report"}
    for i, c in enumerate(config.comparisons):
        if c.target and c.target not in valid_target_names:
            errors.append(
                f"comparisons[{i}].target references unknown target '{c.target}'."
            )
        for slabel in c.selectors:
            if slabel not in valid_selector_labels:
                errors.append(
                    f"comparisons[{i}].selectors references unknown selector "
                    f"'{slabel}'."
                )
        if c.type not in valid_comparison_types:
            errors.append(
                f"comparisons[{i}].type must be one of {sorted(valid_comparison_types)} "
                f"(got '{c.type}')."
            )

    _VALID_EVAL_TYPES = {"logistic_regression", "linear_regression", "knn", "naive_bayes"}
    _CLF_ONLY_TYPES   = {"logistic_regression", "knn", "naive_bayes"}
    _REG_ONLY_TYPES   = {"linear_regression"}

    seen_eval_labels: set[str] = set()
    for i, ev in enumerate(config.evaluators):
        if not ev.label:
            errors.append(f"evaluators[{i}].label must be non-empty.")
        else:
            if ev.label in seen_eval_labels:
                errors.append(f"evaluators[{i}].label '{ev.label}' is duplicated.")
            seen_eval_labels.add(ev.label)

        if ev.type not in _VALID_EVAL_TYPES:
            errors.append(
                f"evaluators[{i}].type must be one of {sorted(_VALID_EVAL_TYPES)} "
                f"(got '{ev.type}')."
            )
        if ev.task_type not in ("classification", "regression"):
            errors.append(
                f"evaluators[{i}].task_type must be 'classification' or 'regression' "
                f"(got '{ev.task_type}')."
            )
        if ev.type in _CLF_ONLY_TYPES and ev.task_type != "classification":
            errors.append(
                f"evaluators[{i}].type '{ev.type}' is only valid for task_type='classification'."
            )
        if ev.type in _REG_ONLY_TYPES and ev.task_type != "regression":
            errors.append(
                f"evaluators[{i}].type '{ev.type}' is only valid for task_type='regression'."
            )

    return errors


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def config_from_dict(d: dict) -> PipelineConfig:
    """Convert a raw dict (e.g. from JSON) into a *PipelineConfig* tree.

    Missing keys use dataclass defaults; extra keys are silently ignored.
    """
    data = DataConfig(**{
        k: v for k, v in d.get("data", {}).items()
        if k in DataConfig.__dataclass_fields__
    })

    preprocessing = PreprocessingConfig(**{
        k: v for k, v in d.get("preprocessing", {}).items()
        if k in PreprocessingConfig.__dataclass_fields__
    })

    targets = [
        TargetConfig(**{
            k: v for k, v in t.items()
            if k in TargetConfig.__dataclass_fields__
        })
        for t in d.get("targets", [])
    ]

    # baseline — handle eval_feature_counts which can be str or list
    baseline_raw = d.get("baseline", {})
    baseline = BaselineConfig(**{
        k: v for k, v in baseline_raw.items()
        if k in BaselineConfig.__dataclass_fields__
    })

    selectors = [
        SelectorConfig(**{
            k: v for k, v in s.items()
            if k in SelectorConfig.__dataclass_fields__
        })
        for s in d.get("selectors", [])
    ]

    comparisons = [
        ComparisonConfig(**{
            k: v for k, v in c.items()
            if k in ComparisonConfig.__dataclass_fields__
        })
        for c in d.get("comparisons", [])
    ]

    score_raw = d.get("score_matrix", {})
    score_matrix = ScoreMatrixConfig(**{
        k: v for k, v in score_raw.items()
        if k in ScoreMatrixConfig.__dataclass_fields__
    })

    output = OutputConfig(**{
        k: v for k, v in d.get("output", {}).items()
        if k in OutputConfig.__dataclass_fields__
    })

    evaluators = [
        EvaluatorConfig(**{
            k: v for k, v in ev.items()
            if k in EvaluatorConfig.__dataclass_fields__
        })
        for ev in d.get("evaluators", [])
    ]

    return PipelineConfig(
        data=data,
        preprocessing=preprocessing,
        targets=targets,
        baseline=baseline,
        selectors=selectors,
        comparisons=comparisons,
        score_matrix=score_matrix,
        output=output,
        evaluators=evaluators,
    )


def config_to_dict(config: PipelineConfig) -> dict:
    """Convert a *PipelineConfig* to a plain dict suitable for JSON serialization."""
    return asdict(config)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_config(filepath: str) -> PipelineConfig:
    """Load a JSON config file, validate it, and return a *PipelineConfig*.

    Raises ``ValueError`` if validation fails.
    """
    with open(filepath, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    config = config_from_dict(raw)
    errors = validate_config(config)
    if errors:
        raise ValueError(
            "Pipeline config validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )
    return config


def save_config(config: PipelineConfig, filepath: str) -> None:
    """Serialize *config* to a JSON file with ``indent=2``."""
    d = config_to_dict(config)
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=2)
        fh.write("\n")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

def default_config() -> PipelineConfig:
    """Return a *PipelineConfig* with all defaults populated."""
    return PipelineConfig()
