"""Argparse-based CLI for the feature-selection pipeline.

Provides two subcommands:

* ``run``      – execute the pipeline (from a config file or quick-run flags)
* ``validate`` – check a config JSON for errors
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from dataclasses import replace

from .config_schema import (
    PipelineConfig,
    SelectorConfig,
    TargetConfig,
    config_from_dict,
    default_config,
    load_config,
    validate_config,
)
from .runner import Pipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Selector presets (shorthand flags → SelectorConfig)
# ---------------------------------------------------------------------------

SELECTOR_PRESETS: dict[str, SelectorConfig] = {
    "mRmR_P": SelectorConfig(
        label="mRmR-Pearson",
        type="mrmr",
        enabled=True,
        params={
            "relevance_method": "pearson",
            "mrmr_score_method": "difference",
            "n_features_to_select": 30,
            "eval_every_k": 2,
        },
        targets=[],
    ),
    "mRmR_MI": SelectorConfig(
        label="mRmR-MI",
        type="mrmr",
        enabled=True,
        params={
            "relevance_method": "mutual_information",
            "mrmr_score_method": "ratio",
            "n_features_to_select": 30,
            "eval_every_k": 2,
        },
        targets=[],
    ),
    "mRmR_RF": SelectorConfig(
        label="mRmR-RF",
        type="mrmr",
        enabled=True,
        params={
            "relevance_method": "random_forest",
            "mrmr_score_method": "ratio",
            "n_features_to_select": 30,
            "eval_every_k": 2,
        },
        targets=[],
    ),
    "mRmR_FS": SelectorConfig(
        label="mRmR-FS",
        type="mrmr",
        enabled=True,
        params={
            "relevance_method": "f_statistic",
            "mrmr_score_method": "ratio",
            "n_features_to_select": 30,
            "eval_every_k": 2,
        },
        targets=[],
    ),
    "IWSS": SelectorConfig(
        label="IWSS",
        type="wrapper",
        enabled=True,
        params={
            "use_su_ranking": True,
            "use_mb_pruning": False,
            "n_features_to_select": 20,
            "test_eval_every_k": 2,
        },
        targets=[],
    ),
    "IWSS_MB": SelectorConfig(
        label="IWSS-MB",
        type="wrapper",
        enabled=True,
        params={
            "use_su_ranking": True,
            "use_mb_pruning": True,
            "mb_threshold": 0.01,
            "n_features_to_select": 20,
            "test_eval_every_k": 2,
        },
        targets=[],
    ),
    "SFS": SelectorConfig(
        label="SFS",
        type="wrapper",
        enabled=True,
        params={
            "use_su_ranking": False,
            "use_mb_pruning": False,
            "n_features_to_select": 20,
            "test_eval_every_k": 2,
        },
        targets=[],
    ),
    "SFS_MB": SelectorConfig(
        label="SFS-MB",
        type="wrapper",
        enabled=True,
        params={
            "use_su_ranking": False,
            "use_mb_pruning": True,
            "mb_threshold": 0.01,
            "n_features_to_select": 20,
            "test_eval_every_k": 2,
        },
        targets=[],
    ),
}

# Map from argparse dest name → preset key
_FLAG_TO_PRESET: dict[str, str] = {
    "mRmR_P": "mRmR_P",
    "mRmR_MI": "mRmR_MI",
    "mRmR_RF": "mRmR_RF",
    "mRmR_FS": "mRmR_FS",
    "IWSS": "IWSS",
    "IWSS_MB": "IWSS_MB",
    "SFS": "SFS",
    "SFS_MB": "SFS_MB",
}


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def _handle_run(args: argparse.Namespace) -> None:
    """Handle the ``run`` subcommand."""
    if args.config:
        logger.info("Loading config from %s", args.config)
        config = load_config(args.config)
        # Allow --output to override even in config mode
        if args.output != "./output":
            config.output.base_dir = args.output
        output_dir = Pipeline(config).run()
        logger.info("Pipeline finished. Output: %s", output_dir)
        return

    # --- Quick-run mode: require --data, --labels, --target ----------------
    missing: list[str] = []
    if not args.data:
        missing.append("--data")
    if not args.labels:
        missing.append("--labels")
    if not args.target:
        missing.append("--target")
    if missing:
        sys.exit(
            f"Error: quick-run mode requires {', '.join(missing)}. "
            "Provide these flags or use --config FILE."
        )

    config = default_config()
    config.data.features_path = args.data
    config.data.labels_path = args.labels
    config.data.sample_id_column = args.sample_id
    config.preprocessing.random_seed = args.seed
    config.targets = [TargetConfig(name=args.target, task_type=args.task)]
    config.output.base_dir = args.output
    config.output.generate_dashboard = not args.no_dashboard
    config.baseline.enabled = not args.no_baseline

    # Collect enabled shorthand selectors
    selectors: list[SelectorConfig] = []
    for flag_dest, preset_key in _FLAG_TO_PRESET.items():
        if getattr(args, flag_dest, False):
            preset = copy.deepcopy(SELECTOR_PRESETS[preset_key])
            preset.targets = [args.target]
            selectors.append(preset)

    # Default to mRmR-Pearson if no shorthand flags given
    if not selectors:
        preset = copy.deepcopy(SELECTOR_PRESETS["mRmR_P"])
        preset.targets = [args.target]
        selectors.append(preset)

    config.selectors = selectors

    logger.info(
        "Quick-run mode: %d selector(s), target=%s",
        len(selectors),
        args.target,
    )
    output_dir = Pipeline(config).run()
    logger.info("Pipeline finished. Output: %s", output_dir)


def _handle_validate(args: argparse.Namespace) -> None:
    """Handle the ``validate`` subcommand."""
    try:
        with open(args.config, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Failed to read config: {exc}", file=sys.stderr)
        sys.exit(1)

    config = config_from_dict(raw)
    errors = validate_config(config)

    if errors:
        print("Config validation failed:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("Config is valid")
        sys.exit(0)


# ---------------------------------------------------------------------------
# Argument parser construction
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with ``run`` and ``validate`` subcommands."""
    parser = argparse.ArgumentParser(
        prog="feature_selection.pipeline",
        description="Feature Selection Pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ----- run subcommand --------------------------------------------------
    run_parser = subparsers.add_parser(
        "run",
        help="Run the feature-selection pipeline",
    )
    run_parser.add_argument(
        "--config",
        metavar="FILE",
        default=None,
        help="Path to pipeline config JSON (overrides all other flags except --output)",
    )
    run_parser.add_argument(
        "--data",
        metavar="FILE",
        default=None,
        help="Path to features CSV (quick-run mode)",
    )
    run_parser.add_argument(
        "--labels",
        metavar="FILE",
        default=None,
        help="Path to labels CSV (quick-run mode)",
    )
    run_parser.add_argument(
        "--target",
        metavar="NAME",
        default=None,
        help="Target column name (quick-run mode)",
    )
    run_parser.add_argument(
        "--task",
        metavar="TYPE",
        default="classification",
        choices=["classification", "regression"],
        help="Task type: classification or regression (default: classification)",
    )
    run_parser.add_argument(
        "--sample-id",
        metavar="COL",
        default="samplename",
        help="Sample ID column name (default: samplename)",
    )
    run_parser.add_argument(
        "--seed",
        metavar="INT",
        type=int,
        default=2,
        help="Random seed (default: 2)",
    )
    run_parser.add_argument(
        "--output",
        metavar="DIR",
        default="./output",
        help="Output base directory (default: ./output)",
    )
    run_parser.add_argument(
        "--no-baseline",
        action="store_true",
        default=False,
        help="Disable baseline evaluation",
    )
    run_parser.add_argument(
        "--no-dashboard",
        action="store_true",
        default=False,
        help="Disable dashboard generation",
    )

    # Shorthand selector flags
    selector_group = run_parser.add_argument_group("shorthand selector flags")
    selector_group.add_argument(
        "--mRmR-P",
        dest="mRmR_P",
        action="store_true",
        default=False,
        help="mRmR with Pearson relevance, difference scoring",
    )
    selector_group.add_argument(
        "--mRmR-MI",
        dest="mRmR_MI",
        action="store_true",
        default=False,
        help="mRmR with mutual information relevance, ratio scoring",
    )
    selector_group.add_argument(
        "--mRmR-RF",
        dest="mRmR_RF",
        action="store_true",
        default=False,
        help="mRmR with random forest relevance, ratio scoring",
    )
    selector_group.add_argument(
        "--mRmR-FS",
        dest="mRmR_FS",
        action="store_true",
        default=False,
        help="mRmR with F-statistic relevance, ratio scoring",
    )
    selector_group.add_argument(
        "--IWSS",
        dest="IWSS",
        action="store_true",
        default=False,
        help="Wrapper with SU ranking, no MB pruning",
    )
    selector_group.add_argument(
        "--IWSS-MB",
        dest="IWSS_MB",
        action="store_true",
        default=False,
        help="Wrapper with SU ranking and MB pruning",
    )
    selector_group.add_argument(
        "--SFS",
        dest="SFS",
        action="store_true",
        default=False,
        help="Wrapper without SU ranking, no MB pruning",
    )
    selector_group.add_argument(
        "--SFS-MB",
        dest="SFS_MB",
        action="store_true",
        default=False,
        help="Wrapper without SU ranking, with MB pruning",
    )

    # ----- validate subcommand ---------------------------------------------
    val_parser = subparsers.add_parser(
        "validate",
        help="Validate a pipeline config JSON file",
    )
    val_parser.add_argument(
        "--config",
        metavar="FILE",
        required=True,
        help="Path to pipeline config JSON",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the feature-selection pipeline."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "run":
        _handle_run(args)
    elif args.command == "validate":
        _handle_validate(args)


if __name__ == "__main__":
    main()
