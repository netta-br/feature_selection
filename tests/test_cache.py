"""Unit tests for SelectorCacheManager (feature_selection/src/pipeline/cache.py)."""

from __future__ import annotations

import copy
import json
import os

import pytest

from feature_selection.src.pipeline.cache import SelectorCacheManager, HASH_FILE
from feature_selection.src.pipeline.config_schema import (
    DataConfig,
    OutputConfig,
    PipelineConfig,
    PreprocessingConfig,
    SelectorConfig,
    TargetConfig,
)
from feature_selection.src.results import SelectionResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_config(
    selector_label: str = "mRmR-P",
    selector_type: str = "mrmr",
    selector_params: dict | None = None,
    target_name: str = "Target",
    task_type: str = "classification",
    base_dir: str = "./output",
    random_seed: int = 42,
    features_path: str = "data/features.csv",
) -> PipelineConfig:
    if selector_params is None:
        selector_params = {
            "relevance_method": "pearson",
            "n_features_to_select": 10,
            "eval_every_k": 1,
        }
    return PipelineConfig(
        data=DataConfig(
            features_path=features_path,
            labels_path="data/labels.csv",
            sample_id_column="samplename",
            auto_transpose=True,
            fillna_value=0,
        ),
        preprocessing=PreprocessingConfig(
            random_seed=random_seed,
            train_pct=0.7,
            val_pct=0.1,
            test_pct=0.2,
            eval_data_source="validation",
        ),
        targets=[TargetConfig(name=target_name, task_type=task_type)],
        selectors=[
            SelectorConfig(
                label=selector_label,
                type=selector_type,
                enabled=True,
                params=selector_params,
                targets=[target_name],
            )
        ],
        output=OutputConfig(base_dir=base_dir),
    )


def _make_result(label: str = "mRmR-P", features: list[str] | None = None) -> SelectionResult:
    return SelectionResult(
        selected_features=features or ["f1", "f2", "f3"],
        performance_history=[{"step": 3, "accuracy": 0.85}],
        stopping_reason="max_features_reached",
        n_steps=3,
        final_metrics={"accuracy": 0.85},
        selection_time_seconds=1.23,
        evaluation_time_seconds=0.45,
        task_type="classification",
        label=label,
    )


def _write_past_run(
    tmp_path,
    run_name: str,
    config: PipelineConfig,
    results: dict[str, SelectionResult] | None = None,
    write_hash_map: bool = True,
) -> str:
    """Create a simulated past run directory under tmp_path."""
    from feature_selection.src.pipeline.config_schema import config_to_dict

    run_dir = str(tmp_path / run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Write pipeline_config.json
    cfg_path = os.path.join(run_dir, "pipeline_config.json")
    with open(cfg_path, "w") as f:
        json.dump(config_to_dict(config), f, indent=2)

    # Write result JSONs
    if results:
        for result_key, result in results.items():
            result_path = os.path.join(run_dir, f"{result_key}.json")
            result.save_as_json(result_path)

    # Write selector_hashes.json if requested
    if write_hash_map:
        mgr = SelectorCacheManager(str(tmp_path), config)
        hashes = mgr.compute_current_hashes()
        with open(os.path.join(run_dir, HASH_FILE), "w") as f:
            json.dump(hashes, f, indent=2)

    return run_dir


# ---------------------------------------------------------------------------
# 1. Hash determinism
# ---------------------------------------------------------------------------

class TestHashDeterminism:
    def test_same_config_same_hash(self):
        """Same config always produces the same hash."""
        cfg = _make_config()
        mgr = SelectorCacheManager("./output", cfg)
        h1 = mgr.compute_current_hashes()
        h2 = mgr.compute_current_hashes()
        assert h1 == h2

    def test_hash_is_hex_string(self):
        """Hash values are 64-char hex strings (SHA-256)."""
        cfg = _make_config()
        mgr = SelectorCacheManager("./output", cfg)
        hashes = mgr.compute_current_hashes()
        for h in hashes.values():
            assert len(h) == 64
            int(h, 16)  # raises ValueError if not valid hex


# ---------------------------------------------------------------------------
# 2. Hash sensitivity to key fields
# ---------------------------------------------------------------------------

class TestHashSensitivity:
    def _get_hash(self, **kwargs) -> str:
        cfg = _make_config(**kwargs)
        mgr = SelectorCacheManager("./output", cfg)
        hashes = mgr.compute_current_hashes()
        assert len(hashes) == 1
        return next(iter(hashes.values()))

    def test_different_random_seed_different_hash(self):
        assert self._get_hash(random_seed=1) != self._get_hash(random_seed=99)

    def test_different_features_path_different_hash(self):
        assert (
            self._get_hash(features_path="a.csv")
            != self._get_hash(features_path="b.csv")
        )

    def test_different_selector_params_different_hash(self):
        h1 = self._get_hash(selector_params={"n_features_to_select": 10})
        h2 = self._get_hash(selector_params={"n_features_to_select": 20})
        assert h1 != h2

    def test_different_selector_type_different_hash(self):
        h1 = self._get_hash(
            selector_type="mrmr",
            selector_params={"n_features_to_select": 10},
        )
        h2 = self._get_hash(
            selector_type="wrapper",
            selector_params={"n_features_to_select": 10},
        )
        assert h1 != h2

    def test_different_target_task_type_different_hash(self):
        assert (
            self._get_hash(task_type="classification")
            != self._get_hash(task_type="regression")
        )


# ---------------------------------------------------------------------------
# 3. Label excluded from hash
# ---------------------------------------------------------------------------

class TestLabelExcludedFromHash:
    def test_different_labels_same_hash(self):
        """Two selectors identical except for label must produce the same hash."""
        cfg_a = _make_config(selector_label="Method-A")
        cfg_b = _make_config(selector_label="Method-B")
        mgr_a = SelectorCacheManager("./output", cfg_a)
        mgr_b = SelectorCacheManager("./output", cfg_b)
        hashes_a = mgr_a.compute_current_hashes()
        hashes_b = mgr_b.compute_current_hashes()
        # Keys differ (label part) but hash values must match
        assert list(hashes_a.values()) == list(hashes_b.values())


# ---------------------------------------------------------------------------
# 4. resolve_from_past_runs — cache hit
# ---------------------------------------------------------------------------

class TestResolveHit:
    def test_hit_loads_result(self, tmp_path):
        """A matching past run with a valid result JSON returns the loaded result."""
        base_dir = str(tmp_path)
        cfg = _make_config(base_dir=base_dir)
        result = _make_result(label="mRmR-P", features=["f1", "f2"])

        _write_past_run(
            tmp_path, "run_20200101_000000", cfg,
            results={"mRmR-P__Target": result},
            write_hash_map=True,
        )

        current_output = str(tmp_path / "run_20200102_000000")
        os.makedirs(current_output)

        mgr = SelectorCacheManager(base_dir, cfg)
        current_hashes = mgr.compute_current_hashes()
        resolved = mgr.resolve_from_past_runs(current_hashes, current_output)

        assert "mRmR-P__Target" in resolved
        assert resolved["mRmR-P__Target"].selected_features == ["f1", "f2"]

    def test_hit_copies_json_to_current_dir(self, tmp_path):
        """Cache hit copies the result JSON into the current run directory."""
        base_dir = str(tmp_path)
        cfg = _make_config(base_dir=base_dir)
        result = _make_result(label="mRmR-P")

        _write_past_run(
            tmp_path, "run_20200101_000000", cfg,
            results={"mRmR-P__Target": result},
        )

        current_output = str(tmp_path / "run_20200102_000000")
        os.makedirs(current_output)

        mgr = SelectorCacheManager(base_dir, cfg)
        current_hashes = mgr.compute_current_hashes()
        mgr.resolve_from_past_runs(current_hashes, current_output)

        assert os.path.isfile(os.path.join(current_output, "mRmR-P__Target.json"))

    def test_hit_relabels_result(self, tmp_path):
        """Cache hit re-labels the result with the current selector's label."""
        base_dir = str(tmp_path)
        past_cfg = _make_config(selector_label="OldName", base_dir=base_dir)
        result = _make_result(label="OldName")

        _write_past_run(
            tmp_path, "run_20200101_000000", past_cfg,
            results={"OldName__Target": result},
        )

        current_output = str(tmp_path / "run_20200102_000000")
        os.makedirs(current_output)

        current_cfg = _make_config(selector_label="NewName", base_dir=base_dir)
        mgr = SelectorCacheManager(base_dir, current_cfg)
        current_hashes = mgr.compute_current_hashes()
        resolved = mgr.resolve_from_past_runs(current_hashes, current_output)

        assert "NewName__Target" in resolved
        assert resolved["NewName__Target"].label == "NewName"


# ---------------------------------------------------------------------------
# 5. resolve_from_past_runs — cache miss (hash mismatch)
# ---------------------------------------------------------------------------

class TestResolveMissHash:
    def test_different_params_returns_empty(self, tmp_path):
        """No cache hit when past and current selector params differ."""
        base_dir = str(tmp_path)

        past_cfg = _make_config(
            base_dir=base_dir,
            selector_params={"n_features_to_select": 5},
        )
        result = _make_result()
        _write_past_run(
            tmp_path, "run_20200101_000000", past_cfg,
            results={"mRmR-P__Target": result},
        )

        current_output = str(tmp_path / "run_20200102_000000")
        os.makedirs(current_output)

        current_cfg = _make_config(
            base_dir=base_dir,
            selector_params={"n_features_to_select": 20},
        )
        mgr = SelectorCacheManager(base_dir, current_cfg)
        current_hashes = mgr.compute_current_hashes()
        resolved = mgr.resolve_from_past_runs(current_hashes, current_output)

        assert resolved == {}


# ---------------------------------------------------------------------------
# 6. resolve_from_past_runs — hash match but JSON file missing
# ---------------------------------------------------------------------------

class TestResolveMissNoFile:
    def test_missing_json_returns_empty(self, tmp_path):
        """Hash matches but result JSON absent → treated as miss."""
        base_dir = str(tmp_path)
        cfg = _make_config(base_dir=base_dir)

        # Write hash map but no result JSON
        _write_past_run(
            tmp_path, "run_20200101_000000", cfg,
            results=None,
            write_hash_map=True,
        )

        current_output = str(tmp_path / "run_20200102_000000")
        os.makedirs(current_output)

        mgr = SelectorCacheManager(base_dir, cfg)
        current_hashes = mgr.compute_current_hashes()
        resolved = mgr.resolve_from_past_runs(current_hashes, current_output)

        assert resolved == {}


# ---------------------------------------------------------------------------
# 7. Newest-first ordering
# ---------------------------------------------------------------------------

class TestNewestFirst:
    def test_newest_run_used_when_multiple_match(self, tmp_path):
        """When multiple past runs match, the newest (lexicographically last) wins."""
        base_dir = str(tmp_path)
        cfg = _make_config(base_dir=base_dir)

        old_result = _make_result(features=["old_f1"])
        new_result = _make_result(features=["new_f1"])

        _write_past_run(
            tmp_path, "run_20200101_000000", cfg,
            results={"mRmR-P__Target": old_result},
        )
        _write_past_run(
            tmp_path, "run_20200202_000000", cfg,
            results={"mRmR-P__Target": new_result},
        )

        current_output = str(tmp_path / "run_20200303_000000")
        os.makedirs(current_output)

        mgr = SelectorCacheManager(base_dir, cfg)
        current_hashes = mgr.compute_current_hashes()
        resolved = mgr.resolve_from_past_runs(current_hashes, current_output)

        assert resolved["mRmR-P__Target"].selected_features == ["new_f1"]


# ---------------------------------------------------------------------------
# 8. Hash map always saved (even when fetch_past_results=False)
# ---------------------------------------------------------------------------

class TestHashMapAlwaysSaved:
    def test_hash_map_written_to_output_dir(self, tmp_path):
        """save_hash_map writes selector_hashes.json into the given directory."""
        cfg = _make_config()
        mgr = SelectorCacheManager(str(tmp_path), cfg)
        hashes = mgr.compute_current_hashes()
        mgr.save_hash_map(str(tmp_path), hashes)

        hash_path = os.path.join(str(tmp_path), HASH_FILE)
        assert os.path.isfile(hash_path)
        with open(hash_path) as f:
            saved = json.load(f)
        assert saved == hashes


# ---------------------------------------------------------------------------
# 9. Backward compat — generates hash map from legacy config
# ---------------------------------------------------------------------------

class TestBackwardCompatGeneratesHashMap:
    def test_generates_hash_map_when_absent(self, tmp_path):
        """Past run with config but no hash map → hash map is created on scan."""
        base_dir = str(tmp_path)
        cfg = _make_config(base_dir=base_dir)
        result = _make_result()

        _write_past_run(
            tmp_path, "run_20200101_000000", cfg,
            results={"mRmR-P__Target": result},
            write_hash_map=False,   # ← no hash file (legacy run)
        )

        hash_path = os.path.join(
            str(tmp_path), "run_20200101_000000", HASH_FILE
        )
        assert not os.path.isfile(hash_path)

        current_output = str(tmp_path / "run_20200102_000000")
        os.makedirs(current_output)

        mgr = SelectorCacheManager(base_dir, cfg)
        current_hashes = mgr.compute_current_hashes()
        resolved = mgr.resolve_from_past_runs(current_hashes, current_output)

        # Hash map must now exist in the legacy run dir
        assert os.path.isfile(hash_path)
        # And the result must have been resolved
        assert "mRmR-P__Target" in resolved


# ---------------------------------------------------------------------------
# 10. Backward compat — hash map not regenerated if already present
# ---------------------------------------------------------------------------

class TestBackwardCompatNoRegeneration:
    def test_hash_map_not_rewritten_when_present(self, tmp_path):
        """Hash map is not rewritten when it already exists (mtime preserved)."""
        base_dir = str(tmp_path)
        cfg = _make_config(base_dir=base_dir)
        result = _make_result()

        _write_past_run(
            tmp_path, "run_20200101_000000", cfg,
            results={"mRmR-P__Target": result},
            write_hash_map=True,
        )

        hash_path = os.path.join(
            str(tmp_path), "run_20200101_000000", HASH_FILE
        )
        mtime_before = os.path.getmtime(hash_path)

        import time; time.sleep(0.05)  # ensure OS mtime resolution

        current_output = str(tmp_path / "run_20200102_000000")
        os.makedirs(current_output)

        mgr = SelectorCacheManager(base_dir, cfg)
        current_hashes = mgr.compute_current_hashes()
        mgr.resolve_from_past_runs(current_hashes, current_output)

        mtime_after = os.path.getmtime(hash_path)
        assert mtime_before == mtime_after
