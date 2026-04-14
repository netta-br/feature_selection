"""Selector result cache — hash-map based lookup across previous pipeline runs.

Each pipeline run writes a ``selector_hashes.json`` file into its output
directory mapping ``"{label}__{target}"`` result keys to SHA-256 hashes of
the label-independent algorithm configuration.  When ``fetch_past_results``
is enabled, the current run scans past run directories for matching hashes
and loads their result JSON files instead of re-running the selectors.

Backward compatibility
----------------------
If a past run directory contains ``pipeline_config.json`` but no
``selector_hashes.json``, the hash map is computed on the fly from the
config and written into that directory so future scans can use it directly.
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
import shutil
from typing import TYPE_CHECKING

from .config_schema import PipelineConfig, SelectorConfig, TargetConfig, config_from_dict

if TYPE_CHECKING:
    from ..results import SelectionResult

log = logging.getLogger(__name__)

HASH_FILE = "selector_hashes.json"


class SelectorCacheManager:
    """Manages selector result caching across pipeline runs.

    Parameters
    ----------
    base_dir:
        The ``output.base_dir`` from the pipeline config — the directory
        that contains individual ``run_*`` subdirectories.
    config:
        The *current* run's :class:`~.config_schema.PipelineConfig`.
    """

    def __init__(self, base_dir: str, config: PipelineConfig) -> None:
        self.base_dir = base_dir
        self.config = config

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute_current_hashes(self) -> dict[str, str]:
        """Return ``{result_key: hash}`` for every enabled selector × target.

        The result key follows the existing pipeline convention:
        ``"{selector.label}__{target_name}"``.

        The hash is label-independent — only algorithm parameters and the
        data/preprocessing config that affect the selector's output are
        included.
        """
        hashes: dict[str, str] = {}
        for selector in self.config.selectors:
            if not selector.enabled:
                continue
            for target_name in selector.targets:
                target_cfg = next(
                    (t for t in self.config.targets if t.name == target_name),
                    None,
                )
                if target_cfg is None:
                    continue
                result_key = f"{selector.label}__{target_name}"
                hashes[result_key] = self._compute_hash(
                    selector, target_name, target_cfg, self.config
                )
        return hashes

    def save_hash_map(self, output_dir: str, hash_map: dict[str, str]) -> None:
        """Write *hash_map* as ``selector_hashes.json`` into *output_dir*."""
        path = os.path.join(output_dir, HASH_FILE)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(hash_map, fh, indent=2)
            log.debug("Saved selector hash map: %s", path)
        except OSError as exc:
            log.warning("Could not write selector hash map to %s: %s", path, exc)

    def resolve_from_past_runs(
        self,
        current_hashes: dict[str, str],
        output_dir: str,
    ) -> dict[str, SelectionResult]:
        """Scan past run directories and return cached results for matching hashes.

        Parameters
        ----------
        current_hashes:
            Mapping of ``result_key → hash`` for the current run (all enabled
            selectors × targets).  Only keys that are *not* yet resolved are
            considered — the caller should pass the full map; resolved keys
            are tracked internally.
        output_dir:
            The *current* run's output directory.  Matched result JSON files
            are copied here so the current run directory is self-contained.

        Returns
        -------
        dict[str, SelectionResult]
            Mapping of ``result_key → loaded result`` for every cache hit.
            Keys absent from this mapping must be computed normally.
        """
        from ..results import SelectionResult  # local import to avoid circularity

        resolved: dict[str, SelectionResult] = {}
        remaining: dict[str, str] = dict(current_hashes)  # result_key → hash

        # Invert current map: hash → list of result_keys (usually 1:1)
        hash_to_keys: dict[str, list[str]] = {}
        for rk, h in remaining.items():
            hash_to_keys.setdefault(h, []).append(rk)

        # Scan past run dirs newest-first (timestamp in dir name → lexicographic sort)
        past_dirs = sorted(
            glob.glob(os.path.join(self.base_dir, "run_*")),
            reverse=True,
        )

        for run_dir in past_dirs:
            if not os.path.isdir(run_dir):
                continue
            # Skip the current output dir — shouldn't exist yet but be safe
            if os.path.abspath(run_dir) == os.path.abspath(output_dir):
                continue

            cfg_path = os.path.join(run_dir, "pipeline_config.json")
            hash_path = os.path.join(run_dir, HASH_FILE)

            past_hashes = self._load_or_generate_hash_map(
                run_dir, cfg_path, hash_path
            )
            if past_hashes is None:
                continue

            # Invert past map: hash → past_result_key
            past_hash_to_key: dict[str, str] = {}
            for past_rk, past_h in past_hashes.items():
                # Keep only first entry if duplicate hashes exist in past run
                if past_h not in past_hash_to_key:
                    past_hash_to_key[past_h] = past_rk

            # Find intersections between remaining hashes and past hashes
            for current_hash, current_keys in list(hash_to_keys.items()):
                if current_hash not in past_hash_to_key:
                    continue

                past_result_key = past_hash_to_key[current_hash]
                past_json = os.path.join(run_dir, f"{past_result_key}.json")

                if not os.path.isfile(past_json):
                    log.debug(
                        "Cache: hash match for '%s' in %s but result file missing",
                        past_result_key,
                        run_dir,
                    )
                    continue

                # Load the result once
                try:
                    result = SelectionResult.load_from_json(past_json)
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "Cache: failed to load %s: %s", past_json, exc
                    )
                    continue

                # Apply the cached result to every current key sharing this hash
                for current_rk in current_keys:
                    if current_rk in resolved:
                        continue  # already resolved from an even newer run

                    # Re-label with the current selector's label
                    current_label = current_rk.split("__")[0]
                    import copy as _copy
                    result_copy = _copy.deepcopy(result)
                    result_copy.label = current_label

                    # Copy JSON into current output dir under the current result key
                    dest_json = os.path.join(output_dir, f"{current_rk}.json")
                    try:
                        result_copy.save_as_json(dest_json)
                    except OSError as exc:
                        log.warning(
                            "Cache: could not copy result JSON to %s: %s",
                            dest_json, exc,
                        )

                    resolved[current_rk] = result_copy
                    log.info(
                        "Cache HIT: '%s' loaded from %s (was '%s')",
                        current_rk, run_dir, past_result_key,
                    )

                # Remove fully resolved hash from tracking
                if all(rk in resolved for rk in current_keys):
                    del hash_to_keys[current_hash]

            if not hash_to_keys:
                log.debug("Cache: all selectors resolved, stopping scan")
                break

        return resolved

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_or_generate_hash_map(
        self,
        run_dir: str,
        cfg_path: str,
        hash_path: str,
    ) -> dict[str, str] | None:
        """Load ``selector_hashes.json`` from *run_dir*, generating it if absent.

        Backward-compatibility path: if ``pipeline_config.json`` exists but
        ``selector_hashes.json`` does not, compute the hash map from the
        config and persist it to *hash_path*.

        Returns the hash map dict, or ``None`` if neither file is readable.
        """
        # Fast path — hash file already present
        if os.path.isfile(hash_path):
            try:
                with open(hash_path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception as exc:  # noqa: BLE001
                log.warning("Cache: could not read %s: %s", hash_path, exc)
                return None

        # Backward-compat path — derive from pipeline_config.json
        if not os.path.isfile(cfg_path):
            return None

        try:
            with open(cfg_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            past_cfg = config_from_dict(raw)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Cache: could not parse past config %s: %s", cfg_path, exc
            )
            return None

        hash_map = self._compute_hash_map_for_config(past_cfg)

        # Persist for future scans (best-effort)
        try:
            with open(hash_path, "w", encoding="utf-8") as fh:
                json.dump(hash_map, fh, indent=2)
            log.info(
                "Cache: generated missing selector_hashes.json for past run %s",
                run_dir,
            )
        except OSError as exc:
            log.warning(
                "Cache: could not write generated hash map to %s: %s",
                hash_path, exc,
            )

        return hash_map

    def _compute_hash_map_for_config(self, cfg: PipelineConfig) -> dict[str, str]:
        """Compute ``{result_key: hash}`` for all enabled selectors in *cfg*."""
        hashes: dict[str, str] = {}
        for selector in cfg.selectors:
            if not selector.enabled:
                continue
            for target_name in selector.targets:
                target_cfg = next(
                    (t for t in cfg.targets if t.name == target_name), None
                )
                if target_cfg is None:
                    continue
                result_key = f"{selector.label}__{target_name}"
                hashes[result_key] = self._compute_hash(
                    selector, target_name, target_cfg, cfg
                )
        return hashes

    def _compute_hash(
        self,
        selector: SelectorConfig,
        target_name: str,
        target_cfg: TargetConfig,
        cfg: PipelineConfig,
    ) -> str:
        """Return the hex SHA-256 of the canonical key dict for this selector run."""
        key_dict = self._build_key_dict(selector, target_name, target_cfg, cfg)
        return self._hash_key_dict(key_dict)

    def _build_key_dict(
        self,
        selector: SelectorConfig,
        target_name: str,
        target_cfg: TargetConfig,
        cfg: PipelineConfig,
    ) -> dict:
        """Assemble the subset of config fields that determine selector output.

        ``label`` is intentionally excluded — two selectors that differ only
        in label produce identical results and should share a cache entry.
        """
        return {
            "data": {
                "features_path":    cfg.data.features_path,
                "labels_path":      cfg.data.labels_path,
                "sample_id_column": cfg.data.sample_id_column,
                "auto_transpose":   cfg.data.auto_transpose,
                "fillna_value":     cfg.data.fillna_value,
            },
            "preprocessing": {
                "random_seed":      cfg.preprocessing.random_seed,
                "train_pct":        cfg.preprocessing.train_pct,
                "val_pct":          cfg.preprocessing.val_pct,
                "test_pct":         cfg.preprocessing.test_pct,
                "eval_data_source": cfg.preprocessing.eval_data_source,
            },
            "selector": {
                "type":   selector.type,
                "params": selector.params,
                # label excluded — does not affect algorithm output
            },
            "target": {
                "name":      target_cfg.name,
                "task_type": target_cfg.task_type,
            },
        }

    @staticmethod
    def _hash_key_dict(d: dict) -> str:
        """Return hex SHA-256 of the JSON-serialized (sorted-keys) *d*."""
        canonical = json.dumps(d, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
