from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class BenchmarkSlice:
    split: str
    start: int
    end: int | None


@dataclass(frozen=True)
class RolloutCommandSpec:
    python_executable: str
    data_path: Path
    data_slice: BenchmarkSlice
    output_dir: Path
    results_file: Path
    log_dir: Path
    generation_config: Path
    model: str
    openai_base_url: str
    rollout_llm_client: str
    rollout_temperature: float
    llm_timeout_seconds: float
    llm_retry_wait_seconds: list[float] = field(default_factory=lambda: [5.0, 10.0, 30.0])
    rollout_num_random_seeds: int = 1
    rollout_repeat: int = 1
    max_turns: int = 30
    workers: int = 1
    skillbank_root: Path | None = None
    skillbank_top_k: int = 0
    selection_log: Path | None = None
    officeqa_docs_dirs: list[Path] = field(default_factory=list)
    officeqa_evaluator: str = "skillopt"
    officeqa_max_completion_tokens: int = 16384
    officeqa_reward_path: Path | None = None
    officeqa_reward_tolerance: float = 0.0
    officeqa_allow_fallback_evaluator: bool = False
    officeqa_use_oracle_context: bool = True
    officeqa_continue_on_infra_error: bool = False


@dataclass(frozen=True)
class EvalCommandSpec:
    python_executable: str
    data_path: Path
    data_slice: BenchmarkSlice
    output_dir: Path
    results_file: Path
    eval_file: Path
    officeqa_reward_path: Path | None = None
    officeqa_reward_tolerance: float = 0.0
    officeqa_allow_fallback_evaluator: bool = False
    officeqa_evaluator: str = "skillopt"


@dataclass(frozen=True)
class ExtractCommandSpec:
    python_executable: str
    log_dir: Path
    eval_file: Path
    records_file: Path


class BenchmarkAdapter(Protocol):
    name: str

    def load_rows(self, data_path: Path, data_slice: BenchmarkSlice) -> list[dict[str, Any]]:
        ...

    def run_rollout(self, spec: RolloutCommandSpec) -> list[str]:
        ...

    def evaluate_results(self, spec: EvalCommandSpec) -> list[str]:
        ...

    def extract_records(self, spec: ExtractCommandSpec) -> list[str]:
        ...

    def write_split_manifest(
        self,
        *,
        data_path: Path,
        run_dir: Path,
        train_slice: BenchmarkSlice,
        heldout_slice: BenchmarkSlice,
    ) -> dict[str, Any]:
        ...

    def write_ordered_records(
        self,
        *,
        source_records: Path,
        output_path: Path,
        manifest_path: Path,
        data_path: Path,
        train_slice: BenchmarkSlice,
    ) -> dict[str, Any]:
        ...


def _task_id_from_row(row: dict[str, Any], fallback: object | None = None) -> str:
    value = row.get("task_id", row.get("id", row.get("uid", row.get("instance_id", fallback))))
    if value is None:
        raise ValueError(f"row has no task id: {row}")
    return str(value)


def _load_json_rows(dataset_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("results") or payload.get("data") or payload.get("instances") or payload.get("items") or []
    else:
        rows = []
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"unsupported dataset format: {dataset_path}")
    return list(rows)


def _load_record_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = []
        for key in ("records", "data", "results", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                rows = value
                break
    else:
        rows = []
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"unsupported records format: {path}")
    return list(rows)


def _write_exact_ordered_records(
    *,
    source_records: Path,
    expected_rows: list[dict[str, Any]],
    output_path: Path,
    manifest_path: Path,
    policy: str,
    source_dataset: str,
    train_slice: BenchmarkSlice,
) -> dict[str, Any]:
    records = _load_record_rows(source_records)
    expected_ids = [_task_id_from_row(row, fallback=index) for index, row in enumerate(expected_rows, start=train_slice.start)]
    by_task_id: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for record in records:
        task_id = _task_id_from_row(record)
        if task_id in by_task_id:
            duplicates.append(task_id)
        by_task_id[task_id] = record
    missing = [task_id for task_id in expected_ids if task_id not in by_task_id]
    extra = [task_id for task_id in by_task_id if task_id not in set(expected_ids)]
    if duplicates or missing or extra:
        raise RuntimeError(
            "records.json does not match the requested train slice exactly: "
            f"duplicates={duplicates[:10]}, missing={missing[:10]}, extra={extra[:10]}"
        )
    ordered = [by_task_id[task_id] for task_id in expected_ids]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")
    source_ids = [_task_id_from_row(record) for record in records]
    manifest = {
        "policy": policy,
        "source_records": str(source_records),
        "ordered_records": str(output_path),
        "source_dataset": source_dataset,
        "train_split": train_slice.split,
        "train_range": [int(train_slice.start), None if train_slice.end is None else int(train_slice.end)],
        "record_count": len(ordered),
        "source_order_equal_dataset_order": source_ids == expected_ids,
        "first_task_ids": expected_ids[:10],
        "last_task_ids": expected_ids[-10:],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


class SpreadsheetBenchAdapter:
    name = "spreadsheetbench"

    def load_rows(self, data_path: Path, data_slice: BenchmarkSlice) -> list[dict[str, Any]]:
        dataset_path = data_path / "dataset.json" if data_path.is_dir() else data_path
        rows = _load_json_rows(dataset_path)
        end = len(rows) if data_slice.end is None else int(data_slice.end)
        return rows[int(data_slice.start):end]

    def write_split_manifest(
        self,
        *,
        data_path: Path,
        run_dir: Path,
        train_slice: BenchmarkSlice,
        heldout_slice: BenchmarkSlice,
    ) -> dict[str, Any]:
        dataset_path = data_path / "dataset.json" if data_path.is_dir() else data_path
        train = self.load_rows(data_path, train_slice)
        heldout = self.load_rows(data_path, heldout_slice)

        def subset(rows: list[dict[str, Any]], start: int) -> list[dict[str, Any]]:
            return [
                {
                    "index": index,
                    "id": str(row.get("id", row.get("task_id", index))),
                    "instruction_type": row.get("instruction_type", ""),
                    "answer_position": row.get("answer_position", ""),
                }
                for index, row in enumerate(rows, start=start)
            ]

        manifest = {
            "benchmark": self.name,
            "source_dataset_json": str(dataset_path.resolve()),
            "policy": "Trace2Skill dataset order / natural task id order; runner still uses start/end indices",
            "train_split": train_slice.split,
            "heldout_split": heldout_slice.split,
            "train_range": [train_slice.start, train_slice.end],
            "heldout_range": [heldout_slice.start, heldout_slice.end],
            "train": subset(train, train_slice.start),
            "heldout": subset(heldout, heldout_slice.start),
        }
        path = run_dir / "split_manifest.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest

    def run_rollout(self, spec: RolloutCommandSpec) -> list[str]:
        cmd = [
            spec.python_executable, "run_spreadsheetbench.py",
            "--data_path", str(spec.data_path),
            "--output_dir", str(spec.output_dir),
            "--agent", "cli_skill_preloaded" if spec.skillbank_root else "cli_only",
        ]
        if spec.skillbank_root:
            cmd.extend(["--skills_dir", str(spec.skillbank_root)])
        cmd.extend([
            "--model", spec.model,
            "--llm_client", spec.rollout_llm_client,
            "--temperature", str(spec.rollout_temperature),
            "--generation_config", str(spec.generation_config),
            "--llm_timeout_seconds", str(spec.llm_timeout_seconds),
            "--llm_retry_wait_seconds", ",".join(str(value) for value in spec.llm_retry_wait_seconds),
            "--num_random_seeds", str(spec.rollout_num_random_seeds),
            "--repeat", str(spec.rollout_repeat),
            "--max_turns", str(spec.max_turns),
            "--start_idx", str(spec.data_slice.start),
            "--workers", str(spec.workers),
            "--results_file", str(spec.results_file),
            "--log_dir", str(spec.log_dir),
            "--log_format", "markdown",
        ])
        if spec.data_slice.end is not None:
            cmd.extend(["--end_idx", str(spec.data_slice.end)])
        return cmd

    def evaluate_results(self, spec: EvalCommandSpec) -> list[str]:
        cmd = [
            spec.python_executable, "evaluate_with_official.py",
            "--data_path", str(spec.data_path),
            "--output_dir", str(spec.output_dir),
            "--start_idx", str(spec.data_slice.start),
            "--results_file", str(spec.eval_file),
        ]
        if spec.data_slice.end is not None:
            cmd.extend(["--end_idx", str(spec.data_slice.end)])
        return cmd

    def extract_records(self, spec: ExtractCommandSpec) -> list[str]:
        return [
            spec.python_executable, "scripts/extract_trace2skill_logs.py",
            "--log-dir", str(spec.log_dir),
            "--results-file", str(spec.eval_file),
            "--output", str(spec.records_file),
        ]

    def write_ordered_records(
        self,
        *,
        source_records: Path,
        output_path: Path,
        manifest_path: Path,
        data_path: Path,
        train_slice: BenchmarkSlice,
    ) -> dict[str, Any]:
        return _write_exact_ordered_records(
            source_records=source_records,
            expected_rows=self.load_rows(data_path, train_slice),
            output_path=output_path,
            manifest_path=manifest_path,
            policy="records are ordered by SpreadsheetBench dataset.json train slice order; no filename sorting or random shuffling",
            source_dataset=str((data_path / "dataset.json" if data_path.is_dir() else data_path).resolve()),
            train_slice=train_slice,
        )


class OfficeQAAdapter:
    name = "officeqa"

    def load_rows(self, data_path: Path, data_slice: BenchmarkSlice) -> list[dict[str, Any]]:
        from .officeqa import load_officeqa_items

        return [
            item.to_dict()
            for item in load_officeqa_items(data_path, split=data_slice.split, start=data_slice.start, end=data_slice.end)
        ]

    def write_split_manifest(
        self,
        *,
        data_path: Path,
        run_dir: Path,
        train_slice: BenchmarkSlice,
        heldout_slice: BenchmarkSlice,
    ) -> dict[str, Any]:
        train = self.load_rows(data_path, train_slice)
        heldout = self.load_rows(data_path, heldout_slice)
        manifest = {
            "benchmark": self.name,
            "source_split_dir": str(data_path.resolve()),
            "policy": "OfficeQA materialized split order; start/end apply within each named split",
            "train_split": train_slice.split,
            "heldout_split": heldout_slice.split,
            "train_range": [train_slice.start, train_slice.end],
            "heldout_range": [heldout_slice.start, heldout_slice.end],
            "train": [
                {"index": idx, "id": row["uid"], "category": row.get("category", "officeqa")}
                for idx, row in enumerate(train, start=train_slice.start)
            ],
            "heldout": [
                {"index": idx, "id": row["uid"], "category": row.get("category", "officeqa")}
                for idx, row in enumerate(heldout, start=heldout_slice.start)
            ],
        }
        path = run_dir / "split_manifest.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest

    def run_rollout(self, spec: RolloutCommandSpec) -> list[str]:
        cmd = [
            spec.python_executable, "scripts/run_officeqa_benchmark.py",
            "--split-dir", str(spec.data_path),
            "--split", spec.data_slice.split,
            "--output_dir", str(spec.output_dir),
            "--results_file", str(spec.results_file),
            "--log_dir", str(spec.log_dir),
            "--model", spec.model,
            "--openai-base-url", spec.openai_base_url,
            "--generation_config", str(spec.generation_config),
            "--max-completion-tokens", str(spec.officeqa_max_completion_tokens),
            "--llm_timeout_seconds", str(spec.llm_timeout_seconds),
            "--llm_retry_wait_seconds", ",".join(str(value) for value in spec.llm_retry_wait_seconds),
            "--max_turns", str(spec.max_turns),
            "--start_idx", str(spec.data_slice.start),
            "--workers", str(spec.workers),
            "--evaluator", spec.officeqa_evaluator,
            "--reward-tolerance", str(spec.officeqa_reward_tolerance),
            "--use-oracle-context", str(bool(spec.officeqa_use_oracle_context)).lower(),
        ]
        if spec.data_slice.end is not None:
            cmd.extend(["--end_idx", str(spec.data_slice.end)])
        for docs_dir in spec.officeqa_docs_dirs:
            cmd.extend(["--docs-dir", str(docs_dir)])
        if spec.officeqa_reward_path:
            cmd.extend(["--reward-path", str(spec.officeqa_reward_path)])
        if spec.officeqa_allow_fallback_evaluator:
            cmd.append("--allow-fallback-evaluator")
        if spec.officeqa_continue_on_infra_error:
            cmd.append("--continue-on-infra-error")
        if spec.skillbank_root:
            cmd.extend(["--skillbank-root", str(spec.skillbank_root)])
            cmd.extend(["--skillbank-top-k", str(spec.skillbank_top_k)])
        if spec.selection_log:
            cmd.extend(["--selection-log", str(spec.selection_log)])
        return cmd

    def evaluate_results(self, spec: EvalCommandSpec) -> list[str]:
        cmd = [
            spec.python_executable, "scripts/evaluate_officeqa_results.py",
            "--results-file", str(spec.results_file),
            "--split-dir", str(spec.data_path),
            "--split", spec.data_slice.split,
            "--output", str(spec.eval_file),
            "--evaluator", spec.officeqa_evaluator,
            "--reward-tolerance", str(spec.officeqa_reward_tolerance),
        ]
        if spec.officeqa_reward_path:
            cmd.extend(["--reward-path", str(spec.officeqa_reward_path)])
        if spec.officeqa_allow_fallback_evaluator:
            cmd.append("--allow-fallback-evaluator")
        return cmd

    def extract_records(self, spec: ExtractCommandSpec) -> list[str]:
        return [
            spec.python_executable, "scripts/extract_officeqa_records.py",
            "--results-file", str(spec.eval_file),
            "--output", str(spec.records_file),
        ]

    def write_ordered_records(
        self,
        *,
        source_records: Path,
        output_path: Path,
        manifest_path: Path,
        data_path: Path,
        train_slice: BenchmarkSlice,
    ) -> dict[str, Any]:
        return _write_exact_ordered_records(
            source_records=source_records,
            expected_rows=self.load_rows(data_path, train_slice),
            output_path=output_path,
            manifest_path=manifest_path,
            policy="records are ordered by OfficeQA materialized split order; gold answers are not included in retrieval queries",
            source_dataset=str(data_path.resolve()),
            train_slice=train_slice,
        )


def get_benchmark_adapter(name: str) -> BenchmarkAdapter:
    normalized = str(name).strip().lower()
    if normalized == "spreadsheetbench":
        return SpreadsheetBenchAdapter()
    if normalized == "officeqa":
        return OfficeQAAdapter()
    raise ValueError(f"unsupported benchmark: {name!r}")
