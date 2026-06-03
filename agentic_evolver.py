#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
import concurrent.futures as cf
import copy
import dataclasses
from dataclasses import dataclass, field
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import traceback
import urllib.error
import urllib.request
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# -----------------------------
# Utility helpers
# -----------------------------


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    tmp = path.with_name(f".{path.name}.{threading.get_ident()}.{time.time_ns()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def short_hash(text: str, n: int = 12) -> str:
    return hashlib.blake2b(text.encode("utf-8", errors="ignore"), digest_size=16).hexdigest()[:n]


def sha_text(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8", errors="ignore"), digest_size=20).hexdigest()


def sha_files(files: Dict[str, str]) -> str:
    h = hashlib.blake2b(digest_size=20)
    for path in sorted(files):
        h.update(path.encode("utf-8", errors="ignore"))
        h.update(b"\0")
        h.update(sha_text(files[path]).encode("ascii"))
        h.update(b"\0")
    return h.hexdigest()


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)


TRUNCATION_MARKER = "\n\n... <TRUNCATED> ...\n\n"


def tail_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[-max_chars:]


def truncate_middle(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= len(TRUNCATION_MARKER):
        return text[:max_chars]
    available = max_chars - len(TRUNCATION_MARKER)
    head = (available + 1) // 2
    tail = available // 2
    return text[:head] + TRUNCATION_MARKER + text[-tail:]


def strip_outer_code_fence(text: str) -> str:
    normalized = normalize_newlines(text).strip()
    if not normalized.startswith("```"):
        return normalize_newlines(text)
    match = re.match(r"^```[A-Za-z0-9_+-]*\s*\n(?P<body>.*)\n```\s*$", normalized, re.DOTALL)
    if match:
        return match.group("body")
    return normalize_newlines(text)


def clean_llm_file_content(text: str) -> str:
    return normalize_newlines(strip_outer_code_fence(text))


def content_looks_incomplete(text: str) -> Optional[str]:
    lowered = text.lower()
    bad_phrases = [
        "... <truncated> ...",
        "omitted for brevity",
        "rest of file unchanged",
        "remaining code unchanged",
        "unchanged code omitted",
        "same as before",
        "<existing code>",
        "<unchanged>",
    ]
    for phrase in bad_phrases:
        if phrase in lowered:
            return phrase
    return None


def safe_relpath(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def deep_get(obj: Any, path: str, default: Any = None) -> Any:
    cur = obj
    for part in path.split("."):
        if isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except Exception:
                return default
        elif isinstance(cur, dict):
            if part not in cur:
                return default
            cur = cur[part]
        else:
            return default
    return cur


def deep_render(obj: Any, mapping: Dict[str, Any]) -> Any:
    if isinstance(obj, dict):
        return {k: deep_render(v, mapping) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deep_render(v, mapping) for v in obj]
    if isinstance(obj, str):
        # Exact-placeholder replacement preserves native type.
        if obj in mapping:
            return mapping[obj]
        out = obj
        for key, value in mapping.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            out = out.replace(key, str(value))
        return out
    return obj


COMMAND_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def render_template(text: str, mapping: Dict[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in mapping:
            return match.group(0)
        return str(mapping[key])

    return COMMAND_PLACEHOLDER_RE.sub(replace, text)


def extract_json_blob(text: str) -> Optional[str]:
    text = text.strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # Find first balanced object or array.
    start_positions = [pos for pos in [text.find("{"), text.find("[")] if pos != -1]
    if not start_positions:
        return None
    start = min(start_positions)
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_jsonish(text: str) -> Optional[Any]:
    blob = extract_json_blob(text)
    if blob is None:
        return None
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        # Last-ditch cleanup of common LLM sins.
        fixed = blob.replace("\u201c", '"').replace("\u201d", '"')
        fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            return None


def parse_jsonish_values(text: str, max_values: int = 20) -> List[Any]:
    values: List[Any] = []
    pos = 0
    while pos < len(text) and len(values) < max_values:
        next_positions = [idx for idx in [text.find("{", pos), text.find("[", pos)] if idx != -1]
        if not next_positions:
            break
        start = min(next_positions)
        blob = extract_json_blob(text[start:])
        if blob is None:
            pos = start + 1
            continue
        try:
            values.append(json.loads(blob))
        except json.JSONDecodeError:
            fixed = blob.replace("\u201c", '"').replace("\u201d", '"')
            fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
            try:
                values.append(json.loads(fixed))
            except json.JSONDecodeError:
                pass
        pos = start + max(1, len(blob))
    return values


def normalize_metric_key(key: str) -> str:
    raw = key.strip().lower()
    raw = raw.replace("%", " percent ")
    raw = raw.replace("/", " per ")
    raw = raw.replace("-", " ")
    normalized = re.sub(r"[^a-z0-9.]+", "_", raw).strip("_.")
    compact = normalized.replace(".", "_")
    leaf = normalized.split(".")[-1] if normalized else ""
    leaf_compact = leaf.replace(".", "_")
    parts = set(p for p in compact.split("_") if p)

    if leaf_compact in {"score", "final_score", "metric_score"}:
        return "score"
    if "fps" in parts:
        has_percent = bool(parts & {"percent", "pct"}) or "pct" in compact or "percent" in compact
        has_low = "low" in parts
        if has_low and ("0_1" in compact or "01pct" in compact or {"0", "1"} <= parts) and has_percent:
            return "fps_0_1pct_low"
        if has_low and has_percent and ("1" in parts or "1pct" in compact or "1percent" in compact):
            return "fps_1pct_low"
        if parts & {"avg", "average", "mean"}:
            return "fps_avg"
        if "min" in parts:
            return "fps_min"
        if "max" in parts:
            return "fps_max"
        return "fps"
    if "frametime" in parts or {"frame", "time"} <= parts or "frame_time" in compact:
        if parts & {"p99", "99"}:
            return "frame_time_p99_ms"
        if parts & {"p95", "95"}:
            return "frame_time_p95_ms"
        if parts & {"avg", "average", "mean"}:
            return "frame_time_avg_ms"
        return "frame_time_ms"
    is_power = bool(parts & {"power", "watt", "watts"}) or compact.endswith("_w")
    if is_power:
        if "gpu" in parts:
            return "gpu_power_w"
        if "cpu" in parts:
            return "cpu_power_w"
        if "package" in parts:
            return "package_power_w"
        if parts & {"avg", "average", "mean"}:
            return "power_avg_w"
        return "power_w"
    if "energy" in parts:
        if parts & {"joule", "joules", "j"}:
            return "energy_j"
        return "energy"
    return compact or "metric"


def collect_numeric_metrics(obj: Any, prefix: str = "") -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                visit(child, child_path)
            return
        if isinstance(value, list):
            for idx, child in enumerate(value[:20]):
                child_path = f"{path}.{idx}" if path else str(idx)
                visit(child, child_path)
            return
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return
        numeric = float(value)
        if not math.isfinite(numeric):
            return
        normalized = normalize_metric_key(path)
        metrics.setdefault(normalized, numeric)
        raw_normalized = re.sub(r"[^a-zA-Z0-9]+", "_", path).strip("_").lower()
        if raw_normalized and raw_normalized != normalized:
            metrics.setdefault(raw_normalized, numeric)

    visit(obj, prefix)
    return metrics


def metric_objective_direction(objectives: Dict[str, str], key: str) -> str:
    explicit = str(objectives.get(key, "")).strip().lower()
    if explicit in {"max", "maximize", "higher", "higher_is_better", "higher-is-better"}:
        return "maximize"
    if explicit in {"min", "minimize", "lower", "lower_is_better", "lower-is-better"}:
        return "minimize"
    lowered = key.lower()
    if lowered == "score" or "fps" in lowered:
        return "maximize"
    if "power" in lowered or "energy" in lowered or "frame_time" in lowered or lowered.endswith("_ms"):
        return "minimize"
    return "observe"


def metric_improvement_from_delta(objectives: Dict[str, str], key: str, delta: float) -> Optional[float]:
    direction = metric_objective_direction(objectives, key)
    if direction == "maximize":
        return delta
    if direction == "minimize":
        return -delta
    return None


def normalize_metric_keys(keys: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for key in keys:
        item = normalize_metric_key(str(key))
        if item and item not in seen:
            seen.add(item)
            normalized.append(item)
    return normalized


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def pick_weighted(items: Sequence[Any], weights: Sequence[float], k: int, rng: random.Random) -> List[Any]:
    if not items:
        return []
    picked: List[Any] = []
    pool_items = list(items)
    pool_weights = [max(0.0, w) for w in weights]
    while pool_items and len(picked) < k:
        total = sum(pool_weights)
        if total <= 0:
            idx = rng.randrange(len(pool_items))
        else:
            r = rng.random() * total
            acc = 0.0
            idx = 0
            for i, w in enumerate(pool_weights):
                acc += w
                if acc >= r:
                    idx = i
                    break
        picked.append(pool_items.pop(idx))
        pool_weights.pop(idx)
    return picked


# -----------------------------
# Config model
# -----------------------------


@dataclass
class LLMConfig:
    endpoint: str
    headers: Dict[str, str] = field(default_factory=lambda: {"Content-Type": "application/json"})
    payload_template: Dict[str, Any] = field(
        default_factory=lambda: {
            "messages": [
                {"role": "system", "content": "${system}"},
                {"role": "user", "content": "${user}"},
            ],
            "temperature": "${temperature}",
            "max_tokens": "${max_output_tokens}",
        }
    )
    response_text_path: str = "choices.0.message.content"
    timeout_sec: int = 180
    temperature_plan: float = 0.85
    temperature_mutate: float = 0.7
    temperature_repair: float = 0.2
    max_output_tokens: int = 128000
    mutation_repair_output_token_multiplier: int = 2
    json_retry_attempts: int = 2
    json_retry_output_token_multiplier: int = 2
    json_retry_include_invalid_excerpt_chars: int = 12000


@dataclass
class EvolutionConfig:
    project_root: str
    work_root: str
    prepare_command: str = ""
    compile_command: str = ""
    deploy_command: str = ""
    device_evaluate_command: str = ""
    device_handoff_ready_marker: str = ""
    evaluate_command: str = ""
    optimization_goal: str = (
        "Maximize the evaluator score by improving sustained FPS, frame time stability, "
        "and useful work per watt without changing externally visible behavior."
    )
    target_fps: float = 120.0
    target_fps_metric: str = "fps_1pct_low"
    metric_notes: str = (
        "Assume wins come from reducing per-frame CPU/GPU work, synchronization stalls, "
        "memory bandwidth, allocations, wakeups, polling, logging, and avoidable device/API calls."
    )
    cxx_standard: str = "C++17"
    primary_metric_keys: List[str] = field(default_factory=lambda: ["fps_1pct_low", "power_w"])
    metric_objectives: Dict[str, str] = field(
        default_factory=lambda: {
            "score": "maximize",
            "fps": "maximize",
            "fps_avg": "maximize",
            "fps_min": "maximize",
            "fps_1pct_low": "maximize",
            "fps_0_1pct_low": "maximize",
            "frame_time_ms": "minimize",
            "frame_time_avg_ms": "minimize",
            "frame_time_p95_ms": "minimize",
            "frame_time_p99_ms": "minimize",
            "power_w": "minimize",
            "power_avg_w": "minimize",
            "cpu_power_w": "minimize",
            "gpu_power_w": "minimize",
            "package_power_w": "minimize",
            "energy_j": "minimize",
        }
    )
    mutable_extensions: List[str] = field(default_factory=lambda: [".cpp", ".cc", ".cxx"])
    mutable_header_extensions: List[str] = field(default_factory=lambda: [".h", ".hh", ".hpp", ".hxx"])
    allow_header_mutations: bool = True
    pair_headers_with_sources: bool = True
    max_paired_headers_per_source: int = 2
    immutable_extensions: List[str] = field(default_factory=lambda: [".h", ".hh", ".hpp", ".hxx"])
    ignore_dirs: List[str] = field(default_factory=lambda: [".git", "build", "dist", "out", "cmake-build-debug", "cmake-build-release"])
    generations: int = 100
    children_per_generation: int = 10
    max_focus_files_per_child: int = 3
    enable_constant_tuning_plans: bool = True
    constant_tuning_plan_fraction: float = 0.2
    max_constant_tuning_plans_per_generation: int = 3
    max_constant_hints_per_file: int = 16
    enable_algorithmic_model_plans: bool = True
    algorithmic_model_plan_fraction: float = 0.5
    max_algorithmic_model_plans_per_generation: int = 6
    max_model_hints_per_file: int = 16
    enable_impact_heuristic_plans: bool = True
    impact_heuristic_plan_fraction: float = 0.3
    max_impact_heuristic_plans_per_generation: int = 4
    reject_trivial_micro_optimizations: bool = True
    max_llm_workers: int = 8
    max_eval_workers: int = 1
    parent_pool_size: int = 8
    elite_keep: int = 16
    archive_keep_per_signature: int = 2
    compile_fail_score: float = -1e9
    max_compile_repairs: int = 20
    planner_children_per_call: int = 10
    enable_llm_summaries: bool = True
    include_full_target_files: bool = True
    max_context_chars_per_file: int = 0
    max_neighbor_chars_total: int = 0
    max_neighbor_chars_per_file: int = 0
    max_repair_error_chars: int = 0
    max_project_summary_files: int = 80
    enable_deep_project_analysis: bool = True
    max_project_analysis_files: int = 0
    max_project_analysis_chars_per_file: int = 0
    max_file_summary_output_tokens: int = 8000
    max_project_analysis_output_tokens: int = 240000
    max_planning_output_tokens: int = 120000
    enable_hypothesis_learning: bool = True
    max_hypothesis_learning_output_tokens: int = 120000
    max_hypothesis_learning_log_chars: int = 40000
    strategy_memory_keep: int = 20
    review_pass: bool = True
    review_single_file_pass: bool = True
    repair_all_files_per_call: bool = True
    pipeline_first_generation_during_baseline: bool = True
    stream_evaluate_candidates: bool = True
    pre_materialize_built_candidates: bool = True
    random_seed: int = 1234
    score_pattern: str = r"SCORE\s*[:=]\s*(-?\d+(?:\.\d+)?)"
    compile_error_markers: List[str] = field(default_factory=lambda: ["compilation_error"])
    keep_failed_candidates: bool = True
    keep_successful_candidates: bool = True
    use_hardlink_copy: bool = False
    command_timeout_sec: int = 0


@dataclass
class FileSummary:
    path: str
    includes: List[str]
    classes: List[str]
    functions: List[str]
    lines: int
    kind: str = ""
    purpose: str = ""
    hot_hints: List[str] = field(default_factory=list)
    numeric_constants: List[str] = field(default_factory=list)
    model_hints: List[str] = field(default_factory=list)
    neighbors: List[str] = field(default_factory=list)
    paired_files: List[str] = field(default_factory=list)


@dataclass
class ChildPlan:
    focus_files: List[str]
    hypothesis: str
    intents: Dict[str, str]
    style: str
    parent_id: str
    plan_id: str


@dataclass
class Candidate:
    candidate_id: str
    generation: int
    parent_id: str
    files: Dict[str, str]
    changed_files: List[str]
    hypothesis: str
    intents: Dict[str, str]
    style: str
    llm_notes: List[str] = field(default_factory=list)
    score: Optional[float] = None
    compile_ok: Optional[bool] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    eval_stage: str = ""
    stdout: str = ""
    stderr: str = ""
    work_dir: Optional[str] = None


@dataclass
class EvalResult:
    score: float
    compile_ok: bool
    return_code: int
    stdout: str
    stderr: str
    work_dir: str
    metrics: Dict[str, float] = field(default_factory=dict)
    stage: str = "evaluate"
    ready_for_device: bool = False


# -----------------------------
# HTTP LLM client
# -----------------------------


class ServerLLMClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self._lock = threading.Lock()

    def _build_payload(self, system_prompt: str, user_prompt: str, *, temperature: float, max_output_tokens: int) -> Dict[str, Any]:
        mapping = {
            "${system}": system_prompt,
            "${user}": user_prompt,
            "${temperature}": temperature,
            "${max_output_tokens}": max_output_tokens,
        }
        return deep_render(copy.deepcopy(self.cfg.payload_template), mapping)

    def complete_text(self, system_prompt: str, user_prompt: str, *, temperature: float, max_output_tokens: Optional[int] = None) -> str:
        payload = self._build_payload(
            system_prompt,
            user_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens or self.cfg.max_output_tokens,
        )
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.cfg.endpoint,
            data=data,
            headers=self.cfg.headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.cfg.timeout_sec) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
            raise RuntimeError(f"LLM HTTPError {exc.code}: {body[:2000]}") from exc
        except Exception as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc

        response_obj = parse_jsonish(raw)
        if response_obj is None:
            # Some servers already return plain text. Humans do this too. Chaos reigns.
            return raw
        value = deep_get(response_obj, self.cfg.response_text_path)
        if value is None:
            return raw
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)

    def complete_json(self, system_prompt: str, user_prompt: str, *, temperature: float, max_output_tokens: Optional[int] = None) -> Any:
        attempts = max(1, int(self.cfg.json_retry_attempts) + 1)
        token_budget = max_output_tokens or self.cfg.max_output_tokens
        last_text = ""
        last_error = ""
        retry_note = ""
        for attempt in range(attempts):
            prompt = user_prompt
            if retry_note:
                prompt = user_prompt + "\n\n" + retry_note
            try:
                text = self.complete_text(
                    system_prompt,
                    prompt,
                    temperature=temperature,
                    max_output_tokens=token_budget,
                )
            except Exception as exc:
                last_error = str(exc)
                text = ""
            last_text = text
            obj = parse_jsonish(text) if text else None
            if obj is not None:
                return obj
            token_budget *= max(1, int(self.cfg.json_retry_output_token_multiplier))
            previous_excerpt = truncate_middle(
                text if text else f"<no response text; last error: {last_error}>",
                max(0, int(self.cfg.json_retry_include_invalid_excerpt_chars)),
            )
            retry_note = textwrap.dedent(
                f"""
                IMPORTANT RETRY:
                Your previous response was not parseable as strict JSON, or it may have been truncated.
                Return exactly one complete JSON object/array matching the requested schema.
                Do not include markdown fences, prose before/after JSON, placeholders, omitted code, or partial content.
                Increase completeness over brevity.

                Previous invalid response excerpt to repair:
                {previous_excerpt}
                """
            ).strip()
        detail = f" Last error: {last_error}" if last_error else ""
        raise RuntimeError(f"LLM did not return valid JSON after {attempts} attempt(s).{detail} Raw head:\n{last_text[:3000]}")


# -----------------------------
# Project indexing and summaries
# -----------------------------


INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^">]+)[">]', re.MULTILINE)
CLASS_RE = re.compile(r'\b(?:class|struct)\s+([A-Za-z_][A-Za-z0-9_]*)')
FUNC_RE = re.compile(
    r'^\s*(?:template\s*<[^;{]+>\s*)?(?:(?:inline|static|virtual|constexpr|friend|extern)\s+)*'
    r'(?:[A-Za-z_][\w:<>,\s\*&~]*?)\s+'
    r'([A-Za-z_~][\w:]*)\s*\([^;{}]*\)\s*(?:const\s*)?(?:noexcept\s*)?(?:->\s*[^{]+)?\{',
    re.MULTILINE,
)
NAMESPACE_RE = re.compile(r'\bnamespace\s+([A-Za-z_][A-Za-z0-9_]*)')
NUMERIC_LITERAL_RE = re.compile(r'(?<![A-Za-z0-9_])[-+]?(?:0x[0-9A-Fa-f]+|\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?:[fFuUlL]+)?(?![A-Za-z0-9_])')
CONSTANT_HINT_RE = re.compile(
    r'#\s*define|\bconstexpr\b|\bconst\b|\benum\b|\bstatic\b|'
    r'\b(?:fps|frame|power|watt|energy|time|interval|timeout|sleep|poll|spin|'
    r'batch|buffer|cache|pool|queue|limit|max|min|threshold|quality|scale|size|count|'
    r'capacity|reserve|samples|iterations|threads|workers)\b',
    re.IGNORECASE,
)
MODEL_DATA_HINT_RE = re.compile(
    r'\b(?:model|predict|prediction|predictor|forecast|estimate|estimator|regress|regression|'
    r'classifier|classify|feature|features|weight|weights|score|confidence|probability|'
    r'filter|kalman|ema|moving_average|average|smooth|smoothing|history|sample|samples|'
    r'dataset|training|train|fit|learn|learning|inference|window|buffer|cache|queue|'
    r'threshold|heuristic|policy|algorithm|cluster|interpolate|extrapolate|lookup|table)\b',
    re.IGNORECASE,
)
TRIVIAL_MICRO_OPT_RE = re.compile(
    r'\b(?:i\+\+|\+\+i|pre[-\s]?increment|post[-\s]?increment|const[-\s]?correct|'
    r'format(?:ting)?|whitespace|rename|naming|style[-\s]?only|cosmetic|'
    r'micro[-\s]?optim(?:i[sz]ation|ize)|tidy|cleanup|clean\s+up)\b',
    re.IGNORECASE,
)
MEANINGFUL_MECHANISM_RE = re.compile(
    r'\b(?:algorithm|model|heuristic|prediction|filter|smooth|history|feature|cache|cached|'
    r'incremental|batch|coalesce|gate|early[-\s]?exit|bounded|window|reuse|precompute|'
    r'allocation|copy|memory|bandwidth|synchroni[sz]ation|wakeups?|polling|sort|scan|'
    r'top[-\s]?k|queue|buffer|data\s+volume|per[-\s]?frame\s+work|power|fps_?1pct|1%\s*low)\b',
    re.IGNORECASE,
)


class ProjectIndex:
    def __init__(self, cfg: EvolutionConfig, llm: ServerLLMClient):
        self.cfg = cfg
        self.project_root = Path(cfg.project_root).resolve()
        self.work_root = Path(cfg.work_root).resolve()
        self.llm = llm
        ensure_dir(self.work_root)
        self.cache_dir = self.work_root / "cache"
        ensure_dir(self.cache_dir)
        self.summary_cache_path = self.cache_dir / "file_summaries.json"
        self.project_analysis_cache_path = self.cache_dir / "project_analysis.json"
        self.mutable_files: List[str] = []
        self.mutable_source_files: List[str] = []
        self.mutable_header_files: List[str] = []
        self.immutable_files: List[str] = []
        self.all_files: List[str] = []
        self.text_by_path: Dict[str, str] = {}
        self.static_summaries: Dict[str, FileSummary] = {}
        self.shared_include_index: Dict[str, List[str]] = {}
        self.paired_headers_by_source: Dict[str, List[str]] = {}
        self.paired_sources_by_header: Dict[str, List[str]] = {}
        self.project_analysis: Dict[str, Any] = {}
        self.project_analysis_text: str = ""
        self._scan_project()
        self._build_static_summaries()
        if self.cfg.enable_llm_summaries:
            self._enrich_summaries_with_llm()
        if self.cfg.enable_deep_project_analysis:
            self._build_deep_project_analysis()

    def is_source_path(self, path: str) -> bool:
        return Path(path).suffix in self.cfg.mutable_extensions

    def is_header_path(self, path: str) -> bool:
        return Path(path).suffix in self.cfg.mutable_header_extensions

    def _iter_source_files(self) -> Iterable[Path]:
        ignored = set(self.cfg.ignore_dirs)
        for root, dirs, files in os.walk(self.project_root):
            dirs[:] = [d for d in dirs if d not in ignored]
            root_path = Path(root)
            for name in files:
                yield root_path / name

    def _scan_project(self) -> None:
        self.mutable_files.clear()
        self.mutable_source_files.clear()
        self.mutable_header_files.clear()
        self.immutable_files.clear()
        self.all_files.clear()
        self.text_by_path.clear()
        for path in self._iter_source_files():
            rel = safe_relpath(path, self.project_root)
            self.all_files.append(rel)
            try:
                text = normalize_newlines(read_text(path))
            except Exception:
                continue
            self.text_by_path[rel] = text
            if self.is_source_path(rel):
                self.mutable_files.append(rel)
                self.mutable_source_files.append(rel)
            elif self.cfg.allow_header_mutations and self.is_header_path(rel):
                self.mutable_files.append(rel)
                self.mutable_header_files.append(rel)
            elif path.suffix in self.cfg.immutable_extensions or self.is_header_path(rel):
                self.immutable_files.append(rel)
        self.mutable_files.sort()
        self.mutable_source_files.sort()
        self.mutable_header_files.sort()
        self.immutable_files.sort()
        self.all_files.sort()

    def _heuristic_hot_hints(self, text: str) -> List[str]:
        hints: List[str] = []
        samples = [
            (r'\bframe\b|\bfps\b|present|swapBuffers|vkQueuePresent|IDXGISwapChain|render\s*\(|tick\s*\(|update\s*\(', 'per-frame path'),
            (r'vkDeviceWaitIdle|vkQueueWaitIdle|glFinish|cudaDeviceSynchronize|clFinish|Present\s*\(', 'CPU/GPU sync stall'),
            (r'memcpy|memmove|std::copy|std::fill|std::sort|std::stable_sort|std::partial_sort|std::nth_element|std::accumulate', 'memory bandwidth / bulk work'),
            (r'\bnew\b|\bdelete\b|std::make_unique|std::make_shared', 'heap traffic'),
            (r'std::vector<|std::string|std::unordered_map|std::map|push_back|emplace_back|insert\s*\(|erase\s*\(|resize\s*\(|reserve\s*\(', 'container heavy'),
            (r'for\s*\(|while\s*\(', 'hot loops'),
            (r'\bmutex\b|std::atomic|std::thread|std::async', 'concurrency sensitive'),
            (r'printf\(|std::cout|std::cerr|LOG\(|spdlog', 'logging overhead'),
            (r'pow\(|sqrt\(|sin\(|cos\(', 'math heavy'),
            (r'sleep_for|Sleep\(|usleep|poll\(|select\(|spin|busy', 'wakeups / idle power'),
            (r'cuda|clEnqueue|vkQueue|egl|gl[A-Z]', 'device / graphics interaction'),
            (r'\bmodel|predict|prediction|feature|history|filter|smooth|estimate|score\b', 'prediction/model/data path'),
            (r'\bthreshold|limit|window|batch|quality|scale|sample|interval|timeout|budget\b', 'tunable heuristic / threshold'),
            (r'\bdirty|cache|cached|generation|version|reuse|incremental|last[A-Z_]|prev[A-Z_]', 'cache/incremental opportunity'),
        ]
        for pattern, label in samples:
            if re.search(pattern, text):
                hints.append(label)
        return hints[:6]

    def _numeric_constant_hints(self, text: str) -> List[str]:
        hints: List[str] = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("//") or stripped.startswith("*"):
                continue
            if "#include" in stripped:
                continue
            if not NUMERIC_LITERAL_RE.search(stripped):
                continue
            if not CONSTANT_HINT_RE.search(stripped):
                continue
            hints.append(f"L{lineno}: {truncate_middle(stripped, 180)}")
            if len(hints) >= max(1, self.cfg.max_constant_hints_per_file):
                break
        return hints

    def _model_data_hints(self, text: str) -> List[str]:
        hints: List[str] = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("//") or stripped.startswith("*"):
                continue
            if not MODEL_DATA_HINT_RE.search(stripped):
                continue
            hints.append(f"L{lineno}: {truncate_middle(stripped, 200)}")
            if len(hints) >= max(1, self.cfg.max_model_hints_per_file):
                break
        return hints

    def summarize_source_text(self, path: str, text: str) -> FileSummary:
        includes = INCLUDE_RE.findall(text)[:40]
        classes = list(dict.fromkeys(CLASS_RE.findall(text)))[:20]
        functions = list(dict.fromkeys(FUNC_RE.findall(text)))[:30]
        lines = text.count("\n") + 1 if text else 0
        kind = "header" if self.is_header_path(path) else "source" if self.is_source_path(path) else "other"
        return FileSummary(
            path=path,
            includes=includes,
            classes=classes,
            functions=functions,
            lines=lines,
            kind=kind,
            purpose="",
            hot_hints=self._heuristic_hot_hints(text),
            numeric_constants=self._numeric_constant_hints(text),
            model_hints=self._model_data_hints(text),
            neighbors=[],
            paired_files=[],
        )

    def _header_include_matches(self, source_path: str, include_name: str, header_path: str) -> bool:
        include_norm = include_name.replace("\\", "/")
        header_norm = header_path.replace("\\", "/")
        if header_norm.endswith(include_norm):
            return True
        if Path(include_norm).name == Path(header_norm).name:
            return True
        source_dir = Path(source_path).parent.as_posix()
        if source_dir and f"{source_dir}/{include_norm}" == header_norm:
            return True
        return False

    def find_paired_headers(self, source_path: str, limit: Optional[int] = None) -> List[str]:
        if not self.cfg.pair_headers_with_sources or not self.is_source_path(source_path):
            return []
        limit = limit if limit is not None else max(1, self.cfg.max_paired_headers_per_source)
        source_summary = self.static_summaries.get(source_path)
        source_stem = Path(source_path).stem
        source_dir = Path(source_path).parent.as_posix()
        scored: Dict[str, float] = {}
        for header in self.mutable_header_files:
            header_path = Path(header)
            score = 0.0
            if header_path.stem == source_stem:
                score += 6.0
            if header_path.parent.as_posix() == source_dir:
                score += 3.0
            if source_summary:
                for inc in source_summary.includes:
                    if self._header_include_matches(source_path, inc, header):
                        score += 10.0
                        break
            if score > 0:
                scored[header] = score
        ranked = sorted(scored.items(), key=lambda kv: (-kv[1], len(kv[0]), kv[0]))
        return [path for path, _ in ranked[:limit]]

    def _build_file_pairings(self) -> None:
        self.paired_headers_by_source.clear()
        self.paired_sources_by_header.clear()
        if not self.cfg.pair_headers_with_sources:
            return
        for source in self.mutable_source_files:
            headers = self.find_paired_headers(source, limit=max(1, self.cfg.max_paired_headers_per_source))
            if not headers:
                continue
            self.paired_headers_by_source[source] = headers
            for header in headers:
                self.paired_sources_by_header.setdefault(header, []).append(source)
        for paths in self.paired_sources_by_header.values():
            paths.sort()

    def paired_files_for(self, path: str) -> List[str]:
        if self.is_source_path(path):
            return list(self.paired_headers_by_source.get(path, []))
        if self.is_header_path(path):
            return list(self.paired_sources_by_header.get(path, []))
        return []

    def expand_focus_with_pairs(self, focus: Sequence[str], max_files: Optional[int] = None) -> List[str]:
        if not self.cfg.pair_headers_with_sources:
            return list(dict.fromkeys(focus))
        limit = max_files if max_files is not None else self.cfg.max_focus_files_per_child + self.cfg.max_paired_headers_per_source
        limit = max(1, limit)
        expanded: List[str] = []
        for path in focus:
            if path not in self.mutable_files or path in expanded:
                continue
            expanded.append(path)
            for paired in self.paired_files_for(path):
                if paired in self.mutable_files and paired not in expanded:
                    expanded.append(paired)
                if len(expanded) >= limit:
                    break
            if len(expanded) >= limit:
                break
        return expanded

    def _build_static_summaries(self) -> None:
        self.static_summaries.clear()
        self.shared_include_index.clear()
        for path in self.mutable_files + self.immutable_files:
            text = self.text_by_path.get(path, "")
            summary = self.summarize_source_text(path, text)
            for inc in summary.includes:
                self.shared_include_index.setdefault(inc, []).append(path)
            self.static_summaries[path] = summary
        self._build_file_pairings()
        for path, summary in self.static_summaries.items():
            summary.paired_files = self.paired_files_for(path)
        for path in self.mutable_files:
            self.static_summaries[path].neighbors = self.find_neighbors(path, limit=8)

    def _load_cached_summaries(self) -> Dict[str, Dict[str, Any]]:
        if not self.summary_cache_path.exists():
            return {}
        try:
            return json.loads(read_text(self.summary_cache_path))
        except Exception:
            return {}

    def _save_cached_summaries(self, data: Dict[str, Dict[str, Any]]) -> None:
        write_text(self.summary_cache_path, json_dumps(data))

    def _project_fingerprint(self) -> str:
        relevant = {path: self.text_by_path[path] for path in self.mutable_files + self.immutable_files if path in self.text_by_path}
        return sha_files(relevant)

    def _load_cached_project_analysis(self) -> Dict[str, Any]:
        if not self.project_analysis_cache_path.exists():
            return {}
        try:
            return json.loads(read_text(self.project_analysis_cache_path))
        except Exception:
            return {}

    def _save_cached_project_analysis(self, payload: Dict[str, Any]) -> None:
        write_text(self.project_analysis_cache_path, json_dumps(payload))

    def _project_analysis_file_rows(self) -> List[Dict[str, Any]]:
        files = list(self.mutable_files + self.immutable_files)
        if self.cfg.max_project_analysis_files > 0:
            files = files[: self.cfg.max_project_analysis_files]
        rows: List[Dict[str, Any]] = []
        for path in files:
            summary = self.static_summaries.get(path)
            if summary is None:
                continue
            row: Dict[str, Any] = {
                "path": path,
                "kind": summary.kind,
                "lines": summary.lines,
                "purpose": summary.purpose,
                "hot_hints": summary.hot_hints,
                "numeric_constants": summary.numeric_constants,
                "model_hints": summary.model_hints,
                "includes": summary.includes[:25],
                "classes": summary.classes[:20],
                "functions": summary.functions[:30],
                "paired_files": summary.paired_files,
                "neighbors": summary.neighbors[:8],
            }
            if self.cfg.max_project_analysis_chars_per_file >= 0:
                text = self.text_by_path.get(path, "")
                if text:
                    row["source_excerpt"] = truncate_middle(text, self.cfg.max_project_analysis_chars_per_file)
            rows.append(row)
        return rows

    def _format_project_analysis(self, analysis: Dict[str, Any]) -> str:
        if not analysis:
            return ""
        sections = [
            ("Architecture", analysis.get("architecture")),
            ("Execution/Data Flow", analysis.get("execution_flow")),
            ("Hotspot Map", analysis.get("hotspot_map")),
            ("Model/Data Paths", analysis.get("model_and_data_paths")),
            ("Source/Header Contracts", analysis.get("source_header_contracts")),
            ("Optimization Levers", analysis.get("optimization_levers")),
            ("Risk Boundaries", analysis.get("risk_boundaries")),
            ("Planning Strategy", analysis.get("planning_strategy")),
        ]
        lines: List[str] = []
        for title, value in sections:
            if not value:
                continue
            lines.append(f"### {title}")
            if isinstance(value, str):
                lines.append(value)
            else:
                lines.append(json_dumps(value))
        return "\n".join(lines)

    def _fallback_project_analysis(self) -> Dict[str, Any]:
        hot_files = []
        for path in self.mutable_files:
            summary = self.static_summaries.get(path)
            if summary and summary.hot_hints:
                hot_files.append({"path": path, "kind": summary.kind, "hot_hints": summary.hot_hints, "paired_files": summary.paired_files})
        return {
            "architecture": "LLM deep project analysis was unavailable. Use file summaries, source/header pairings, includes, and history as the current project model.",
            "execution_flow": [],
            "hotspot_map": hot_files[:40],
            "model_and_data_paths": [
                {"files": [row["path"]], "model_or_algorithm": "detected from textual hints", "data_used": "unknown", "possible_issue": ", ".join(row.get("hot_hints", [])), "candidate_redesign": "reduce model/data work or use incremental/cached state"}
                for row in hot_files[:20]
                if "prediction/model/data path" in row.get("hot_hints", [])
            ],
            "source_header_contracts": self.paired_headers_by_source,
            "optimization_levers": [
                "Reduce per-frame work and synchronization in files tagged with per-frame, device, memory, allocation, or logging hints.",
                "For prediction/model/data paths, reduce data scanned per frame, use bounded history, cached aggregates, cheaper features, or simpler/incremental algorithms.",
                "Keep source/header edits coordinated through paired_files.",
            ],
            "risk_boundaries": [
                "Preserve external behavior and public contracts unless all affected paired files are in focus.",
                "Avoid broad rewrites without a concrete evaluator-visible mechanism.",
            ],
            "planning_strategy": "Prefer small source/header pairs with clear runtime mechanism, compile safety, and measurable FPS/frame-time/power upside.",
        }

    def _build_deep_project_analysis(self) -> None:
        fingerprint = self._project_fingerprint()
        cached = self._load_cached_project_analysis()
        if cached.get("fingerprint") == fingerprint and isinstance(cached.get("analysis"), dict):
            self.project_analysis = cached["analysis"]
            self.project_analysis_text = self._format_project_analysis(self.project_analysis)
            return

        system_prompt = textwrap.dedent(
            """
            You are a senior C++ performance engineer building a whole-project mental model for an evolutionary optimizer.
            Analyze source files and headers together. Think like an engineer before coding: architecture, dataflow, ownership, prediction/model/data paths, source/header contracts, likely frame-time/power hotspots, risk boundaries, and high-leverage mutation strategies.
            Return strict JSON only.
            """
        ).strip()
        user_prompt = textwrap.dedent(
            f"""
            Optimization objective:
            {self.cfg.optimization_goal}

            Metric notes:
            {self.cfg.metric_notes}

            Source/header pairings:
            {json_dumps(self.paired_headers_by_source)}

            File inventory:
            {json_dumps(self._project_analysis_file_rows())}

            Return JSON object with these keys:
            {{
              "architecture": "concise but deep architecture map",
              "execution_flow": ["important runtime/frame/update/dataflow paths"],
              "hotspot_map": [
                {{"files": ["source.cpp", "source.hpp"], "mechanism": "why this may affect FPS/frame-time/power", "confidence": "low|medium|high"}}
              ],
              "model_and_data_paths": [
                {{"files": ["source.cpp"], "model_or_algorithm": "prediction/filter/data algorithm", "data_used": "history/features/buffers scanned", "possible_issue": "too much data or expensive update", "candidate_redesign": "new model/algorithm idea"}}
              ],
              "source_header_contracts": [
                {{"files": ["source.cpp", "source.hpp"], "contract": "what must stay coordinated", "safe_mutations": ["specific ideas"]}}
              ],
              "optimization_levers": [
                {{"lever": "specific lever", "files": ["..."], "expected_metric_effect": "FPS/frame-time/power reason", "risk": "low|medium|high"}}
              ],
              "risk_boundaries": ["things the mutation workers must preserve"],
              "planning_strategy": "how future child plans should choose files and edits"
            }}
            """
        ).strip()
        try:
            obj = self.llm.complete_json(
                system_prompt,
                user_prompt,
                temperature=0.2,
                max_output_tokens=self.cfg.max_project_analysis_output_tokens,
            )
            if not isinstance(obj, dict):
                raise RuntimeError("project analysis is not an object")
            self.project_analysis = obj
        except Exception:
            self.project_analysis = self._fallback_project_analysis()
        self.project_analysis_text = self._format_project_analysis(self.project_analysis)
        self._save_cached_project_analysis({"fingerprint": fingerprint, "analysis": self.project_analysis, "timestamp": now_ts()})

    def _enrich_summaries_with_llm(self) -> None:
        cache = self._load_cached_summaries()
        dirty = False
        to_process: List[str] = []
        for path in self.mutable_files:
            text = self.text_by_path[path]
            current_sha = sha_text(text)
            cached = cache.get(path)
            if cached and cached.get("sha") == current_sha:
                self.static_summaries[path].purpose = cached.get("purpose", "")
                self.static_summaries[path].hot_hints = cached.get("hot_hints", self.static_summaries[path].hot_hints)
                continue
            to_process.append(path)

        if not to_process:
            return

        system_prompt = (
            "You summarize C++ source or header files for an evolutionary optimizer. "
            "Return strict JSON with keys purpose and hot_hints. "
            "Be concrete and infer what the file likely controls. "
            "For hot_hints, call out per-frame work, FPS/frame-time risk, CPU/GPU sync, memory bandwidth, allocations, and power-sensitive idle/wakeup behavior when visible."
        )

        def worker(path: str) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
            summary = self.static_summaries[path]
            text = self.text_by_path[path]
            user_prompt = textwrap.dedent(
                f"""
                File: {path}
                Kind: {summary.kind}
                Includes: {summary.includes[:15]}
                Classes: {summary.classes[:10]}
                Functions: {summary.functions[:20]}

                Source:
                {truncate_middle(text, self.cfg.max_context_chars_per_file)}

                Return JSON:
                {{
                  "purpose": "one or two sentences",
                  "hot_hints": ["short phrases"]
                }}
                """
            ).strip()
            try:
                obj = self.llm.complete_json(
                    system_prompt,
                    user_prompt,
                    temperature=0.2,
                    max_output_tokens=self.cfg.max_file_summary_output_tokens,
                )
                if not isinstance(obj, dict):
                    raise RuntimeError("summary is not an object")
                return path, obj, None
            except Exception as exc:
                return path, None, str(exc)

        with cf.ThreadPoolExecutor(max_workers=min(self.cfg.max_llm_workers, max(1, len(to_process)))) as pool:
            futures = [pool.submit(worker, path) for path in to_process]
            for fut in cf.as_completed(futures):
                path, obj, err = fut.result()
                if obj:
                    self.static_summaries[path].purpose = str(obj.get("purpose", "")).strip()
                    hot_hints = obj.get("hot_hints", [])
                    if isinstance(hot_hints, list):
                        self.static_summaries[path].hot_hints = [str(x) for x in hot_hints][:8]
                    cache[path] = {
                        "sha": sha_text(self.text_by_path[path]),
                        "purpose": self.static_summaries[path].purpose,
                        "hot_hints": self.static_summaries[path].hot_hints,
                    }
                    dirty = True
                elif err:
                    # Fall back silently. The model being flaky is not exactly a breaking news event.
                    pass
        if dirty:
            self._save_cached_summaries(cache)

    def find_neighbors(self, path: str, limit: int = 8) -> List[str]:
        summary = self.static_summaries.get(path)
        if summary is None:
            return []
        candidates: Dict[str, float] = {}
        stem = Path(path).stem
        directory = str(Path(path).parent)
        for other in self.mutable_files + self.immutable_files:
            if other == path:
                continue
            score = 0.0
            if other in self.paired_files_for(path):
                score += 12.0
            if Path(other).stem == stem:
                score += 5.0
            if str(Path(other).parent) == directory:
                score += 1.5
            other_summary = self.static_summaries.get(other)
            if not other_summary:
                continue
            shared_includes = len(set(summary.includes) & set(other_summary.includes))
            score += 0.5 * shared_includes
            if any(inc.endswith(Path(other).name) for inc in summary.includes):
                score += 3.0
            if any(inc.endswith(Path(path).name) for inc in other_summary.includes):
                score += 2.0
            if set(summary.functions) & set(other_summary.functions):
                score += 2.0
            if score > 0:
                candidates[other] = score
        ranked = sorted(candidates.items(), key=lambda kv: (-kv[1], kv[0]))
        return [path for path, _ in ranked[:limit]]

    def snapshot(self) -> Dict[str, str]:
        return {k: self.text_by_path[k] for k in self.mutable_files}

    def update_from_candidate(self, candidate_files: Dict[str, str]) -> None:
        for path, text in candidate_files.items():
            self.text_by_path[path] = normalize_newlines(text)
        self._build_static_summaries()

    def project_summary_text(self, limit_files: Optional[int] = None, current_files: Optional[Dict[str, str]] = None) -> str:
        rows: List[str] = []
        limit = limit_files or self.cfg.max_project_summary_files
        for path in self.mutable_files[:limit]:
            summary = self.static_summaries[path]
            if current_files is not None and path in current_files:
                current_summary = self.summarize_source_text(path, current_files[path])
                current_summary.purpose = summary.purpose
                current_summary.neighbors = summary.neighbors
                if not current_summary.hot_hints:
                    current_summary.hot_hints = summary.hot_hints
                summary = current_summary
            purpose = summary.purpose or "No LLM summary yet."
            hot = ", ".join(summary.hot_hints[:4]) if summary.hot_hints else "none"
            constants = " | constants=" + "; ".join(summary.numeric_constants[:4]) if summary.numeric_constants else ""
            models = " | model/data=" + "; ".join(summary.model_hints[:4]) if summary.model_hints else ""
            funcs = ", ".join(summary.functions[:8]) if summary.functions else "none"
            paired = ", ".join(summary.paired_files[:3]) if summary.paired_files else "none"
            rows.append(f"- {path} ({summary.kind}): {purpose} | hints={hot}{constants}{models} | funcs={funcs} | paired={paired}")
        return "\n".join(rows)


# -----------------------------
# Archive and state
# -----------------------------


class Archive:
    def __init__(self, cfg: EvolutionConfig):
        self.cfg = cfg
        self.work_root = Path(cfg.work_root)
        ensure_dir(self.work_root)
        self.state_path = self.work_root / "state.json"
        self.history_path = self.work_root / "history.jsonl"
        self.file_stats_path = self.work_root / "file_stats.json"
        self.strategy_memory_path = self.work_root / "strategy_memory.json"
        self.snapshots_dir = self.work_root / "snapshots"
        ensure_dir(self.snapshots_dir)
        self.elites: List[Candidate] = []
        self.best: Optional[Candidate] = None
        self.history_meta: List[Dict[str, Any]] = []
        self.file_stats: Dict[str, Dict[str, float]] = {}
        self.strategy_memory: Dict[str, Any] = {}
        self.by_signature: Dict[str, List[Tuple[float, str]]] = {}
        self.load()

    def load(self) -> None:
        if self.file_stats_path.exists():
            try:
                self.file_stats = json.loads(read_text(self.file_stats_path))
            except Exception:
                self.file_stats = {}
        if self.strategy_memory_path.exists():
            try:
                memory = json.loads(read_text(self.strategy_memory_path))
                if isinstance(memory, dict):
                    self.strategy_memory = memory
            except Exception:
                self.strategy_memory = {}
        if self.history_path.exists():
            self.history_meta.clear()
            with self.history_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self.history_meta.append(json.loads(line))
                    except Exception:
                        continue
        if self.state_path.exists():
            try:
                state = json.loads(read_text(self.state_path))
                elite_ids = state.get("elite_ids", [])
                best_id = state.get("best_id")
                loaded = []
                for candidate_id in elite_ids:
                    cand = self._load_candidate_snapshot(candidate_id)
                    if cand is not None:
                        loaded.append(cand)
                self.elites = sorted(loaded, key=lambda c: float(c.score if c.score is not None else -1e300), reverse=True)[: self.cfg.elite_keep]
                if best_id:
                    self.best = next((c for c in self.elites if c.candidate_id == best_id), None)
                    if self.best is None:
                        self.best = self._load_candidate_snapshot(best_id)
            except Exception:
                self.elites = []
                self.best = None

    def save_file_stats(self) -> None:
        write_text(self.file_stats_path, json_dumps(self.file_stats))

    def save_strategy_memory(self) -> None:
        write_text(self.strategy_memory_path, json_dumps(self.strategy_memory))

    def record_generation_learning(self, generation: int, analysis: Dict[str, Any]) -> None:
        if not isinstance(analysis, dict):
            return
        memory = dict(self.strategy_memory)
        reviews = list(memory.get("generation_reviews", []))
        reviews.append(
            {
                "generation": generation,
                "timestamp": now_ts(),
                "analysis": analysis,
            }
        )
        keep = max(1, int(self.cfg.strategy_memory_keep))
        memory["generation_reviews"] = reviews[-keep:]
        for key in [
            "planning_directive",
            "validated_hypotheses",
            "rejected_hypotheses",
            "fps_power_lessons",
            "promote_next",
            "avoid_next",
            "file_priorities",
            "metric_tradeoffs",
            "risk_controls",
        ]:
            if key in analysis:
                memory[key] = analysis[key]
        memory["updated_at"] = now_ts()
        self.strategy_memory = memory
        self.save_strategy_memory()

    def strategy_memory_text(self) -> str:
        if not self.strategy_memory:
            return "No measured hypothesis memory yet. Treat this generation as exploration, but require each child to state a measurable FPS/frame-time/power mechanism."
        memory = dict(self.strategy_memory)
        reviews = list(memory.get("generation_reviews", []))
        memory["generation_reviews"] = reviews[-min(8, len(reviews)) :]
        return json_dumps(memory)

    def _snapshot_path(self, candidate_id: str) -> Path:
        return self.snapshots_dir / f"{candidate_id}.json.gz"

    def _save_candidate_snapshot(self, candidate: Candidate) -> None:
        payload = {
            "candidate_id": candidate.candidate_id,
            "generation": candidate.generation,
            "parent_id": candidate.parent_id,
            "files": candidate.files,
            "changed_files": candidate.changed_files,
            "hypothesis": candidate.hypothesis,
            "intents": candidate.intents,
            "style": candidate.style,
            "llm_notes": candidate.llm_notes,
            "score": candidate.score,
            "compile_ok": candidate.compile_ok,
            "metrics": candidate.metrics,
            "eval_stage": candidate.eval_stage,
            "stdout": candidate.stdout[-50000:],
            "stderr": candidate.stderr[-50000:],
            "work_dir": candidate.work_dir,
        }
        path = self._snapshot_path(candidate.candidate_id)
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)

    def _load_candidate_snapshot(self, candidate_id: str) -> Optional[Candidate]:
        path = self._snapshot_path(candidate_id)
        if not path.exists():
            return None
        try:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                payload = json.load(fh)
            return Candidate(**payload)
        except Exception:
            return None

    def _save_state(self) -> None:
        state = {
            "best_id": self.best.candidate_id if self.best else None,
            "elite_ids": [c.candidate_id for c in self.elites[: self.cfg.elite_keep]],
            "history_count": len(self.history_meta),
            "timestamp": now_ts(),
        }
        write_text(self.state_path, json_dumps(state))

    def _candidate_signature(self, candidate: Candidate) -> str:
        items = sorted(candidate.changed_files)
        return "|".join(items) if items else "<no-change>"

    def _candidate_meta(self, candidate: Candidate) -> Dict[str, Any]:
        return {
            "candidate_id": candidate.candidate_id,
            "generation": candidate.generation,
            "parent_id": candidate.parent_id,
            "changed_files": candidate.changed_files,
            "hypothesis": candidate.hypothesis,
            "style": candidate.style,
            "score": candidate.score,
            "compile_ok": candidate.compile_ok,
            "metrics": candidate.metrics,
            "eval_stage": candidate.eval_stage,
            "work_dir": candidate.work_dir,
            "llm_notes": candidate.llm_notes,
            "signature": self._candidate_signature(candidate),
            "timestamp": now_ts(),
        }

    def register(self, candidate: Candidate) -> None:
        meta = self._candidate_meta(candidate)
        with self.history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
        self.history_meta.append(meta)
        signature = meta["signature"]
        bucket = self.by_signature.setdefault(signature, [])
        bucket.append((float(candidate.score or self.cfg.compile_fail_score), candidate.candidate_id))
        bucket.sort(key=lambda kv: kv[0], reverse=True)
        del bucket[self.cfg.archive_keep_per_signature :]

        self.elites.append(candidate)
        self.elites.sort(key=lambda c: float(c.score if c.score is not None else -1e300), reverse=True)
        self.elites = self.elites[: self.cfg.elite_keep]
        if self.best is None or float(candidate.score or -1e300) > float(self.best.score or -1e300):
            self.best = candidate
        self._update_file_stats(candidate)
        self.save_file_stats()
        self._save_candidate_snapshot(candidate)
        self._save_state()

    def _update_file_stats(self, candidate: Candidate) -> None:
        parent_score = None
        for elite in self.elites:
            if elite.candidate_id == candidate.parent_id:
                parent_score = elite.score
                break
        delta = 0.0
        if candidate.score is not None and parent_score is not None:
            delta = candidate.score - parent_score
        for path in candidate.changed_files:
            stats = self.file_stats.setdefault(path, {
                "attempts": 0.0,
                "compile_failures": 0.0,
                "improvements": 0.0,
                "best_delta": 0.0,
                "last_delta": 0.0,
            })
            stats["attempts"] += 1.0
            if not candidate.compile_ok:
                stats["compile_failures"] += 1.0
            if delta > 0:
                stats["improvements"] += 1.0
            stats["best_delta"] = max(stats["best_delta"], delta)
            stats["last_delta"] = delta

    def select_parent(self, rng: random.Random) -> Optional[Candidate]:
        if not self.elites:
            return None
        pool = self.elites[: max(1, min(self.cfg.parent_pool_size, len(self.elites)))]
        weights = []
        best_score = max(float(c.score or -1e9) for c in pool)
        for candidate in pool:
            base = math.exp(clamp(((candidate.score or -1e9) - best_score) / 10.0, -8.0, 3.0))
            novelty = 1.0 + 0.1 * len(candidate.changed_files)
            weights.append(base * novelty)
        picked = pick_weighted(pool, weights, 1, rng)
        return picked[0] if picked else pool[0]


# -----------------------------
# Prompt construction
# -----------------------------


class PromptFactory:
    MUTATION_STYLES = [
        "reduce per-frame CPU work and frame-time spikes",
        "remove CPU/GPU synchronization stalls",
        "reduce memory bandwidth and cache misses",
        "lower wakeups, polling, and idle power",
        "avoid repeated device or rendering API calls",
        "reduce allocations and copies",
        "tighten hot loops and branches",
        "cache or precompute repeated work",
        "improve memory locality",
        "avoid redundant device or API calls",
        "replace expensive abstractions in hot code paths",
        "remove logging or debug overhead from hot paths",
        "simplify control flow while preserving behavior",
        "tune performance-sensitive numeric constants and thresholds",
        "redesign prediction, filtering, or model update algorithm",
        "reduce model input/history data while preserving prediction quality",
        "enable inlining, vectorization, or devirtualization on measured hot paths",
    ]

    def __init__(self, cfg: EvolutionConfig, index: ProjectIndex, archive: Archive, llm: ServerLLMClient):
        self.cfg = cfg
        self.index = index
        self.archive = archive
        self.llm = llm

    def optimization_guidance(self) -> str:
        return textwrap.dedent(
            f"""
            Primary optimization objective:
            {self.cfg.optimization_goal}

            FPS target:
            Keep {normalize_metric_key(self.cfg.target_fps_metric)} at or above {self.cfg.target_fps:g} when possible. If it is below target, prioritize raising FPS/1% low FPS. If it is at or above target, prefer reducing power without dropping below target.

            Metric guidance:
            {self.cfg.metric_notes}

            Primary parsed navigation metrics:
            {json_dumps(normalize_metric_keys(self.cfg.primary_metric_keys))}

            Parsed evaluator metric objectives:
            {json_dumps(self.cfg.metric_objectives)}

            Prefer changes with a plausible measurable path to:
            - higher sustained FPS or lower average frame time
            - fewer frame-time spikes and less jitter
            - less CPU/GPU synchronization, driver chatter, and blocking waits
            - lower memory traffic, heap traffic, cache misses, and redundant copies
            - cheaper or better prediction/model/data update algorithms with less history, fewer features, or less per-frame inference work
            - lower active CPU/GPU time, wakeups, polling, thermal load, and power draw

            Compile success is a gate, not the objective. A useful child must have a concrete
            runtime hypothesis for FPS, frame time, or power; do not spend candidates on
            cosmetic rewrites or compile-only cleanups.
            """
        ).strip()

    def cxx_correctness_contract(self) -> str:
        return textwrap.dedent(
            f"""
            C++ correctness contract:
            - Generated code must compile as {self.cfg.cxx_standard}; do not use C++20/C++23-only features such as concepts, ranges, std::span, std::format, consteval, requires, designated initializers, or spaceship comparisons.
            - Prefer the project's existing APIs, types, namespaces, helper functions, ownership conventions, and include style. Do not invent methods, fields, namespaces, macros, libraries, build flags, or external dependencies.
            - Preserve function signatures, overload sets, virtual overrides, template parameters, const/reference/noexcept qualifiers, visibility, ABI-relevant data layout, and public behavior unless every affected source/header file is in the focus list.
            - When adding a standard-library type or algorithm, add the required C++17 header in the same file if that file owns includes; otherwise reuse existing includes.
            - Keep namespace, class, and preprocessor scopes balanced. Avoid introducing ODR hazards in headers; header definitions must be inline, constexpr, template, or otherwise consistent with the existing pattern.
            - Before returning JSON, mentally run a compile audit: matching braces/parentheses, semicolons, includes, names in scope, declaration/definition consistency, member initialization, return paths, type conversions, and warning-prone signed/unsigned or narrowing changes.
            - If a risky rewrite would need unknown context, choose a smaller safe edit that preserves the optimization hypothesis instead of guessing.
            """
        ).strip()

    def deep_project_context(self) -> str:
        if not self.index.project_analysis_text:
            return "No cached deep project analysis is available; rely on file summaries, pairings, neighbors, and history."
        return self.index.project_analysis_text

    def evolution_learning_context(self) -> str:
        return self.archive.strategy_memory_text()

    def learned_file_weight(self, path: str) -> float:
        priorities = self.archive.strategy_memory.get("file_priorities", [])
        if not isinstance(priorities, list):
            return 1.0
        weight = 1.0
        for item in priorities:
            if not isinstance(item, dict):
                continue
            item_path = str(item.get("path", ""))
            if item_path != path:
                continue
            priority = str(item.get("priority", "")).lower()
            if priority in {"critical", "very_high", "very high", "high"}:
                weight *= 1.75
            elif priority in {"medium", "moderate"}:
                weight *= 1.35
            elif priority in {"low"}:
                weight *= 1.1
        return weight

    def constant_tuning_target_count(self) -> int:
        if not self.cfg.enable_constant_tuning_plans:
            return 0
        if self.cfg.children_per_generation <= 1:
            return 0
        fraction = clamp(float(self.cfg.constant_tuning_plan_fraction), 0.0, 1.0)
        target = int(self.cfg.children_per_generation * fraction)
        if target <= 0 and fraction > 0:
            target = 1
        return min(max(0, int(self.cfg.max_constant_tuning_plans_per_generation)), target)

    def algorithmic_model_target_count(self) -> int:
        if not self.cfg.enable_algorithmic_model_plans:
            return 0
        if self.cfg.children_per_generation <= 1:
            return 0
        fraction = clamp(float(self.cfg.algorithmic_model_plan_fraction), 0.0, 1.0)
        target = int(self.cfg.children_per_generation * fraction)
        if target <= 0 and fraction > 0:
            target = 1
        return min(max(0, int(self.cfg.max_algorithmic_model_plans_per_generation)), target)

    def impact_heuristic_target_count(self) -> int:
        if not self.cfg.enable_impact_heuristic_plans:
            return 0
        if self.cfg.children_per_generation <= 1:
            return 0
        fraction = clamp(float(self.cfg.impact_heuristic_plan_fraction), 0.0, 1.0)
        target = int(self.cfg.children_per_generation * fraction)
        if target <= 0 and fraction > 0:
            target = 1
        return min(max(0, int(self.cfg.max_impact_heuristic_plans_per_generation)), target)

    def is_trivial_micro_plan(self, row: Dict[str, Any]) -> bool:
        if not self.cfg.reject_trivial_micro_optimizations:
            return False
        text = json_dumps(row)
        return bool(TRIVIAL_MICRO_OPT_RE.search(text)) and not bool(MEANINGFUL_MECHANISM_RE.search(text))

    def algorithmic_model_children(self, parent: Candidate, count: int, generation: int, rng: random.Random) -> List[ChildPlan]:
        if count <= 0:
            return []
        paths = [
            path
            for path in self.index.mutable_files
            if self.index.static_summaries[path].model_hints
            or "prediction/model/data path" in self.index.static_summaries[path].hot_hints
        ]
        if not paths:
            paths = [
                path
                for path in self.index.mutable_files
                if any(hint in self.index.static_summaries[path].hot_hints for hint in ["per-frame path", "container heavy", "math heavy", "memory bandwidth / bulk work"])
            ]
        if not paths:
            return []

        weights: List[float] = []
        for path in paths:
            summary = self.index.static_summaries[path]
            stats = self.archive.file_stats.get(path, {})
            attempts = float(stats.get("attempts", 0.0))
            fails = float(stats.get("compile_failures", 0.0))
            improvements = float(stats.get("improvements", 0.0))
            best_delta = float(stats.get("best_delta", 0.0))
            model_bonus = 1.0 + min(5, len(summary.model_hints)) * 0.3
            hot_bonus = 1.0 + 0.2 * len(summary.hot_hints)
            fail_penalty = 1.0 / (1.0 + fails)
            history_bonus = 1.0 + clamp(best_delta, 0.0, 50.0) / 22.0 + improvements * 0.12
            exploration_bonus = 1.6 / (1.0 + attempts * 0.2)
            weights.append(max(0.1, model_bonus * hot_bonus * fail_penalty * history_bonus * exploration_bonus * self.learned_file_weight(path)))

        children: List[ChildPlan] = []
        limit = max(self.cfg.max_focus_files_per_child, self.cfg.max_focus_files_per_child + self.cfg.max_paired_headers_per_source)
        templates = [
            (
                "algorithmic model redesign",
                "Replace, simplify, or restructure a prediction/filtering/model update algorithm so it scans less data, stores less history, or computes fewer features while preserving useful prediction behavior.",
            ),
            (
                "model data reduction",
                "Reduce the amount of model input/history data used per frame, for example by bounded windows, cached aggregates, cheaper features, early exits, or incremental updates.",
            ),
            (
                "prediction quality per watt",
                "Change the model/update policy to improve fps_1pct_low and lower power_w by reducing unnecessary inference/update work without making predictions unstable.",
            ),
        ]
        for idx in range(count):
            focus_base = pick_weighted(paths, weights, 1, rng)
            if not focus_base:
                break
            focus = self.index.expand_focus_with_pairs(focus_base, max_files=limit)
            style, premise = templates[idx % len(templates)]
            summary = self.index.static_summaries[focus_base[0]]
            model_hints = summary.model_hints[: self.cfg.max_model_hints_per_file]
            intents: Dict[str, str] = {}
            for path in focus:
                path_summary = self.index.static_summaries[path]
                hints = "; ".join(path_summary.model_hints[:8]) if path_summary.model_hints else "; ".join(path_summary.hot_hints[:6])
                paired = ", ".join(path_summary.paired_files[:3]) if path_summary.paired_files else "none"
                intents[path] = (
                    f"{premise} Inspect whether this file uses too much data, history, features, filtering, prediction state, or model update work. "
                    f"Relevant model/data hints: {hints or 'none detected'}; paired files: {paired}. "
                    "A useful edit may introduce a small helper, change state/update policy, change feature/data retention, or swap the algorithm, but it must remain focused and measurable."
                )
            hypothesis = (
                f"{style} in {', '.join(focus)}: {premise} "
                "Expected effect: improve fps_1pct_low and/or lower power_w by reducing per-frame model/data work or improving prediction stability. "
                f"Model/data evidence: {'; '.join(model_hints[:8]) if model_hints else 'hot file selected from fallback hints'}."
            )
            children.append(
                ChildPlan(
                    focus_files=focus,
                    hypothesis=hypothesis,
                    intents=intents,
                    style=style,
                    parent_id=parent.candidate_id,
                    plan_id=f"g{generation:04d}_algo_{idx:02d}",
                )
            )
        return children

    def constant_tuning_children(self, parent: Candidate, count: int, generation: int, rng: random.Random) -> List[ChildPlan]:
        if count <= 0:
            return []
        paths = [path for path in self.index.mutable_files if self.index.static_summaries[path].numeric_constants]
        if not paths:
            return []
        weights: List[float] = []
        for path in paths:
            summary = self.index.static_summaries[path]
            stats = self.archive.file_stats.get(path, {})
            attempts = float(stats.get("attempts", 0.0))
            fails = float(stats.get("compile_failures", 0.0))
            improvements = float(stats.get("improvements", 0.0))
            best_delta = float(stats.get("best_delta", 0.0))
            hot_bonus = 1.0 + 0.25 * len(summary.hot_hints)
            constant_bonus = 1.0 + min(3, len(summary.numeric_constants)) * 0.2
            fail_penalty = 1.0 / (1.0 + fails)
            history_bonus = 1.0 + clamp(best_delta, 0.0, 50.0) / 25.0 + improvements * 0.1
            unexplored_bonus = 1.4 / (1.0 + attempts * 0.25)
            weights.append(max(0.1, hot_bonus * constant_bonus * fail_penalty * history_bonus * unexplored_bonus * self.learned_file_weight(path)))

        children: List[ChildPlan] = []
        limit = max(self.cfg.max_focus_files_per_child, self.cfg.max_focus_files_per_child + self.cfg.max_paired_headers_per_source)
        for idx in range(count):
            focus_base = pick_weighted(paths, weights, 1, rng)
            if not focus_base:
                break
            focus = self.index.expand_focus_with_pairs(focus_base, max_files=limit)
            summary = self.index.static_summaries[focus_base[0]]
            constants = summary.numeric_constants[: self.cfg.max_constant_hints_per_file]
            intents: Dict[str, str] = {}
            for path in focus:
                path_summary = self.index.static_summaries[path]
                constant_text = "; ".join(path_summary.numeric_constants[:8]) if path_summary.numeric_constants else "paired declarations or call-site assumptions"
                intents[path] = (
                    "Run a deliberately small numeric-constant experiment for measured fps_1pct_low and power_w. "
                    f"Candidate constants/thresholds: {constant_text}. "
                    "Change only a very small number of existing numeric values or named constants unless a paired declaration must stay consistent."
                )
            hypothesis = (
                f"Constant tuning experiment in {', '.join(focus)}: adjust one or a few existing performance-sensitive numeric constants "
                "to test whether fps_1pct_low improves and/or power_w decreases without changing externally visible behavior. "
                f"Candidate constants: {'; '.join(constants[:8])}."
            )
            children.append(
                ChildPlan(
                    focus_files=focus,
                    hypothesis=hypothesis,
                    intents=intents,
                    style="targeted constant tuning",
                    parent_id=parent.candidate_id,
                    plan_id=f"g{generation:04d}_const_{idx:02d}",
                )
            )
        return children

    def impact_heuristic_children(self, parent: Candidate, count: int, generation: int, rng: random.Random) -> List[ChildPlan]:
        if count <= 0:
            return []
        impact_labels = {
            "per-frame path",
            "CPU/GPU sync stall",
            "memory bandwidth / bulk work",
            "heap traffic",
            "container heavy",
            "hot loops",
            "wakeups / idle power",
            "device / graphics interaction",
            "prediction/model/data path",
            "tunable heuristic / threshold",
            "cache/incremental opportunity",
        }
        paths = [
            path
            for path in (self.index.mutable_source_files or self.index.mutable_files)
            if any(hint in impact_labels for hint in self.index.static_summaries[path].hot_hints)
        ]
        if not paths:
            paths = list(self.index.mutable_source_files or self.index.mutable_files)
        if not paths:
            return []

        weights: List[float] = []
        for path in paths:
            summary = self.index.static_summaries[path]
            stats = self.archive.file_stats.get(path, {})
            attempts = float(stats.get("attempts", 0.0))
            fails = float(stats.get("compile_failures", 0.0))
            improvements = float(stats.get("improvements", 0.0))
            best_delta = float(stats.get("best_delta", 0.0))
            hot_bonus = 1.0 + 0.35 * len(summary.hot_hints)
            model_bonus = 1.3 if summary.model_hints else 1.0
            constant_bonus = 1.15 if summary.numeric_constants else 1.0
            fail_penalty = 1.0 / (1.0 + fails)
            history_bonus = 1.0 + clamp(best_delta, 0.0, 50.0) / 18.0 + improvements * 0.18
            exploration_bonus = 1.7 / (1.0 + attempts * 0.25)
            weights.append(max(0.1, hot_bonus * model_bonus * constant_bonus * fail_penalty * history_bonus * exploration_bonus * self.learned_file_weight(path)))

        templates = [
            (
                "impact heuristic: cache or incremental update",
                "Replace repeated per-frame recomputation with cached derived state, dirty flags, generation counters, or incremental updates.",
            ),
            (
                "impact heuristic: gate or early-exit expensive work",
                "Add a correctness-preserving guard, threshold, or early exit so expensive work is skipped when inputs/state did not meaningfully change.",
            ),
            (
                "impact heuristic: reduce data movement and allocation",
                "Move allocation/copy/container growth out of hot paths, reuse buffers, reserve based on known bounds, or avoid full data movement.",
            ),
            (
                "impact heuristic: batch or coalesce device/API work",
                "Coalesce repeated device/API calls, logging, polling, wakeups, or synchronization into fewer operations while preserving observable behavior.",
            ),
            (
                "impact heuristic: bounded scan or cheaper selection",
                "Replace full scans/sorts/history walks with bounded windows, incremental aggregates, top-k/early-stop logic, or cheaper feature selection.",
            ),
        ]

        children: List[ChildPlan] = []
        limit = max(self.cfg.max_focus_files_per_child, self.cfg.max_focus_files_per_child + self.cfg.max_paired_headers_per_source)
        for idx in range(count):
            focus_base = pick_weighted(paths, weights, 1, rng)
            if not focus_base:
                break
            focus = self.index.expand_focus_with_pairs(focus_base, max_files=limit)
            if focus and len(focus) < self.cfg.max_focus_files_per_child:
                neighbors = self.index.find_neighbors(focus[0], limit=5)
                neighbor_candidates = [p for p in neighbors if p in self.index.mutable_files and p not in focus]
                if neighbor_candidates:
                    focus.append(rng.choice(neighbor_candidates))
                    focus = self.index.expand_focus_with_pairs(focus, max_files=limit)
            focus = focus[:limit]
            style, premise = templates[idx % len(templates)]
            intents: Dict[str, str] = {}
            evidence_parts: List[str] = []
            for path in focus:
                summary = self.index.static_summaries[path]
                hints = "; ".join(summary.hot_hints[:6])
                constants = "; ".join(summary.numeric_constants[:5]) if summary.numeric_constants else "none detected"
                model_hints = "; ".join(summary.model_hints[:5]) if summary.model_hints else "none detected"
                paired = ", ".join(summary.paired_files[:3]) if summary.paired_files else "none"
                evidence_parts.append(f"{path}: hot={hints or 'none'} constants={constants} model/data={model_hints}")
                intents[path] = (
                    f"{premise} Use existing code structure and evidence instead of syntax-only micro-optimizations. "
                    f"Hot hints: {hints or 'none detected'}; constants: {constants}; model/data hints: {model_hints}; paired files: {paired}. "
                    "A valid edit should change actual runtime work, data volume, update frequency, allocation/copy behavior, or device/API interaction."
                )
            hypothesis = (
                f"{style} across {', '.join(focus)}: {premise} "
                "Expected effect: improve fps_1pct_low and/or power_w by changing runtime work, not by cosmetic or compiler-noise edits. "
                f"Evidence: {' | '.join(evidence_parts[:4])}."
            )
            children.append(
                ChildPlan(
                    focus_files=focus,
                    hypothesis=hypothesis,
                    intents=intents,
                    style=style,
                    parent_id=parent.candidate_id,
                    plan_id=f"g{generation:04d}_impact_{idx:02d}",
                )
            )
        return children

    def heuristic_children(self, parent: Candidate, count: int, generation: int, rng: random.Random) -> List[ChildPlan]:
        paths = list(self.index.mutable_source_files or self.index.mutable_files)
        weights: List[float] = []
        for path in paths:
            stats = self.archive.file_stats.get(path, {})
            attempts = float(stats.get("attempts", 0.0))
            fails = float(stats.get("compile_failures", 0.0))
            improvements = float(stats.get("improvements", 0.0))
            best_delta = float(stats.get("best_delta", 0.0))
            unexplored_bonus = 2.0 / (1.0 + attempts)
            fail_penalty = 1.0 / (1.0 + fails)
            impact_bonus = 1.0 + clamp(best_delta, 0.0, 50.0) / 20.0 + improvements * 0.15
            if attempts >= 3.0 and improvements <= 0.0 and best_delta <= 0.0:
                impact_bonus *= 0.55
            learned_bonus = self.learned_file_weight(path)
            same_stem_bonus = 1.25 if Path(path).stem in {Path(p).stem for p in parent.changed_files} else 1.0
            weights.append(max(0.1, unexplored_bonus * fail_penalty * impact_bonus * learned_bonus * same_stem_bonus))

        children: List[ChildPlan] = []
        for idx in range(count):
            k = rng.randint(1, max(1, self.cfg.max_focus_files_per_child))
            focus = pick_weighted(paths, weights, k, rng)
            focus = self.index.expand_focus_with_pairs(
                focus,
                max_files=max(self.cfg.max_focus_files_per_child, self.cfg.max_focus_files_per_child + self.cfg.max_paired_headers_per_source),
            )
            # With some probability, add one strong neighbor for cross-file consistency.
            if focus and rng.random() < 0.6 and len(focus) < self.cfg.max_focus_files_per_child:
                neighbors = self.index.find_neighbors(focus[0], limit=4)
                neighbor_candidates = [p for p in neighbors if p in self.index.mutable_files and p not in focus]
                if neighbor_candidates:
                    focus.append(rng.choice(neighbor_candidates))
                    focus = self.index.expand_focus_with_pairs(
                        focus,
                        max_files=max(self.cfg.max_focus_files_per_child, self.cfg.max_focus_files_per_child + self.cfg.max_paired_headers_per_source),
                    )
            focus = focus[: max(self.cfg.max_focus_files_per_child, self.cfg.max_focus_files_per_child + self.cfg.max_paired_headers_per_source)]
            style = rng.choice(self.MUTATION_STYLES)
            mechanism = (
                "choose a real mechanism such as caching repeated work, bounding history, adding an early exit, "
                "reducing allocation/copy volume, coalescing repeated device/API calls, or making an update incremental"
            )
            intents: Dict[str, str] = {}
            for path in focus:
                summary = self.index.static_summaries.get(path)
                purpose = summary.purpose or f"{summary.kind if summary else 'C++'} file"
                hints = ", ".join(summary.hot_hints[:3]) if summary and summary.hot_hints else "general runtime cost"
                paired = ", ".join(summary.paired_files[:3]) if summary and summary.paired_files else "none"
                intents[path] = (
                    f"Optimize this file for FPS/frame-time and power by focusing on {style}; "
                    f"likely hotspots: {hints}; role: {purpose}; paired files: {paired}. "
                    f"Performance-sensitive constants: {'; '.join(summary.numeric_constants[:5]) if summary and summary.numeric_constants else 'none detected'}. "
                    f"To make this useful, {mechanism}. Do not spend this child on i++/++i, formatting, renaming, or other syntax-only micro-edits."
                )
            hypothesis = (
                f"Apply {style} in a compile-safe way across {', '.join(focus)} while preserving externally visible behavior "
                f"and improving sustained FPS, frame-time stability, or power efficiency. Mechanism requirement: {mechanism}."
            )
            children.append(
                ChildPlan(
                    focus_files=focus,
                    hypothesis=hypothesis,
                    intents=intents,
                    style=style,
                    parent_id=parent.candidate_id,
                    plan_id=f"g{generation:04d}_heur_{idx:02d}",
                )
            )
        return children

    def plan_children(self, parent: Candidate, generation: int, rng: random.Random) -> List[ChildPlan]:
        guidance = self.optimization_guidance()
        cxx_contract = self.cxx_correctness_contract()
        deep_context = self.deep_project_context()
        learning_context = self.evolution_learning_context()
        system_prompt = textwrap.dedent(
            f"""
            You are the planning agent for project-level C++ evolution.
            Your job is to propose diverse, high-value child mutation plans after doing careful engineering analysis.

            {guidance}

            {cxx_contract}

            Engineering workflow:
            - First reason from the deep project model, measured hypothesis memory, source/header pairings, recent outcomes, and file statistics.
            - Form a concrete performance hypothesis before choosing files.
            - Prefer edits that remove work from hot paths, reduce synchronization, improve locality, or lower wakeups/power.
            - Explore new algorithms, models, and heuristics: batching/coalescing, cached or incremental state, bounded history/windows, cheaper feature selection, early exits, data reduction, update-policy changes, and replacement of full scans/sorts with cheaper maintained summaries.
            - Treat prior measured outcomes as evidence: promote hypotheses that improved score and avoid repeating hypotheses that only compiled or reduced score.
            - Pair headers with sources when declarations, layout, inline/template code, constants, or helper APIs may need to move together.
            - Avoid random file selection. Every selected file must have a role in the hypothesis.

            Constraints:
            - Only mutable source/header files may be edited.
            - Header edits are allowed when the header is in focus_files.
            - When a source file has paired headers, usually include the paired header in the same child plan so API, inline, template, and declaration changes stay coordinated.
            - Keep each child focused on 1 to a few files.
            - Bias toward likely FPS, frame-time, and power wins that the evaluator can measure.
            - Compile success is required, but compile-only candidates are failures of planning.
            - Avoid changing public interfaces unless the paired source/header files are listed together and the compatibility risk is justified.
            - Be diverse across children.
            - Do not waste children on formatting-only, naming-only, i++ to ++i, const-only, reserve-only, or speculative rewrites with no runtime mechanism.
            - A plan is invalid if its expected benefit could plausibly be optimized away by the compiler without changing runtime work, data volume, update frequency, synchronization, allocation/copy behavior, or device/API interaction.
            Return strict JSON only.
            """
        ).strip()

        best_score = self.archive.best.score if self.archive.best else None
        file_stats_rows = []
        for path in self.index.mutable_files[:80]:
            stats = self.archive.file_stats.get(path, {})
            summary = self.index.static_summaries[path]
            file_stats_rows.append(
                {
                    "path": path,
                    "purpose": summary.purpose or "",
                    "hot_hints": summary.hot_hints,
                    "attempts": stats.get("attempts", 0),
                    "compile_failures": stats.get("compile_failures", 0),
                    "improvements": stats.get("improvements", 0),
                    "best_delta": stats.get("best_delta", 0),
                    "last_delta": stats.get("last_delta", 0),
                    "kind": summary.kind,
                    "numeric_constants": summary.numeric_constants,
                    "model_hints": summary.model_hints,
                    "paired_files": summary.paired_files,
                    "neighbors": summary.neighbors[:4],
                }
            )
        recent = self.archive.history_meta[-10:]
        user_prompt = textwrap.dedent(
            f"""
            Current best score: {best_score}
            Parent candidate: {parent.candidate_id}
            Parent score: {parent.score}
            Parent changed files: {parent.changed_files}

            Deep project analysis:
            {deep_context}

            Measured hypothesis learning memory:
            {learning_context}

            Mutable file summaries:
            {json_dumps(file_stats_rows[: self.cfg.max_project_summary_files])}

            Recent history:
            {json_dumps(recent)}

            Produce {self.cfg.planner_children_per_call} diverse child plans.
            Each plan should name the concrete runtime mechanism: per-frame CPU work, GPU/driver sync, memory traffic, allocation churn, wakeups/polling, logging, or redundant device/API calls.
            Prefer small compile-safe edits that preserve output while improving FPS, frame-time stability, or power per frame. Do not propose children whose only credible value is making code compile more cleanly.
            Include deeper algorithm/model/data updates, especially around prediction models, filtering/smoothing, feature/history use, data volume, caching, incremental updates, and model update policy.
            Include impact heuristics such as batching/coalescing repeated calls, dirty flags, cached derived state, bounded history windows, early exits, moving allocations/copies out of frame loops, and replacing full scans/sorts with cheaper incremental summaries.
            Do not propose trivial micro-optimizations such as i++ to ++i, local renaming, formatting, const-only edits, reserve-only edits, or generic cleanup unless they are only a tiny part of a larger runtime-work mechanism.
            Include some low-blast-radius numeric constant or threshold experiments when the file summaries expose performance-sensitive constants.
            If a plan targets a .cpp/.cc/.cxx file with paired_files, include the paired header unless the change is provably source-local.
            Header changes may adjust declarations, inline/template code, constants, data layout, helper APIs, and member fields only when the paired source edit is included and compatibility is preserved.

            Return JSON object:
            {{
              "project_hypothesis": "overall search direction",
              "children": [
                {{
                  "focus_files": ["path1.cpp", "path1.hpp"],
                  "hypothesis": "why this combination may improve FPS/frame-time/power score",
                  "mechanism": "specific runtime mechanism being optimized",
                  "expected_metric_effect": "how the evaluator should improve",
                  "risk_controls": ["compile/semantic safety checks the mutation should preserve"],
                  "style": "short style label",
                  "intents": {{"path1.cpp": "specific intent", "path1.hpp": "specific paired-header intent"}}
                }}
              ]
            }}
            """
        ).strip()

        llm_plans: List[ChildPlan] = []
        try:
            obj = self.llm.complete_json(
                system_prompt,
                user_prompt,
                temperature=self.llm.cfg.temperature_plan,
                max_output_tokens=self.cfg.max_planning_output_tokens,
            )
            children = obj.get("children", []) if isinstance(obj, dict) else []
            for idx, row in enumerate(children[: self.cfg.children_per_generation]):
                if not isinstance(row, dict):
                    continue
                if self.is_trivial_micro_plan(row):
                    continue
                focus = [str(p) for p in row.get("focus_files", []) if isinstance(p, str)]
                focus = [p for p in focus if p in self.index.mutable_files]
                if not focus:
                    continue
                focus = self.index.expand_focus_with_pairs(
                    focus,
                    max_files=max(self.cfg.max_focus_files_per_child, self.cfg.max_focus_files_per_child + self.cfg.max_paired_headers_per_source),
                )
                focus = focus[: max(self.cfg.max_focus_files_per_child, self.cfg.max_focus_files_per_child + self.cfg.max_paired_headers_per_source)]
                intents_raw = row.get("intents", {})
                intents = {}
                for p in focus:
                    summary = self.index.static_summaries.get(p)
                    paired = ", ".join(summary.paired_files[:3]) if summary and summary.paired_files else "none"
                    intents[p] = str(
                        intents_raw.get(
                            p,
                            f"{row.get('hypothesis', 'Optimize carefully.')} Paired files: {paired}.",
                        )
                    )
                mechanism = str(row.get("mechanism", "")).strip()
                expected = str(row.get("expected_metric_effect", "")).strip()
                risk_controls = row.get("risk_controls", [])
                if isinstance(risk_controls, list):
                    risk_text = "; ".join(str(x) for x in risk_controls if str(x).strip())
                else:
                    risk_text = str(risk_controls).strip()
                hypothesis = str(row.get("hypothesis", "")).strip() or "Optimize for better score."
                if mechanism:
                    hypothesis += f" Mechanism: {mechanism}."
                if expected:
                    hypothesis += f" Expected metric effect: {expected}."
                if risk_text:
                    hypothesis += f" Risk controls: {risk_text}."
                llm_plans.append(
                    ChildPlan(
                        focus_files=focus,
                        hypothesis=hypothesis,
                        intents=intents,
                        style=str(row.get("style", "general optimization")).strip() or "general optimization",
                        parent_id=parent.candidate_id,
                        plan_id=f"g{generation:04d}_plan_{idx:02d}",
                    )
                )
        except Exception:
            pass

        remaining_slots = self.cfg.children_per_generation
        algorithm_target = min(self.algorithmic_model_target_count(), remaining_slots)
        remaining_slots -= algorithm_target
        impact_target = min(self.impact_heuristic_target_count(), remaining_slots)
        remaining_slots -= impact_target
        constant_target = min(self.constant_tuning_target_count(), remaining_slots)
        remaining_slots -= constant_target
        constant_plans = self.constant_tuning_children(parent, constant_target, generation, rng)
        algorithm_plans = self.algorithmic_model_children(parent, algorithm_target, generation, rng)
        impact_plans = self.impact_heuristic_children(parent, impact_target, generation, rng)
        llm_limit = max(0, self.cfg.children_per_generation - len(algorithm_plans) - len(impact_plans) - len(constant_plans))
        plans: List[ChildPlan] = algorithm_plans + impact_plans + constant_plans + llm_plans[:llm_limit]
        need = self.cfg.children_per_generation - len(plans)
        if need > 0:
            plans.extend(self.heuristic_children(parent, need, generation, rng))
        return plans[: self.cfg.children_per_generation]

    def build_file_context(self, current_files: Dict[str, str], target_path: str) -> str:
        summary = self.index.static_summaries[target_path]
        neighbors = summary.neighbors[:8]
        pieces: List[str] = []
        used_chars = 0
        for neighbor in neighbors:
            text = current_files.get(neighbor)
            if text is None:
                text = self.index.text_by_path.get(neighbor, "")
            if not text:
                continue
            snippet = truncate_middle(text, self.cfg.max_neighbor_chars_per_file)
            chunk = f"\n\n### Neighbor: {neighbor}\n{snippet}"
            if self.cfg.max_neighbor_chars_total > 0 and used_chars + len(chunk) > self.cfg.max_neighbor_chars_total:
                break
            pieces.append(chunk)
            used_chars += len(chunk)
        return "".join(pieces)

    def mutation_prompt(self, current_files: Dict[str, str], plan: ChildPlan, target_path: str) -> Tuple[str, str]:
        current = current_files[target_path]
        static_summary = self.index.static_summaries[target_path]
        summary = self.index.summarize_source_text(target_path, current)
        summary.purpose = static_summary.purpose
        summary.neighbors = static_summary.neighbors
        if not summary.hot_hints:
            summary.hot_hints = static_summary.hot_hints
        target_content = current if self.cfg.include_full_target_files else truncate_middle(current, self.cfg.max_context_chars_per_file)
        neighbors_text = self.build_file_context(current_files, target_path)
        project_summary = self.index.project_summary_text(
            limit_files=self.cfg.max_project_summary_files,
            current_files=current_files,
        )
        deep_context = self.deep_project_context()
        learning_context = self.evolution_learning_context()
        guidance = self.optimization_guidance()
        cxx_contract = self.cxx_correctness_contract()
        system_prompt = textwrap.dedent(
            f"""
            You are a C++ mutation worker inside an evolutionary optimizer.
            You may modify exactly one target source or header file.

            {guidance}

            {cxx_contract}

            Engineering workflow:
            - Read the deep project analysis, measured hypothesis memory, focus-file hypothesis, paired files, target file, and neighbor context before editing.
            - Identify the smallest local code change that satisfies the hypothesis.
            - Keep declarations, definitions, layout, inline/template code, and call sites consistent across the focus files.
            - Prefer concrete runtime-work reduction over broad rewrites or compile-only tidy-ups.
            - For algorithmic/model/impact heuristic styles, the primary edit must change actual work: data volume, update frequency, caching/reuse, history/window size, feature selection, allocation/copy behavior, synchronization, or device/API call count.
            - If the style is targeted constant tuning, change only one or a few existing numeric values/named constants and leave structure alone unless a paired declaration must stay consistent.
            - If the style is algorithmic model redesign, model data reduction, or prediction quality per watt, inspect data volume and update policy. You may change internal prediction/filtering/smoothing/model algorithms, cached aggregates, bounded history, feature selection, or helper state when the change is focused and compatible with paired files.
            - If the style is an impact heuristic, prefer dirty flags, cached derived values, incremental aggregates, bounded scans/history, early exits, buffer reuse, batched/coalesced calls, or cheaper selection over syntax-only changes.

            Rules:
            - Edit only the target file.
            - Header edits are allowed only when the target file is a header.
            - Public interface, layout, inline, template, and declaration changes are allowed only when they are compatible with the listed focus files and preserve external behavior.
            - Prefer compile-safe, local, concrete changes.
            - Preserve externally visible behavior, output correctness, timing contracts, and numerical stability.
            - Do not invent APIs or call functions/classes/members that are not present in the target, neighbor context, project summary, or standard C++17 library.
            - If the edit touches declarations, member fields, template code, inline functions, or public constants, verify the paired source/header contract from the focus list before emitting content.
            - Do not use i++ to ++i, formatting, renaming, const-only edits, local variable reshuffling, or reserve-only edits as the main optimization. Those are invalid unless attached to a larger mechanism that changes real runtime work.
            - Do not claim expected score gain from compiler-noise micro-edits. The summary must name the real runtime-work mechanism.
            - Do not replace code with placeholders, summaries, omitted regions, or comments saying the rest is unchanged.
            - Return the complete new file content, every line needed to compile.
            - Keep existing line endings, includes, namespace structure, and formatting style where practical.
            - Return strict JSON only.
            - The JSON must contain the complete new file content.
            """
        ).strip()
        user_prompt = textwrap.dedent(
            f"""
            Overall project summary:
            {project_summary}

            Deep project analysis:
            {deep_context}

            Measured hypothesis learning memory:
            {learning_context}

            Child hypothesis:
            {plan.hypothesis}

            Focus files for this child: {plan.focus_files}
            Target file: {target_path}
            Paired files for target: {self.index.paired_files_for(target_path)}
            Style: {plan.style}
            File-specific intent: {plan.intents.get(target_path, '')}
            Performance-sensitive numeric constants in target:
            {json_dumps(summary.numeric_constants)}
            Prediction/model/data hints in target:
            {json_dumps(summary.model_hints)}

            Static summary:
            {json_dumps(dataclasses.asdict(summary))}

            Current target file content:
            {target_content}
            {neighbors_text}

            Return JSON:
            {{
              "path": "{target_path}",
              "content": "full new file content",
              "edit_summary": "short concrete summary including expected FPS/frame-time/power mechanism",
              "compile_self_check": "brief C++17 compile audit: names, includes, signatures, scopes, return paths",
              "risk": "low|medium|high"
            }}
            """
        ).strip()
        return system_prompt, user_prompt

    def review_prompt(self, base_files: Dict[str, str], candidate_files: Dict[str, str], plan: ChildPlan) -> Tuple[str, str]:
        guidance = self.optimization_guidance()
        cxx_contract = self.cxx_correctness_contract()
        deep_context = self.deep_project_context()
        learning_context = self.evolution_learning_context()
        system_prompt = textwrap.dedent(
            f"""
            You review a bundle of C++ edits for cross-file consistency.
            You may replace any of the listed focus files with corrected full contents.
            Do not touch files outside the focus list.
            {guidance}
            {cxx_contract}
            Source/header pairs may both be edited when both are in the focus list.
            Preserve or improve the intended FPS/frame-time/power mechanism while fixing consistency issues.
            Audit every changed file as drop-in {self.cfg.cxx_standard} code before returning it.
            Reject placeholder or omitted content; any returned content must be a complete file.
            Return strict JSON only.
            """
        ).strip()
        changed = []
        for path in plan.focus_files:
            before = base_files.get(path, "")
            after = candidate_files.get(path, before)
            if before != after:
                if not self.cfg.include_full_target_files:
                    before = truncate_middle(before, self.cfg.max_context_chars_per_file)
                    after = truncate_middle(after, self.cfg.max_context_chars_per_file)
                changed.append({"path": path, "before": before, "after": after})
        user_prompt = textwrap.dedent(
            f"""
            Hypothesis: {plan.hypothesis}
            Style: {plan.style}
            Focus files: {plan.focus_files}

            Deep project analysis:
            {deep_context}

            Measured hypothesis learning memory:
            {learning_context}

            Review these proposed changes and fix C++17 compile risks, missing includes, scope mistakes, mismatches between names, signatures, calls, declarations, member fields, and assumptions if needed.
            Keep edits focused on compile safety, semantic preservation, and the concrete FPS/frame-time/power mechanism.

            Changes:
            {json_dumps(changed)}

            Return JSON:
            {{
              "review_notes": "brief notes including any FPS/frame-time/power concern",
              "compile_self_check": "brief C++17 compile audit across returned files",
              "changes": [
                {{"path": "focus/file.cpp", "content": "complete corrected file content"}}
              ]
            }}
            """
        ).strip()
        return system_prompt, user_prompt

    def repair_prompt(self, candidate: Candidate, target_path: str, compile_stderr: str) -> Tuple[str, str]:
        guidance = self.optimization_guidance()
        cxx_contract = self.cxx_correctness_contract()
        deep_context = self.deep_project_context()
        learning_context = self.evolution_learning_context()
        system_prompt = textwrap.dedent(
            f"""
            You are a C++ repair worker.
            You are given compiler or linker errors after an optimization mutation.
            Repair only the target source/header file.
            {guidance}
            {cxx_contract}
            Preserve the intended optimization where possible, but prioritize restoring a successful build.
            Use the compiler/linker output as evidence. Fix the actual broken names, includes, signatures, scopes, types, and declarations; do not guess unrelated rewrites.
            Do not use placeholders or omit unchanged code. Return the complete repaired file content.
            Return strict JSON only.
            """
        ).strip()
        summary = self.index.static_summaries.get(target_path)
        target_content = candidate.files[target_path]
        if not self.cfg.include_full_target_files:
            target_content = truncate_middle(target_content, self.cfg.max_context_chars_per_file)
        user_prompt = textwrap.dedent(
            f"""
            Candidate hypothesis: {candidate.hypothesis}
            Candidate changed files: {candidate.changed_files}
            Target file: {target_path}
            Paired files for target: {self.index.paired_files_for(target_path)}
            File summary: {json_dumps(dataclasses.asdict(summary)) if summary else '{}'}

            Deep project analysis:
            {deep_context}

            Measured hypothesis learning memory:
            {learning_context}

            Compiler / linker output tail (stdout + stderr):
            {tail_text(compile_stderr, self.cfg.max_repair_error_chars)}

            Current target file content:
            {target_content}

            Return JSON:
            {{
              "path": "{target_path}",
              "content": "full repaired file content",
              "compile_self_check": "brief C++17 compile audit for the repaired file",
              "repair_summary": "what was fixed"
            }}
            """
        ).strip()
        return system_prompt, user_prompt

    def repair_bundle_prompt(self, candidate: Candidate, compile_stderr: str) -> Tuple[str, str]:
        guidance = self.optimization_guidance()
        cxx_contract = self.cxx_correctness_contract()
        deep_context = self.deep_project_context()
        learning_context = self.evolution_learning_context()
        system_prompt = textwrap.dedent(
            f"""
            You are a C++ bundle repair worker.
            You are given compiler or linker errors after a multi-file optimization mutation.
            Repair any or all changed source/header files in one coordinated response.
            {guidance}
            {cxx_contract}
            Preserve the intended optimization where possible, but prioritize restoring a successful build.
            Use the compiler/linker output as evidence. Fix the actual broken names, includes, signatures, scopes, types, declarations, and source/header mismatches; do not guess unrelated rewrites.
            Do not touch files outside the changed file list.
            Do not use placeholders or omit unchanged code inside any returned file. Every returned content field must be a complete file.
            Return strict JSON only.
            """
        ).strip()
        files = []
        for path in candidate.changed_files:
            content = candidate.files.get(path, "")
            if not self.cfg.include_full_target_files:
                content = truncate_middle(content, self.cfg.max_context_chars_per_file)
            summary = self.index.static_summaries.get(path)
            files.append(
                {
                    "path": path,
                    "summary": dataclasses.asdict(summary) if summary else {},
                    "paired_files": self.index.paired_files_for(path),
                    "content": content,
                }
            )
        user_prompt = textwrap.dedent(
            f"""
            Candidate hypothesis: {candidate.hypothesis}
            Candidate style: {candidate.style}
            Candidate changed files: {candidate.changed_files}

            Deep project analysis:
            {deep_context}

            Measured hypothesis learning memory:
            {learning_context}

            Compiler / linker output (stdout + stderr):
            {tail_text(compile_stderr, self.cfg.max_repair_error_chars)}

            Changed file contents:
            {json_dumps(files)}

            Return JSON:
            {{
              "repair_summary": "what was fixed across the changed files",
              "compile_self_check": "brief C++17 compile audit across the repaired files",
              "changes": [
                {{"path": "changed/file.cpp", "content": "complete repaired file content"}}
              ]
            }}
            """
        ).strip()
        return system_prompt, user_prompt


# -----------------------------
# Evaluator
# -----------------------------


class Evaluator:
    def __init__(self, cfg: EvolutionConfig):
        self.cfg = cfg
        self.project_root = Path(cfg.project_root).resolve()
        self.work_root = Path(cfg.work_root).resolve()
        ensure_dir(self.work_root / "candidates")

    def _copy_tree(self, src: Path, dst: Path, *, allow_hardlinks: bool = True) -> None:
        if dst.exists():
            shutil.rmtree(dst)
        copy_function = os.link if (self.cfg.use_hardlink_copy and allow_hardlinks) else shutil.copy2
        shutil.copytree(src, dst, copy_function=copy_function)

    def _candidate_root(self, candidate_id: str) -> Path:
        return self.work_root / "candidates" / candidate_id

    def _existing_parent_dir(self, candidate: Candidate) -> Optional[Path]:
        if candidate.parent_id in ("", "<none>", "baseline"):
            return None
        parent_root = self._candidate_root(candidate.parent_id)
        if parent_root.exists() and parent_root.is_dir():
            return parent_root
        return None

    def _materialize_source_dir(self, candidate: Candidate, parent_dir: Optional[Path]) -> Path:
        if parent_dir is not None and parent_dir.resolve() != self._candidate_root(candidate.candidate_id).resolve():
            return parent_dir
        return self.project_root

    def _candidate_meta_path(self, candidate: Candidate) -> Path:
        return self._candidate_root(candidate.candidate_id) / "agentic_evolve_meta.json"

    def _is_materialized_candidate_current(self, candidate: Candidate) -> bool:
        candidate_root = self._candidate_root(candidate.candidate_id)
        meta_path = self._candidate_meta_path(candidate)
        if not candidate_root.exists() or not meta_path.exists():
            return False
        try:
            meta = json.loads(read_text(meta_path))
        except Exception:
            return False
        return (
            meta.get("candidate_id") == candidate.candidate_id
            and meta.get("files_sha") == sha_files(candidate.files)
            and meta.get("rewrite_policy") == self._materialize_rewrite_policy(candidate, self._existing_parent_dir(candidate))
            and candidate.work_dir == str(candidate_root)
        )

    def ensure_materialized_candidate(self, candidate: Candidate) -> Path:
        if self._is_materialized_candidate_current(candidate):
            return self._candidate_root(candidate.candidate_id)
        return self.materialize_candidate(candidate)

    def _materialize_rewrite_policy(self, candidate: Candidate, parent_dir: Optional[Path]) -> str:
        if not candidate.changed_files:
            return "copy_only_no_changes"
        if parent_dir is not None or candidate.parent_id in ("", "<none>", "baseline"):
            return "rewrite_changed_files"
        return "rewrite_all_tracked_fallback"

    def _materialize_rewrite_paths(self, candidate: Candidate, parent_dir: Optional[Path]) -> List[str]:
        policy = self._materialize_rewrite_policy(candidate, parent_dir)
        if policy == "copy_only_no_changes":
            return []
        if policy == "rewrite_changed_files":
            return [path for path in candidate.changed_files if path in candidate.files]
        # Candidate.files is the cumulative mutable-file snapshot for this lineage.
        # Rewrite all tracked mutable files only when the parent directory is gone
        # and we must reconstruct inherited mutations from the snapshot.
        return list(candidate.files.keys())

    def materialize_candidate(self, candidate: Candidate) -> Path:
        candidate_root = self._candidate_root(candidate.candidate_id)
        parent_dir = self._existing_parent_dir(candidate)
        source_dir = self._materialize_source_dir(candidate, parent_dir)
        self._copy_tree(source_dir, candidate_root, allow_hardlinks=source_dir.resolve() == self.project_root)
        rewrite_policy = self._materialize_rewrite_policy(candidate, parent_dir)
        for rel_path in self._materialize_rewrite_paths(candidate, parent_dir):
            content = candidate.files.get(rel_path)
            if content is None:
                continue
            write_text(candidate_root / rel_path, content)
        meta = {
            "candidate_id": candidate.candidate_id,
            "generation": candidate.generation,
            "parent_id": candidate.parent_id,
            "files_sha": sha_files(candidate.files),
            "rewrite_policy": rewrite_policy,
            "parent_dir": str(parent_dir) if parent_dir is not None else "",
            "materialize_source_dir": str(source_dir),
            "changed_files": candidate.changed_files,
            "hypothesis": candidate.hypothesis,
            "style": candidate.style,
            "intents": candidate.intents,
            "timestamp": now_ts(),
        }
        write_text(candidate_root / "agentic_evolve_meta.json", json_dumps(meta))
        candidate.work_dir = str(candidate_root)
        return candidate_root

    def _run_command(self, command: str, *, cwd: Path, env: Dict[str, str], timeout: Optional[int] = None) -> subprocess.CompletedProcess:
        effective_timeout = timeout
        if effective_timeout is None and self.cfg.command_timeout_sec > 0:
            effective_timeout = self.cfg.command_timeout_sec
        try:
            return subprocess.run(
                command,
                cwd=str(cwd),
                shell=True,
                env=env,
                text=True,
                capture_output=True,
                timeout=effective_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            stderr += f"\nCommand timed out after {effective_timeout} second(s): {command}\n"
            return subprocess.CompletedProcess(command, 124, stdout=stdout, stderr=stderr)

    def _run_command_with_ready_marker(
        self,
        command: str,
        *,
        cwd: Path,
        env: Dict[str, str],
        ready_marker: str,
        ready_event: threading.Event,
        timeout: Optional[int] = None,
    ) -> subprocess.CompletedProcess:
        effective_timeout = timeout
        if effective_timeout is None and self.cfg.command_timeout_sec > 0:
            effective_timeout = self.cfg.command_timeout_sec
        stdout_parts: List[str] = []
        stderr_parts: List[str] = []
        marker = ready_marker.lower()

        proc = subprocess.Popen(
            command,
            cwd=str(cwd),
            shell=True,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
        )

        def reader(pipe: Any, parts: List[str]) -> None:
            try:
                for line in iter(pipe.readline, ""):
                    parts.append(line)
                    if marker and marker in line.lower():
                        ready_event.set()
            finally:
                try:
                    pipe.close()
                except Exception:
                    pass

        threads = [
            threading.Thread(target=reader, args=(proc.stdout, stdout_parts), daemon=True),
            threading.Thread(target=reader, args=(proc.stderr, stderr_parts), daemon=True),
        ]
        for thread in threads:
            thread.start()

        try:
            return_code = proc.wait(timeout=effective_timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            return_code = 124
            stderr_parts.append(f"\nCommand timed out after {effective_timeout} second(s): {command}\n")
        finally:
            for thread in threads:
                thread.join(timeout=1)
            ready_event.set()

        return subprocess.CompletedProcess(command, return_code, stdout="".join(stdout_parts), stderr="".join(stderr_parts))

    def _parse_score(self, stdout: str, stderr: str) -> Optional[float]:
        combined = stdout + "\n" + stderr
        m = re.search(self.cfg.score_pattern, combined)
        if m:
            return float(m.group(1))
        obj = parse_jsonish(stdout)
        if isinstance(obj, dict) and "score" in obj:
            try:
                return float(obj["score"])
            except Exception:
                pass
        obj = parse_jsonish(combined)
        if isinstance(obj, dict) and "score" in obj:
            try:
                return float(obj["score"])
            except Exception:
                pass
        floats = re.findall(r'-?\d+(?:\.\d+)?', stdout)
        if floats:
            try:
                return float(floats[-1])
            except Exception:
                pass
        return None

    def _parse_metrics(self, stdout: str, stderr: str) -> Dict[str, float]:
        combined = stdout + "\n" + stderr
        metrics: Dict[str, float] = {}
        for obj in parse_jsonish_values(combined):
            for key, value in collect_numeric_metrics(obj).items():
                metrics.setdefault(key, value)
        return metrics

    def _has_compile_error_marker(self, stdout: str, stderr: str) -> bool:
        combined = (stdout + "\n" + stderr).lower()
        return any(str(marker).lower() in combined for marker in self.cfg.compile_error_markers if str(marker).strip())

    def uses_staged_device_pipeline(self) -> bool:
        return bool(
            self.cfg.compile_command.strip()
            or self.cfg.deploy_command.strip()
            or self.cfg.device_evaluate_command.strip()
        )

    def _evaluation_context(self, candidate: Candidate) -> Tuple[Path, Dict[str, str], Dict[str, str], Path, Optional[Path]]:
        parent_dir = self._existing_parent_dir(candidate)
        source_dir = self._materialize_source_dir(candidate, parent_dir)
        candidate_root = self.ensure_materialized_candidate(candidate)
        candidate_name = candidate.candidate_id
        lib_name = candidate_name
        env = os.environ.copy()
        env.update(
            {
                "EVOLVE_CANDIDATE_DIR": str(candidate_root),
                "EVOLVE_PARENT_DIR": str(parent_dir) if parent_dir is not None else "",
                "EVOLVE_MATERIALIZE_SOURCE_DIR": str(source_dir),
                "EVOLVE_PROJECT_ROOT": str(self.project_root),
                "EVOLVE_WORK_ROOT": str(self.work_root),
                "EVOLVE_CANDIDATE_ID": candidate.candidate_id,
                "EVOLVE_CANDIDATE_NAME": candidate_name,
                "EVOLVE_LIB_NAME": lib_name,
                "EVOLVE_GENERATION": str(candidate.generation),
                "EVOLVE_PARENT_ID": candidate.parent_id,
            }
        )
        template_vars = {
            "candidate_dir": str(candidate_root),
            "parent_dir": str(parent_dir) if parent_dir is not None else "",
            "materialize_source_dir": str(source_dir),
            "project_root": str(self.project_root),
            "work_root": str(self.work_root),
            "candidate_id": candidate.candidate_id,
            "candidate_name": candidate_name,
            "lib_name": lib_name,
            "generation": str(candidate.generation),
            "parent_id": candidate.parent_id,
        }
        return candidate_root, env, template_vars, source_dir, parent_dir

    def compile_and_deploy(self, candidate: Candidate) -> EvalResult:
        candidate_root, env, template_vars, _, _ = self._evaluation_context(candidate)
        stdout = ""
        stderr = ""
        if self.cfg.prepare_command.strip():
            prepare_command = render_template(self.cfg.prepare_command, template_vars)
            prepare_proc = self._run_command(prepare_command, cwd=candidate_root, env=env)
            stdout += prepare_proc.stdout
            stderr += prepare_proc.stderr
            if prepare_proc.returncode != 0:
                return EvalResult(
                    score=self.cfg.compile_fail_score,
                    compile_ok=False,
                    return_code=prepare_proc.returncode,
                    stdout=stdout,
                    stderr=stderr,
                    work_dir=str(candidate_root),
                    metrics={},
                    stage="prepare",
                )

        if self.cfg.compile_command.strip():
            compile_command = render_template(self.cfg.compile_command, template_vars)
            compile_proc = self._run_command(compile_command, cwd=candidate_root, env=env)
            stdout += compile_proc.stdout
            stderr += compile_proc.stderr
            if compile_proc.returncode != 0:
                return EvalResult(
                    score=self.cfg.compile_fail_score,
                    compile_ok=False,
                    return_code=compile_proc.returncode,
                    stdout=stdout,
                    stderr=stderr,
                    work_dir=str(candidate_root),
                    metrics={},
                    stage="compile",
                )

        if self.cfg.deploy_command.strip():
            deploy_command = render_template(self.cfg.deploy_command, template_vars)
            deploy_proc = self._run_command(deploy_command, cwd=candidate_root, env=env)
            stdout += deploy_proc.stdout
            stderr += deploy_proc.stderr
            if deploy_proc.returncode != 0:
                return EvalResult(
                    score=self.cfg.compile_fail_score,
                    compile_ok=False,
                    return_code=deploy_proc.returncode,
                    stdout=stdout,
                    stderr=stderr,
                    work_dir=str(candidate_root),
                    metrics={},
                    stage="deploy",
                )

        metrics = self._parse_metrics(stdout, stderr)
        score = self._parse_score(stdout, stderr)
        return EvalResult(
            score=score if score is not None else self.cfg.compile_fail_score,
            compile_ok=True,
            return_code=0,
            stdout=stdout,
            stderr=stderr,
            work_dir=str(candidate_root),
            metrics=metrics,
            stage="compile_deploy",
            ready_for_device=bool(self.cfg.device_evaluate_command.strip()),
        )

    def evaluate_on_device(
        self,
        candidate: Candidate,
        compile_result: EvalResult,
        handoff_ready_event: Optional[threading.Event] = None,
    ) -> EvalResult:
        if not self.cfg.device_evaluate_command.strip():
            if handoff_ready_event is not None:
                handoff_ready_event.set()
            score = compile_result.score if compile_result.score != self.cfg.compile_fail_score else self._parse_score(compile_result.stdout, compile_result.stderr)
            compile_ok = compile_result.compile_ok and score is not None
            return dataclasses.replace(
                compile_result,
                score=score if (compile_ok and score is not None) else self.cfg.compile_fail_score,
                compile_ok=compile_ok,
                metrics=compile_result.metrics if compile_ok else {},
                stage="evaluate",
                ready_for_device=False,
            )

        candidate_root, env, template_vars, _, _ = self._evaluation_context(candidate)
        device_command = render_template(self.cfg.device_evaluate_command, template_vars)
        marker = self.cfg.device_handoff_ready_marker.strip()
        if handoff_ready_event is not None and marker:
            device_proc = self._run_command_with_ready_marker(
                device_command,
                cwd=candidate_root,
                env=env,
                ready_marker=marker,
                ready_event=handoff_ready_event,
            )
        else:
            device_proc = self._run_command(device_command, cwd=candidate_root, env=env)
            if handoff_ready_event is not None:
                handoff_ready_event.set()
        stdout = compile_result.stdout + device_proc.stdout
        stderr = compile_result.stderr + device_proc.stderr
        metrics = self._parse_metrics(stdout, stderr)
        score = self._parse_score(device_proc.stdout, device_proc.stderr)
        if score is None:
            score = metrics.get("score")
        compile_ok = compile_result.compile_ok and device_proc.returncode == 0 and score is not None
        return EvalResult(
            score=score if (compile_ok and score is not None) else self.cfg.compile_fail_score,
            compile_ok=compile_ok,
            return_code=device_proc.returncode,
            stdout=stdout,
            stderr=stderr,
            work_dir=str(candidate_root),
            metrics=metrics if compile_ok else {},
            stage="device",
            ready_for_device=False,
        )

    def evaluate(self, candidate: Candidate) -> EvalResult:
        if self.uses_staged_device_pipeline():
            compile_result = self.compile_and_deploy(candidate)
            if not compile_result.compile_ok:
                return compile_result
            return self.evaluate_on_device(candidate, compile_result)

        candidate_root, env, template_vars, _, _ = self._evaluation_context(candidate)

        prepare_stdout = ""
        prepare_stderr = ""
        if self.cfg.prepare_command.strip():
            prepare_command = render_template(self.cfg.prepare_command, template_vars)
            prepare_proc = self._run_command(prepare_command, cwd=candidate_root, env=env)
            prepare_stdout = prepare_proc.stdout
            prepare_stderr = prepare_proc.stderr
            if prepare_proc.returncode != 0:
                return EvalResult(
                    score=self.cfg.compile_fail_score,
                    compile_ok=False,
                    return_code=prepare_proc.returncode,
                    stdout=prepare_stdout,
                    stderr=prepare_stderr,
                    work_dir=str(candidate_root),
                    metrics={},
                    stage="prepare",
                )

        eval_stdout = ""
        eval_stderr = ""
        if self.cfg.evaluate_command.strip():
            evaluate_command = render_template(self.cfg.evaluate_command, template_vars)
            eval_proc = self._run_command(evaluate_command, cwd=candidate_root, env=env)
            eval_stdout = eval_proc.stdout
            eval_stderr = eval_proc.stderr
            metrics = self._parse_metrics(eval_stdout, eval_stderr)
            score = self._parse_score(eval_stdout, eval_stderr)
            if score is None:
                score = metrics.get("score")
            compile_ok = eval_proc.returncode == 0 and score is not None
            stage = "compile" if (not compile_ok and self._has_compile_error_marker(eval_stdout, eval_stderr)) else "evaluate"
            return EvalResult(
                score=score if (compile_ok and score is not None) else self.cfg.compile_fail_score,
                compile_ok=compile_ok,
                return_code=eval_proc.returncode,
                stdout=prepare_stdout + eval_stdout,
                stderr=prepare_stderr + eval_stderr,
                work_dir=str(candidate_root),
                metrics=metrics if compile_ok else {},
                stage=stage,
            )

        raise RuntimeError("evaluate_command is empty. Set it in config.")

    def cleanup_candidate_dir(self, result: EvalResult, *, keep: bool) -> None:
        if keep:
            return
        try:
            shutil.rmtree(result.work_dir)
        except Exception:
            pass


# -----------------------------
# Evolver
# -----------------------------


class Evolver:
    def __init__(self, cfg: EvolutionConfig, llm_cfg: LLMConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.random_seed)
        self.llm = ServerLLMClient(llm_cfg)
        self.index = ProjectIndex(cfg, self.llm)
        self.archive = Archive(cfg)
        self.prompts = PromptFactory(cfg, self.index, self.archive, self.llm)
        self.evaluator = Evaluator(cfg)
        self.base_files = self.index.snapshot()
        self.run_log_path = Path(cfg.work_root) / "run.log"
        ensure_dir(Path(cfg.work_root))

    def log(self, message: str) -> None:
        line = f"[{now_ts()}] {message}"
        print(line, flush=True)
        with self.run_log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def mutation_repair_output_tokens(self) -> int:
        multiplier = max(1, int(self.llm.cfg.mutation_repair_output_token_multiplier))
        return int(self.llm.cfg.max_output_tokens) * multiplier

    def apply_eval_result(self, candidate: Candidate, result: EvalResult, *, cleanup: bool = True) -> Candidate:
        candidate.score = result.score
        candidate.compile_ok = result.compile_ok
        candidate.metrics = result.metrics
        candidate.eval_stage = result.stage
        candidate.stdout = result.stdout
        candidate.stderr = result.stderr
        candidate.work_dir = result.work_dir
        if cleanup:
            self.evaluator.cleanup_candidate_dir(
                result,
                keep=self.cfg.keep_successful_candidates if result.compile_ok else self.cfg.keep_failed_candidates,
            )
        return candidate

    def candidate_output_text(self, candidate: Candidate) -> str:
        if candidate.stdout and candidate.stderr:
            return candidate.stdout + "\n" + candidate.stderr
        return candidate.stdout or candidate.stderr or ""

    def has_compile_error_marker(self, candidate: Candidate) -> bool:
        combined = self.candidate_output_text(candidate).lower()
        return any(str(marker).lower() in combined for marker in self.cfg.compile_error_markers if str(marker).strip())

    def is_repairable_failure(self, candidate: Candidate) -> bool:
        if candidate.compile_ok:
            return False
        if self.has_compile_error_marker(candidate):
            return True
        if self.evaluator.uses_staged_device_pipeline():
            return candidate.eval_stage in {"prepare", "compile", "deploy", "no-change"}
        return True

    def analyze_generation_outcomes(self, generation: int, evaluated: List[Candidate]) -> None:
        if not self.cfg.enable_hypothesis_learning or not evaluated:
            return
        score_by_id: Dict[str, Optional[float]] = {}
        metrics_by_id: Dict[str, Dict[str, float]] = {}
        for elite in self.archive.elites:
            score_by_id[elite.candidate_id] = elite.score
            metrics_by_id[elite.candidate_id] = dict(elite.metrics)
        for meta in self.archive.history_meta:
            cid = meta.get("candidate_id")
            if isinstance(cid, str):
                score_by_id[cid] = meta.get("score")
                raw_metrics = meta.get("metrics", {})
                if isinstance(raw_metrics, dict):
                    metrics_by_id[cid] = {
                        str(key): float(value)
                        for key, value in raw_metrics.items()
                        if isinstance(value, (int, float)) and not isinstance(value, bool)
                    }

        primary_metric_keys = normalize_metric_keys(self.cfg.primary_metric_keys)
        outcome_rows: List[Dict[str, Any]] = []
        for candidate in sorted(evaluated, key=lambda c: float(c.score if c.score is not None else -1e300), reverse=True):
            parent_score = score_by_id.get(candidate.parent_id)
            delta = None
            try:
                if parent_score is not None and candidate.score is not None:
                    delta = float(candidate.score) - float(parent_score)
            except Exception:
                delta = None
            parent_metrics = metrics_by_id.get(candidate.parent_id, {})
            metric_deltas: Dict[str, float] = {}
            metric_improvements: Dict[str, float] = {}
            for key, value in candidate.metrics.items():
                if key in parent_metrics:
                    try:
                        delta_value = float(value) - float(parent_metrics[key])
                        metric_deltas[key] = delta_value
                        improvement = metric_improvement_from_delta(self.cfg.metric_objectives, key, delta_value)
                        if improvement is not None:
                            metric_improvements[key] = improvement
                    except Exception:
                        pass
            outcome_rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "parent_id": candidate.parent_id,
                    "parent_score": parent_score,
                    "score": candidate.score,
                    "score_delta_from_parent": delta,
                    "primary_metrics": {key: candidate.metrics[key] for key in primary_metric_keys if key in candidate.metrics},
                    "primary_metric_deltas_from_parent": {key: metric_deltas[key] for key in primary_metric_keys if key in metric_deltas},
                    "primary_metric_improvements_from_parent": {
                        key: metric_improvements[key] for key in primary_metric_keys if key in metric_improvements
                    },
                    "metrics": candidate.metrics,
                    "metric_deltas_from_parent": metric_deltas,
                    "metric_improvements_from_parent": metric_improvements,
                    "compile_ok": candidate.compile_ok,
                    "changed_files": candidate.changed_files,
                    "hypothesis": candidate.hypothesis,
                    "style": candidate.style,
                    "llm_notes": candidate.llm_notes,
                    "stdout_tail": tail_text(candidate.stdout, self.cfg.max_hypothesis_learning_log_chars),
                    "stderr_tail": tail_text(candidate.stderr, self.cfg.max_hypothesis_learning_log_chars),
                }
            )

        file_stats_rows = []
        for path, stats in sorted(
            self.archive.file_stats.items(),
            key=lambda kv: (
                -float(kv[1].get("improvements", 0.0)),
                -float(kv[1].get("best_delta", 0.0)),
                float(kv[1].get("compile_failures", 0.0)),
            ),
        )[:200]:
            summary = self.index.static_summaries.get(path)
            file_stats_rows.append(
                {
                    "path": path,
                    "stats": stats,
                    "kind": summary.kind if summary else "",
                    "purpose": summary.purpose if summary else "",
                    "hot_hints": summary.hot_hints if summary else [],
                    "paired_files": summary.paired_files if summary else [],
                }
            )

        system_prompt = textwrap.dedent(
            f"""
            You are the performance research lead for an evolutionary C++ optimizer.
            Your task is to learn from measured candidate outcomes and improve the next search step.

            {self.prompts.optimization_guidance()}

            Rules:
            - Treat compile success as a gate only; do not count compile-only cleanup as a validated optimization.
            - Prefer explanations grounded in score deltas, parsed benchmark metrics, stdout/stderr metrics, changed files, and hypotheses.
            - This benchmark's primary navigation metrics are configured separately. Focus on them when present and ignore absent metric families.
            - Use metric_improvements_from_parent as directional evidence: positive means better according to metric_objectives, negative means worse.
            - Interpret {normalize_metric_key(self.cfg.target_fps_metric)} against target FPS {self.cfg.target_fps:g}: below target means FPS recovery is urgent; above target means power reductions are valuable only if target FPS is preserved.
            - Be explicit about FPS/frame-time/power mechanisms and about which hypotheses should be retried, refined, or abandoned.
            - Do not invent metrics that are not present. If stdout/stderr do not expose FPS/power separately, infer cautiously from the score and say what evidence is missing.
            - Return strict JSON only.
            """
        ).strip()
        user_prompt = textwrap.dedent(
            f"""
            Generation: {generation}
            Current best: {self.archive.best.candidate_id if self.archive.best else None}
            Current best score: {self.archive.best.score if self.archive.best else None}

            Deep project analysis:
            {self.prompts.deep_project_context()}

            Metric objectives:
            {json_dumps(self.cfg.metric_objectives)}

            FPS target:
            {{"metric": "{normalize_metric_key(self.cfg.target_fps_metric)}", "target": {self.cfg.target_fps}}}

            Primary navigation metrics:
            {json_dumps(primary_metric_keys)}
            Only these parsed metrics are expected from this benchmark. If other metrics are absent, ignore them. Use score for ranking, and use the primary metrics to explain whether a score change came from better 1% low FPS, lower power, or a tradeoff between them.

            Previous measured hypothesis memory:
            {self.archive.strategy_memory_text()}

            File statistics:
            {json_dumps(file_stats_rows)}

            Evaluated candidate outcomes:
            {json_dumps(outcome_rows)}

            Analyze the generation as an experiment. Identify what actually helped the evaluator score, what only helped compile or failed, and what hypotheses should be tested next.

            Return JSON object:
            {{
              "planning_directive": "concise directive for the next planner, focused on FPS/frame-time/power score wins",
              "validated_hypotheses": [
                {{"hypothesis": "what seems to work", "evidence": "candidate ids and score deltas", "next_test": "specific follow-up"}}
              ],
              "rejected_hypotheses": [
                {{"hypothesis": "what to avoid", "evidence": "candidate ids/errors/deltas", "avoidance_rule": "specific constraint"}}
              ],
              "fps_power_lessons": ["specific measured lesson"],
              "metric_tradeoffs": [
                {{"candidate_id": "candidate", "evidence": "which metrics improved or regressed", "interpretation": "what tradeoff means for next search"}}
              ],
              "promote_next": ["specific mutation direction to try next"],
              "avoid_next": ["specific mutation direction to avoid next"],
              "file_priorities": [
                {{"path": "relative/file.cpp", "priority": "high|medium|low", "reason": "why this file is promising or risky", "paired_files": ["relative/file.hpp"]}}
              ],
              "risk_controls": ["compile and semantic controls for next mutations"]
            }}
            """
        ).strip()
        try:
            obj = self.llm.complete_json(
                system_prompt,
                user_prompt,
                temperature=0.2,
                max_output_tokens=self.cfg.max_hypothesis_learning_output_tokens,
            )
        except Exception as exc:
            self.log(f"Generation {generation}: hypothesis learning failed: {exc}")
            return
        if not isinstance(obj, dict):
            self.log(f"Generation {generation}: hypothesis learning returned non-object JSON; ignored")
            return
        self.archive.record_generation_learning(generation, obj)
        directive = str(obj.get("planning_directive", "")).strip()
        if directive:
            self.log(f"Generation {generation}: hypothesis learning directive: {truncate_middle(directive, 1200)}")

    def _make_baseline_candidate(self) -> Candidate:
        return Candidate(
            candidate_id="baseline",
            generation=-1,
            parent_id="<none>",
            files=dict(self.base_files),
            changed_files=[],
            hypothesis="Baseline project state.",
            intents={},
            style="baseline",
        )

    def _loaded_baseline(self) -> Optional[Candidate]:
        if self.archive.best is None:
            return None
        for elite in self.archive.elites:
            if elite.candidate_id == "baseline":
                return elite
        return None

    def _evaluate_baseline_candidate(self, baseline: Candidate) -> Candidate:
        self.log("Evaluating baseline candidate.")
        result = self.evaluator.evaluate(baseline)
        self.apply_eval_result(baseline, result, cleanup=False)
        self.archive.register(baseline)
        self.evaluator.cleanup_candidate_dir(
            result,
            keep=self.cfg.keep_successful_candidates if result.compile_ok else self.cfg.keep_failed_candidates,
        )
        self.log(f"Baseline score={baseline.score} compile_ok={baseline.compile_ok}")
        return baseline

    def ensure_baseline(self) -> Candidate:
        loaded = self._loaded_baseline()
        if loaded is not None:
            return loaded
        # History may have been loaded, but a baseline snapshot is not available.
        # Re-evaluate baseline so we have a live in-memory candidate.
        baseline = self._make_baseline_candidate()
        return self._evaluate_baseline_candidate(baseline)

    def prepare_candidate_dir(self, candidate: Candidate) -> None:
        if not self.cfg.pre_materialize_built_candidates or not candidate.changed_files:
            return
        self.evaluator.materialize_candidate(candidate)

    def mutate_one_file(self, current_files: Dict[str, str], plan: ChildPlan, target_path: str) -> Tuple[str, Optional[str], str]:
        system_prompt, user_prompt = self.prompts.mutation_prompt(current_files, plan, target_path)
        try:
            obj = self.llm.complete_json(
                system_prompt,
                user_prompt,
                temperature=self.llm.cfg.temperature_mutate,
                max_output_tokens=self.mutation_repair_output_tokens(),
            )
            if not isinstance(obj, dict):
                raise RuntimeError("mutation response is not an object")
            path = str(obj.get("path", target_path))
            if path != target_path:
                raise RuntimeError(f"worker tried to edit {path}, expected {target_path}")
            content = obj.get("content")
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("worker did not return content")
            content = clean_llm_file_content(content)
            incomplete_marker = content_looks_incomplete(content)
            if incomplete_marker:
                raise RuntimeError(f"worker returned incomplete file content marker: {incomplete_marker}")
            note = f"{target_path}: {obj.get('edit_summary', '').strip()} (risk={obj.get('risk', 'n/a')})"
            return target_path, content, note
        except Exception as exc:
            return target_path, None, f"{target_path}: mutation failed: {exc}"

    def review_candidate(self, base_files: Dict[str, str], candidate_files: Dict[str, str], plan: ChildPlan) -> Tuple[Dict[str, str], List[str]]:
        changed_focus = [path for path in plan.focus_files if candidate_files.get(path) != base_files.get(path)]
        if not self.cfg.review_pass or not changed_focus:
            return candidate_files, []
        if len(plan.focus_files) <= 1 and not self.cfg.review_single_file_pass:
            return candidate_files, []
        system_prompt, user_prompt = self.prompts.review_prompt(base_files, candidate_files, plan)
        notes: List[str] = []
        try:
            obj = self.llm.complete_json(
                system_prompt,
                user_prompt,
                temperature=0.3,
                max_output_tokens=self.mutation_repair_output_tokens(),
            )
            if not isinstance(obj, dict):
                return candidate_files, notes
            notes.append(str(obj.get("review_notes", "")).strip())
            notes.append(str(obj.get("compile_self_check", "")).strip())
            changes = obj.get("changes", [])
            if isinstance(changes, list):
                for row in changes:
                    if not isinstance(row, dict):
                        continue
                    path = str(row.get("path", ""))
                    content = row.get("content")
                    if path in plan.focus_files and isinstance(content, str) and content.strip():
                        cleaned = clean_llm_file_content(content)
                        if content_looks_incomplete(cleaned):
                            notes.append(f"{path}: review returned incomplete content; ignored")
                            continue
                        candidate_files[path] = cleaned
            return candidate_files, [n for n in notes if n]
        except Exception as exc:
            return candidate_files, [f"review failed: {exc}"]

    def build_candidate(self, generation: int, parent: Candidate, plan: ChildPlan) -> Candidate:
        current_files = dict(parent.files)
        candidate_id = f"g{generation:04d}_{short_hash(plan.plan_id + parent.candidate_id + str(time.time_ns()), 10)}"
        notes: List[str] = []
        changed: List[str] = []
        with cf.ThreadPoolExecutor(max_workers=min(self.cfg.max_llm_workers, max(1, len(plan.focus_files)))) as pool:
            futures = [pool.submit(self.mutate_one_file, current_files, plan, path) for path in plan.focus_files]
            for fut in cf.as_completed(futures):
                path, new_content, note = fut.result()
                notes.append(note)
                if new_content is not None and new_content != current_files[path]:
                    current_files[path] = new_content
                    changed.append(path)
        changed = sorted(set(changed))
        candidate_files = dict(parent.files)
        for path in changed:
            candidate_files[path] = current_files[path]
        candidate_files, review_notes = self.review_candidate(dict(parent.files), candidate_files, plan)
        notes.extend(review_notes)
        changed = sorted([path for path in plan.focus_files if candidate_files.get(path) != parent.files.get(path)])
        return Candidate(
            candidate_id=candidate_id,
            generation=generation,
            parent_id=parent.candidate_id,
            files=candidate_files,
            changed_files=changed,
            hypothesis=plan.hypothesis,
            intents=plan.intents,
            style=plan.style,
            llm_notes=notes,
        )

    def build_and_prepare_candidate(self, generation: int, parent: Candidate, plan: ChildPlan) -> Candidate:
        candidate = self.build_candidate(generation, parent, plan)
        try:
            self.prepare_candidate_dir(candidate)
        except Exception as exc:
            candidate.llm_notes.append(f"pre-materialize failed: {exc}")
        return candidate

    def build_generation(self, generation: int, parent: Candidate) -> List[Candidate]:
        self.log(f"Generation {generation}: selected parent {parent.candidate_id} score={parent.score}")
        plans = self.prompts.plan_children(parent, generation, self.rng)
        self.log(f"Generation {generation}: created {len(plans)} child plans")
        build_workers = min(self.cfg.max_llm_workers, max(1, len(plans)))
        built: List[Candidate] = []
        with cf.ThreadPoolExecutor(max_workers=build_workers) as pool:
            futures = [pool.submit(self.build_and_prepare_candidate, generation, parent, plan) for plan in plans]
            for fut in cf.as_completed(futures):
                try:
                    candidate = fut.result()
                    built.append(candidate)
                    prepared = f" prepared={candidate.work_dir}" if candidate.work_dir else ""
                    self.log(
                        f"Built {candidate.candidate_id} changed={candidate.changed_files}{prepared} notes={candidate.llm_notes[:2]}"
                    )
                except Exception:
                    self.log("Candidate build crashed:\n" + traceback.format_exc())
        return built

    def evaluate_staged_pipeline(
        self,
        *,
        build_futures: Optional[Dict[cf.Future[Candidate], None]] = None,
        built: Optional[List[Candidate]] = None,
        repair_workers: int,
    ) -> List[Candidate]:
        pending_builds: Dict[cf.Future[Candidate], None] = dict(build_futures or {})
        compile_queue: deque[Tuple[Candidate, int]] = deque((candidate, 0) for candidate in (built or []))
        evaluated: List[Candidate] = []

        def collect_build(future: cf.Future[Candidate]) -> None:
            try:
                candidate = future.result()
            except Exception:
                self.log("Candidate build crashed:\n" + traceback.format_exc())
                return
            prepared = f" prepared={candidate.work_dir}" if candidate.work_dir else ""
            self.log(f"Built {candidate.candidate_id} changed={candidate.changed_files}{prepared} notes={candidate.llm_notes[:2]}")
            compile_queue.append((candidate, 0))

        with (
            cf.ThreadPoolExecutor(max_workers=1) as compile_pool,
            cf.ThreadPoolExecutor(max_workers=1) as device_pool,
            cf.ThreadPoolExecutor(max_workers=repair_workers) as repair_pool,
        ):
            compile_future: Optional[cf.Future[Tuple[Candidate, EvalResult]]] = None
            compile_attempt = 0
            device_futures: Dict[cf.Future[Candidate], threading.Event] = {}
            repair_futures: Dict[cf.Future[Tuple[Candidate, bool]], int] = {}

            while pending_builds or compile_queue or compile_future is not None or device_futures or repair_futures:
                ready_builds = [future for future in pending_builds if future.done()]
                for future in ready_builds:
                    pending_builds.pop(future, None)
                    collect_build(future)

                ready_repairs = [future for future in repair_futures if future.done()]
                for future in ready_repairs:
                    attempt = repair_futures.pop(future)
                    try:
                        repaired_candidate, repaired_any = future.result()
                    except Exception:
                        self.log("Candidate repair crashed:\n" + traceback.format_exc())
                        continue
                    if repaired_any:
                        compile_queue.append((repaired_candidate, attempt))
                        self.log(f"Repair produced new contents for {repaired_candidate.candidate_id}; queued staged compile")
                    elif attempt < self.cfg.max_compile_repairs:
                        next_attempt = attempt + 1
                        retry_future = repair_pool.submit(self.repair_candidate_once, repaired_candidate, next_attempt)
                        repair_futures[retry_future] = next_attempt
                        self.log(
                            f"Repair attempt {attempt}/{self.cfg.max_compile_repairs} produced no usable changes for "
                            f"{repaired_candidate.candidate_id}; queued retry {next_attempt}/{self.cfg.max_compile_repairs}"
                        )
                    else:
                        evaluated.append(repaired_candidate)
                        self.log(
                            f"Repair produced no usable changes for {repaired_candidate.candidate_id}; "
                            f"exhausted {self.cfg.max_compile_repairs} attempt(s)"
                        )

                if compile_future is not None and compile_future.done():
                    try:
                        candidate, compile_result = compile_future.result()
                    except Exception:
                        self.log("Candidate staged compile/deploy crashed:\n" + traceback.format_exc())
                        candidate = None
                        compile_result = None
                    compile_future = None
                    if candidate is not None and compile_result is not None:
                        self.log(
                            f"Compiled/deployed {candidate.candidate_id} stage={compile_result.stage} "
                            f"ok={compile_result.compile_ok}"
                        )
                        if compile_result.compile_ok:
                            handoff_ready = threading.Event()
                            if self.cfg.deploy_command.strip():
                                handoff_ready.set()
                            future = device_pool.submit(self.device_evaluate_candidate_once, candidate, compile_result, handoff_ready)
                            device_futures[future] = handoff_ready
                            if handoff_ready.is_set():
                                if self.cfg.deploy_command.strip():
                                    self.log(f"Queued device evaluation for {candidate.candidate_id}; deploy handoff complete, compile lane can continue")
                                elif self.cfg.device_handoff_ready_marker.strip():
                                    self.log(f"Queued device evaluation for {candidate.candidate_id}; device handoff marker already observed, compile lane can continue")
                                else:
                                    self.log(f"Queued device evaluation for {candidate.candidate_id}; device command already finished, compile lane can continue")
                            elif self.cfg.device_handoff_ready_marker.strip():
                                self.log(
                                    f"Queued device evaluation for {candidate.candidate_id}; compile lane waits for marker "
                                    f"{self.cfg.device_handoff_ready_marker!r}"
                                )
                            else:
                                self.log(
                                    f"Queued device evaluation for {candidate.candidate_id}; compile lane waits for device command to finish "
                                    "because no deploy_command or device_handoff_ready_marker is configured"
                                )
                        elif (
                            self.is_repairable_failure(candidate)
                            and candidate.changed_files
                            and compile_attempt < self.cfg.max_compile_repairs
                        ):
                            next_attempt = compile_attempt + 1
                            future = repair_pool.submit(self.repair_candidate_once, candidate, next_attempt)
                            repair_futures[future] = next_attempt
                            self.log(
                                f"Queued repair {next_attempt}/{self.cfg.max_compile_repairs} for "
                                f"{candidate.candidate_id} after staged {compile_result.stage} failure"
                            )
                        else:
                            evaluated.append(candidate)

                ready_devices = [future for future in device_futures if future.done()]
                for future in ready_devices:
                    handoff_ready = device_futures.pop(future, None)
                    try:
                        candidate = future.result()
                    except Exception:
                        self.log("Candidate device evaluation crashed:\n" + traceback.format_exc())
                        if handoff_ready is not None:
                            handoff_ready.set()
                        continue
                    if handoff_ready is not None:
                        handoff_ready.set()
                    evaluated.append(candidate)
                    self.log(
                        f"Device evaluated {candidate.candidate_id} score={candidate.score} "
                        f"compile_ok={candidate.compile_ok} stage={candidate.eval_stage}"
                    )

                handoff_ready_for_all_devices = all(event.is_set() for event in device_futures.values())
                if compile_future is None and compile_queue and handoff_ready_for_all_devices:
                    candidate, compile_attempt = compile_queue.popleft()
                    compile_future = compile_pool.submit(self.compile_candidate_for_device_once, candidate)
                    self.log(
                        f"Queued staged compile/deploy for {candidate.candidate_id} "
                        f"(repair attempts used={compile_attempt}/{self.cfg.max_compile_repairs})"
                    )
                    continue
                if compile_future is None and compile_queue and not handoff_ready_for_all_devices:
                    self.log("Compile lane waiting for device handoff before preparing the next candidate")

                pending: List[cf.Future[Any]] = []
                if pending_builds:
                    pending.extend(pending_builds.keys())
                if compile_future is not None:
                    pending.append(compile_future)
                pending.extend(device_futures.keys())
                pending.extend(repair_futures.keys())
                if pending:
                    wait_timeout = 0.2 if compile_queue and not handoff_ready_for_all_devices else None
                    cf.wait(tuple(pending), return_when=cf.FIRST_COMPLETED, timeout=wait_timeout)

        return evaluated

    def evaluate_build_futures_streaming(
        self,
        build_futures: Dict[cf.Future[Candidate], None],
        *,
        repair_workers: int,
    ) -> List[Candidate]:
        if self.evaluator.uses_staged_device_pipeline():
            return self.evaluate_staged_pipeline(build_futures=build_futures, repair_workers=repair_workers)
        eval_queue: deque[Tuple[Candidate, int]] = deque()
        evaluated: List[Candidate] = []

        def collect_build(future: cf.Future[Candidate]) -> None:
            try:
                candidate = future.result()
            except Exception:
                self.log("Candidate build crashed:\n" + traceback.format_exc())
                return
            prepared = f" prepared={candidate.work_dir}" if candidate.work_dir else ""
            self.log(f"Built {candidate.candidate_id} changed={candidate.changed_files}{prepared} notes={candidate.llm_notes[:2]}")
            eval_queue.append((candidate, 0))

        def collect_repair(
            future: cf.Future[Tuple[Candidate, bool]],
            attempt: int,
            repair_pool: cf.ThreadPoolExecutor,
            repair_futures: Dict[cf.Future[Tuple[Candidate, bool]], int],
        ) -> None:
            try:
                repaired_candidate, repaired_any = future.result()
            except Exception:
                self.log("Candidate repair crashed:\n" + traceback.format_exc())
                return
            if repaired_any:
                eval_queue.append((repaired_candidate, attempt))
                self.log(f"Repair produced new contents for {repaired_candidate.candidate_id}; queued re-evaluation")
            elif attempt < self.cfg.max_compile_repairs:
                next_attempt = attempt + 1
                retry_future = repair_pool.submit(self.repair_candidate_once, repaired_candidate, next_attempt)
                repair_futures[retry_future] = next_attempt
                self.log(
                    f"Repair attempt {attempt}/{self.cfg.max_compile_repairs} produced no usable changes for "
                    f"{repaired_candidate.candidate_id}; queued retry {next_attempt}/{self.cfg.max_compile_repairs}"
                )
            else:
                evaluated.append(repaired_candidate)
                self.log(
                    f"Repair produced no usable changes for {repaired_candidate.candidate_id}; "
                    f"exhausted {self.cfg.max_compile_repairs} attempt(s)"
                )

        with cf.ThreadPoolExecutor(max_workers=repair_workers) as repair_pool:
            repair_futures: Dict[cf.Future[Tuple[Candidate, bool]], int] = {}

            while build_futures or eval_queue or repair_futures:
                ready_builds = [future for future in build_futures if future.done()]
                for future in ready_builds:
                    build_futures.pop(future, None)
                    collect_build(future)

                ready_repairs = [future for future in repair_futures if future.done()]
                for future in ready_repairs:
                    attempt = repair_futures.pop(future)
                    collect_repair(future, attempt, repair_pool, repair_futures)

                if eval_queue:
                    candidate, repair_attempts = eval_queue.popleft()
                    candidate = self.evaluate_candidate_once(candidate)
                    self.log(f"Evaluated {candidate.candidate_id} score={candidate.score} compile_ok={candidate.compile_ok}")
                    if (
                        self.is_repairable_failure(candidate)
                        and candidate.changed_files
                        and repair_attempts < self.cfg.max_compile_repairs
                    ):
                        next_attempt = repair_attempts + 1
                        future = repair_pool.submit(self.repair_candidate_once, candidate, next_attempt)
                        repair_futures[future] = next_attempt
                        self.log(
                            f"Queued repair {next_attempt}/{self.cfg.max_compile_repairs} for {candidate.candidate_id} while evaluator continues"
                        )
                    else:
                        evaluated.append(candidate)
                    continue

                pending = tuple(build_futures.keys()) + tuple(repair_futures.keys())
                if pending:
                    cf.wait(pending, return_when=cf.FIRST_COMPLETED)

        return evaluated

    def build_and_evaluate_generation_streaming(self, generation: int, parent: Candidate) -> List[Candidate]:
        self.log(f"Generation {generation}: selected parent {parent.candidate_id} score={parent.score}")
        plans = self.prompts.plan_children(parent, generation, self.rng)
        self.log(f"Generation {generation}: created {len(plans)} child plans")
        if not plans:
            return []

        build_workers = min(self.cfg.max_llm_workers, max(1, len(plans)))
        repair_workers = max(1, min(self.cfg.max_llm_workers, len(plans)))
        with cf.ThreadPoolExecutor(max_workers=build_workers) as build_pool:
            build_futures: Dict[cf.Future[Candidate], None] = {
                build_pool.submit(self.build_and_prepare_candidate, generation, parent, plan): None for plan in plans
            }
            return self.evaluate_build_futures_streaming(build_futures, repair_workers=repair_workers)

    def evaluate_candidate_once(self, candidate: Candidate) -> Candidate:
        if not candidate.changed_files:
            candidate.score = self.cfg.compile_fail_score
            candidate.compile_ok = False
            candidate.metrics = {}
            candidate.eval_stage = "no-change"
            candidate.stderr = "No file changed."
            return candidate
        result = self.evaluator.evaluate(candidate)
        return self.apply_eval_result(candidate, result)

    def compile_candidate_for_device_once(self, candidate: Candidate) -> Tuple[Candidate, EvalResult]:
        if not candidate.changed_files:
            result = EvalResult(
                score=self.cfg.compile_fail_score,
                compile_ok=False,
                return_code=1,
                stdout="",
                stderr="No file changed.",
                work_dir=candidate.work_dir or "",
                metrics={},
                stage="no-change",
            )
            self.apply_eval_result(candidate, result, cleanup=False)
            return candidate, result
        result = self.evaluator.compile_and_deploy(candidate)
        # Keep successful compile/deploy candidate dirs until the device stage finishes.
        self.apply_eval_result(candidate, result, cleanup=not result.compile_ok)
        if result.compile_ok:
            candidate.score = None
        return candidate, result

    def device_evaluate_candidate_once(
        self,
        candidate: Candidate,
        compile_result: EvalResult,
        handoff_ready_event: Optional[threading.Event] = None,
    ) -> Candidate:
        result = self.evaluator.evaluate_on_device(candidate, compile_result, handoff_ready_event)
        return self.apply_eval_result(candidate, result)

    def repair_candidate_once(self, candidate: Candidate, attempt: int) -> Tuple[Candidate, bool]:
        if not candidate.changed_files:
            return candidate, False
        self.log(f"Repair LLM attempt {attempt}/{self.cfg.max_compile_repairs} for {candidate.candidate_id}")
        repaired_any = False
        repair_output = self.candidate_output_text(candidate)
        if self.cfg.repair_all_files_per_call:
            system_prompt, user_prompt = self.prompts.repair_bundle_prompt(candidate, repair_output)
            repairs, note = self._repair_file_bundle(system_prompt, user_prompt, candidate.changed_files)
            candidate.llm_notes.append(note)
            for path, content in repairs.items():
                if content != candidate.files[path]:
                    candidate.files[path] = content
                    repaired_any = True
        else:
            with cf.ThreadPoolExecutor(max_workers=min(self.cfg.max_llm_workers, len(candidate.changed_files))) as pool:
                futures = []
                for path in candidate.changed_files:
                    system_prompt, user_prompt = self.prompts.repair_prompt(candidate, path, repair_output)
                    futures.append(pool.submit(self._repair_one_file, path, system_prompt, user_prompt))
                for fut in cf.as_completed(futures):
                    path, content, note = fut.result()
                    candidate.llm_notes.append(note)
                    if content and content != candidate.files[path]:
                        candidate.files[path] = content
                        repaired_any = True
        if repaired_any:
            try:
                self.prepare_candidate_dir(candidate)
            except Exception as exc:
                candidate.llm_notes.append(f"repair pre-materialize failed: {exc}")
        return candidate, repaired_any

    def repair_candidate(self, candidate: Candidate) -> Candidate:
        if not candidate.changed_files:
            return candidate
        for attempt in range(self.cfg.max_compile_repairs):
            if candidate.compile_ok:
                return candidate
            candidate, repaired_any = self.repair_candidate_once(candidate, attempt + 1)
            if not repaired_any:
                self.log(
                    f"Repair attempt {attempt + 1}/{self.cfg.max_compile_repairs} produced no usable changes "
                    f"for {candidate.candidate_id}; trying remaining repair budget"
                )
                continue
            result = self.evaluator.evaluate(candidate)
            self.apply_eval_result(candidate, result)
        return candidate

    def _repair_file_bundle(self, system_prompt: str, user_prompt: str, allowed_paths: Sequence[str]) -> Tuple[Dict[str, str], str]:
        allowed = set(allowed_paths)
        repairs: Dict[str, str] = {}
        notes: List[str] = []
        try:
            obj = self.llm.complete_json(
                system_prompt,
                user_prompt,
                temperature=self.llm.cfg.temperature_repair,
                max_output_tokens=self.mutation_repair_output_tokens(),
            )
            if not isinstance(obj, dict):
                raise RuntimeError("bundle repair response is not an object")
            summary = str(obj.get("repair_summary", "")).strip()
            if summary:
                notes.append(summary)
            self_check = str(obj.get("compile_self_check", "")).strip()
            if self_check:
                notes.append(f"compile self-check: {self_check}")
            changes = obj.get("changes", [])
            if not isinstance(changes, list):
                raise RuntimeError("bundle repair response has no changes list")
            for row in changes:
                if not isinstance(row, dict):
                    continue
                path = str(row.get("path", ""))
                if path not in allowed:
                    notes.append(f"{path}: bundle repair tried to edit a non-changed file; ignored")
                    continue
                content = row.get("content")
                if not isinstance(content, str) or not content.strip():
                    notes.append(f"{path}: bundle repair returned empty content; ignored")
                    continue
                content = clean_llm_file_content(content)
                incomplete_marker = content_looks_incomplete(content)
                if incomplete_marker:
                    notes.append(f"{path}: bundle repair returned incomplete marker {incomplete_marker}; ignored")
                    continue
                repairs[path] = content
            return repairs, "bundle repair: " + ("; ".join(notes) if notes else f"{len(repairs)} file(s) returned")
        except Exception as exc:
            return {}, f"bundle repair failed: {exc}"

    def _repair_one_file(self, path: str, system_prompt: str, user_prompt: str) -> Tuple[str, Optional[str], str]:
        try:
            obj = self.llm.complete_json(
                system_prompt,
                user_prompt,
                temperature=self.llm.cfg.temperature_repair,
                max_output_tokens=self.mutation_repair_output_tokens(),
            )
            if not isinstance(obj, dict):
                raise RuntimeError("repair response is not an object")
            out_path = str(obj.get("path", path))
            if out_path != path:
                raise RuntimeError(f"repair worker tried to edit {out_path}, expected {path}")
            content = obj.get("content")
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("repair worker did not return content")
            content = clean_llm_file_content(content)
            incomplete_marker = content_looks_incomplete(content)
            if incomplete_marker:
                raise RuntimeError(f"repair worker returned incomplete file content marker: {incomplete_marker}")
            summary = str(obj.get("repair_summary", "")).strip()
            self_check = str(obj.get("compile_self_check", "")).strip()
            note_parts = [part for part in [summary, f"compile self-check: {self_check}" if self_check else ""] if part]
            return path, content, f"{path}: {'; '.join(note_parts)}"
        except Exception as exc:
            return path, None, f"{path}: repair failed: {exc}"

    def evaluate_candidate(self, candidate: Candidate) -> Candidate:
        candidate = self.evaluate_candidate_once(candidate)
        if self.is_repairable_failure(candidate) and self.cfg.max_compile_repairs > 0:
            candidate = self.repair_candidate(candidate)
        return candidate

    def evaluate_generation_candidates(self, generation: int, built: List[Candidate]) -> List[Candidate]:
        if self.evaluator.uses_staged_device_pipeline():
            repair_workers = max(1, min(self.cfg.max_llm_workers, max(1, len(built))))
            return self.evaluate_staged_pipeline(built=built, repair_workers=repair_workers)

        if max(1, self.cfg.max_eval_workers) != 1:
            evaluated: List[Candidate] = []
            with cf.ThreadPoolExecutor(max_workers=max(1, self.cfg.max_eval_workers)) as pool:
                futures = [pool.submit(self.evaluate_candidate, candidate) for candidate in built]
                for fut in cf.as_completed(futures):
                    candidate = fut.result()
                    evaluated.append(candidate)
                    self.log(
                        f"Evaluated {candidate.candidate_id} score={candidate.score} compile_ok={candidate.compile_ok}"
                    )
            return evaluated

        queue: deque[Tuple[Candidate, int]] = deque((candidate, 0) for candidate in built)
        repair_futures: Dict[cf.Future[Tuple[Candidate, bool]], int] = {}
        evaluated: List[Candidate] = []
        repair_workers = max(1, min(self.cfg.max_llm_workers, max(1, len(built))))

        with cf.ThreadPoolExecutor(max_workers=repair_workers) as repair_pool:
            while queue or repair_futures:
                while queue:
                    candidate, repair_attempts = queue.popleft()
                    candidate = self.evaluate_candidate_once(candidate)
                    self.log(
                        f"Evaluated {candidate.candidate_id} score={candidate.score} compile_ok={candidate.compile_ok}"
                    )
                    if (
                        self.is_repairable_failure(candidate)
                        and candidate.changed_files
                        and repair_attempts < self.cfg.max_compile_repairs
                    ):
                        next_attempt = repair_attempts + 1
                        future = repair_pool.submit(self.repair_candidate_once, candidate, next_attempt)
                        repair_futures[future] = next_attempt
                        self.log(
                            f"Queued repair {next_attempt}/{self.cfg.max_compile_repairs} for {candidate.candidate_id} while evaluator continues"
                        )
                    else:
                        evaluated.append(candidate)

                if repair_futures:
                    done, _ = cf.wait(tuple(repair_futures.keys()), return_when=cf.FIRST_COMPLETED)
                    for future in done:
                        attempt = repair_futures.pop(future)
                        try:
                            repaired_candidate, repaired_any = future.result()
                        except Exception:
                            self.log("Candidate repair crashed:\n" + traceback.format_exc())
                            continue
                        if repaired_any:
                            queue.append((repaired_candidate, attempt))
                            self.log(f"Repair produced new contents for {repaired_candidate.candidate_id}; queued re-evaluation")
                        elif attempt < self.cfg.max_compile_repairs:
                            next_attempt = attempt + 1
                            retry_future = repair_pool.submit(self.repair_candidate_once, repaired_candidate, next_attempt)
                            repair_futures[retry_future] = next_attempt
                            self.log(
                                f"Repair attempt {attempt}/{self.cfg.max_compile_repairs} produced no usable changes for "
                                f"{repaired_candidate.candidate_id}; queued retry {next_attempt}/{self.cfg.max_compile_repairs}"
                            )
                        else:
                            evaluated.append(repaired_candidate)
                            self.log(
                                f"Repair produced no usable changes for {repaired_candidate.candidate_id}; "
                                f"exhausted {self.cfg.max_compile_repairs} attempt(s)"
                            )

        return evaluated

    def generation_summary(self, generation: int, evaluated: List[Candidate]) -> str:
        ranked = sorted(evaluated, key=lambda c: float(c.score if c.score is not None else -1e300), reverse=True)
        lines = [f"Generation {generation}: best child score={ranked[0].score if ranked else None}"]
        summary_metric_keys = normalize_metric_keys(self.cfg.primary_metric_keys)
        for cand in ranked[: min(5, len(ranked))]:
            metrics = {key: cand.metrics[key] for key in summary_metric_keys if key in cand.metrics}
            metric_text = f" metrics={metrics}" if metrics else ""
            lines.append(
                f"  {cand.candidate_id} score={cand.score} compile_ok={cand.compile_ok}{metric_text} files={cand.changed_files} style={cand.style}"
            )
        return "\n".join(lines)

    def run(self) -> None:
        evaluated_generation_zero: Optional[List[Candidate]] = None
        loaded_baseline = self._loaded_baseline()
        if loaded_baseline is not None:
            baseline = loaded_baseline
        else:
            baseline = self._make_baseline_candidate()
            if self.cfg.pipeline_first_generation_during_baseline and self.cfg.generations > 0:
                self.log("Starting baseline evaluation and streaming generation 0 build/evaluation from the baseline snapshot.")
                with cf.ThreadPoolExecutor(max_workers=1) as baseline_eval_pool:
                    baseline_future = baseline_eval_pool.submit(self._evaluate_baseline_candidate, baseline)
                    try:
                        self.log(f"Generation 0: selected parent {baseline.candidate_id} score={baseline.score}")
                        plans = self.prompts.plan_children(baseline, 0, self.rng)
                        self.log(f"Generation 0: created {len(plans)} child plans")
                        if plans:
                            build_workers = min(self.cfg.max_llm_workers, max(1, len(plans)))
                            repair_workers = max(1, min(self.cfg.max_llm_workers, len(plans)))
                            with cf.ThreadPoolExecutor(max_workers=build_workers) as build_pool:
                                build_futures: Dict[cf.Future[Candidate], None] = {
                                    build_pool.submit(self.build_and_prepare_candidate, 0, baseline, plan): None for plan in plans
                                }
                                baseline = baseline_future.result()
                                evaluated_generation_zero = self.evaluate_build_futures_streaming(
                                    build_futures,
                                    repair_workers=repair_workers,
                                )
                        else:
                            baseline = baseline_future.result()
                            evaluated_generation_zero = []
                    except Exception:
                        self.log("Generation 0 streaming build/evaluation crashed:\n" + traceback.format_exc())
                        evaluated_generation_zero = None
                        baseline = baseline_future.result()
            else:
                baseline = self._evaluate_baseline_candidate(baseline)

        if self.archive.best is None:
            self.archive.best = baseline
        if not self.archive.elites:
            self.archive.elites.append(baseline)
        self.log(
            f"Project has {len(self.index.mutable_files)} mutable files "
            f"({len(self.index.mutable_source_files)} sources, {len(self.index.mutable_header_files)} headers), "
            f"{len(self.index.immutable_files)} immutable context files, and "
            f"{sum(len(v) for v in self.index.paired_headers_by_source.values())} source/header pairings."
        )
        for generation in range(self.cfg.generations):
            if generation == 0 and evaluated_generation_zero is not None:
                evaluated = evaluated_generation_zero
                self.log(f"Generation 0: using {len(evaluated)} candidates streamed during baseline evaluation")
            else:
                parent = self.archive.select_parent(self.rng) or baseline
                if self.cfg.stream_evaluate_candidates and max(1, self.cfg.max_eval_workers) == 1:
                    evaluated = self.build_and_evaluate_generation_streaming(generation, parent)
                else:
                    built = self.build_generation(generation, parent)
                    evaluated = self.evaluate_generation_candidates(generation, built)
            for candidate in evaluated:
                self.archive.register(candidate)
            self.log(self.generation_summary(generation, evaluated))
            if self.archive.best:
                best_metric_keys = normalize_metric_keys(self.cfg.primary_metric_keys)
                best_metrics = {
                    key: self.archive.best.metrics[key]
                    for key in best_metric_keys
                    if key in self.archive.best.metrics
                }
                metric_text = f" metrics={best_metrics}" if best_metrics else ""
                self.log(
                    f"Global best after generation {generation}: {self.archive.best.candidate_id} score={self.archive.best.score}{metric_text} files={self.archive.best.changed_files}"
                )
            self.analyze_generation_outcomes(generation, evaluated)
        self.log("Search finished.")
        if self.archive.best:
            self.log(f"BEST: {self.archive.best.candidate_id} score={self.archive.best.score}")


# -----------------------------
# Config loading / CLI
# -----------------------------


def load_config(path: Path) -> Tuple[EvolutionConfig, LLMConfig]:
    raw = json.loads(read_text(path))
    llm_cfg = LLMConfig(**raw["llm"])
    evo_raw = {k: v for k, v in raw.items() if k != "llm"}
    evo_cfg = EvolutionConfig(**evo_raw)
    return evo_cfg, llm_cfg


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Project-aware agentic LLM evolver for C++ codebases.")
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    args = parser.parse_args(argv)
    cfg, llm_cfg = load_config(Path(args.config))
    evolver = Evolver(cfg, llm_cfg)
    evolver.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
