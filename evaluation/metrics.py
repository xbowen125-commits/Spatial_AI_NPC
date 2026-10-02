"""汇总 interactions.jsonl 中的稳定性指标。"""

import argparse
import json
from pathlib import Path

try:
    from .interaction_logger import DEFAULT_LOG_PATH
except ImportError:
    # 允许直接运行：python evaluation/metrics.py
    from interaction_logger import DEFAULT_LOG_PATH


LATENCY_FIELDS = (
    "stt_latency",
    "agent_latency",
    "tts_latency",
    "total_latency",
)


def calculate_metrics(path=DEFAULT_LOG_PATH):
    totals = {field: 0 for field in LATENCY_FIELDS}
    latency_counts = {field: 0 for field in LATENCY_FIELDS}
    total = 0
    llm_successes = 0
    fallbacks = 0
    action_errors = 0
    fact_violations = 0
    invalid_lines = 0

    path = Path(path)
    if not path.exists():
        return _empty_metrics(str(path))

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                invalid_lines += 1
                continue

            total += 1
            source = record.get("agent_source")
            if source == "llm" and not record.get("fallback_reason"):
                llm_successes += 1
            if source == "rule" or record.get("fallback_reason"):
                fallbacks += 1
            action_errors += int(record.get("action_error_count", 0) or 0)
            fact_violations += int(record.get("fact_violation_count", 0) or 0)

            latency = record.get("latency", {})
            for field in LATENCY_FIELDS:
                value = latency.get(field)
                if isinstance(value, (int, float)) and value >= 0:
                    totals[field] += value
                    latency_counts[field] += 1

    averages = {
        field: round(totals[field] / latency_counts[field], 2)
        if latency_counts[field]
        else 0.0
        for field in LATENCY_FIELDS
    }
    return {
        "log_path": str(path),
        "total_interactions": total,
        "llm_success_rate": round(llm_successes / total, 4) if total else 0.0,
        "fallback_rate": round(fallbacks / total, 4) if total else 0.0,
        "average_latency_ms": averages,
        "action_error_count": action_errors,
        "fact_violation_count": fact_violations,
        "invalid_log_lines": invalid_lines,
    }


def _empty_metrics(path):
    return {
        "log_path": path,
        "total_interactions": 0,
        "llm_success_rate": 0.0,
        "fallback_rate": 0.0,
        "average_latency_ms": {field: 0.0 for field in LATENCY_FIELDS},
        "action_error_count": 0,
        "fact_violation_count": 0,
        "invalid_log_lines": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统计 NPC Interaction JSONL")
    parser.add_argument("path", nargs="?", default=str(DEFAULT_LOG_PATH))
    args = parser.parse_args()
    print(json.dumps(calculate_metrics(args.path), ensure_ascii=False, indent=2))
