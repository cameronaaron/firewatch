#!/usr/bin/env python3
"""Production readiness audit runner for Firewatch.

This script runs the core functional gates required before deployment:
- Python regression suite
- Edge worker test suite
- Frontend and worker JavaScript syntax checks
- Optional live edge endpoint probes
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SCORECARD_GRADE_ORDER = {"C": 0, "B": 1, "A": 2}


@dataclass
class GateResult:
    name: str
    command: str
    passed: bool
    details: str


def run_command_gate(name: str, command: list[str]) -> GateResult:
    process = subprocess.run(
        command,
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
    )
    output = "\n".join(part for part in [process.stdout, process.stderr] if part).strip()
    if not output:
        output = "No output"

    if process.returncode == 0:
        details = summarize_output(output)
        return GateResult(name=name, command=" ".join(command), passed=True, details=details)

    details = summarize_output(output)
    return GateResult(name=name, command=" ".join(command), passed=False, details=details)


def summarize_output(output: str, max_lines: int = 12) -> str:
    lines = [line for line in output.splitlines() if line.strip()]
    if len(lines) <= max_lines:
        return "\n".join(lines)
    head = lines[: max_lines - 2]
    tail = lines[-1:]
    omitted = len(lines) - len(head) - len(tail)
    return "\n".join([*head, f"... ({omitted} lines omitted) ...", *tail])


def probe_edge_endpoint(base_url: str, path: str, timeout_seconds: int) -> GateResult:
    normalized_base = base_url.rstrip("/")
    target = f"{normalized_base}{path}"
    command = ["curl", "-fsS", "--max-time", str(timeout_seconds), target]
    process = subprocess.run(command, cwd=ROOT_DIR, capture_output=True, text=True)
    if process.returncode == 0:
        return GateResult(
            name=f"edge probe {path}",
            command=" ".join(command),
            passed=True,
            details="HTTP 2xx",
        )

    stderr = (process.stderr or process.stdout or "curl probe failed").strip()
    return GateResult(
        name=f"edge probe {path}",
        command=" ".join(command),
        passed=False,
        details=summarize_output(stderr, max_lines=4),
    )


def parse_json_payload(text: str) -> dict | None:
    raw = text.strip()
    if not raw:
        return None

    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(raw):
        if char != "{":
            continue
        try:
            candidate, end_index = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(candidate, dict):
            continue
        trailing = raw[index + end_index :].strip()
        if not trailing:
            return candidate
    return None


def scorecard_grade_meets_minimum(actual_grade: str, minimum_grade: str) -> bool:
    actual = str(actual_grade or "").upper()
    minimum = str(minimum_grade or "").upper()
    if actual not in SCORECARD_GRADE_ORDER or minimum not in SCORECARD_GRADE_ORDER:
        return False
    return SCORECARD_GRADE_ORDER[actual] >= SCORECARD_GRADE_ORDER[minimum]


def run_model_readiness_gate(
    *,
    require_uplift: bool,
    min_uplift_pct: float,
    require_min_r2: bool,
    min_best_r2: float,
    require_production_ready: bool,
    min_scorecard_grade: str | None = None,
    require_informative: bool = True,
) -> GateResult:
    command = [sys.executable, "-m", "app.ml.benchmark_cli", "--output", "json", "report"]
    process = subprocess.run(command, cwd=ROOT_DIR, capture_output=True, text=True)

    combined_output = (
        "\n".join(part for part in [process.stdout, process.stderr] if part).strip() or "No output"
    )

    if process.returncode != 0:
        return GateResult(
            name="model readiness gate",
            command=" ".join(command),
            passed=False,
            details=summarize_output(combined_output),
        )

    payload = parse_json_payload(process.stdout) or parse_json_payload(combined_output)
    if not payload:
        return GateResult(
            name="model readiness gate",
            command=" ".join(command),
            passed=False,
            details="Could not parse benchmark report JSON from CLI output",
        )

    report = payload.get("report") if isinstance(payload, dict) else None
    if not isinstance(report, dict):
        return GateResult(
            name="model readiness gate",
            command=" ".join(command),
            passed=False,
            details="Benchmark report payload is missing the 'report' object",
        )

    informative = bool(report.get("dataset_is_informative"))
    can_benchmark = bool(report.get("can_benchmark", True))
    improvement = float(report.get("best_improvement_pct") or 0.0)
    sample_count = int(report.get("sample_count") or 0)
    group_count = int(report.get("group_count") or 0)
    best_model = str(report.get("best_model") or "unknown")
    baseline_model = str(report.get("baseline_model") or "unknown")
    best_r2 = float((report.get("best_model_metrics") or {}).get("r2") or 0.0)
    scorecard = report.get("scorecard") if isinstance(report.get("scorecard"), dict) else None
    scorecard_grade = str((scorecard or {}).get("grade") or "unknown")
    production_ready = bool((scorecard or {}).get("production_ready"))
    state_of_the_art = bool((scorecard or {}).get("state_of_the_art"))
    assessment = str((scorecard or {}).get("assessment") or "")
    time_split_test_r2 = float(
        (((report.get("time_split") or {}).get("best_model_metrics") or {}).get("test_r2")) or 0.0
    )

    details = (
        f"can_benchmark={can_benchmark}, informative={informative}, samples={sample_count}, "
        f"groups={group_count}, "
        f"best_model={best_model}, baseline_model={baseline_model}, "
        f"improvement_pct={improvement:.2f}, min_uplift_pct={min_uplift_pct:.2f}, "
        f"best_r2={best_r2:.4f}, min_best_r2={min_best_r2:.4f}, "
        f"time_split_test_r2={time_split_test_r2:.4f}, scorecard_grade={scorecard_grade}, "
        f"production_ready={production_ready}, state_of_the_art={state_of_the_art}"
    )
    if min_scorecard_grade:
        details = f"{details}, min_scorecard_grade={str(min_scorecard_grade).upper()}"
    if assessment:
        details = f"{details}, assessment={assessment}"

    failures: list[str] = []

    if require_informative and not informative:
        failures.append("Dataset not informative for model readiness decision")

    if require_uplift and improvement < min_uplift_pct:
        failures.append("Model uplift below threshold")

    if require_min_r2 and best_r2 < min_best_r2:
        failures.append("Best model R2 below threshold")

    if require_production_ready:
        if not scorecard:
            failures.append("Benchmark report is missing the production-readiness scorecard")
        elif not production_ready:
            failures.append("Model failed the production-ready scorecard")

    if min_scorecard_grade:
        minimum_grade = str(min_scorecard_grade).upper()
        if not scorecard:
            failures.append(
                "Benchmark report is missing the scorecard grade required for promotion"
            )
        elif not scorecard_grade_meets_minimum(scorecard_grade, minimum_grade):
            failures.append(
                f"Model scorecard grade {scorecard_grade} is below required grade {minimum_grade}"
            )

    if failures:
        return GateResult(
            name="model readiness gate",
            command=" ".join(command),
            passed=False,
            details=f"{'; '.join(failures)}. {details}",
        )

    return GateResult(
        name="model readiness gate",
        command=" ".join(command),
        passed=True,
        details=f"Model readiness thresholds satisfied. {details}",
    )


def render_report(results: Iterable[GateResult]) -> int:
    result_list = list(results)
    failed = [result for result in result_list if not result.passed]

    report = {
        "summary": {
            "total": len(result_list),
            "passed": len(result_list) - len(failed),
            "failed": len(failed),
            "status": "pass" if not failed else "fail",
        },
        "checks": [
            {
                "name": result.name,
                "command": result.command,
                "status": "pass" if result.passed else "fail",
                "details": result.details,
            }
            for result in result_list
        ],
    }

    print(json.dumps(report, indent=2))

    if failed:
        print("\nFAILED CHECKS:", file=sys.stderr)
        for result in failed:
            print(f"- {result.name}", file=sys.stderr)
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Firewatch production readiness checks.")
    parser.add_argument(
        "--edge-base-url",
        default="",
        help="Optional deployed edge base URL to probe (for example https://<project>.pages.dev)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=12,
        help="HTTP timeout in seconds for edge probes.",
    )
    parser.add_argument(
        "--require-model-uplift",
        action="store_true",
        help="Fail audit when benchmarked model uplift is below threshold.",
    )
    parser.add_argument(
        "--min-uplift-pct",
        type=float,
        default=1.0,
        help="Minimum required best_improvement_pct from benchmark report.",
    )
    parser.add_argument(
        "--require-min-r2",
        action="store_true",
        help="Fail audit when the benchmark best-model R2 is below threshold.",
    )
    parser.add_argument(
        "--require-production-ready",
        action="store_true",
        help="Fail audit when the benchmark scorecard is not production-ready.",
    )
    parser.add_argument(
        "--min-scorecard-grade",
        choices=["A", "B", "C"],
        default=None,
        help="Fail audit when the benchmark scorecard grade falls below this floor.",
    )
    parser.add_argument(
        "--min-best-r2",
        type=float,
        default=0.05,
        help="Minimum required best-model R2 from benchmark report.",
    )
    return parser.parse_args()


def command_gates() -> list[tuple[str, list[str]]]:
    """The test and syntax gates, run with this interpreter and every JS test file."""
    js_tests = sorted(str(path.relative_to(ROOT_DIR)) for path in ROOT_DIR.glob("tests/*.mjs"))
    return [
        ("python regression tests", [sys.executable, "-m", "pytest", "-q"]),
        ("javascript tests", ["node", "--test", *js_tests]),
        ("frontend syntax", ["node", "--check", "app/static/app.js"]),
        ("worker syntax", ["node", "--check", "app/static/_worker.js"]),
    ]


def main() -> int:
    args = parse_args()

    checks: list[GateResult] = [run_command_gate(name, cmd) for name, cmd in command_gates()]

    if (
        args.require_model_uplift
        or args.require_min_r2
        or args.require_production_ready
        or args.min_scorecard_grade
    ):
        checks.append(
            run_model_readiness_gate(
                require_uplift=args.require_model_uplift,
                min_uplift_pct=args.min_uplift_pct,
                require_min_r2=args.require_min_r2,
                min_best_r2=args.min_best_r2,
                require_production_ready=args.require_production_ready,
                min_scorecard_grade=args.min_scorecard_grade,
            )
        )

    if args.edge_base_url:
        checks.extend(
            [
                probe_edge_endpoint(args.edge_base_url, "/edge/v1/health", args.timeout_seconds),
                probe_edge_endpoint(
                    args.edge_base_url, "/edge/v1/health/diagnostics", args.timeout_seconds
                ),
                probe_edge_endpoint(args.edge_base_url, "/docs", args.timeout_seconds),
                probe_edge_endpoint(args.edge_base_url, "/openapi.json", args.timeout_seconds),
            ]
        )

    return render_report(checks)


if __name__ == "__main__":
    raise SystemExit(main())
