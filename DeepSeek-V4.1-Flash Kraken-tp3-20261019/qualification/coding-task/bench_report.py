#!/usr/bin/env python3
"""Analyze JSON Lines benchmark results and summarize throughput by model/concurrency.

Input rows are expected to look like::

    {"model": "model-a", "concurrency": 1, "prompt_tokens": 32000,
     "output_tokens": 4096, "elapsed_seconds": 18.2,
     "accepted_tokens": 3100, "drafted_tokens": 3600}

Only the Python standard library is used.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, TextIO, Tuple


REQUIRED_FIELDS: Tuple[str, ...] = (
    "model",
    "concurrency",
    "prompt_tokens",
    "output_tokens",
    "elapsed_seconds",
    "accepted_tokens",
    "drafted_tokens",
)


class ValidationError(ValueError):
    """Raised when a benchmark row cannot be parsed or validated."""


@dataclass(frozen=True)
class BenchmarkRun:
    """A single validated benchmark run, with derived metrics."""

    model: str
    concurrency: int
    prompt_tokens: float
    output_tokens: float
    elapsed_seconds: float
    accepted_tokens: float
    drafted_tokens: float

    @property
    def generation_throughput(self) -> float:
        """Output tokens generated per second."""
        return self.output_tokens / self.elapsed_seconds

    @property
    def total_throughput(self) -> float:
        """Prompt plus output tokens processed per second."""
        return (self.prompt_tokens + self.output_tokens) / self.elapsed_seconds

    @property
    def acceptance_rate(self) -> float:
        """Draft acceptance rate as a percentage (0.0 when nothing was drafted)."""
        if self.drafted_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.drafted_tokens * 100.0

    @property
    def draft_acceptance_rate(self) -> float:
        """Alias for :attr:`acceptance_rate`."""
        return self.acceptance_rate

    @property
    def rejected_tokens(self) -> float:
        """Number of drafted tokens that were not accepted."""
        return self.drafted_tokens - self.accepted_tokens


@dataclass(frozen=True)
class GroupStats:
    """Aggregated statistics for one model/concurrency group."""

    model: str
    concurrency: int
    runs: int
    mean_generation_tps: float
    median_generation_tps: float
    min_generation_tps: float
    max_generation_tps: float
    stdev_generation_tps: float
    mean_acceptance_rate: float
    total_accepted_tokens: float
    total_rejected_tokens: float


def _require_number(data: Mapping[str, Any], field: str) -> float:
    """Return a finite numeric field as a float, or raise ValidationError."""
    if field not in data:
        raise ValidationError("missing required field '{}'".format(field))

    value = data[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError("field '{}' must be a number".format(field))

    number = float(value)
    if not math.isfinite(number):
        raise ValidationError("field '{}' must be a finite number".format(field))
    return number


def run_from_mapping(data: Mapping[str, Any]) -> BenchmarkRun:
    """Validate a decoded JSON object and convert it into a BenchmarkRun."""
    if not isinstance(data, Mapping):
        raise ValidationError("row must be a JSON object")

    missing = [field for field in REQUIRED_FIELDS if field not in data]
    if missing:
        raise ValidationError(
            "missing required field(s): {}".format(", ".join(missing))
        )

    model = data["model"]
    if not isinstance(model, str) or not model.strip():
        raise ValidationError("field 'model' must be a non-empty string")

    concurrency_number = _require_number(data, "concurrency")
    if not concurrency_number.is_integer():
        raise ValidationError("field 'concurrency' must be an integer")
    concurrency = int(concurrency_number)
    if concurrency < 1:
        raise ValidationError("field 'concurrency' must be at least 1")

    prompt_tokens = _require_number(data, "prompt_tokens")
    output_tokens = _require_number(data, "output_tokens")
    elapsed_seconds = _require_number(data, "elapsed_seconds")
    accepted_tokens = _require_number(data, "accepted_tokens")
    drafted_tokens = _require_number(data, "drafted_tokens")

    for field, value in (
        ("prompt_tokens", prompt_tokens),
        ("output_tokens", output_tokens),
        ("accepted_tokens", accepted_tokens),
        ("drafted_tokens", drafted_tokens),
    ):
        if value < 0:
            raise ValidationError("field '{}' must not be negative".format(field))

    if elapsed_seconds <= 0:
        raise ValidationError("field 'elapsed_seconds' must be greater than 0")

    if accepted_tokens > drafted_tokens:
        raise ValidationError(
            "field 'accepted_tokens' must not exceed 'drafted_tokens'"
        )

    return BenchmarkRun(
        model=model,
        concurrency=concurrency,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        elapsed_seconds=elapsed_seconds,
        accepted_tokens=accepted_tokens,
        drafted_tokens=drafted_tokens,
    )


def parse_json_line(line: str) -> BenchmarkRun:
    """Parse one JSONL line into a validated BenchmarkRun."""
    try:
        data = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValidationError("invalid JSON: {}".format(exc.msg)) from exc

    if not isinstance(data, dict):
        raise ValidationError("row must be a JSON object")

    return run_from_mapping(data)


def parse_jsonl(
    lines: Iterable[str],
    source: str = "<input>",
    error_stream: Optional[TextIO] = None,
) -> List[BenchmarkRun]:
    """Parse an iterable of JSONL lines, reporting invalid rows to a stream.

    Blank lines are ignored. Invalid rows do not stop processing of later rows.
    """
    runs: List[BenchmarkRun] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            runs.append(parse_json_line(line))
        except ValidationError as exc:
            if error_stream is not None:
                print(
                    "{}:{}: {}".format(source, line_number, exc),
                    file=error_stream,
                )
    return runs


def load_runs(path: str, error_stream: Optional[TextIO] = None) -> List[BenchmarkRun]:
    """Read and validate a JSONL file. OSError is propagated to the caller."""
    with open(path, "r", encoding="utf-8") as handle:
        return parse_jsonl(handle, source=path, error_stream=error_stream)


def filter_runs(
    runs: Iterable[BenchmarkRun],
    *,
    model: Optional[str] = None,
    min_acceptance: Optional[float] = None,
) -> List[BenchmarkRun]:
    """Filter individual runs by model and/or minimum acceptance percentage."""
    selected: List[BenchmarkRun] = []
    for run in runs:
        if model is not None and run.model != model:
            continue
        if min_acceptance is not None and run.acceptance_rate < min_acceptance:
            continue
        selected.append(run)
    return selected


def summarize_group(
    model: str, concurrency: int, runs: Sequence[BenchmarkRun]
) -> GroupStats:
    """Calculate aggregate statistics for a non-empty group of runs."""
    if not runs:
        raise ValueError("cannot summarize an empty group")

    throughputs = [run.generation_throughput for run in runs]
    acceptance_rates = [run.acceptance_rate for run in runs]

    return GroupStats(
        model=model,
        concurrency=concurrency,
        runs=len(runs),
        mean_generation_tps=statistics.mean(throughputs),
        median_generation_tps=float(statistics.median(throughputs)),
        min_generation_tps=min(throughputs),
        max_generation_tps=max(throughputs),
        stdev_generation_tps=statistics.pstdev(throughputs),
        mean_acceptance_rate=statistics.mean(acceptance_rates),
        total_accepted_tokens=sum(run.accepted_tokens for run in runs),
        total_rejected_tokens=sum(run.rejected_tokens for run in runs),
    )


def group_runs(runs: Iterable[BenchmarkRun]) -> List[GroupStats]:
    """Group runs by (model, concurrency), sorted by mean throughput descending."""
    buckets: Dict[Tuple[str, int], List[BenchmarkRun]] = {}
    for run in runs:
        key = (run.model, run.concurrency)
        buckets.setdefault(key, []).append(run)

    groups = [
        summarize_group(model, concurrency, group_runs_)
        for (model, concurrency), group_runs_ in buckets.items()
    ]
    groups.sort(
        key=lambda group: (
            -group.mean_generation_tps,
            group.model,
            group.concurrency,
        )
    )
    return groups


def top_groups(groups: Sequence[GroupStats], count: Optional[int]) -> List[GroupStats]:
    """Return at most ``count`` groups (all groups when count is None)."""
    if count is None:
        return list(groups)
    return list(groups[:count])


def _table_rows(groups: Sequence[GroupStats]) -> Tuple[List[str], List[List[str]]]:
    headers = [
        "MODEL",
        "CONC",
        "RUNS",
        "AVG_TPS",
        "MEDIAN_TPS",
        "MIN_TPS",
        "MAX_TPS",
        "STDDEV",
        "ACCEPT%",
        "ACCEPTED",
        "REJECTED",
    ]
    rows: List[List[str]] = []
    for group in groups:
        rows.append(
            [
                group.model,
                str(group.concurrency),
                str(group.runs),
                "{:.1f}".format(group.mean_generation_tps),
                "{:.1f}".format(group.median_generation_tps),
                "{:.1f}".format(group.min_generation_tps),
                "{:.1f}".format(group.max_generation_tps),
                "{:.2f}".format(group.stdev_generation_tps),
                "{:.1f}".format(group.mean_acceptance_rate),
                "{:.0f}".format(group.total_accepted_tokens),
                "{:.0f}".format(group.total_rejected_tokens),
            ]
        )
    return headers, rows


def format_table(groups: Sequence[GroupStats]) -> str:
    """Render groups as a fixed-width text table."""
    headers, rows = _table_rows(groups)
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def render(row: Sequence[str]) -> str:
        cells = []
        for index, cell in enumerate(row):
            if index == 0:
                cells.append(cell.ljust(widths[index]))
            else:
                cells.append(cell.rjust(widths[index]))
        return "  ".join(cells).rstrip()

    lines = [render(headers)]
    lines.extend(render(row) for row in rows)
    return "\n".join(lines)


def format_json(groups: Sequence[GroupStats]) -> str:
    """Render groups as a JSON array with native numeric values."""
    return json.dumps([asdict(group) for group in groups], indent=2)


def format_csv(groups: Sequence[GroupStats]) -> str:
    """Render groups as CSV, including a header row."""
    field_names = [field.name for field in fields(GroupStats)]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=field_names, lineterminator="\n")
    writer.writeheader()
    for group in groups:
        writer.writerow(asdict(group))
    return output.getvalue().rstrip("\n")


def format_output(groups: Sequence[GroupStats], output_format: str) -> str:
    """Dispatch to the requested output formatter."""
    if output_format == "table":
        return format_table(groups)
    if output_format == "json":
        return format_json(groups)
    if output_format == "csv":
        return format_csv(groups)
    raise ValueError("unsupported output format: {}".format(output_format))


def positive_int(value: str) -> int:
    """argparse type for a positive integer."""
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench_report.py",
        description=(
            "Analyze JSON Lines benchmark results and summarize generation "
            "throughput by model and concurrency."
        ),
        epilog=(
            "Examples:\n"
            "  python3 bench_report.py results.jsonl\n"
            "  python3 bench_report.py results.jsonl --model model-a --top 5\n"
            "  python3 bench_report.py results.jsonl --min-acceptance 80 "
            "--format json\n"
            "\n"
            "Each input line must be a JSON object with these fields:\n"
            "  model, concurrency, prompt_tokens, output_tokens, "
            "elapsed_seconds, accepted_tokens, drafted_tokens\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "path",
        help="path to the input JSON Lines file",
    )
    parser.add_argument(
        "--top",
        type=positive_int,
        metavar="N",
        default=None,
        help="show only the N groups with the highest mean generation throughput",
    )
    parser.add_argument(
        "--min-acceptance",
        type=float,
        metavar="PERCENT",
        default=None,
        help=(
            "include only individual runs whose draft acceptance rate is at "
            "least PERCENT"
        ),
    )
    parser.add_argument(
        "--model",
        metavar="NAME",
        default=None,
        help="include only runs for the specified model",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=("table", "json", "csv"),
        default="table",
        help="output format (default: table)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        runs = load_runs(args.path, error_stream=sys.stderr)
    except (OSError, UnicodeError) as exc:
        print(
            "error: cannot read '{}': {}".format(args.path, exc),
            file=sys.stderr,
        )
        return 1

    runs = filter_runs(
        runs,
        model=args.model,
        min_acceptance=args.min_acceptance,
    )
    if not runs:
        print("error: no valid rows after filtering", file=sys.stderr)
        return 1

    groups = top_groups(group_runs(runs), args.top)
    print(format_output(groups, args.output_format))
    return 0


if __name__ == "__main__":
    sys.exit(main())
