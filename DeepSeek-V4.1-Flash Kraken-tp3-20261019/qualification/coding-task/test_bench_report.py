#!/usr/bin/env python3
"""Tests for bench_report.py.

These tests use only the standard library and plain ``assert`` statements.
They can be collected by pytest or run directly with::

    python3 test_bench_report.py
"""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
import traceback
from contextlib import redirect_stderr, redirect_stdout

import bench_report as br


def make_row(**overrides):
    row = {
        "model": "model-a",
        "concurrency": 1,
        "prompt_tokens": 32000,
        "output_tokens": 4096,
        "elapsed_seconds": 18.2,
        "accepted_tokens": 3100,
        "drafted_tokens": 3600,
    }
    row.update(overrides)
    return row


def make_run(**overrides):
    return br.run_from_mapping(make_row(**overrides))


def assert_close(actual, expected, tolerance=1e-9):
    assert abs(actual - expected) <= tolerance, (actual, expected)


def test_parse_valid_line():
    run = br.parse_json_line(json.dumps(make_row()))
    assert run.model == "model-a"
    assert run.concurrency == 1
    assert run.prompt_tokens == 32000
    assert run.output_tokens == 4096
    assert run.elapsed_seconds == 18.2
    assert run.accepted_tokens == 3100
    assert run.drafted_tokens == 3600


def test_validation_rejects_missing_fields():
    for field in br.REQUIRED_FIELDS:
        row = make_row()
        del row[field]
        try:
            br.run_from_mapping(row)
        except br.ValidationError as exc:
            assert field in str(exc)
        else:
            raise AssertionError("missing field {!r} was accepted".format(field))


def test_validation_rejects_negative_token_counts():
    for field in (
        "prompt_tokens",
        "output_tokens",
        "accepted_tokens",
        "drafted_tokens",
    ):
        try:
            make_run(**{field: -1})
        except br.ValidationError as exc:
            assert field in str(exc)
        else:
            raise AssertionError("negative {!r} was accepted".format(field))


def test_validation_rejects_bad_elapsed_and_acceptance():
    for row in (
        make_row(elapsed_seconds=0),
        make_row(elapsed_seconds=-1.5),
        make_row(accepted_tokens=3601, drafted_tokens=3600),
    ):
        try:
            br.run_from_mapping(row)
        except br.ValidationError:
            pass
        else:
            raise AssertionError("invalid row was accepted: {!r}".format(row))


def test_validation_rejects_malformed_json_and_non_objects():
    for line in ("{not json", "[]", "42", '"text"'):
        try:
            br.parse_json_line(line)
        except br.ValidationError:
            pass
        else:
            raise AssertionError("invalid line was accepted: {!r}".format(line))


def test_parse_jsonl_keeps_valid_rows_and_reports_line_numbers():
    lines = [
        json.dumps(make_row()),
        "",
        json.dumps(make_row(accepted_tokens=4000)),
        "{broken",
        json.dumps(make_row(model="model-b", concurrency=2)),
    ]
    errors = io.StringIO()
    runs = br.parse_jsonl(lines, source="sample.jsonl", error_stream=errors)

    assert len(runs) == 2
    assert runs[0].model == "model-a"
    assert runs[1].model == "model-b"
    error_text = errors.getvalue()
    assert "sample.jsonl:3:" in error_text
    assert "sample.jsonl:4:" in error_text


def test_calculations():
    run = make_run()
    assert_close(run.generation_throughput, 4096 / 18.2)
    assert_close(run.total_throughput, (32000 + 4096) / 18.2)
    assert_close(run.acceptance_rate, 3100 / 3600 * 100)
    assert_close(run.draft_acceptance_rate, run.acceptance_rate)
    assert_close(run.rejected_tokens, 500)


def test_zero_drafted_tokens_has_zero_acceptance_rate():
    run = make_run(accepted_tokens=0, drafted_tokens=0)
    assert run.acceptance_rate == 0.0
    assert run.rejected_tokens == 0


def test_grouping_statistics():
    runs = [
        make_run(output_tokens=100, elapsed_seconds=1, accepted_tokens=50,
                 drafted_tokens=100),
        make_run(output_tokens=200, elapsed_seconds=1, accepted_tokens=75,
                 drafted_tokens=100),
        make_run(output_tokens=300, elapsed_seconds=1, accepted_tokens=100,
                 drafted_tokens=100),
    ]
    groups = br.group_runs(runs)
    assert len(groups) == 1
    group = groups[0]
    assert group.model == "model-a"
    assert group.concurrency == 1
    assert group.runs == 3
    assert_close(group.mean_generation_tps, 200.0)
    assert_close(group.median_generation_tps, 200.0)
    assert_close(group.min_generation_tps, 100.0)
    assert_close(group.max_generation_tps, 300.0)
    assert_close(group.stdev_generation_tps, (20000 / 3) ** 0.5)
    assert_close(group.mean_acceptance_rate, 75.0)
    assert_close(group.total_accepted_tokens, 225.0)
    assert_close(group.total_rejected_tokens, 75.0)


def test_grouping_separates_models_and_concurrency():
    runs = [
        make_run(model="model-a", concurrency=1),
        make_run(model="model-a", concurrency=8),
        make_run(model="model-b", concurrency=1),
        make_run(model="model-a", concurrency=1),
    ]
    groups = br.group_runs(runs)
    keys = {(group.model, group.concurrency): group.runs for group in groups}
    assert keys == {
        ("model-a", 1): 2,
        ("model-a", 8): 1,
        ("model-b", 1): 1,
    }


def test_group_sorting_is_mean_throughput_descending():
    runs = [
        make_run(model="slow", output_tokens=100, elapsed_seconds=1),
        make_run(model="fast", output_tokens=300, elapsed_seconds=1),
        make_run(model="medium", output_tokens=200, elapsed_seconds=1),
    ]
    groups = br.group_runs(runs)
    assert [group.model for group in groups] == ["fast", "medium", "slow"]


def test_filter_by_model_and_minimum_acceptance():
    runs = [
        make_run(model="model-a", accepted_tokens=50, drafted_tokens=100),
        make_run(model="model-a", accepted_tokens=90, drafted_tokens=100),
        make_run(model="model-b", accepted_tokens=100, drafted_tokens=100),
    ]

    by_model = br.filter_runs(runs, model="model-a")
    assert len(by_model) == 2
    assert all(run.model == "model-a" for run in by_model)

    by_acceptance = br.filter_runs(runs, min_acceptance=90)
    assert len(by_acceptance) == 2
    assert all(run.acceptance_rate >= 90 for run in by_acceptance)

    combined = br.filter_runs(
        runs, model="model-a", min_acceptance=90
    )
    assert len(combined) == 1
    assert combined[0].acceptance_rate == 90


def test_top_groups():
    runs = [
        make_run(model="slow", output_tokens=100, elapsed_seconds=1),
        make_run(model="fast", output_tokens=300, elapsed_seconds=1),
        make_run(model="medium", output_tokens=200, elapsed_seconds=1),
    ]
    groups = br.group_runs(runs)
    assert [group.model for group in br.top_groups(groups, 2)] == [
        "fast",
        "medium",
    ]
    assert br.top_groups(groups, None) == groups


def test_json_output_uses_numeric_values():
    groups = br.group_runs([make_run()])
    decoded = json.loads(br.format_json(groups))
    assert isinstance(decoded, list)
    assert len(decoded) == 1
    group = decoded[0]
    for field in (
        "runs",
        "concurrency",
        "mean_generation_tps",
        "median_generation_tps",
        "min_generation_tps",
        "max_generation_tps",
        "stdev_generation_tps",
        "mean_acceptance_rate",
        "total_accepted_tokens",
        "total_rejected_tokens",
    ):
        assert isinstance(group[field], (int, float)), field
        assert not isinstance(group[field], str), field


def test_csv_output_has_header_and_rows():
    groups = br.group_runs(
        [make_run(model="model-a", concurrency=1),
         make_run(model="model-b", concurrency=2)]
    )
    rows = list(csv.reader(io.StringIO(br.format_csv(groups))))
    assert rows[0][0] == "model"
    assert "mean_generation_tps" in rows[0]
    assert len(rows) == 3


def test_table_output_contains_headers_and_values():
    groups = br.group_runs([make_run()])
    table = br.format_table(groups)
    assert "MODEL" in table
    assert "AVG_TPS" in table
    assert "ACCEPT%" in table
    assert "model-a" in table


def test_main_success_and_file_error():
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "results.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(make_row()) + "\n")

        output = io.StringIO()
        with redirect_stdout(output):
            code = br.main([path, "--format", "json"])
        assert code == 0
        assert json.loads(output.getvalue())[0]["model"] == "model-a"

        errors = io.StringIO()
        with redirect_stderr(errors):
            code = br.main([os.path.join(directory, "missing.jsonl")])
        assert code != 0
        assert "cannot read" in errors.getvalue()


def test_main_returns_nonzero_when_no_rows_after_filtering():
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "invalid.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(make_row(accepted_tokens=9999)) + "\n")

        errors = io.StringIO()
        with redirect_stderr(errors):
            code = br.main([path])
        assert code != 0
        assert "no valid rows" in errors.getvalue()


def _run_all() -> int:
    tests = [
        (name, value)
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    failures = 0
    for name, test in tests:
        try:
            test()
        except Exception:
            failures += 1
            print("FAIL {}".format(name))
            traceback.print_exc()
        else:
            print("PASS {}".format(name))
    print("\n{} passed, {} failed".format(len(tests) - failures, failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
