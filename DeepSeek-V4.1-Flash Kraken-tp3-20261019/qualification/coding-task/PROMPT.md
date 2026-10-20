# Coding task prompt

Sent once to the server running `azallaza/ds41-kraken-tp3:20261019.2` through a coding client; the files in
this folder are the model's output, unedited.

> Build a Python 3 CLI tool called `bench_report.py` that analyzes benchmark results from a JSON Lines file.
>
> Each line of the input file contains one benchmark run with fields like:
>
> ```json
> {"model": "model-a", "concurrency": 1, "prompt_tokens": 32000, "output_tokens": 4096,
>  "elapsed_seconds": 18.2, "accepted_tokens": 3100, "drafted_tokens": 3600}
> ```
>
> Requirements:
>
> 1. Read a JSONL file passed as the first positional argument.
> 2. Validate every row.
>    - Required fields: model, concurrency, prompt_tokens, output_tokens, elapsed_seconds, accepted_tokens, drafted_tokens
>    - Reject rows with missing fields, negative token counts, elapsed_seconds <= 0, or accepted_tokens > drafted_tokens.
>    - Invalid rows should be reported to stderr with their line number, but valid rows should still be processed.
> 3. For every valid row, calculate:
>    - generation throughput = output_tokens / elapsed_seconds
>    - total throughput = (prompt_tokens + output_tokens) / elapsed_seconds
>    - draft acceptance rate = accepted_tokens / drafted_tokens * 100
>    - rejected draft tokens = drafted_tokens - accepted_tokens
> 4. Group results by: model + concurrency
> 5. For each group calculate: number of runs, mean / median / minimum / maximum generation throughput,
>    standard deviation, mean draft acceptance rate, total accepted tokens, total rejected draft tokens.
> 6. Add command-line options:
>    - `--top N`: show the N groups with the highest mean generation throughput.
>    - `--min-acceptance PERCENT`: only include individual runs whose draft acceptance rate is at least this value.
>    - `--model NAME`: only include results for the specified model.
>    - `--format table|json|csv`: default is table.
> 7. Table output should look roughly like:
>
>    ```
>    MODEL        CONC  RUNS  AVG_TPS  MEDIAN_TPS  MAX_TPS  ACCEPT%
>    model-a      1     8     312.4    309.8       355.1    87.3
>    model-a      8     8     901.7    895.4       950.2    74.1
>    ```
>
>    Sort by mean generation throughput descending.
> 8. JSON output must contain numeric values as numbers, not formatted strings.
> 9. CSV output must include a header row.
> 10. Use only the Python standard library.
> 11. Organize the program into reusable functions instead of putting everything in main().
> 12. Use dataclasses and type hints.
> 13. Add useful --help text and return a non-zero exit code if the file cannot be opened or if there are zero
>     valid rows after filtering.
> 14. Create a small pytest-style test file named test_bench_report.py, but do not depend on pytest-specific
>     features. The tests should also run directly with: `python3 test_bench_report.py`
>
> Test parsing, validation, calculations, grouping, filtering, and sorting.
>
> At the end, briefly explain the program structure and provide 3 example CLI commands.
