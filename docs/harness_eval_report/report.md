# Eval harness report

**Result: 10/10 tasks passed** with `claude-haiku-5-5`, for $0.0071 in all, in one run on 2026-11-13. This is one run, not a measurement: with 10 tasks, each task is 10 points of pass rate.

## What it is

- **Tasks:** 10 tiny Python repos in [`evals/tasks/`](../../evals/tasks/). Each has `repo/` (where the agent starts), `task.md` (the prompt) and `check/` (hidden tests and their `pytest.ini`). The toy repo from the first live run is `toy_median`.
- **Runner:** [`evals/run.py`](../../evals/run.py). For each task it:
  1. copies `repo/` to a temporary folder and runs `git init`;
  2. runs the agent in the sandbox (`system_v2`) with fixed budgets of 20 steps, $0.25 and 300 s;
  3. copies `check/` over the result;
  4. runs `python -m pytest -q --noconftest -c pytest.ini <check tests>` in the sandbox. The task passes on exit code 0.
- **Results:** one JSON line per task in [`evals/results/2026-11-13-claude-haiku-5-5.jsonl`](../../evals/results/2026-11-13-claude-haiku-5-5.jsonl). Trajectories stay in `~/.swe-agent/runs/`.

## Results

| Task | Kind | Pass | Steps | Cost | Time |
| --- | --- | --- | --- | --- | --- |
| `toy_median` | missed edge case (even length) | yes | 4 | $0.0005 | 4.1 s |
| `chunks_off_by_one` | off by one | yes | 3 | $0.0005 | 4.4 s |
| `moving_average_off_by_one` | off by one | yes | 4 | $0.0004 | 4.2 s |
| `word_count_edges` | missed edge cases | yes | 4 | $0.0006 | 5.7 s |
| `merge_intervals_edges` | missed edge cases | yes | 4 | $0.0008 | 6.9 s |
| `slugify_readme` | feature described in the README | yes | 3 | $0.0008 | 6.0 s |
| `inventory_remove_readme` | feature described in the README | yes | 4 | $0.0009 | 6.5 s |
| `discount_two_files` | change across two files | yes | 4 | $0.0008 | 6.3 s |
| `kelvin_two_files` | change across two files | yes | 4 | $0.0008 | 5.8 s |
| `renamed_import` | import error | yes | 4 | $0.0009 | 5.6 s |

Every run stopped with `submitted`, after 3 or 4 steps. In `discount_two_files` the agent also added its own tests to `test_invoice.py`. Grading copies `check/` back first, so those tests were not part of the grade.

## Checks on the harness

These run with no API key, in `tests/test_evals.py`:

- Every task's `check/` fails on its untouched repo, so doing nothing cannot pass.
- With a scripted model: a fix passes, while doing nothing, editing the test, adding a `conftest.py` that skips every test, and adding a `pytest.toml` with `--collect-only` all fail. The last two pass plain `pytest -q` with exit code 0, which is why grading uses `--noconftest -c pytest.ini`.
- A reference fix for each task was checked to pass, on the host and in the sandbox (not committed).

## Limits

- The tasks are easy. 10/10 shows that the harness works end to end. It says little about the model, and the set cannot tell two good models apart.
- One run per task. A second run could differ.

## Reproduce

Paid. The cap is $0.25 per task, and this run cost under a cent.

```bash
docker build -t swe-agent-sandbox -f docker/sandbox.Dockerfile docker/    # once
uv run python -m evals.run                    # all tasks
uv run python -m evals.run --task toy_median  # one task
```
