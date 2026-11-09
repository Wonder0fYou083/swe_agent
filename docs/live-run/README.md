# A live run, start to finish

One real run of `swe-agent` against the Anthropic API: from the command to the fixed bug, with its trace and the prompt-cache numbers.

- **Date:** 2026-11-09 · **Model:** `claude-haiku-5-5` · **Prompt:** `system_v1` · **swe-agent:** 0.0.0
- **Result:** bug fixed in 4 steps, 5.4 s, $0.00074
- **Traces:** [cached.jsonl](cached.jsonl), [no-cache.jsonl](no-cache.jsonl)

## The bug

[tests/fixtures/toy_repo](../../tests/fixtures/toy_repo) is a four-file repository. `median` in `stats.py` returns the upper middle value for even-length lists, so one of its five tests fails:

```text
>       assert median([4, 1, 3, 2]) == 2.5
E       assert 3 == 2.5
1 failed, 4 passed in 0.01s
```

## The command

The agent edits files, so it works on a copy with its own git history:

```bash
rm -rf /tmp/toy && cp -R tests/fixtures/toy_repo /tmp/toy && git -C /tmp/toy init -q && git -C /tmp/toy add -A && git -C /tmp/toy commit -qm init
uv run --env-file .env swe-agent run "make the tests pass" --cwd /tmp/toy --max-cost 0.25
```

## What happens inside

1. **Key.** `uv run --env-file .env` puts `ANTHROPIC_API_KEY` in the environment. `run` in [cli/main.py](../../src/swe_agent/cli/main.py) reads it and removes it from the environment, so the agent's shell commands cannot see it.
2. **Setup.** The CLI loads [prompts/system_v1.md](../../prompts/system_v1.md), then builds the model adapter and the agent with its budgets: 30 steps, $0.25, 600 s. A model with no price entry fails here.
3. **Session start.** `Agent.run` in [core/agent.py](../../src/swe_agent/core/agent.py) opens `runs/<session_id>.jsonl` and writes a `session_start` record. The history is one user message: the task.
4. **Model call.** The budgets are checked. Then [models/anthropic.py](../../src/swe_agent/models/anthropic.py) posts the system prompt, the history and the `bash` tool spec to `/v1/messages`. [models/http.py](../../src/swe_agent/models/http.py) retries rate limits, server errors and network errors with backoff, up to 4 attempts.
5. **Response.** Text and tool calls become provider-neutral blocks. Any other block, such as thinking, is kept as it came and sent back unchanged on the next call. Token usage is priced from [models/pricing.py](../../src/swe_agent/models/pricing.py).
6. **Tools.** Each tool call runs through [tools/bash.py](../../src/swe_agent/tools/bash.py): a fresh `/bin/bash -c` in `--cwd`, killed with its process group after 30 s, output capped at 10,000 characters. All results go back to the model in one user message.
7. **Step record.** The step is written to the trajectory and printed. The loop returns to 4.
8. **Stop.** A reply with no tool call ends the run as `submitted`. A used-up budget, a refusal, a `max_tokens` cut-off or an API error ends it under its own name. A `session_end` record is written, and the exit code is 0 only for `submitted`.

## The run

The terminal shows each step as it finishes. Command output is cut to 5 lines on screen; the trace has all of it.

```text
[step 1] $0.0002 so far
I'll start by looking at the repository layout and running the test suite to see what's failing.
$ ls -la && git status 2>/dev/null | head -20 && git log --oneline 2>/dev/null | head -5
  exit 0
  total 32
  drwxr-xr-x@   7 kaiwenwu  wheel   224 Nov  9 14:32 .
  drwxrwxrwt  251 root      wheel  8032 Nov  9 14:32 ..
  drwxr-xr-x   12 kaiwenwu  wheel   384 Nov  9 14:32 .git
  -rw-r--r--@   1 kaiwenwu  wheel    23 Nov  9 14:32 pytest.ini
  ... 6 more lines
$ ls -R . 2>/dev/null | grep -v '^\.git' | head -100
  exit 0
  pytest.ini
  README.md
  stats.py
  test_stats.py

[step 2] $0.0003 so far
$ cat README.md; echo ----; cat pytest.ini; echo ----; cat stats.py; echo ----; cat test_stats.py; echo ----; python3 -m pytest -q 2>&1 | tail -30
  exit 0
  # stats

  Small statistics helpers.

  Run the tests with:
  ... 64 more lines

[step 3] $0.0006 so far
The one failure is the even-length median: it returns the upper middle element instead of averaging the two middle values. I'll fix that in `stats.py`.
$ cd . && python3 - <<'EOF'
import re
p='stats.py'
s=open(p).read()
old="""    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle]
"""
new="""    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2
"""
assert s.count(old)==1
open(p,'w').write(s.replace(old,new))
EOF
python3 -m pytest -q 2>&1 | tail -3
  exit 0
  .....                                                                    [100%]
  5 passed in 0.00s

[step 4] $0.0007 so far
All 5 tests pass now.

The only failure was `median` on even-length lists. It returned the upper of the two middle values (`3` for `[4, 1, 3, 2]`) instead of their average. In `stats.py`, I changed it so odd-length lists still return the middle element and even-length lists return the mean of the two middle elements. I didn't touch the tests.

stopped: submitted after 4 steps, $0.0007
trajectory: /path/to/swe_agent/runs/f4b4e3680ed04ffea123a1dd5c4cf5cd.jsonl
```

| Step | What the agent did | Outcome |
| --- | --- | --- |
| 1 | Listed the files and the git state, with two tool calls in one reply | 4 files, clean tree |
| 2 | Read every file and ran the tests | 1 failed, 4 passed |
| 3 | Patched `median` with a Python script and re-ran the tests | 5 passed |
| 4 | Summarized the change, with no tool call | `submitted` |

Model calls took 4.9 s of the 5.4 s; the commands took 0.5 s.

## The fix

`git -C /tmp/toy diff` after the run. Only `stats.py` changed:

```diff
@@ -14,4 +14,6 @@ def median(values: list[float]) -> float:
         raise ValueError("median of an empty list")
     ordered = sorted(values)
     middle = len(ordered) // 2
-    return ordered[middle]
+    if len(ordered) % 2 == 1:
+        return ordered[middle]
+    return (ordered[middle - 1] + ordered[middle]) / 2
```

## The trace

A trajectory is one JSON object per line: one `session_start`, one `step` per model call, one `session_end`. The schema is in [telemetry/trace.py](../../src/swe_agent/telemetry/trace.py).

Step 2 of [cached.jsonl](cached.jsonl), with long strings shortened:

```json
{
  "type": "step",
  "session_id": "f4b4e3680ed04ffea123a1dd5c4cf5cd",
  "step": 2,
  "parent_step": 1,
  "request_messages": 3,
  "assistant": {
    "role": "assistant",
    "content": [
      {
        "type": "tool_call",
        "id": "toolu_01FHxKb45S5Co1DeiE4aFCbP",
        "name": "bash",
        "arguments": {"command": "cat README.md; echo ----; cat …"}
      }
    ]
  },
  "appended": true,
  "stop_reason": "tool_use",
  "usage": {"input_tokens": 2, "output_tokens": 117, "cache_read_tokens": 805, "cache_write_tokens": 611},
  "cost_usd": 0.000143125,
  "model_started_ns": 852314061477791,
  "model_first_token_ns": null,
  "model_ended_ns": 852315033466291,
  "attempts": 1,
  "tools": [
    {
      "tool_call_id": "toolu_01FHxKb45S5Co1DeiE4aFCbP",
      "tool_name": "bash",
      "command": "cat README.md; echo ----; cat …",
      "observation": "exit code: 0\n<output>\n# stats\n\nSmall sta…",
      "is_error": false,
      "exit_code": 0,
      "timed_out": false,
      "truncated_chars": 0,
      "started_ns": 852315033649875,
      "ended_ns": 852315274781375
    }
  ]
}
```

- `observation` is exactly the text the model received.
- Every `*_ns` field is a monotonic clock reading. `session_start` holds `t0_ns` and the only wall-clock time, `started_at`.
- `model_first_token_ns` is null because responses are not streamed yet.
- Step 1 contains a thinking block. The API returns it with empty text and a signature, and still counts the thinking in `output_tokens`.

Per-step tokens and cost from a trace (columns: step, uncached input, cache write, cache read, output, cost):

```bash
jq -r 'select(.type=="step") | [.step, .usage.input_tokens, .usage.cache_write_tokens, .usage.cache_read_tokens, .usage.output_tokens, .cost_usd] | @tsv' docs/live-run/cached.jsonl
```

## Cache and no cache

Prices for `claude-haiku-5-5` per million tokens, from [models/pricing.py](../../src/swe_agent/models/pricing.py): input $0.10, cache write $0.125, cache read $0.01, output $0.50.

### With caching (the default)

Every request carries a top-level `cache_control: {"type": "ephemeral"}`, which asks the API to cache the prompt up to its last block. Trace: [cached.jsonl](cached.jsonl).

| Step | Prompt tokens | Uncached input | Cache write | Cache read | Output | Cost |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 809 | 4 | 805 | 0 | 195 | $0.000199 |
| 2 | 1,418 | 2 | 611 | 805 | 117 | $0.000143 |
| 3 | 2,245 | 2 | 827 | 1,416 | 310 | $0.000273 |
| 4 | 2,598 | 2 | 353 | 2,243 | 119 | $0.000126 |
| Total | 7,070 | 10 | 2,596 | 4,464 | 741 | $0.00074 |

Step 1 writes its whole prompt to the cache. Each later step reads everything the step before it wrote and read (805, then 805 + 611 = 1,416, then 1,416 + 827 = 2,243) and writes only the new messages.

### Without caching (`--no-cache`)

The same command with `--no-cache` added. Trace: [no-cache.jsonl](no-cache.jsonl).

| Step | Prompt tokens | Uncached input | Cache write | Cache read | Output | Cost |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 809 | 809 | 0 | 0 | 83 | $0.000122 |
| 2 | 1,418 | 1,418 | 0 | 0 | 92 | $0.000188 |
| 3 | 2,248 | 2,248 | 0 | 0 | 270 | $0.000360 |
| Total | 4,475 | 4,475 | 0 | 0 | 445 | $0.00067 |

Both cache counters stay at 0, and every prompt token is billed at the full input price.

This run started from the copy the first run had already fixed, because the copy was not reset in between. The agent found the tests passing, changed nothing and stopped after 3 steps. It shows the counters, but its total cannot be compared with the cached run.

### The same four steps at both prices

Pricing the cached run's prompts at the plain input rate gives the like-for-like comparison:

| Step | Prompt tokens | With cache | Without cache |
| --- | --- | --- | --- |
| 1 | 809 | $0.000199 | $0.000178 |
| 2 | 1,418 | $0.000143 | $0.000200 |
| 3 | 2,245 | $0.000273 | $0.000380 |
| 4 | 2,598 | $0.000126 | $0.000319 |
| Total | 7,070 | $0.00074 | $0.00108 |

Caching costs 11% more on step 1, because a write is 1.25 times the input price. It costs less on every step after that, because a read is 0.1 times the input price. Over the whole run it is 31% cheaper. This is one run, not a measurement.
