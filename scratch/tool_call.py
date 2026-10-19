import json
import os
import subprocess
import sys
from pathlib import Path

import httpx

API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-haiku-5-5"
REPO_ROOT = Path(__file__).resolve().parent.parent  # this file lives in <repo>/scratch/

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    sys.exit("no valid api key :(")

BASH_TOOL = {
    "name": "bash",
    "description": (
        "Run shell commands in bash. "
        "Run the shell commands from the repository root. "
        "Return stdout, stderr, and exit code. "
        "Decide what to do next given the result. "
        "commands are stopped after 30 seconds."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "the shell command to run in bash, e.g. ls src",
            },
        },
        "required": ["command"],
    },
}

headers = {
    "x-api-key": api_key,
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
}

body = {
    "model": MODEL,
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "how many python files are in this repository?"}],
    "tools": [BASH_TOOL],
}

response = httpx.post(API_URL, headers=headers, json=body, timeout=60)
print("status: ", response.status_code)
# check before .json(): an error body has no stop_reason and may not even be JSON
if response.status_code != 200:
    sys.exit(f"API error: {response.text}")
print(json.dumps(response.json(), indent=2))

data = response.json()


def append_one_tool_result(tool_call: dict) -> dict:

    command = tool_call["input"]["command"]
    result = None
    stdout = None
    stderr = None
    returncode = 0
    try:
        result = subprocess.run(
            command,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            errors="replace",  # binary output must not crash the script
            timeout=30,
            cwd=REPO_ROOT,
        )
    except subprocess.TimeoutExpired as e:
        stdout = (e.stdout or b"").decode(errors="replace")
        stderr = (e.stderr or b"").decode(errors="replace")
        returncode = -9

    if result is not None:
        stdout = result.stdout
        stderr = result.stderr
        returncode = result.returncode

    tool_result = {
        "type": "tool_result",
        "tool_use_id": tool_call["id"],
        "content": json.dumps(
            {
                "stdout": stdout,
                "stderr": stderr,
                "returncode": returncode,
                "timeout": result is None,
            }
        ),
        "is_error": returncode != 0,
    }

    return tool_result


if data["stop_reason"] == "tool_use":
    tool_calls = [block for block in data["content"] if block["type"] == "tool_use"]

    if len(tool_calls) == 0:
        sys.exit("No tool_use block found (wtf):(")

    body["messages"].append({"role": "assistant", "content": data["content"]})

    tool_results = []
    for tool_call in tool_calls:
        print("\nModel wants to run: ", tool_call["input"]["command"])
        tool_results.append(append_one_tool_result(tool_call))
    body["messages"].append({"role": "user", "content": tool_results})

    print(json.dumps(body, indent=2))
    response = httpx.post(API_URL, headers=headers, json=body, timeout=60)
    print("status: ", response.status_code)
    if response.status_code != 200:
        sys.exit(f"API error: {response.text}")
    print(json.dumps(response.json(), indent=2))
