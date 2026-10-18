import json
import os
import sys

import httpx

API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-haiku-5-5"

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    sys.exit("no valid api key :(")

"""
TOOL_NAME = {
    "name": "...",  # short identifier the model will use to call it; letters, digits, _ and -
    "description": "...",  # plain English: what it does, when to use it, what it returns
    "input_schema": {  # JSON Schema describing the arguments the model must send
        "type": "object",  # always "object": the arguments are a set of named fields
        "properties": {
            "<field_name>": {
                "type": "...",  # string | integer | number | boolean | array | object
                "description": "...",  # what this field means, with an example
            },
            # ...more fields if the tool needs them
        },
        "required": ["<field_name>"],  # fields that must always be sent; leave optional ones out
    },
}
"""

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
print(json.dumps(response.json(), indent=2))

data = response.json()

tool_call = None
if data["stop_reason"] == "tool_use":
    for block in data["content"]:
        if block["type"] == "tool_use":
            tool_call = block

    if tool_call is None:
        sys.exit("No tool_use block found")

    print("\nModel wants to run: ", tool_call["input"]["command"])
    output = input("Paste the command's output: ")

    body["messages"].append({"role": "assistant", "content": data["content"]})

    body["messages"].append(
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call["id"],
                    "content": output,
                }
            ],
        }
    )

    print("\nSending messages: ")
    print(json.dumps(body, indent=2))

    response = httpx.post(API_URL, headers=headers, json=body, timeout=60)
    data = response.json()
    print("status:", response.status_code)
    print(json.dumps(data, indent=2))
