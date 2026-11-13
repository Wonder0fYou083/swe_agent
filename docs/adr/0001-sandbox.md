# 0001 · Run every bash command in a fresh Docker container

**Status:** accepted · **Date:** 2026-11-13

## Context

The agent's only tool, `bash`, ran each command on the host with the user's full rights. The commands come from a model that reads untrusted text (repository files, test output), so one bad command could delete files outside the repository, send data out over the network, or fork-bomb the machine. The shell also inherited the host environment, so `env` could print any secret that was in it.

The sandbox has to stop those things without changing what the model sees: one tool, one command per call, a fresh shell each time, the repository as the working folder.

## Decision

`run_bash_in_sandbox` in `src/swe_agent/tools/bash.py` has the same signature as `run_bash`. It starts one container per command with the Docker CLI, through `subprocess`:

```text
docker run --rm --name swe-<uuid> --network none
    --read-only -v <repo>:/workspace -w /workspace
    --tmpfs /tmp:size=64m -e HOME=/tmp
    --memory 1g --memory-swap 1g --cpus 1 --pids-limit 256
    --user <uid>:<gid> --cap-drop ALL --security-opt no-new-privileges
    --pull never swe-agent-sandbox bash -c <command>
```

| Flag | Why |
| --- | --- |
| `--rm`, `--name swe-<uuid>` | The container is removed after the command. The name lets a timeout find it. |
| `--network none` | No network. A connect fails at once with "Network is unreachable". |
| `--read-only`, `-v <repo>:/workspace` | The repository is the only writable host path. Nothing else from the host is in the container. |
| `--tmpfs /tmp:size=64m`, `-e HOME=/tmp` | Scratch space that lives in memory and is gone after the command, and a home a non-root user can write. |
| `--memory 1g --memory-swap 1g` | A memory cap with no swap. Going over kills the command with exit code 137. |
| `--cpus 1`, `--pids-limit 256` | A CPU cap, and a process cap that stops a fork bomb from spreading. |
| `--user`, `--cap-drop ALL`, `no-new-privileges` | The command is not root, has no capabilities, and setuid binaries cannot raise its privileges. |
| `--pull never` | A missing image fails at once with exit code 125, instead of a pull from Docker Hub. |

The command is the last element of the argv list, so no shell on the host parses it.

**On timeout,** the tool runs `docker rm -f <name>` and then collects the output. The daemon owns the container, so killing the `docker run` client leaves the container running. Removing the container kills every process in it, so a background child can no longer outlive the call.

**The image** is `docker/sandbox.Dockerfile`: `python:3.12-slim` plus pytest 9.1.1, the version in `uv.lock`. It is 235 MB, has GNU tools and no git. The build needs the network; the containers have none.

**The CLI** uses the sandbox by default, with the `system_v2` prompt (Linux, GNU tools, no network, `/workspace`). `--no-sandbox` runs commands on the host with `system_v1`, the macOS prompt. Before the first model call, the CLI runs `docker image inspect swe-agent-sandbox` and stops with the build command if Docker or the image is missing. Without that check, the model would see exit code 125 on every command and could fix neither.

## Consequences

- **Tested:** `tests/test_sandbox.py` (10 tests, marked `docker`) checks that each attempt fails cleanly. That means a normal result, an end within the timeout plus 5 s, no container left behind, and no change next to the workspace. The attempts are:
  - writing outside the workspace (`touch /etc/x`, `touch /x`, `touch ../x`)
  - reaching the network
  - a fork bomb, which ends by timeout in about 3.3 s with a 3 s timeout
  - using 2 GiB of memory (exit code 137)
  - running past the timeout
  - reading host secrets
  - running as root (`id -u` is the host uid)

  The `sandbox_argv` tests check every flag without Docker, so they also run in CI. CI has no image, so the `docker` tests are skipped there.
- **Cost:** a container starts in about 0.11 s once warm, and 0.23 s the first time (Docker 29.9.2, Docker Desktop, linux/arm64).
- **The model cannot install anything** (no network) and has no git. The prompt says so.
- **File ownership on macOS:** Docker Desktop shows every file in the mount as owned by the host user, even one root wrote. So on a Mac only `id -u` inside the container shows that `--user` works. On Linux the ownership on disk is real.
- **Not protected:**
  - The workspace itself: the agent can still delete or rewrite any file in it, `.git` included.
  - Files in the workspace that the host runs later, such as git hooks, a Makefile or `conftest.py`: they run with full rights once you run them outside the container.
  - The kernel: it is shared with the container. On macOS it is the Docker Desktop VM's kernel.
  - Ctrl-C during a command: it skips the timeout branch, so that container can outlive the agent.
- **`--no-sandbox`** has none of these protections. It still removes the API keys from the environment before the first command.
- **Podman** takes the same flags, but the code calls `docker` and Podman is untested.

## Alternatives considered

- **One long-lived container, with `docker exec` per command.** Starts faster and keeps installed state. Rejected because state would leak from one command to the next, and a timed-out `exec` leaves its processes running inside the container. It is also the "persistent shell" feature, which is out of scope.
- **The Docker SDK for Python.** Rejected: it adds a dependency, and the CLI flags also work with Podman.
- **OS sandboxes** (Seatbelt on macOS, bubblewrap on Linux). They need no VM or daemon. Rejected because they mean writing a policy for each platform, and macOS has no cgroup limits for memory or processes. A tested container runtime is safer than isolation code written here.
