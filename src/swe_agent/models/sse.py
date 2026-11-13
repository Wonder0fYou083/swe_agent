"""Parse Server-Sent Events, the format streaming APIs use."""

from collections.abc import Iterable, Iterator


def iter_sse(lines: Iterable[str]) -> Iterator[tuple[str | None, str]]:
    """Yield (event name, data) for each event. A blank line ends an event."""
    event: str | None = None
    data: list[str] = []
    for line in lines:
        if line == "":
            if data:
                yield event, "\n".join(data)
            event, data = None, []
            continue
        if line.startswith(":"):  # a comment, used as a keep-alive
            continue
        field, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if field == "event":
            event = value
        elif field == "data":
            data.append(value)
    if data:
        yield event, "\n".join(data)
