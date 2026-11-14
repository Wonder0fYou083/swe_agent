import random
import re
from collections.abc import Generator
from contextlib import contextmanager

import pytest

from swe_agent.cli.splash import (
    _ENTER,  # pyright: ignore[reportPrivateUsage]
    _LEAVE,  # pyright: ignore[reportPrivateUsage]
    BLACK,
    DRAIN_S,
    MAX_PARTICLES,
    Move,
    Press,
    Quit,
    Release,
    Scene,
    Start,
    parse_input,
    render,
    run_splash,
)

ROW_START = re.compile(r"\x1b\[\d+;1H")
ESCAPE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def press(col: int, row: int) -> bytes:
    """An SGR mouse report for a left-button press at 1-based (col, row)."""
    return f"\x1b[<0;{col};{row}M".encode()


def release(col: int, row: int) -> bytes:
    return f"\x1b[<0;{col};{row}m".encode()


def hover(col: int, row: int) -> bytes:
    return f"\x1b[<35;{col};{row}M".encode()


def drag(col: int, row: int) -> bytes:
    return f"\x1b[<32;{col};{row}M".encode()


# --- parse_input


def test_a_press_and_a_release_carry_their_0_based_cell() -> None:
    assert parse_input(press(10, 5) + release(10, 5)) == ([Press(9, 4), Release(9, 4)], b"")


@pytest.mark.parametrize("report", [hover(3, 7), drag(3, 7)])
def test_hovering_and_dragging_are_moves(report: bytes) -> None:
    assert parse_input(report) == ([Move(2, 6)], b"")


@pytest.mark.parametrize("report", [b"\x1b[<64;3;7M", b"\x1b[<65;3;7M"])
def test_the_scroll_wheel_is_ignored(report: bytes) -> None:
    assert parse_input(report) == ([], b"")


@pytest.mark.parametrize(("key", "event"), [(b"q", Quit()), (b"\r", Start()), (b" ", Start())])
def test_keys(key: bytes, event: Quit | Start) -> None:
    assert parse_input(key) == ([event], b"")


@pytest.mark.parametrize("key", [b"\x1b[A", b"\x1bOB", b"x", b"\x1b[I"])
def test_arrow_keys_focus_reports_and_other_keys_do_nothing(key: bytes) -> None:
    assert parse_input(key) == ([], b"")


def test_a_lone_esc_is_left_over_since_a_mouse_report_may_follow() -> None:
    assert parse_input(b"\x1b") == ([], b"\x1b")


def test_esc_followed_by_a_plain_key_is_the_esc_key() -> None:
    assert parse_input(b"\x1bx") == ([Quit()], b"")


def test_a_report_split_across_two_reads_is_parsed_once_it_is_whole() -> None:
    first, rest = hover(40, 12)[:6], hover(40, 12)[6:]

    events, pending = parse_input(hover(1, 1) + first)
    assert events == [Move(0, 0)]
    assert pending == first

    assert parse_input(pending + rest) == ([Move(39, 11)], b"")


# --- the scene


def scene(cols: int = 80, rows: int = 24) -> Scene:
    return Scene(cols, rows, title="swe-agent 0.0.0", hint="click", rng=random.Random(0), ambient=0)


def test_the_centre_of_the_screen_is_the_black_shadow() -> None:
    s = scene()
    pixels = s.compose(t=1.0)
    assert pixels[(s.h // 2) * s.w + s.w // 2] == BLACK


def test_a_frame_draws_every_row_and_column_once() -> None:
    frame = render(scene(80, 24), 1.0, truecolor=True)

    rows = ROW_START.split(frame)[1:]
    assert len(rows) == 24
    visible = [ESCAPE.sub("", row) for row in rows]
    assert all(len(row) == 80 for row in visible)
    assert "swe-agent 0.0.0" in visible[21]


@pytest.mark.parametrize(("cols", "rows"), [(10, 4), (1, 1), (300, 90)])
def test_any_terminal_size_renders(cols: int, rows: int) -> None:
    frame = render(scene(cols, rows), 1.0, truecolor=True, fall=0.5)
    assert len(ROW_START.split(frame)) == rows + 1


def test_without_truecolor_only_the_256_color_palette_is_used() -> None:
    frame = render(scene(), 1.0, truecolor=False)
    assert "38;2;" not in frame
    assert "48;2;" not in frame
    assert "48;5;" in frame


def test_at_the_end_of_the_fall_the_whole_screen_is_black() -> None:
    assert set(scene().compose(t=1.0, fall=1.0)) == {BLACK}


def test_stardust_from_the_edge_of_the_screen_falls_in_before_it_expires() -> None:
    s = scene()
    edge = [(x + 0.5, y) for x in range(0, s.w, 4) for y in (0.5, s.h - 0.5)]
    edge += [(x, y + 0.5) for y in range(0, s.h, 8) for x in (0.5, s.w - 0.5)]
    for x, y in edge[:50]:
        s.spawn(x, y, count=1)
    assert len(s.particles) == 50

    for _ in range(12 * 30):  # twelve seconds at 30 frames a second
        s.step(1 / 30)

    assert s.particles == []
    assert s.removed == {"swallowed": 50, "escaped": 0, "expired": 0, "dropped": 0}


def test_stardust_runs_the_same_way_round_as_the_disk() -> None:
    s = scene()
    s.spawn(s.cx + 3 * s.r_sh, s.cy, count=1)  # right of the hole
    assert s.particles[0].vy > 0  # heading down the screen: clockwise, like the disk


def test_a_flick_of_the_mouse_cannot_fling_stardust_out_of_orbit() -> None:
    s = scene()
    s.spawn(s.cx + 3 * s.r_sh, s.cy, vx=1e6, vy=0, count=1)
    p = s.particles[0]
    assert (p.vx**2 + p.vy**2) ** 0.5 < 1.4 * s.orbit_speed(3 * s.r_sh)


def test_the_oldest_stardust_is_dropped_past_the_cap() -> None:
    s = scene()
    s.spawn(1, 1, count=MAX_PARTICLES + 100)
    assert len(s.particles) == MAX_PARTICLES
    assert s.removed["dropped"] == 100


def test_stardust_drifts_in_by_itself() -> None:
    s = Scene(80, 24, rng=random.Random(0), ambient=3.0)
    s.step(1.0)
    assert len(s.particles) == 3


# --- run_splash


class FakeTerminal:
    """Plays back scripted input. Each read moves the fake clock on by its timeout."""

    def __init__(self, reads: list[bytes | BaseException]) -> None:
        self.reads = reads
        self.now = 0.0
        self.written: list[str] = []
        self.raw_mode: list[str] = []
        self.reads_after_leave = 0

    def clock(self) -> float:
        return self.now

    def size(self) -> tuple[int, int]:
        return 40, 12

    def read(self, timeout: float) -> bytes:
        self.now += max(timeout, 1e-3)
        if _LEAVE in self.written:
            self.reads_after_leave += 1
        if not self.reads:
            return b""
        item = self.reads.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def write(self, text: str) -> None:
        self.written.append(text)

    @contextmanager
    def raw(self) -> Generator[None]:
        self.raw_mode.append("on")
        try:
            yield
        finally:
            self.raw_mode.append("off")


def splash(term: FakeTerminal) -> bool:
    return run_splash(
        term, title="swe-agent", truecolor=True, clock=term.clock, rng=random.Random(0)
    )


def test_a_click_starts_after_the_fall_in() -> None:
    term = FakeTerminal([press(5, 5), release(5, 5)])

    assert splash(term) is True
    assert term.now >= 0.8  # the fall-in played first


def test_a_drag_does_not_start_and_q_quits() -> None:
    term = FakeTerminal([press(5, 5), drag(9, 5), release(9, 5), b"q"])
    assert splash(term) is False


def test_moving_the_mouse_sheds_stardust_where_the_cursor_is(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawned: list[tuple[float, float, float, float]] = []

    def record(self: Scene, x: float, y: float, vx: float = 0.0, vy: float = 0.0) -> None:
        spawned.append((x, y, vx, vy))

    monkeypatch.setattr(Scene, "spawn", record)
    term = FakeTerminal([hover(5, 5), hover(9, 5), b"q"])

    assert splash(term) is False

    # Cell (4, 4) holds pixels (4, 8) and (4, 9); the stardust starts in the cell's middle.
    assert spawned[0] == (4.5, 9.0, 0.0, 0.0)
    x, y, vx, vy = spawned[1]
    assert (x, y) == (8.5, 9.0)
    assert vx > 0  # carries some of the cursor's rightward speed
    assert vy == 0


@pytest.mark.parametrize("key", [b"\r", b" "])
def test_enter_and_space_start(key: bytes) -> None:
    assert splash(FakeTerminal([key])) is True


def test_esc_with_nothing_after_it_quits() -> None:
    assert splash(FakeTerminal([b"\x1b", b""])) is False


def test_the_terminal_is_set_up_then_put_back() -> None:
    term = FakeTerminal([b"q"])

    splash(term)

    assert term.written[0] == _ENTER
    assert term.written[-1] == _LEAVE
    assert term.raw_mode == ["on", "off"]
    assert term.reads_after_leave > 0  # stray mouse reports were drained


def test_ctrl_c_passes_through_after_the_terminal_is_put_back() -> None:
    term = FakeTerminal([hover(5, 5), KeyboardInterrupt()])

    with pytest.raises(KeyboardInterrupt):
        splash(term)

    assert term.written[-1] == _LEAVE
    assert term.raw_mode == ["on", "off"]
    assert term.reads_after_leave > 0
    assert term.now >= DRAIN_S
