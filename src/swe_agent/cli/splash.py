"""The start page of bare `swe-agent`: a black hole drawn in the terminal.

The hole sits in the middle of the screen with a spinning accretion disk, and the stars
behind it are bent by a point-mass lens. Moving the mouse sheds stardust that spirals in.
A click, Enter or Space starts a session; q, Esc or Ctrl-C quits.

Everything that decides what appears is pure and tested without a terminal: `parse_input`,
`Scene` and `render`. `PosixTerminal` is the only part that touches the TTY, and
`run_splash` owns setup and teardown, so even a crash leaves the terminal usable.
"""

import itertools
import math
import os
import random
import re
import select
import sys
import time
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from typing import Protocol

RGB = tuple[float, float, float]
BLACK: RGB = (0.0, 0.0, 0.0)

HINT = "click anywhere to start · q to quit"

# --- input -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Move:
    """The mouse is over cell (x, y), 0-based, with or without a button held."""

    x: int
    y: int


@dataclass(frozen=True, slots=True)
class Press:
    x: int
    y: int


@dataclass(frozen=True, slots=True)
class Release:
    x: int
    y: int


@dataclass(frozen=True, slots=True)
class Start:
    """Enter or Space."""


@dataclass(frozen=True, slots=True)
class Quit:
    """q, Esc or Ctrl-C."""


Event = Move | Press | Release | Start | Quit

ESC = b"\x1b"
# SGR mouse report: ESC [ < button ; column ; row, then M for press/motion or m for release.
_SGR_MOUSE = re.compile(rb"\x1b\[<(\d+);(\d+);(\d+)([Mm])")
# Any other escape sequence: CSI (arrow keys, focus reports, ...) or SS3 (ESC O A).
_OTHER_ESCAPE = re.compile(rb"\x1b(?:\[[0-?]*[ -/]*[@-~]|O.)", re.DOTALL)
# The start of an escape sequence whose rest has not arrived yet, a lone ESC included.
_UNFINISHED = re.compile(rb"\x1b(?:\[[0-?]*[ -/]*|O)?\Z")
_START_KEYS = frozenset(b"\r\n ")
_QUIT_KEYS = frozenset(b"qQ\x03")


def parse_input(buf: bytes) -> tuple[list[Event], bytes]:
    """The events in `buf`, and any unfinished escape sequence at its end to read again.

    A lone ESC at the end is left over too: the caller treats it as the Esc key when
    nothing follows it within a frame, since it may be the first byte of a mouse report.
    """
    events: list[Event] = []
    i = 0
    while i < len(buf):
        if buf[i] != ESC[0]:
            if buf[i] in _START_KEYS:
                events.append(Start())
            elif buf[i] in _QUIT_KEYS:
                events.append(Quit())
            i += 1
        elif mouse := _SGR_MOUSE.match(buf, i):
            if (event := _mouse_event(mouse)) is not None:
                events.append(event)
            i = mouse.end()
        elif other := _OTHER_ESCAPE.match(buf, i):
            i = other.end()
        elif _UNFINISHED.match(buf, i):
            return events, buf[i:]
        else:  # ESC and then a plain key: the Esc key, then that key
            events.append(Quit())
            i += 1
    return events, b""


def _mouse_event(report: re.Match[bytes]) -> Event | None:
    button, x, y = int(report[1]), int(report[2]) - 1, int(report[3]) - 1
    if button & 64:  # the scroll wheel
        return None
    if button & 32:  # motion, with or without a button held
        return Move(x, y)
    return Press(x, y) if report[4] == b"M" else Release(x, y)


# --- the scene -------------------------------------------------------------------------
# Distances are in pixels. A terminal cell is about twice as tall as it is wide, so each
# cell holds two square pixels, drawn with the upper-half block.

SHADOW = 0.11  # radius of the black shadow, as a fraction of the screen's shorter side
LENS = 1.35  # Einstein radius of the lens that bends the stars, in shadow radii
FLATTEN = 0.3  # the disk is seen nearly edge-on: its height over its width
DISK_IN, DISK_OUT = 1.3, 3.6  # inner and outer disk radius, in shadow radii
DISK_BANDS = 7  # rings that each turn rigidly, so the texture never shears into noise
SPIN = 2.2  # disk angular speed at one shadow radius, rad/s; Keplerian, so ∝ r^-1.5
GRAVITY = 170.0  # G·M in shadow radii³/s²: an orbit at 3 shadow radii takes about 2.5 s
DRAG = 0.3  # 1/s: stardust from the edge of the screen falls in within about 4 s
MAX_AGE = 12.0  # seconds before a particle fades out, in case it never falls in
MAX_PARTICLES = 600
STAR_DENSITY = 0.012  # chance that a pixel of the unbent sky holds a star

# Disk temperature, from the inner edge (white-hot) to the outer edge (deep red).
_RAMP: list[tuple[float, RGB]] = [
    (0.0, (255.0, 250.0, 235.0)),
    (0.25, (255.0, 214.0, 140.0)),
    (0.5, (250.0, 150.0, 60.0)),
    (0.75, (200.0, 80.0, 25.0)),
    (1.0, (110.0, 30.0, 10.0)),
]
_STAR_TINTS: list[RGB] = [
    (255.0, 255.0, 255.0),
    (200.0, 215.0, 255.0),
    (255.0, 230.0, 200.0),
    (235.0, 240.0, 255.0),
]


@dataclass(slots=True)
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    tx: float  # where it was one frame ago, for the trail
    ty: float
    age: float = 0.0


@dataclass(slots=True)
class _DiskPixel:
    index: int
    band: float  # the ring's middle radius, in shadow radii
    phi: float  # angle around the disk
    omega: float  # the ring's angular speed, rad/s
    color: RGB


@dataclass(slots=True)
class _Star:
    index: int
    color: RGB
    phase: float
    rate: float


@dataclass
class Scene:
    """One terminal size's worth of black hole: precomputed layers, plus live stardust."""

    cols: int
    rows: int
    title: str = ""
    hint: str = ""
    rng: random.Random = field(default_factory=random.Random)
    ambient: float = 3.0  # stardust per second that drifts in from the edges by itself
    particles: list[Particle] = field(default_factory=list[Particle])
    removed: dict[str, int] = field(
        default_factory=lambda: {"swallowed": 0, "escaped": 0, "expired": 0, "dropped": 0}
    )

    def __post_init__(self) -> None:
        self.w, self.h = max(1, self.cols), max(2, 2 * self.rows)
        self.cx, self.cy = self.w / 2, self.h / 2
        self.r_sh = max(1.5, SHADOW * min(self.w, self.h))
        self.reach = math.hypot(self.w, self.h) / 2  # from the centre to a corner
        self.gm = GRAVITY * self.r_sh**3
        self._ambient_due = 0.0
        self.r2: list[float] = []  # squared distance of each pixel from the centre
        self.static: list[RGB] = []  # glow, photon ring and shadow; never changes
        self.stars: list[_Star] = []
        self.disk: list[_DiskPixel] = []
        self._build()

    # -- precomputed layers

    def _build(self) -> None:
        lens2 = (LENS * self.r_sh) ** 2
        ring_width = max(0.6, 0.06 * self.r_sh)
        for py in range(self.h):
            for px in range(self.w):
                index = len(self.static)
                dx, dy = px + 0.5 - self.cx, py + 0.5 - self.cy
                r2 = dx * dx + dy * dy
                r = math.sqrt(r2)
                self.r2.append(r2)
                u = r / self.r_sh
                if u < 1:
                    self.static.append(BLACK)
                else:
                    glow = 0.2 * math.exp(-(u - 1) / 1.3)
                    ring = 0.9 * math.exp(-(((r - 1.05 * self.r_sh) / ring_width) ** 2))
                    self.static.append(
                        (
                            255 * glow + 255 * ring,
                            120 * glow + 230 * ring,
                            45 * glow + 190 * ring,
                        )
                    )
                    self._add_star(index, dx, dy, r2, lens2)
                self._add_disk(index, dx, dy, r, u)

    def _add_star(self, index: int, dx: float, dy: float, r2: float, lens2: float) -> None:
        """Sample the unbent sky where this pixel's light came from: θ - θE²/θ."""
        k = 1 - lens2 / r2
        h = _hash(math.floor(self.cx + dx * k), math.floor(self.cy + dy * k))
        if h % 10_000 >= STAR_DENSITY * 10_000:
            return
        magnification = min(3.0, 1 / max(1e-6, abs(1 - (lens2 / r2) ** 2)))
        brightness = 200 * (0.15 + 0.85 * ((h >> 8) & 255) / 255) ** 3 * magnification
        tint = _STAR_TINTS[(h >> 16) & 3]
        color = (tint[0] * brightness / 255, tint[1] * brightness / 255, tint[2] * brightness / 255)
        phase = ((h >> 4) & 1023) / 1023 * math.tau
        self.stars.append(_Star(index, color, phase, 1.5 + ((h >> 20) & 15) / 5))

    def _add_disk(self, index: int, dx: float, dy: float, r: float, u: float) -> None:
        # The disk itself: its near half (lower on screen) passes in front of the shadow,
        # its far half is hidden behind it.
        rho = math.hypot(dx, dy / FLATTEN) / self.r_sh
        if DISK_IN <= rho <= DISK_OUT and (dy > 0 or u >= 1):
            phi = math.atan2(dy / FLATTEN, dx)
            self._disk_pixel(index, rho, phi, _disk_color(rho, math.cos(phi), 1.0))
        # The far half again, bent up over the top of the shadow (and, thinner, under it).
        if r > 0 and 1.02 <= u <= 1.85:
            up = max(0.0, -dy / r)
            thickness = 0.3 + 0.55 * up
            depth = (u - 1.02) / thickness
            if depth <= 1:
                rho = DISK_IN + depth * 0.6 * (DISK_OUT - DISK_IN)  # its hot inner part
                phi = math.atan2(dy, dx)
                gain = 0.45 + 0.45 * up
                self._disk_pixel(index, rho, phi, _disk_color(rho, math.cos(phi), gain))

    def _disk_pixel(self, index: int, rho: float, phi: float, color: RGB) -> None:
        width = (DISK_OUT - DISK_IN) / DISK_BANDS
        band = DISK_IN + (min(DISK_BANDS - 1, int((rho - DISK_IN) / width)) + 0.5) * width
        self.disk.append(_DiskPixel(index, band, phi, SPIN * band**-1.5, color))

    # -- stardust

    def orbit_speed(self, r: float) -> float:
        """The speed of a circular orbit at `r` pixels from the centre."""
        return math.sqrt(self.gm / max(r, 1e-6))

    def spawn(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0, count: int = 2) -> None:
        """Stardust at pixel (x, y): slower than a circular orbit, plus some of (vx, vy)."""
        dx, dy = x - self.cx, y - self.cy
        r = math.hypot(dx, dy)
        if r < 1.05 * self.r_sh:
            return
        speed = self.orbit_speed(r)
        extra = math.hypot(vx, vy)
        if extra > 0.4 * speed:  # a flick of the mouse should not fling it away
            vx, vy = vx * 0.4 * speed / extra, vy * 0.4 * speed / extra
        for _ in range(count):
            s = speed * self.rng.uniform(0.65, 0.95)
            px, py = x + self.rng.uniform(-1, 1), y + self.rng.uniform(-1, 1)
            # (-dy, dx) runs the same way round as the disk.
            self.particles.append(Particle(px, py, -dy / r * s + vx, dx / r * s + vy, px, py))
        overflow = len(self.particles) - MAX_PARTICLES
        if overflow > 0:
            del self.particles[:overflow]
            self.removed["dropped"] += overflow

    def step(self, dt: float, fall: float = 0.0) -> None:
        """Move the stardust on by `dt` seconds. `fall` (0 to 1) pulls everything in."""
        self._spawn_ambient(dt)
        gm = self.gm * (1 + 12 * fall)
        n = max(1, math.ceil(dt / 0.008))
        h = dt / n
        damp = math.exp(-(DRAG + 3 * fall) * h)
        shadow2 = self.r_sh**2
        far2 = (2 * self.reach) ** 2
        alive: list[Particle] = []
        for p in self.particles:
            p.tx, p.ty = p.x, p.y
            p.age += dt
            fate = None
            for _ in range(n):
                dx, dy = p.x - self.cx, p.y - self.cy
                r2 = dx * dx + dy * dy
                if r2 < shadow2:
                    fate = "swallowed"
                    break
                a = -gm / (r2 * math.sqrt(r2))
                p.vx = (p.vx + a * dx * h) * damp
                p.vy = (p.vy + a * dy * h) * damp
                p.x += p.vx * h
                p.y += p.vy * h
            if fate is None:
                r2 = (p.x - self.cx) ** 2 + (p.y - self.cy) ** 2
                if r2 < shadow2:
                    fate = "swallowed"
                elif r2 > far2:
                    fate = "escaped"
                elif p.age >= MAX_AGE:
                    fate = "expired"
            if fate is None:
                alive.append(p)
            else:
                self.removed[fate] += 1
        self.particles = alive

    def _spawn_ambient(self, dt: float) -> None:
        self._ambient_due += dt * self.ambient
        while self._ambient_due >= 1:
            self._ambient_due -= 1
            angle = self.rng.uniform(0, math.tau)
            r = self.reach * self.rng.uniform(0.7, 1.0)
            self.spawn(self.cx + r * math.cos(angle), self.cy + r * math.sin(angle), count=1)

    # -- one frame

    def compose(self, t: float, fall: float = 0.0) -> list[RGB]:
        """Every pixel's color at time `t`, row by row."""
        buf = self.static.copy()
        for star in self.stars:
            k = 0.7 + 0.3 * math.sin(star.rate * t + star.phase)
            r, g, b = buf[star.index]
            sr, sg, sb = star.color
            buf[star.index] = (r + sr * k, g + sg * k, b + sb * k)
        for d in self.disk:
            a = d.phi - d.omega * t
            texture = (
                0.75
                + 0.17 * math.sin(5 * a + 3.1 * d.band)
                + 0.08 * math.sin(13 * a - 7.3 * d.band + 1.7)
            )
            r, g, b = buf[d.index]
            cr, cg, cb = d.color
            buf[d.index] = (r + cr * texture, g + cg * texture, b + cb * texture)
        self._draw_particles(buf)
        if fall > 0:  # the shadow swells until it covers the screen
            edge = self.r_sh + (1.05 * self.reach - self.r_sh) * min(1.0, fall) ** 3
            edge2 = edge * edge
            for i, r2 in enumerate(self.r2):
                if r2 < edge2:
                    buf[i] = BLACK
        return buf

    def _draw_particles(self, buf: list[RGB]) -> None:
        hot = 2.5 * self.orbit_speed(3 * self.r_sh)
        for p in self.particles:
            fade = min(1.0, p.age / 0.25, MAX_AGE - p.age)
            if fade <= 0:
                continue
            heat = min(1.0, math.hypot(p.vx, p.vy) / hot)  # amber when slow, white when fast
            color = (255 * fade, (160 + 80 * heat) * fade, (70 + 185 * heat) * fade)
            self._add(buf, p.x, p.y, color, 1.0)
            self._add(buf, (p.x + p.tx) / 2, (p.y + p.ty) / 2, color, 0.5)
            self._add(buf, p.tx, p.ty, color, 0.25)

    def _add(self, buf: list[RGB], x: float, y: float, color: RGB, k: float) -> None:
        ix, iy = math.floor(x), math.floor(y)
        if 0 <= ix < self.w and 0 <= iy < self.h:
            i = iy * self.w + ix
            r, g, b = buf[i]
            buf[i] = (r + color[0] * k, g + color[1] * k, b + color[2] * k)

    def text(self, t: float, fall: float = 0.0) -> dict[tuple[int, int], tuple[str, RGB]]:
        """The title and the hint, centred near the bottom: {(row, col): (char, color)}."""
        if self.rows < 6:
            return {}
        fade = max(0.0, 1 - fall)
        pulse = (0.65 + 0.35 * math.sin(2.4 * t)) * fade
        lines = [
            (self.rows - 3, self.title, (235 * fade, 235 * fade, 240 * fade)),
            (self.rows - 2, self.hint, (200 * pulse, 200 * pulse, 210 * pulse)),
        ]
        cells: dict[tuple[int, int], tuple[str, RGB]] = {}
        for row, line, color in lines:
            line = line[: self.cols]
            left = (self.cols - len(line)) // 2
            for col, char in enumerate(line, start=left):
                cells[(row, col)] = (char, color)
        return cells


def _hash(x: int, y: int) -> int:
    h = (x * 374_761_393 + y * 668_265_263) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1_274_126_177) & 0xFFFFFFFF
    return h ^ (h >> 16)


def _disk_color(rho: float, cos_phi: float, gain: float) -> RGB:
    """The disk's color at radius `rho`. The left side moves toward the viewer, so Doppler
    beaming makes it brighter and hotter."""
    t = (rho - DISK_IN) / (DISK_OUT - DISK_IN)
    beta = math.sqrt(DISK_IN / rho)  # orbital speed, 1 at the inner edge
    boost = (1 - 0.3 * beta * cos_phi) ** 3
    heat = min(1.0, max(0.0, t + 0.18 * beta * cos_phi))
    fade = 1 - _smoothstep(0.65, 1.0, t)  # brightest at the inner edge, faint at the outer
    r, g, b = _ramp(heat)
    k = boost * fade * gain
    return (r * k, g * k, b * k)


def _ramp(t: float) -> RGB:
    for (t0, c0), (t1, c1) in itertools.pairwise(_RAMP):
        if t <= t1:
            f = (t - t0) / (t1 - t0)
            return (
                c0[0] + (c1[0] - c0[0]) * f,
                c0[1] + (c1[1] - c0[1]) * f,
                c0[2] + (c1[2] - c0[2]) * f,
            )
    return _RAMP[-1][1]


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    f = min(1.0, max(0.0, (x - edge0) / (edge1 - edge0)))
    return f * f * (3 - 2 * f)


# --- drawing ---------------------------------------------------------------------------

_SYNC_ON, _SYNC_OFF = "\x1b[?2026h", "\x1b[?2026l"  # terminals without it ignore it
_TOP_HALF = "▀"


def render(scene: Scene, t: float, *, truecolor: bool, fall: float = 0.0) -> str:
    """One frame: every row, from its first column, with the text drawn over it."""
    pixels = scene.compose(t, fall)
    cells = scene.text(t, fall)
    key = _truecolor_key if truecolor else _cube_key
    w = scene.w
    out = [_SYNC_ON]
    for row in range(scene.rows):
        out.append(f"\x1b[{row + 1};1H")
        fg = bg = ""
        for col in range(scene.cols):
            top, bottom = pixels[2 * row * w + col], pixels[(2 * row + 1) * w + col]
            want_fg = fg
            if (row, col) in cells:
                char, ink = cells[(row, col)]
                want_fg = key(ink)
                want_bg = key(tuple(0.175 * (a + b) for a, b in zip(top, bottom, strict=True)))
            else:
                top_key, bottom_key = key(top), key(bottom)
                if top_key == bottom_key:
                    char, want_bg = " ", top_key
                else:
                    char, want_fg, want_bg = _TOP_HALF, top_key, bottom_key
            if want_bg != bg:
                out.append(f"\x1b[48;{want_bg}m")
                bg = want_bg
            if want_fg != fg:
                out.append(f"\x1b[38;{want_fg}m")
                fg = want_fg
            out.append(char)
        out.append("\x1b[0m")
    out.append(_SYNC_OFF)
    return "".join(out)


def _truecolor_key(color: tuple[float, ...]) -> str:
    r, g, b = (min(255, int(v)) for v in color)
    return f"2;{r};{g};{b}"


def _cube_key(color: tuple[float, ...]) -> str:
    """The nearest color in the xterm 256-color cube."""
    r, g, b = (_cube_level(v) for v in color)
    return f"5;{16 + 36 * r + 6 * g + b}"


def _cube_level(v: float) -> int:
    v = int(v)
    if v < 48:
        return 0
    if v < 115:
        return 1
    return min(5, (v - 35) // 40)


def supports_truecolor() -> bool:
    return os.environ.get("COLORTERM", "").lower() in {"truecolor", "24bit"}


# --- the terminal ----------------------------------------------------------------------

# Alternate screen, hidden cursor, no autowrap, mouse reports for buttons, drags and
# hovering, in SGR encoding. _LEAVE undoes each one.
_ENTER = "\x1b[?1049h\x1b[?25l\x1b[?7l\x1b[?1002h\x1b[?1003h\x1b[?1006h\x1b[2J"
_LEAVE = "\x1b[?1006l\x1b[?1003l\x1b[?1002l\x1b[?7h\x1b[?25h\x1b[0m\x1b[?1049l"

FRAME_S = 1 / 30
FALL_S = 0.8
DRAIN_S = 0.08  # mouse reports already on their way when tracking is switched off


class Terminal(Protocol):
    def size(self) -> tuple[int, int]:
        """(columns, rows)."""
        ...

    def read(self, timeout: float) -> bytes:
        """Whatever the keyboard and mouse sent, waiting up to `timeout` s; b"" if nothing."""
        ...

    def write(self, text: str) -> None: ...

    def raw(self) -> AbstractContextManager[None]:
        """Keys arrive one at a time and are not echoed. Ctrl-C still interrupts."""
        ...


class PosixTerminal:
    """The real terminal on stdin and stdout."""

    def __init__(self) -> None:
        self.fd_in = sys.stdin.fileno()
        self.fd_out = sys.stdout.fileno()

    def size(self) -> tuple[int, int]:
        size = os.get_terminal_size(self.fd_out)
        return size.columns, size.lines

    def read(self, timeout: float) -> bytes:
        ready, _, _ = select.select([self.fd_in], [], [], timeout)
        return os.read(self.fd_in, 65_536) if ready else b""

    def write(self, text: str) -> None:
        data = memoryview(text.encode())
        while data:
            data = data[os.write(self.fd_out, data) :]

    @contextmanager
    def raw(self) -> Generator[None]:
        import termios  # POSIX only, so imported here and not when the CLI loads
        import tty

        sys.stdout.flush()
        saved = termios.tcgetattr(self.fd_in)
        tty.setcbreak(self.fd_in)
        try:
            yield
        finally:
            # TCSAFLUSH also drops any input still unread, so none of it reaches the prompt.
            termios.tcsetattr(self.fd_in, termios.TCSAFLUSH, saved)


def run_splash(
    term: Terminal,
    *,
    title: str,
    truecolor: bool | None = None,
    clock: Callable[[], float] = time.monotonic,
    rng: random.Random | None = None,
) -> bool:
    """Show the start page until the user starts (True) or quits (False).

    KeyboardInterrupt passes through, but only after the terminal is put back.
    """
    if truecolor is None:
        truecolor = supports_truecolor()
    with term.raw():
        term.write(_ENTER)
        try:
            return _animate(term, title, truecolor, clock, rng or random.Random())
        finally:
            term.write(_LEAVE)
            # Reports the terminal sent before it read _LEAVE are still arriving. Read and
            # drop them, or they would show up as text like [<35;40;12M in the prompt.
            deadline = clock() + DRAIN_S
            while (left := deadline - clock()) > 0:
                term.read(left)


def _animate(
    term: Terminal,
    title: str,
    truecolor: bool,
    clock: Callable[[], float],
    rng: random.Random,
) -> bool:
    began = last = clock()
    scene: Scene | None = None
    pending = b""
    pressed: tuple[int, int] | None = None  # where a button went down, until it comes up
    cursor: tuple[float, float, float] | None = None  # time and pixel of the last move
    falling_since: float | None = None
    while True:
        now = clock()
        cols, rows = term.size()
        if scene is None or (scene.cols, scene.rows) != (cols, rows):
            scene = Scene(cols, rows, title=title, hint=HINT, rng=rng)
        fall = 0.0
        if falling_since is not None:
            fall = (now - falling_since) / FALL_S
            if fall >= 1:
                return True
        scene.step(min(now - last, 0.1), fall)
        last = now
        term.write(render(scene, now - began, truecolor=truecolor, fall=fall))

        data = term.read(max(0.0, FRAME_S - (clock() - now)))
        if not data and pending == ESC:
            return False  # Esc on its own, not the start of a mouse report
        events, pending = parse_input(pending + data)
        if falling_since is not None:
            continue
        for event in events:
            match event:
                case Quit():
                    return False
                case Start():
                    falling_since = now
                case Press(x, y):
                    pressed = (x, y)
                case Release(x, y):
                    if pressed == (x, y):  # a click, not the end of a drag
                        falling_since = now
                    pressed = None
                case Move(x, y):
                    if pressed is not None and pressed != (x, y):
                        pressed = None
                    px, py = x + 0.5, 2 * y + 1.0
                    vx = vy = 0.0
                    if cursor is not None:
                        gap = max(now - cursor[0], FRAME_S)
                        vx, vy = (px - cursor[1]) / gap, (py - cursor[2]) / gap
                    scene.spawn(px, py, 0.3 * vx, 0.3 * vy)
                    cursor = (now, px, py)
            if falling_since is not None:
                break
