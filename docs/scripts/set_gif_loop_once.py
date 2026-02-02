#!/usr/bin/env python3
"""Set GIFs in docs/unit_tests to play once (loop=1) for doc animation behavior."""
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Pillow required: pip install Pillow")

DOCS = Path(__file__).resolve().parent.parent
UNIT_TESTS = DOCS / "unit_tests"

for d in UNIT_TESTS.iterdir():
    if not d.is_dir():
        continue
    gif = d / "alpha.gif"
    if not gif.exists():
        continue
    img = Image.open(gif)
    frames = []
    durations = []
    try:
        while True:
            frames.append(img.copy().convert("P"))
            durations.append(img.info.get("duration", 100))
            img.seek(img.tell() + 1)
    except EOFError:
        pass
    if not frames:
        continue
    frames[0].save(
        gif,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=1,
    )
    print(gif)
