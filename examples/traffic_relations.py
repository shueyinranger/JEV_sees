"""Ask whether each detected person is wearing a high-visibility safety vest."""

from __future__ import annotations

import sys
from pathlib import Path

from jev_sees import Sees

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "assets" / "vest_2.mp4"
# vest_2.mp4 is 30 fps and 398 frames, about 13.3 seconds.
WINDOW = (0, 397)
QUESTION = "Is this person wearing a high-visibility safety vest?"


def main() -> int:
    start, end = WINDOW
    Sees(vision="jev").probability(
        VIDEO,
        QUESTION,
        start=start,
        end=end,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
