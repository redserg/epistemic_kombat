"""Launcher for Epistemic Kombat console game."""
from __future__ import annotations

import sys

from main import main


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:
        sys.exit(0)
