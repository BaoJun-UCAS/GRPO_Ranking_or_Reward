#!/usr/bin/env python3
"""Prepare/train/audit/generate/judge a controlled two-arm advantage experiment."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from open_r1.comparison import main

if __name__ == '__main__':
    raise SystemExit(main())
