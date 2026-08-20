from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kmagnn.evaluate import evaluate_checkpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained KMAGNN checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device")
    args = parser.parse_args()
    results = evaluate_checkpoint(args.checkpoint, device=args.device)
    print(json.dumps({str(k): v for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()

