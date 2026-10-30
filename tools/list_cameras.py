"""List the cameras that can be opened, with the index to pass to `python -m fingercursor --camera N`.

Device names are shown for the dshow backend when the optional `pygrabber` package is installed.

Examples:
  python tools/list_cameras.py
  python tools/list_cameras.py --backend msmf --max-index 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fingercursor.camera import BACKENDS, device_names, list_cameras  # noqa: E402
from fingercursor.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.toml")
    ap.add_argument("--backend", choices=sorted(BACKENDS), help="capture backend (default: camera.backend in config)")
    ap.add_argument("--max-index", type=int, default=10, help="probe indices 0..N-1 (default: 10)")
    args = ap.parse_args()

    backend = args.backend or load_config(args.config, []).camera.backend
    print(f"probing indices 0..{args.max_index - 1} with backend={backend} ...\n")
    cams = list_cameras(backend, args.max_index)
    if not cams:
        print("no camera found")
        return

    print(f"{'index':>5}  {'size':>10}  {'fps':>5}  {'frame':<6} name")
    for c in cams:
        print(f"{c['index']:>5}  {c['width']}x{c['height']:<5}  {c['fps']:>5g}  "
              f"{'ok' if c['frame_ok'] else 'FAIL':<6} {c['name'] or '-'}")
    if backend == "dshow" and device_names() is None:
        print("\n(device names unavailable: pip install pygrabber)")
    print("\nuse one with:  python -m fingercursor --camera <index>")


if __name__ == "__main__":
    main()
