"""Self-contained 3D replay viewer for logged Naigos rollouts.

    python -m naigos.demo.viewer runs/demo/demo.json --open

Writes a single HTML file next to the JSON with the trajectory data inlined, so
it opens straight off the filesystem with no server and no network beyond the
three.js CDN script. Everything it draws -- terrain, threat envelopes, aircraft
tracks, detection state -- comes from `demo.json`, which comes from
`env.rollout`. Nothing is scripted or interpolated: the aircraft are at the
positions they were logged at, and a track that stops stops because that
aircraft was lost on that step.

This is `prompt.md` s10 step 8: a 3D replay of actual logged rollouts, with live
counters, untrained against trained.
"""

from __future__ import annotations

import argparse
import json
import webbrowser
from pathlib import Path

TEMPLATE = (Path(__file__).parent / "assets" / "viewer.html").read_text


def build(data_path: Path, out_path: Path | None = None) -> Path:
    data = json.loads(data_path.read_text())
    out_path = out_path or data_path.with_suffix(".html")
    html = (Path(__file__).parent / "assets" / "viewer.html").read_text()
    # a plain replace, not a format string: the template is full of CSS braces
    html = html.replace("/*__NAIGOS_DATA__*/null", json.dumps(data, separators=(",", ":")))
    out_path.write_text(html)
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="naigos.demo.viewer", description=__doc__)
    ap.add_argument("data", nargs="?", default="runs/demo/demo.json", help="demo.json from naigos.demo.replay")
    ap.add_argument("--out", default=None, help="output .html (default: alongside the JSON)")
    ap.add_argument("--open", action="store_true", help="open it in the default browser")
    a = ap.parse_args(argv)

    src = Path(a.data)
    if not src.exists():
        raise SystemExit(
            f"{src} not found. Generate it first:\n"
            f"  python -m naigos.demo.replay --checkpoint checkpoints/theatre_1000.pkl"
        )
    out = build(src, Path(a.out) if a.out else None)
    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out}  ({size_mb:.1f} MB, self-contained)")
    if a.open:
        webbrowser.open(out.resolve().as_uri())
    else:
        print(f"open it with:  open {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
