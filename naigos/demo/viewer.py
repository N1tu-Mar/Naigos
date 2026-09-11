"""Static 3D replay export: the live globe, in one file, with no server.

    python -m naigos.demo.viewer runs/demo/demo.json --open

Writes a single HTML file next to the JSON. It is `assets/cesium.html` -- the
same page `naigos.demo.live` serves -- with the three routes it would otherwise
fetch inlined at the `__EMBED__` substitution point. So the artifact draws the
globe the live viewer draws: the simulation's own terrain, threat envelopes,
LOS rays with their refraction and pinch points, the evidence banner, and real
georeferencing.

This replaces a separate three.js viewer that rendered the same recordings in a
local ENU box. Two renderers meant two visual languages, two sets of bugs, and
only one of them georeferenced -- and the one shipped as the portfolio artifact
was the weaker one. Everything it drew, `cesium.html` already drew better.

Nothing here is scripted or interpolated by this module: the aircraft are at the
positions `env.rollout` logged, and a track that stops stops because that
aircraft was lost on that step. The page does interpolate *between* logged
samples for display -- linearly, at the simulation timestep, with availability
ending where the track ends. See the replay block in `assets/cesium.html`.

The export is deliberately CREDENTIAL-FREE and evidence-grade. It resolves the
visual config with no ion token and no Google key, which lands on keyless
OpenStreetMap over the simulation's own DEM: `evidence_grade` is True, nothing
token-shaped can reach a file that gets committed, and the artifact renders the
same for everyone who opens it.
"""

from __future__ import annotations

import argparse
import json
import webbrowser
from pathlib import Path

from . import imagery as imagery_mod

ASSETS = Path(__file__).parent / "assets"


def build(data_path: Path, out_path: Path | None = None, aoi: str | None = None) -> Path:
    """Render `demo.json` into a self-contained Cesium page."""
    # Imported here rather than at module scope: `live` pulls in jax and the RL
    # package, and `--help` should not pay for a compiler it will not use.
    from .live import render_page, static_payload

    out_path = out_path or data_path.with_suffix(".html")

    # No credentials, on purpose. See the module docstring.
    visual = imagery_mod.resolve_visual_config(
        "physics", ion_token=None, google_api_key=None, imagery="osm")
    html = render_page(visual, ion_token=None, google_api_key=None)

    payload = static_payload(data_path, aoi=aoi)
    marker = "/*__EMBED__*/null"
    if marker not in html:
        raise SystemExit(f"{ASSETS / 'cesium.html'} has no {marker} substitution point")
    html = html.replace(marker, json.dumps(payload, separators=(",", ":")))

    out_path.write_text(html)
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="naigos.demo.viewer", description=__doc__)
    ap.add_argument("data", nargs="?", default="runs/demo/demo.json",
                    help="demo.json from naigos.demo.replay")
    ap.add_argument("--out", default=None, help="output .html (default: alongside the JSON)")
    ap.add_argument("--aoi", default=None,
                    help="refuse to export unless the recording was made on this theatre")
    ap.add_argument("--open", action="store_true",
                    help="serve it on 127.0.0.1 and open it in the default browser")
    ap.add_argument("--port", type=int, default=8766, help="local port for --open")
    a = ap.parse_args(argv)

    src = Path(a.data)
    if not src.exists():
        raise SystemExit(
            f"{src} not found. Generate it first:\n"
            f"  python -m naigos.demo.replay --checkpoint checkpoints/theatre_1000.pkl"
        )
    out = build(src, Path(a.out) if a.out else None, aoi=a.aoi)
    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out}  ({size_mb:.1f} MB, self-contained)")
    print("terrain: the simulation's own heightmap, embedded -- what occludes on screen "
          "is what occluded in the model")
    print("imagery: OpenStreetMap, keyless. No credential is written into the artifact.")
    if a.open:
        serve(out, a.port)
    else:
        print(f"view it with:  python -m naigos.demo.viewer {src} --open")
        print("  (or any static server: python -m http.server -d "
              f"{out.parent} -- then /{out.name})")
    return 0


def serve(page: Path, port: int) -> None:
    """Serve the artifact's directory on loopback and open it.

    Not `file://`: CesiumJS builds the terrain mesh and every static geometry in
    web workers, and a page opened from the filesystem has an opaque origin the
    browser will not start them for. The globe then never appears -- the
    envelopes and aircraft float over black space -- which is what opening the
    artifact directly did before this existed. Loopback only; nothing is
    exposed beyond this machine.
    """
    import functools
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(page.parent.resolve()))
    handler.log_message = lambda *a: None
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{port}/{page.name}"
    print(f"\n{url}   (static replay, served locally; ctrl-c to stop)")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
