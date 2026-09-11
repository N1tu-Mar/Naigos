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

``--visual urban-presentation`` exports the same recording with the local city
layer (``naigos.demo.urban``) embedded: OSM building footprints and road
centrelines, extruded over the same DEM, opening on the oblique city camera.
Still credential-free -- the provider path needs a key and is never taken
here -- and still self-contained: the layer is inside the file, so viewing it
makes no request for data. It is a PRESENTATION export (``evidence_grade`` is
False) and it refuses to build without the local cache rather than shipping an
empty city.
"""

from __future__ import annotations

import argparse
import json
import webbrowser
from pathlib import Path

from . import imagery as imagery_mod

ASSETS = Path(__file__).parent / "assets"


STATIC_VISUAL_MODES = ("physics", imagery_mod.URBAN_MODE)


def build(data_path: Path, out_path: Path | None = None, aoi: str | None = None,
          visual_mode: str = "physics", camera: str | None = None,
          atmosphere: str | None = None, ambience: str | None = None,
          ambience_setting: str | None = None, visual_seed: int = 0) -> Path:
    """Render `demo.json` into a self-contained Cesium page.

    The presentation -- atmosphere profile, the opt-in fictional ambience
    stream (generated here, once, from ``visual_seed`` and embedded, so every
    play of the file shows the same effects), the scenario framing and the
    checkpoint disclosure recorded with the rollout -- resolves through the
    same ``naigos.demo.presentation`` the live server uses.
    """
    # Imported here rather than at module scope: `live` pulls in jax and the RL
    # package, and `--help` should not pay for a compiler it will not use.
    from . import camera as camera_mod
    from . import presentation as presentation_mod
    from . import urban as urban_mod
    from .live import render_page, static_payload

    if visual_mode not in STATIC_VISUAL_MODES:
        raise SystemExit(f"a static export is credential-free; --visual must be one of "
                         f"{', '.join(STATIC_VISUAL_MODES)}")
    urban = visual_mode == imagery_mod.URBAN_MODE
    out_path = out_path or (data_path.with_name(data_path.stem + "_urban.html") if urban
                            else data_path.with_suffix(".html"))

    rec = json.loads(data_path.read_text())
    theatre = aoi or rec.get("theatre")
    try:
        presentation = presentation_mod.resolve(
            theatre, visual_mode, atmosphere=atmosphere, ambience=ambience,
            ambience_setting=ambience_setting, visual_seed=visual_seed,
            checkpoint=rec.get("checkpoint"), layout_seed=rec.get("seed"))
    except presentation_mod.PresentationError as e:
        raise SystemExit(f"naigos.demo.viewer: {e}")
    urban_status = urban_mod.load(theatre) if theatre else None
    if urban and not (urban_status and urban_status.available):
        raise SystemExit(
            f"--visual urban-presentation embeds the local city layer, and there is none: "
            f"{urban_status.reason if urban_status else 'the recording names no theatre'}")

    # No credentials, on purpose. See the module docstring.
    visual = imagery_mod.resolve_visual_config(
        visual_mode, ion_token=None, google_api_key=None, imagery="osm",
        local_urban=bool(urban and urban_status.available))
    html = render_page(visual, ion_token=None, google_api_key=None)

    # The city focus is passed in every mode (so --camera urban-overview frames
    # the same place in a physics export); the layer itself only in urban mode.
    payload = static_payload(data_path, aoi=aoi, urban_status=urban_status,
                             camera_key=camera_mod.preset_key(camera, urban), embed_urban=urban,
                             presentation=presentation)
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
    ap.add_argument("--visual", choices=STATIC_VISUAL_MODES, default="physics",
                    help="physics (default, evidence-grade) or urban-presentation (embeds the "
                         "local OSM city layer; presentation only)")
    ap.add_argument("--camera", default=None,
                    help="opening camera: terrain-overview, urban-overview, street-canyon, "
                         "follow-aircraft, analysis-topdown, or a city's coastal-corridor / "
                         "valley-overview")
    ap.add_argument("--atmosphere", default="auto",
                    help="presentation-only atmosphere profile (auto: the theatre's own under "
                         "urban-presentation, neutral under physics)")
    ap.add_argument("--ambience", default="off", choices=("off", "conflict_ambience"),
                    help="embed the FICTIONAL ambience VFX stream (presentation modes only)")
    ap.add_argument("--ambience-setting", default="sustained", choices=("sparse", "sustained"))
    ap.add_argument("--visual-seed", type=int, default=0,
                    help="seed for the ambience stream; the same seed embeds the same effects")
    a = ap.parse_args(argv)

    src = Path(a.data)
    if not src.exists():
        raise SystemExit(
            f"{src} not found. Generate it first:\n"
            f"  python -m naigos.demo.replay --checkpoint checkpoints/theatre_1000.pkl"
        )
    out = build(src, Path(a.out) if a.out else None, aoi=a.aoi, visual_mode=a.visual,
                camera=a.camera, atmosphere=a.atmosphere, ambience=a.ambience,
                ambience_setting=a.ambience_setting, visual_seed=a.visual_seed)
    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out}  ({size_mb:.1f} MB, self-contained)")
    if a.visual == imagery_mod.URBAN_MODE:
        print("terrain: the simulation's own heightmap, embedded -- the ground on screen is the "
              "ground the model used; the buildings standing on it are not")
    else:
        print("terrain: the simulation's own heightmap, embedded -- what occludes on screen "
              "is what occluded in the model")
    if a.visual == imagery_mod.URBAN_MODE:
        print(f"urban: OSM buildings and roads embedded (ODbL, (c) OpenStreetMap contributors). "
              f"PRESENTATION ONLY -- {imagery_mod.BUILDING_LOS_NOTE}")
    print("imagery: OpenStreetMap, keyless. No credential is written into the artifact.")
    if a.ambience != "off":
        print(f"ambience: conflict_ambience ({a.ambience_setting}, visual seed {a.visual_seed}) "
              "embedded -- FICTIONAL presentation VFX, not simulated events")
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
