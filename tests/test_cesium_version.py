"""One Cesium version, named once, and no build tooling behind it.

The repository had three answers to "which CesiumJS is this?": package.json asked
for `^1.145.0`, a committed node_modules held 1.145.0, and the page that actually
runs pinned 1.144 from the CDN. Only the third one was ever executed. The other
two were drift with a 132 MB footprint -- 3907 tracked files that nothing
imported, which two test files had already had to skip by name in order to scan
the tree.

The CDN URL is the source of truth because it is the build the browser loads.
package.json exists to pin the version that URL must use, and these tests are
what keep the two from separating again.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PACKAGE_JSON = REPO / "package.json"
ASSETS = REPO / "naigos" / "demo" / "assets"
PAGE = ASSETS / "cesium.html"

CDN = re.compile(r"https://cesium\.com/downloads/cesiumjs/releases/([0-9.]+)/([^\"']+)")


def declared_version() -> str:
    return json.loads(PACKAGE_JSON.read_text())["dependencies"]["cesium"]


def cdn_version(npm_version: str) -> str:
    """The release directory on the CDN for an npm version.

    npm requires three components; the CDN publishes `1.145`, not `1.145.0`
    (`/releases/1.145.0/` is a 404). So a `.0` patch is dropped and anything
    else is used verbatim -- Cesium does occasionally ship a patch release, and
    that one would have its own directory.
    """
    return npm_version[:-2] if npm_version.endswith(".0") else npm_version


def test_the_declared_version_is_exact():
    """A caret range means the answer changes on someone else's machine. The
    page hard-codes one build; the manifest has to name that build, not a family
    it belongs to."""
    v = declared_version()
    assert re.fullmatch(r"\d+\.\d+\.\d+", v), f"expected an exact version, got {v!r}"


@pytest.mark.parametrize("asset", sorted(ASSETS.glob("*.html")), ids=lambda p: p.name)
def test_every_cesium_url_uses_the_declared_version(asset):
    urls = CDN.findall(asset.read_text())
    assert urls, f"{asset.name} loads no CesiumJS at all"
    want = cdn_version(declared_version())
    for version, path in urls:
        assert version == want, (
            f"{asset.name} loads {path} from CesiumJS {version}, "
            f"but package.json declares {declared_version()} (CDN: {want})"
        )


def test_the_page_loads_the_library_and_its_stylesheet_from_the_same_build():
    """A widgets.css from a different release than Cesium.js is a class of bug
    that shows up as subtly broken widget chrome and nothing else."""
    text = PAGE.read_text()
    scripts = re.findall(r'<script[^>]*src="([^"]+)"', text)
    styles = re.findall(r'<link[^>]*href="([^"]+)"', text)
    assert len(scripts) == 1, f"expected exactly one external script, got {scripts}"
    assert len(styles) == 1, f"expected exactly one external stylesheet, got {styles}"
    assert CDN.match(scripts[0]) and scripts[0].endswith("/Build/Cesium/Cesium.js")
    assert CDN.match(styles[0]) and styles[0].endswith("/Widgets/widgets.css")
    assert CDN.match(scripts[0]).group(1) == CDN.match(styles[0]).group(1)


def test_the_version_is_named_in_exactly_one_place_per_file():
    """Not a style point: the script and the stylesheet are two URLs, so a
    careless bump updates one and leaves the other. They are checked together
    above; this asserts there is no third copy hiding somewhere."""
    versions = {v for v, _ in CDN.findall(PAGE.read_text())}
    assert versions == {cdn_version(declared_version())}


# --- no bundler, and no vendored copy ---------------------------------------------------


def test_node_modules_is_not_tracked():
    """132 MB of build output that nothing imports. Untracked and ignored."""
    tracked = subprocess.run(
        ["git", "ls-files", "node_modules"], cwd=REPO,
        capture_output=True, text=True, check=True).stdout.split()
    assert not tracked, f"{len(tracked)} node_modules files are still tracked"
    assert "node_modules/" in (REPO / ".gitignore").read_text().splitlines()


def test_nothing_imports_the_package():
    """The CDN build is the one that runs. A module-graph import would be a
    second, silently different copy -- and would need the bundler this project
    deliberately does not have."""
    offenders = []
    for path in sorted((REPO / "naigos").rglob("*")):
        if not path.is_file() or path.suffix not in (".py", ".html", ".js"):
            continue
        text = path.read_text()
        for pattern in (r"""from\s+["']cesium["']""", r"""require\(["']cesium["']\)""",
                        r"""import\(["']cesium["']\)""", r"node_modules/cesium"):
            if re.search(pattern, text):
                offenders.append(f"{path.relative_to(REPO)}: {pattern}")
    assert not offenders, f"the npm package is being imported: {offenders}"


def test_no_build_tooling_was_introduced():
    """The page is a static asset the server substitutes into. Keeping it that
    way is what lets a reader open cesium.html and see the whole thing."""
    pkg = json.loads(PACKAGE_JSON.read_text())
    assert "scripts" not in pkg, "package.json grew a build step"
    assert "devDependencies" not in pkg
    assert set(pkg["dependencies"]) == {"cesium"}
    for config in ("webpack.config.js", "vite.config.js", "rollup.config.js",
                   "tsconfig.json", "esbuild.config.js"):
        assert not (REPO / config).exists(), f"{config} implies a bundler"
