# File-defined theatres

One JSON per theatre, named `<aoi>.json`. `naigos.research.aoi` discovers every
file here at import and registers it beside the built-in AOIs, so adding a
theatre never edits a shared table.

A file-defined theatre is built with the same research agent:

    uv run naigos-research --aoi <aoi>

but in isolation: it writes only `components/aoi/<aoi>/` and
`docs/theatres/<aoi>/DATA.md`, never the top-level `components/*.json` or
`docs/DATA.md` that the default theatre owns.

| key | meaning |
| --- | --- |
| `name` | the AOI name; must equal the file name |
| `west` `south` `east` `north` | WGS84 degrees; the fingerprint hashes these |
| `country` | ISO code |
| `dem_source` | `copernicus_dem` outside CONUS |
| `rationale` | why this terrain |
| `bounds_policy` | how the box was chosen and what it deliberately leaves out |
| `scenario` | the framing shown wherever the theatre is drawn |
| `exclude_military_airfields` | drop airfields whose published name marks them military |
| `protected_zones` | boxes the project stays out of: `name`, box, `buffer_m`, `policies`, `reason` |

Zone policies: `extraction` (no visual geometry kept), `airfield` (no start
point), `camera` (no preset sits in, aims at or frames it), `ambience` (no
presentation effect originates there), `render_cutout` (no imagery or provider
3D tiles drawn over it). Each consumer enforces only the policies a zone names.

Every theatre here is a notional simulation envelope, not a digital twin: threat
layouts are procedural and random per seed, and no file here describes any real
force, facility or condition.
