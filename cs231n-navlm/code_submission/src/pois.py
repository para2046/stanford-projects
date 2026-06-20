"""POI table extraction from OpenStreetMap (DEV_MANUAL §2.3).

Builds the single v2 POI table — point landmarks + way/area features
(streets, river, lake, bridges) — with real geometry, via osmnx
Overpass over the project bbox.

  clean_name()  — validate / normalise an OSM name (pure, unit-tested)
  extract()     — the osmnx query; writes data/cities/zurich/pois.json

    python -m src.pois            # extract the POI table
"""

import json
import sys
import unicodedata
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

NAME_BLOCKLIST = {"", "unknown", "unnamed", "n/a", "none"}
MIN_NAME_LEN, MAX_NAME_LEN = 3, 40

# point landmarks
POINT_TAGS = {
    "tourism": True, "historic": True, "railway": "station",
    "amenity": ["theatre", "museum", "place_of_worship", "townhall",
                "marketplace", "library"],
}
# way / area features — kept WITH geometry
WAY_TAGS = {
    "highway": ["primary", "secondary", "tertiary", "residential",
                "pedestrian", "living_street"],
    "waterway": "river", "natural": "water", "man_made": "bridge",
}
# OSM tags carrying alternative names for the same place.
# English / German only — Chinese names are not collected as aliases.
ALIAS_TAGS = ["alt_name", "short_name", "official_name", "loc_name",
              "name:en", "name:de"]

# OSM tag keys that say what a POI *is*, in priority order.
OSM_KIND_KEYS = ["tourism", "historic", "amenity", "railway", "man_made",
                 "waterway", "natural", "leisure", "highway", "place"]


def osm_kind(row):
    """The POI's primary OSM tag as 'key=value' (e.g. 'tourism=museum',
    'amenity=place_of_worship', 'highway=primary'), or '' if none.
    `row`: a dict-like of OSM tags. Pure — unit-tested."""
    for k in OSM_KIND_KEYS:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return f"{k}={v.strip()}"
    return ""


# OSM tag -> human-readable descriptor (for the spoken answers)
OSM_KIND_LABEL = {
    "tourism=attraction": "a landmark", "tourism=viewpoint": "a viewpoint",
    "tourism=museum": "a museum", "tourism=gallery": "an art gallery",
    "tourism=hotel": "a hotel", "tourism=artwork": "a public artwork",
    "tourism=zoo": "a zoo", "tourism=theme_park": "a theme park",
    "historic=castle": "a castle", "historic=monument": "a monument",
    "historic=memorial": "a memorial",
    "amenity=place_of_worship": "a church", "amenity=townhall": "the town hall",
    "amenity=theatre": "a theatre", "amenity=cinema": "a cinema",
    "amenity=museum": "a museum", "amenity=library": "a library",
    "amenity=marketplace": "a market square",
    "amenity=university": "a university", "amenity=college": "a college",
    "railway=station": "a railway station", "man_made=bridge": "a bridge",
    "waterway=river": "a river", "natural=water": "a body of water",
    "leisure=park": "a park", "leisure=garden": "a garden",
    "leisure=stadium": "a stadium", "place=square": "a square",
}
LABEL_BY_KEY = {"highway": "a street", "tourism": "a place of interest",
                "historic": "a historic site", "amenity": "a local amenity",
                "leisure": "a leisure spot", "place": "a place",
                "railway": "a transport stop", "waterway": "a waterway",
                "natural": "a natural feature", "man_made": "a structure"}


def kind_label(osm_kind_tag):
    """Human-readable descriptor for a POI's OSM tag — e.g.
    'amenity=theatre' -> 'a theatre'. Pure — unit-tested."""
    if not osm_kind_tag:
        return "a place"
    if osm_kind_tag in OSM_KIND_LABEL:
        return OSM_KIND_LABEL[osm_kind_tag]
    return LABEL_BY_KEY.get(osm_kind_tag.split("=", 1)[0], "a place")


def clean_name(name):
    """Return a normalised POI name, or None if it should be dropped.
    Drops blocklisted / too-short / too-long / non-capitalised names."""
    if not name or not isinstance(name, str):
        return None
    n = " ".join(name.split())
    if n.lower() in NAME_BLOCKLIST:
        return None
    if not (MIN_NAME_LEN <= len(n) <= MAX_NAME_LEN):
        return None
    if not n[0].isupper():
        return None
    return n


def collect_aliases(name, tags):
    """Deduped alias list for a POI — the canonical name plus the OSM
    alternative-name tag values (ALIAS_TAGS, English/German only).
    `tags`: a dict-like of OSM tags. Pure — unit-tested."""
    cands = [name]
    for t in ALIAS_TAGS:
        v = tags.get(t)
        if isinstance(v, str) and v.strip():     # skip NaN / missing tags
            cands += [s.strip() for s in v.split(";")]
    seen, out = set(), []
    for c in cands:
        c = " ".join((c or "").split())
        if not c or c.lower() in NAME_BLOCKLIST:
            continue
        if c.lower() not in seen:
            seen.add(c.lower())
            out.append(c)
    return out


def fold(s):
    """Lowercase + strip diacritics — for lenient name matching, so
    'Grossmunster' matches 'Grossmünster'. Pure."""
    s = unicodedata.normalize("NFKD", str(s or "").lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def resolve_poi(query, pois):
    """Resolve a place name to a POI via its name + aliases.

    `pois`: list of dicts, each with 'name' and (optional) 'aliases'.
    Match order: exact (diacritic-folded) on name/alias, then substring
    either direction. Returns the POI dict or None. Pure — unit-tested.
    """
    q = fold(" ".join((query or "").split()))
    if not q:
        return None
    for p in pois:                              # exact (diacritic-folded)
        for n in [p["name"]] + list(p.get("aliases", [])):
            if q == fold(n):
                return p
    for p in pois:                              # substring, either direction
        for n in [p["name"]] + list(p.get("aliases", [])):
            nf = fold(n)
            if q in nf or nf in q:
                return p
    return None


def extract(bbox=config.POI_BBOX):
    """Query OSM for point + way/area POIs; write pois.json. Needs osmnx."""
    import osmnx as ox

    w, s, e, n = bbox
    rows = []
    for tags, kind_group in ((POINT_TAGS, "point"), (WAY_TAGS, "way")):
        gdf = ox.features_from_bbox((w, s, e, n), tags=tags)
        for _, row in tqdm(gdf.iterrows(), total=len(gdf),
                           desc=f"[pois] {kind_group}", unit="osm"):
            name = clean_name(row.get("name"))
            if not name:
                continue
            geom = row.geometry
            ok = osm_kind(row)
            desc = row.get("description")
            rows.append({
                "name": name,
                "aliases": collect_aliases(name, row),
                "kind_group": kind_group,
                "osm_kind": ok,
                "kind_label": kind_label(ok),
                "description": desc.strip() if isinstance(desc, str) else "",
                "lat": geom.centroid.y, "lon": geom.centroid.x,
                "geometry": geom.wkt,
            })

    # dedupe by name (keep first)
    seen, uniq = set(), []
    for r in rows:
        if r["name"] not in seen:
            seen.add(r["name"])
            uniq.append(r)

    out = config.CITY_DIR / "pois.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(uniq, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"[pois] {len(uniq)} POIs -> {out}")
    return uniq


def main():
    extract()


if __name__ == "__main__":
    main()
