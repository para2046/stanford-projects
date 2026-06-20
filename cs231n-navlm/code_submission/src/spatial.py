"""Spatial queries over the OSM POI table.

Used by GPS recovery (DEV_MANUAL §2.5) to cross-check the VLM's named
place against the OSM POI **nearest to the DINOv2-matched SV pano**.
Point-to-line distance is used for streets / rivers — so a long street
like Bahnhofstrasse is matched correctly anywhere along it, not just
near its centroid.

    idx = build_poi_index(pois)             # one-time, ~1 s
    poi, d_m = nearest_poi_m(lat, lon, idx) # O(log N) per query
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# ── local planar projection — centred on the POI bbox so the whole
#    project area is within a few km of origin; metric distances are
#    accurate to <0.1 % over this scope.
_LAT0 = (config.POI_BBOX[1] + config.POI_BBOX[3]) / 2
_LAT_M = 111_320.0
_LON_M = _LAT_M * math.cos(math.radians(_LAT0))


def _proj_lonlat(x, y, z=None):
    """shapely.ops.transform callback: (lon, lat[, z]) -> metres."""
    if z is None:
        return (x * _LON_M, y * _LAT_M)
    return (x * _LON_M, y * _LAT_M, z)


def build_poi_index(pois):
    """Parse each POI's WKT geometry, project to local metres, index.

    Returns an opaque dict {tree, geoms, pois}. POIs whose geometry
    can't be parsed are silently skipped (logged absence in caller)."""
    from shapely import wkt
    from shapely.ops import transform
    from shapely.strtree import STRtree

    geoms, keep = [], []
    for p in pois:
        wkt_str = p.get("geometry")
        if not wkt_str:
            continue
        try:
            g_m = transform(_proj_lonlat, wkt.loads(wkt_str))
            geoms.append(g_m)
            keep.append(p)
        except Exception:
            continue
    name_to_idx = {p["name"]: i for i, p in enumerate(keep)}
    tree = STRtree(geoms) if geoms else None
    return {"tree": tree, "geoms": geoms, "pois": keep,
            "name_to_idx": name_to_idx}


def poi_geometry_m(name, index):
    """The projected (metres) geometry for a POI by name, or None."""
    i = index["name_to_idx"].get(name)
    return index["geoms"][i] if i is not None else None


def distance_pois_m(name_a, name_b, index):
    """Point-to-geometry distance between two named POIs (metres).
    Uses the polyline/polygon — so 'Bahnhofstrasse' vs 'Badergasse'
    returns the *shortest* distance between the two streets, not the
    centroid-to-centroid distance. inf if either name isn't indexed."""
    ga = poi_geometry_m(name_a, index)
    gb = poi_geometry_m(name_b, index)
    if ga is None or gb is None:
        return float("inf")
    return float(ga.distance(gb))


def nearest_poi_m(lat, lon, index):
    """(poi_dict, distance_metres) for the POI whose projected geometry
    is closest to (lat, lon). (None, inf) if the index is empty."""
    if not index["tree"] or not index["geoms"]:
        return None, float("inf")
    from shapely.geometry import Point
    pt_m = Point(lon * _LON_M, lat * _LAT_M)
    i = int(index["tree"].nearest(pt_m))                # shapely 2.x
    g = index["geoms"][i]
    return index["pois"][i], float(pt_m.distance(g))


def name_matches_poi(query, poi):
    """True iff `query` equals `poi['name']` or any alias, after
    lower-casing and diacritic-folding (so 'Grossmunster' matches
    'Grossmünster', 'ETH' matches 'ETH Zürich' alias, etc.). Pure."""
    if not query or not poi:
        return False
    from src.pois import fold
    q = fold(query.lower().strip())
    if q == fold(poi.get("name", "").lower().strip()):
        return True
    for a in poi.get("aliases", []):
        if a and q == fold(str(a).lower().strip()):
            return True
    return False
