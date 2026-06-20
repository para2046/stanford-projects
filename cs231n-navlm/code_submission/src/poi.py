"""The 27 candidate POIs used by the video POI-scan — with GPS + names.

This is the hand-picked tier-1 ("L1") shortlist of iconic Zurich
landmarks + scenery that the VLM POI-scan chooses from. The *names* come
from reference/toolbox/scan_video_pois_multi.py:CANDIDATE_POIS (a
hardcoded list — there is no extraction code, it was hand-curated);
coordinates here are resolved from the curated POI tables.

    python -m src.poi --list      # print the 27 (English / 中文 / GPS)
    python -m src.poi --map       # write viz/poi_candidates_map.html

The map uses an emoji "signature icon" per POI kind.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# (english, 中文, lat, lon, kind)
CANDIDATE_POIS = [
    ("Hauptbahnhof",      "苏黎世中央车站",   47.37802, 8.54023, "station"),
    ("Lindenhof",         "林登霍夫山丘",     47.37280, 8.54149, "hill"),
    ("Paradeplatz",       "阅兵广场",         47.36953, 8.53866, "square"),
    ("Münsterhof",        "明斯特霍夫广场",   47.37072, 8.54128, "square"),
    ("Fraumünster",       "圣母大教堂",       47.37005, 8.54148, "church"),
    ("Grossmünster",      "大教堂",           47.37018, 8.54425, "church"),
    ("St. Peter",         "圣彼得教堂",       47.37154, 8.54126, "church"),
    ("Bellevueplatz",     "贝尔维尤广场",     47.36695, 8.54513, "square"),
    ("Sechseläutenplatz", "六鸣节广场",       47.36620, 8.54615, "square"),
    ("Bürkliplatz",       "比尔克利广场",     47.36615, 8.54153, "square"),
    ("Quaibrücke",        "码头桥",           47.36593, 8.54367, "bridge"),
    ("Münsterbrücke",     "大教堂桥",         47.36970, 8.54200, "bridge"),
    ("Rathausbrücke",     "市政厅桥",         47.37117, 8.54158, "bridge"),
    ("Rathaus",           "市政厅",           47.37160, 8.54280, "civic"),
    ("Stadthaus",         "市政府大楼",       47.36931, 8.54121, "civic"),
    ("Opernhaus",         "苏黎世歌剧院",     47.36548, 8.54683, "culture"),
    ("Kunsthaus",         "苏黎世美术馆",     47.37021, 8.54793, "museum"),
    ("Landesmuseum",      "瑞士国家博物馆",   47.37926, 8.54021, "museum"),
    ("Polyterrasse",      "联邦理工观景台",   47.37610, 8.54652, "hill"),
    ("Globus",            "高乐斯百货",       47.37563, 8.54058, "store"),
    ("Jelmoli",           "耶尔莫利百货",     47.37480, 8.53846, "store"),
    ("Bahnhofstrasse",    "班霍夫大街",       47.37367, 8.53924, "street"),
    ("Niederdorfstrasse", "下村街",           47.37318, 8.54417, "street"),
    ("Limmatquai",        "利马特河滨道",     47.37200, 8.54330, "street"),
    ("Rennweg",           "伦韦格街",         47.37326, 8.54000, "street"),
    ("Limmat river",      "利马特河",         47.37100, 8.54200, "water"),
    ("Lake Zurich",       "苏黎世湖",         47.36500, 8.54500, "water"),
]

KIND_ICON = {
    "station": "🚉", "hill": "🌳", "square": "🟦", "church": "⛪",
    "bridge": "🌉", "civic": "🏛️", "culture": "🎭", "museum": "🖼️",
    "store": "🏬", "street": "🛣️", "water": "🌊",
}


def print_list():
    print(f"{len(CANDIDATE_POIS)} candidate POIs "
          f"(from scan_video_pois_multi.py:CANDIDATE_POIS)\n")
    print(f"  {'#':>2}  {'English':18s} {'中文':12s} {'kind':8s} GPS")
    for i, (en, zh, lat, lon, kind) in enumerate(CANDIDATE_POIS, 1):
        print(f"  {i:>2}  {en:18s} {zh:12s} {kind:8s} {lat:.5f}, {lon:.5f}")


def write_map():
    config.VIZ_DIR.mkdir(parents=True, exist_ok=True)
    out = config.VIZ_DIR / "poi_candidates_map.html"
    clat = sum(p[2] for p in CANDIDATE_POIS) / len(CANDIDATE_POIS)
    clon = sum(p[3] for p in CANDIDATE_POIS) / len(CANDIDATE_POIS)

    markers = []
    for en, zh, lat, lon, kind in CANDIDATE_POIS:
        icon = KIND_ICON.get(kind, "📍")
        markers.append(
            f"L.marker([{lat},{lon}],{{icon:L.divIcon({{"
            f"html:'<div style=\"font-size:22px\">{icon}</div>',"
            f"className:'',iconSize:[24,24]}})}})"
            f".bindPopup('<b>{en}</b><br>{zh}<br><i>{kind}</i>').addTo(map);"
        )
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>NavLM — 27 candidate POIs</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body,#map{{height:100%;margin:0}}</style></head><body>
<div id="map"></div><script>
var map=L.map('map').setView([{clat},{clon}],15);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
 {{maxZoom:19,attribution:'© OpenStreetMap'}}).addTo(map);
{chr(10).join(markers)}
</script></body></html>"""
    out.write_text(html, encoding="utf-8")
    print(f"map ({len(CANDIDATE_POIS)} POIs) -> {out}")


def main():
    ap = argparse.ArgumentParser(description="candidate POIs — list / map")
    ap.add_argument("--list", action="store_true", help="print the 27 POIs")
    ap.add_argument("--map", action="store_true", help="write the HTML map")
    # parse_known_args so a stray '/' separator (from copy-pasting
    # `--list / --map`) is ignored rather than erroring out
    args, _ = ap.parse_known_args()
    if args.map:
        write_map()
    if args.list or not args.map:
        print_list()


if __name__ == "__main__":
    main()
