"""Mirror the Omniverse scene assets (UR3, biped_demo) to a local directory.

The Isaac scene loads three Omniverse S3 assets: the ground environment, the
UR3 and biped_demo.  They are normally served by the Omniverse resolver's
asset cache; when that cache is unavailable (or the resolver's network path is
broken) the scene cannot build.  ``ecg_scene`` prefers a local mirror at
``~/isaac_assets/mirror`` (override with ROBOECG_ASSET_DIR), and this script
fetches it with plain HTTP (the S3 bucket allows listing).

Usage: python3 scripts/mirror_scene_assets.py
"""
import urllib.request, urllib.parse, pathlib
import xml.etree.ElementTree as ET

NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
BASE = "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
PREFIXES = [
    "Assets/Isaac/6.0/Isaac/Robots/UniversalRobots/ur3/",
    "Assets/Isaac/5.0/Isaac/People/Characters/biped_demo/",
]
DEST = pathlib.Path("/home/jxy/isaac_assets/mirror")
total = 0
for prefix in PREFIXES:
    token = None
    while True:
        url = f"{BASE}/?list-type=2&prefix={urllib.parse.quote(prefix)}&max-keys=200"
        if token:
            url += f"&continuation-token={urllib.parse.quote(token)}"
        root = ET.fromstring(urllib.request.urlopen(url, timeout=60).read())
        truncated = root.findtext(f"{NS}IsTruncated") == "true"
        token = root.findtext(f"{NS}NextContinuationToken")
        for content in root.findall(f"{NS}Contents"):
            key = content.findtext(f"{NS}Key")
            size = int(content.findtext(f"{NS}Size") or 0)
            if "/.thumbs/" in key:
                continue
            out = DEST / key
            if out.exists() and out.stat().st_size == size:
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(
                f"{BASE}/{urllib.parse.quote(key)}", out
            )
            total += size
            print(f"{size:>10} {key}", flush=True)
        if not truncated:
            break
print(f"mirrored total {total/1e6:.1f} MB")
