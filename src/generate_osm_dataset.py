import os
import requests
import io
import math
from PIL import Image, ImageDraw
import osmnx as ox
import geopandas as gpd
from shapely.geometry import box
from tqdm import tqdm


CITIES = {
    "kyoto": "Kyoto, Japan",
    "osaka": "Osaka, Japan",
    "sapporo": "Sapporo, Japan",
    "fukuoka": "Fukuoka, Japan"
}

OUT_DIR = "../data_osm/train"
os.makedirs(f"{OUT_DIR}/image", exist_ok=True)
os.makedirs(f"{OUT_DIR}/mask", exist_ok=True)


TILE_URL = "https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg"
ZOOM = 18  
PATCH_SIZE = 512  
PATCH_M = 300     
ox.settings.overpass_endpoint = "https://overpass.kumi.systems/api/interpreter"



def download_tile(x, y, z):
    url = TILE_URL.format(z=z, x=x, y=y)
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        return None
    return Image.open(io.BytesIO(r.content)).convert("RGB")

def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)

def draw_mask(gdf, bounds, out_size=PATCH_SIZE):
    minx, miny, maxx, maxy = bounds
    mask = Image.new("L", (out_size, out_size), 0)
    draw = ImageDraw.Draw(mask)

    sub = gdf.cx[minx:maxx, miny:maxy]
    if sub.empty:
        return mask

    for geom in sub.geometry:
        if geom.is_empty:
            continue
        if geom.geom_type == "Polygon":
            pts = [( (x - minx)/(maxx-minx)*out_size,
                     (maxy - y)/(maxy-miny)*out_size)
                    for x, y in geom.exterior.coords]
            draw.polygon(pts, fill=255)
        elif geom.geom_type == "MultiPolygon":
            for p in geom.geoms:
                pts = [( (x - minx)/(maxx-minx)*out_size,
                         (maxy - y)/(maxy-miny)*out_size)
                        for x, y in p.exterior.coords]
                draw.polygon(pts, fill=255)
    return mask


for city_tag, city_name in CITIES.items():
    print(f"--- {city_name} ---")

    point = ox.geocode(city_name)
    gdf = ox.features_from_point(point, dist=2000, tags={"building": True})
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    gdf = gdf.to_crs(epsg=4326) 

    bounds = gdf.total_bounds 
    minx, miny, maxx, maxy = bounds
    cx, cy = (minx + maxx)/2, (miny + maxy)/2

    xt, yt = deg2num(cy, cx, ZOOM)
    tile_range = range(xt-1, xt+2)
    row_range = range(yt-1, yt+2)

    idx = 0
    for x in tqdm(tile_range, desc=f"{city_tag}"):
        for y in row_range:
            img = download_tile(x, y, ZOOM)
            if img is None:
                continue
            lat1, lon1 = num2deg(x, y, ZOOM)
            lat2, lon2 = num2deg(x+1, y+1, ZOOM)
            bounds = (lon1, lat2, lon2, lat1) 

            mask = draw_mask(gdf, bounds)

            base = f"{city_tag}_{idx:03d}.png"
            img.save(f"{OUT_DIR}/image/{base}")
            mask.save(f"{OUT_DIR}/mask/{base}")
            idx += 1

    print(f"Saved {idx} image/mask pairs for {city_tag}")
