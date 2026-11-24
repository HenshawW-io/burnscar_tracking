from pystac_client import Client
import stackstac
import osmnx as ox
import geopandas as gpd
import rioxarray # noqa: F401
import numpy as np
from shapely import box
from shapely.affinity import scale
import xarray as xr
import time
import datetime
import rasterio.features
import shapely
import argparse
from pathlib import Path

URL = 'https://earth-search.aws.element84.com/v1'
COLLECTION = 'sentinel-2-c1-l2a'
SCL_CLASSES = [2, 4, 5, 7]
SQ_M_PER_SQ_MI = 2589988
SPECKLE_SIZE = 25
DNBR_THRESHOLDS = [
            #   < -0.25         = 0
    -0.25,  #   -0.25 to -0.10  = 1
    -0.10,  #   -0.10 to 0.10   = 2
    0.10,   #   0.10 to 0.27    = 3
    0.27,   #   0.27 to 0.44    = 4
    0.44,   #   0.44 to 0.66    = 5
    0.66    #   >= 0.66         = 6
]

'''
Burn severity index table (Key & Benson, USGS FIREMON)
--------------------------
dNBR	    Class	Meaning
< -0.25	        1	High regrowth
-0.25 to -0.10	2	Low regrowth
-0.10 to 0.10	3	Unburned
0.10 to 0.27	4	Low severity
0.27 to 0.44	5	Moderate-low severity
0.44 to 0.66	6	Moderate-high severity
>= 0.66	        7	High severity
'''

def load(da, tries=3):
    for attempt in range(tries):
        try:
            return da.compute()
        except RuntimeError as e:
            if attempt == tries - 1:
                raise
            print(f'read failed ({str(e)[:80]}...), retrying')
            time.sleep(5 * (attempt + 1))

def sieve_until_stable(a, size, connectivity=8, max_passes=10, mask=None):
    for _ in range(max_passes):
        out = rasterio.features.sieve(a, size=size, connectivity=connectivity, mask=mask)
        if np.array_equal(out, a):
            return out
        a = out
    return a

def to_cog(da, path, nodata, epsg):
    (da.reset_coords(drop=True) # drop stackstac's extra metadata coords
    .rio.write_crs(f'EPSG:{epsg}')
    .rio.write_nodata(nodata)
    .rio.to_raster(path, driver='COG', compress='DEFLATE'))

def get_bbox(place, zoom):
    # get the boundary polygon of the burn
    aoi = ox.geocoder.geocode_to_gdf(place)
    aoi_geom = aoi.geometry
    bbox = list(scale(box(*aoi_geom[0].bounds), xfact=zoom, yfact=zoom).bounds)
    return bbox

def search_items(bbox, start, end, max_cloud):
    # search the catalog for items over the AOI within the search window
    catalog = Client.open(URL)
    items = catalog.search(
        collections=[COLLECTION],
        bbox=bbox,
        datetime=f'{start}/{end}',
        query={'eo:cloud_cover': {'lt': max_cloud}},
        sortby='properties.eo:cloud_cover',
        limit=200,
    ).item_collection()
    print(f'{len(items)} scene(s) found')
    if len(items) == 0:
        raise ValueError('no scenes found; check --place, the search window, or --max-cloud-cover')
    return items

def get_epsg(items):
    # retrieve the EPSG code for the band assets
    codes = {it.properties['proj:code'] for it in items} # putting them into a set so that only unique elements are stored
    if len(codes) > 1:
        raise ValueError(f'items span multiple UTM zones: {codes}; pick one')
    epsg = int(codes.pop().split(':')[1])   # 32630
    return epsg

def load_cube(items, bbox, epsg):
    cube = stackstac.stack(
        items, 
        assets=['nir08', 'swir22', 'scl'], 
        resolution=20, 
        bounds_latlon=bbox, 
        epsg=epsg,
        xy_coords='center',
    )
    cube = load(cube)
    return cube

def compute_indices(cube, fire_start, fire_end):
    # calculate Normalized Burn Ratio (NBR)
    valid_classes = cube.sel(band='scl').isin(SCL_CLASSES) # dark areas, vegetation, bare soil, unclassified. Burn scars are often classified as any of these
    nir = cube.sel(band='nir08').where(valid_classes)
    swir = cube.sel(band='swir22').where(valid_classes)
    nbr = (nir - swir) / (nir + swir)

    # calculate DNBR and RBR
    pre_scenes = nbr.sel(time=slice(None, fire_start))
    n_pre = pre_scenes.sizes['time']
    print(f'{n_pre} scene(s) found pre-fire')
    if n_pre == 0:
        raise ValueError(f'no clear scenes before {fire_start}; use an earlier --search-start')
    pre = pre_scenes.median('time') # gets the median nbr over all the dates before start of the fire to eliminate effects of clouds

    post_scenes = nbr.sel(time=slice(fire_end, None))
    n_post = post_scenes.sizes['time']
    print(f'{n_post} scene(s) found post-fire')
    if n_post == 0:
        raise ValueError(f'no clear scenes after {fire_end}; use a later --search-end')
    post = post_scenes.median('time')
    
    dnbr = pre - post
    rbr = dnbr / (pre + 1.001)
    return dnbr, rbr


def classify_severity(dnbr):
    # Key & Benson classes: 1-2 regrowth, 3 unburned, 4-7 low to high severity, 0 nodata
    severity = xr.where(dnbr.isnull(), 0, # sets all null values to 0
                        np.digitize(dnbr, 
                        DNBR_THRESHOLDS
                        ) + 1).astype('uint8') # classes 1-7
    return severity

def burn_perimeter(severity, min_patch_sq_mi, epsg, min_severity):
    transform = severity.rio.transform()

    # produce binary mask of burn area: did that pixel get burned or not?
    burned = (severity.values >= min_severity).astype('uint8') # default 5 excludes dry farmland that registers as class 4 (see Le Porge, France)
    burned = sieve_until_stable(burned, SPECKLE_SIZE) # remove pixel-level speckle (25 px ~ 0.004 sq mi)
    burned_gdf = gpd.GeoDataFrame(geometry=[
        shapely.geometry.shape(geom) 
        for geom, _ in rasterio.features.shapes(
            burned, 
            mask=burned == 1, 
            connectivity=8, 
            transform=transform)
            ], crs=epsg)
    burned_gdf['area_sq_mi'] = burned_gdf.area / SQ_M_PER_SQ_MI
    burned_gdf = burned_gdf[burned_gdf.area_sq_mi >= min_patch_sq_mi].reset_index(drop=True)
    print(f'{len(burned_gdf)} polygons, {burned_gdf.area_sq_mi.sum():.2f} sq mi burned')
    return burned_gdf

def severity_polygons(severity, perimeter, epsg, min_severity):
    # producing burn severity vector: within the burn area, how severe was the burn at each pixel?
    in_scar = rasterio.features.rasterize(
        perimeter.geometry, out_shape=severity.shape, transform=severity.rio.transform(),
        fill=0, default_value=1, dtype='uint8',
    ) == 1
    inside = np.where(in_scar, severity.values, 0).astype('uint8')
    arr = sieve_until_stable(inside, size=SPECKLE_SIZE, mask=in_scar) # mask says that pixels outside the mask do not belong to any region and cannot be merged into
    burn_severity = gpd.GeoDataFrame(
        [{'severity': int(v), 'geometry': shapely.geometry.shape(g)}
        for g, v in rasterio.features.shapes(
            arr, 
            mask=arr >= min_severity, 
            connectivity=8,
            transform=severity.rio.transform())],
        columns=['severity', 'geometry'], # explicitly listing the columns allows this is handle an empty result
        geometry='geometry',
        crs=epsg,
    )
    return burn_severity

def main():
    parser = argparse.ArgumentParser(description='Map a burn scar from Sentinel-2')
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument('--place', help='region of interest in format of City, Country or City, State, Country, surrounded by quotes, e.g., "Le Porge, France"')
    location.add_argument('--bbox', nargs=4, type=float, metavar=('XMIN', 'YMIN', 'XMAX', 'YMAX'), help='bounding box in lon/lat, e.g., -118.21 34.14 -118.05 34.24')
    parser.add_argument('--fire-start', required=True, help='start date of the fire in the format YYYY-MM-DD, e.g., 2026-07-21')
    parser.add_argument('--fire-end', required=True, help='end date of the fire in the format YYYY-MM-DD, e.g., 2026-09-04')
    parser.add_argument('--outdir', required=True, help='folder to store the outputs. Will be created if not exist')
    parser.add_argument('--min-patch-sq-mi', type=float, default=0.5, help='minimum patch area for burn analysis. Any cluster of pixels smaller than this will be masked out')
    parser.add_argument('--zoom', type=float, default=1, help='zoom the AOI bbox out by this factor. Useful if the burn is too large to fit in the regular bounds. Only used with --place')
    parser.add_argument('--max-cloud-cover', type=int, default=20, help='maximum allowable percentage of cloud cover in scenes')
    parser.add_argument('--search-start', help='start date of the search in the catalog in the format YYYY-MM-DD, e.g., 2026-06-01. Defaults to 30 days pre-fire')
    parser.add_argument('--search-end', help='end date of the search in the catalog in the format YYYY-MM-DD, e.g., 2026-09-30. Defaults to 30 days post-fire')
    parser.add_argument('--min-severity', type=int, choices=[4,5,6,7], default=5, help='minimum burn severity (4-7, higher = more severe)')

    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if not args.search_start:
        args.search_start = datetime.date.fromisoformat(args.fire_start) - datetime.timedelta(days=30)
    if not args.search_end:
        args.search_end = datetime.date.fromisoformat(args.fire_end) + datetime.timedelta(days=30)

    # 1. Get bounding box of AOI
    if args.bbox:
        bbox = args.bbox
    else:
        bbox = get_bbox(args.place, args.zoom)

    # 2. Retrieve items from STAC catalog intersecting that bbox within the search window
    items = search_items(bbox, args.search_start, args.search_end, args.max_cloud_cover)

    # 3. Retrieve EPSG codes of items
    epsg = get_epsg(items)

    # 4. Stack all important assets from each item into a cube
    cube = load_cube(items, bbox, epsg)

    # 5. Calculate burn indices using the cube
    dnbr, rbr = compute_indices(cube, args.fire_start, args.fire_end)

    # 6. Reclassify the DNBR raster into severity bins
    severity = classify_severity(dnbr)

    # 7. Produce a mask of burn presence in areas larger than minimum patch size
    burn_mask = burn_perimeter(severity, args.min_patch_sq_mi, epsg, args.min_severity)
    edge = box(*severity.rio.bounds()).exterior
    if burn_mask.intersects(edge).any():
        print('WARNING: burn touches the AOI edge and may be cut off; increase --zoom or enlarge --bbox')

    # 8. Polygonize severity raster by classification to produce severity polygons
    burn_severity = severity_polygons(severity, burn_mask, epsg, args.min_severity)
    
    # 9. Write outputs
    to_cog(dnbr.astype('float32'), outdir / 'dnbr.tif', np.nan, epsg)
    to_cog(rbr.astype('float32'), outdir / 'rbr.tif', np.nan, epsg)
    to_cog(severity, outdir / 'severity.tif', 0, epsg)
    burn_mask.to_parquet(outdir / 'burn_perimeter.parquet')
    burn_severity.to_parquet(outdir / 'severity_polygons.parquet')

if __name__ == '__main__':
    main()