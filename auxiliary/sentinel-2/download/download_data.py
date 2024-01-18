#!/usr/bin/env python3

import geopandas as gpd
from sentinelsat.sentinel import SentinelAPI
import rasterio
import matplotlib.pyplot as plt
from rasterio import plot
from rasterio.plot import show
from rasterio.mask import mask
from osgeo import gdal
from pathlib import Path
import os

JSONPath = Path.cwd()
JSONFile = 'California.json'
JSONFullPath = JSONPath / JSONFile

USERNAME = os.environ['COPERNICUS_USER_NAME']
PASSWORD = os.environ['COPERNICUS_PASSWORD']

ExternalPath = Path('/run/media/dryglicki/External_01/sentinel-2')

boundary = gpd.read_file(JSONFullPath)

footprint = None

for i in boundary['geometry']:
    footprint = i

api = SentinelAPI(USERNAME, PASSWORD, 'https://scihub.copernicus.eu/dhus')

products = api.query(footprint,
        date = ('20200812', '20201122'),
        platformname = 'Sentinel-2',
        processinglevel = 'Level-2A')

print(api.get_products_size(products))

print("Converting to GeoPandas DataFrame")
gdf = api.to_geodataframe(products)
print("Saving to GeoJSON")
gdf.to_file('california-L2A-all-clouds.geojson', driver='GeoJSON')
#gdf_sorted = gdf.sort_values(['cloudcoverpercentage'], ascending=[True])
#check = api.trigger_offline_retrieval('da93d7e0-cdeb-4079-b8e6-4ac1aeb96e60')
#print(check)
#api.download('da93d7e0-cdeb-4079-b8e6-4ac1aeb96e60')
