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

USERNAME = os.environ['COPERNICUS_USER_NAME']
PASSWORD = os.environ['COPERNICUS_PASSWORD']

ExternalPath = Path('/run/media/dryglicki/External_01/sentinel-2')

api = SentinelAPI(USERNAME, PASSWORD, 'https://scihub.copernicus.eu/dhus')

for JSONFile in JSONPath.glob('*Oregon*.json'):
    print('Currently processing: ',JSONFile)
    boundary = gpd.read_file(JSONFile)
    footprint = None
    for i in boundary['geometry']:
        footprint = i

    products = api.query(footprint,
        date = ('20221008', '20221017'),
        platformname = 'Sentinel-2',
        processinglevel = 'Level-2A' ) #,
#       cloudcoverpercentage = (0,75))

    join01 = '_'.join(JSONFile.name.split('.')[:-1])
    outName = f'gdf_{join01}.geojson'
    print('Writing out to: ',outName)

    print("Converting to GeoPandas DataFrame")
    gdf = api.to_geodataframe(products)
    print("Saving to GeoJSON")
    gdf.to_file(outName, driver='GeoJSON')
#gdf_sorted = gdf.sort_values(['cloudcoverpercentage'], ascending=[True])
#check = api.trigger_offline_retrieval('da93d7e0-cdeb-4079-b8e6-4ac1aeb96e60')
#print(check)
#api.download('da93d7e0-cdeb-4079-b8e6-4ac1aeb96e60')
