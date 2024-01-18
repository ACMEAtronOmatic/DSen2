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
import numpy as np
import time

JSONPath = Path.cwd()
JSONFile = 'California.json'
GeoJSONFile = 'california-L2A-all-clouds.geojson'
JSONFullPath = JSONPath / JSONFile
GJFullPath = JSONPath / GeoJSONFile

outPath = Path('/run/media/dryglicki/External_01/sentinel-2')

USERNAME = os.environ['COPERNICUS_USER_NAME']
PASSWORD = os.environ['COPERNICUS_PASSWORD']

ExternalPath = Path('/run/media/dryglicki/External_01/sentinel-2')
boundary = gpd.read_file(JSONFullPath)
footprint = None

for i in boundary['geometry']:
    footprint = i

api = SentinelAPI(USERNAME, PASSWORD, 'https://scihub.copernicus.eu/dhus')

gdf = gpd.read_file(GJFullPath)

print(gdf['index'])
for col in gdf.columns:
    print(col)

TopIDs = gdf.sort_values(['size'], ascending=[True]).iloc[:10,].loc[:,['index']].reset_index()
downloaded = [False]*len(TopIDs)
staged = [True]*len(TopIDs)
TopIDs['staged'] = staged
TopIDs['download'] = downloaded

firstcheck = []
# first pass is to check for staging.
for index, row in TopIDs.iterrows():
    print(row['index'])
    try:
        res = api.trigger_offline_retrieval(str(row['index']))
        firstcheck.append(res)
    except Exception as e:
        print("Encountered following exception: ",e)
        firstcheck.append(np.nan)
#   firstcheck.append(api.trigger_offline_retrieval(row['index']))

TopIDs.loc[:,['staged']] = firstcheck
TopIDs.dropna(axis=0, subset=['staged'], inplace=True)
print(TopIDs['staged'])

#api.download('147bcaa7-1522-4479-88ad-676dcb83bb43')

#print(np.sum(TopIDs['download'].values == False))

while np.sum(TopIDs['download'].values == False) > 0:
    newReady = []
    newDL = []
    for index, row in TopIDs.iterrows():
        print("Currently processing: ",row['index'])
        if row['staged'] == False and row['download'] == False:
            print("Downloading: ",row['index'])
            api.download(row['index'])
            newDL.append(True)
            newReady.append(False)
        elif row['staged'] == False and row['download'] == True:
            newDL.append(True)
            newReady.append(False)
        else:
            try:
                res = api.trigger_offline_retrieval(str(row['index']))
                newReady.append(res)
                newDL.append(False)
            except Exception as e:
                print(e)
                newReady.append(True)
                newDL.append(False)

    TopIDs.loc[:,['staged']] = newReady
    TopIDs.loc[:,['download']] = newDL

    totalDL = np.sum(TopIDs['download'].values == True)
    print(f'Total number of files downloaded so far: {totalDL}')
    print("Sleeping for 5 minutes...")
    time.sleep(300)

    


# After first pass, now downloading.
#for index, row in TopIDs.iterrows()
