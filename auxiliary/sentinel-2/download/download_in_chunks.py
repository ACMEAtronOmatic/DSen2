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
import gc

JSONPath = Path.cwd()
GeoJSONFile = 'gdf_GJ_Oregon.geojson'
GJFullPath = JSONPath / GeoJSONFile

outPath = Path('/run/media/dryglicki/External_01/sentinel-2')

USERNAME = os.environ['COPERNICUS_USER_NAME']
PASSWORD = os.environ['COPERNICUS_PASSWORD']

ExternalPath = Path('/run/media/dryglicki/External_01/sentinel-2/source/Oregon')
if not ExternalPath.exists(): ExternalPath.mkdir(parents=True)
#boundary = gpd.read_file(JSONFullPath)
#footprint = None

#for i in boundary['geometry']:
#    footprint = i

api = SentinelAPI(USERNAME, PASSWORD, 'https://scihub.copernicus.eu/dhus')

print("What is concurrent download limit?")
print(api.concurrent_dl_limit)
print()
print("What is the concurrent LTA trigger limit?")
print(api.concurrent_lta_trigger_limit)
print()

print("Currently ingesting: ",GJFullPath)

gdf = gpd.read_file(GJFullPath)

#print(gdf['index'])
#for col in gdf.columns:
#    print(col)

chunksize = int(api.concurrent_lta_trigger_limit)
nchunks = len(gdf) // chunksize 

gdfSort =  gdf.sort_values(['size'], ascending=[True])
startChunk = 0

for chunk in range(startChunk,nchunks,1):
    print(f'Currently working on chunk {chunk}.')
    first = chunk*chunksize
    last = (chunk+1)*chunksize
    chunkgdf = gdfSort.iloc[first:last].loc[:,['index']].reset_index()
    downloaded = [False]*len(chunkgdf)
    staged = [True]*len(chunkgdf)
    chunkgdf['download'] = downloaded
    chunkgdf['staged'] = staged
    firstcheck = []
    # first pass is to check for staging.
    for index, row in chunkgdf.iterrows():
        print(row['index'])
        try:
            res = api.trigger_offline_retrieval(str(row['index']))
            firstcheck.append(res)
        except Exception as e:
            print("Encountered following exception: ",e)
            firstcheck.append(np.nan)

    chunkgdf.loc[:,['staged']] = firstcheck
    chunkgdf.dropna(axis=0, subset=['staged'], inplace=True)

    while np.sum(chunkgdf['download'].values == False) > 0:
        print("Sleeping for 3 minutes...")
        time.sleep(180)
        newReady = []
        newDL = []
        for index, row in chunkgdf.iterrows():
            print("Currently processing: ",row['index'])
            if row['staged'] == False and row['download'] == False:
                print("Downloading: ",row['index'])
                try:
                    api.download(row['index'], directory_path=ExternalPath)
                    newDL.append(True)
                    newReady.append(False)
                except Exception as e:
                    print(e)
                    newDL.append(False)
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
    
        chunkgdf.loc[:,['staged']] = newReady
        chunkgdf.loc[:,['download']] = newDL
    
        totalDL = np.sum(chunkgdf['download'].values == True)
        print(f'Total number of files downloaded so far: {totalDL}')

    del chunkgdf
    gc.collect()
