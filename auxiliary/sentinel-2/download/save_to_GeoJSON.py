#!/usr/bin/env python3

import geopandas as gpd
from sentinelsat.sentinel import SentinelAPI
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
import os

USERNAME = os.environ['COPERNICUS_USER_NAME']
PASSWORD = os.environ['COPERNICUS_PASSWORD']

api = SentinelAPI(USERNAME, PASSWORD, 'https://scihub.copernicus.eu/dhus')

def writeout(jsonFile, startDate, endDate):
    print('Currently processing: ',JSONFile)
    boundary = gpd.read_file(JSONFile)
    footprint = None
    for i in boundary['geometry']:
        footprint = i

    products = api.query(footprint,
        date = ('20221008', '20221017'),
        platformname = 'Sentinel-2',
        processinglevel = 'Level-2A' ) #,
        cloudcoverpercentage = (0,99))

    join01 = '_'.join(JSONFile.name.split('.')[:-1])
    outName = f'gdf_{join01}.geojson'
    print('Writing out to: ',outName)

    print("Converting to GeoPandas DataFrame")
    gdf = api.to_geodataframe(products)
    print("Saving to GeoJSON")
    gdf.to_file(outName, driver='GeoJSON')


def main():
    parser = ArgumentParser('Create GeoJSON files from JSON files from geojson.io')
    parser.add_argument('json_directory', help = 'Path to JSON files.')
    parser.add_argument('geojson_directory', help = 'Path for output geopandas GeoJSON files.')
    parser.add_argument('-sd', '--start_date', default = '20220101', help = 'Start date, YYYYMMDD. Default: 20220101')
    parser.add_argument('-ed', '--end_date', default = '20221231', help = 'End date, YYYYMMDD. Default: 20221231')
    parser.add_argument('-np', '--num_procs', default = 4, type=int, help = 'Number of threads (integer). Default: 4')

    args = parser.parse_args()
    jsonPath = Path(args.json_directory)
    if not jsonPath.is_dir(): raise NotADirectoryError(f'{jsonPath} is not a directory.')

    geojsonPath = Path(args.geojson_directory)
    if not geojsonPath.is_dir(): geojsonPath.mkdir(parents=True, exist_ok = True)

    startDate = datetime.strptime(args.start_date, '%Y%m%d')
    endDate   = datetime.strptime(args.end_date, '%Y%m%d')

    nproc = args.num_procs

    jsonList = sorted(jsonPath.glob('*json'))

    Parallel(n_jobs = nproc)(delayed(writeout)(jsonFile, startDate, endDate) for jsonFile in tqdm(jsonList))

if __name__ == '__main__':
    try:
        main()
        sys.exit(0)
    except KeyboardInterrupt:
        sys.exit(-1)



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
