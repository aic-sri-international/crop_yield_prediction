import ee
import time
import sys
import numpy as np
import pandas as pd
import itertools
import os
import urllib

ee.Initialize()

def export_oneimage(img,folder,name,scale,crs):
  task = ee.batch.Export.image(img, name, {
      'driveFolder':folder,
      'driveFileNamePrefix':name,
      'scale':scale,
      'crs':crs
  })
  task.start()
  print 'Task Registered: ', task.status()

# Locations_final is an export from the county_region Feature Collection
locations = pd.read_csv('locations_final.csv',header=None)


# Transforms an Image Collection with 1 band per Image into a single Image with items as bands
# Author: Jamie Vleeshouwer

def appendBand(current, previous):
    # Rename the band
    previous=ee.Image(previous)
    # Select the day time and night time temperature layers
    current = current.select([0,4])
    # Append it to the result (Note: only return current item on first element/iteration)
    accum = ee.Algorithms.If(ee.Algorithms.IsEqual(previous,None), current, previous.addBands(ee.Image(current)))
    # Return the accumulation
    return accum

# County shape layer in Google Earth Engine Feature Collection.
county_region = ee.FeatureCollection('ft:1S4EB6319wWW2sWQDPhDvmSBIVrD3iEmCLYB7nMM')

imgcoll = ee.ImageCollection('MODIS/MYD11A2') \
    .filterBounds(ee.Geometry.Rectangle(-106.5, 50,-64, 23))\
    .filterDate('2002-12-31','2016-8-4')
img=imgcoll.iterate(appendBand)
img=ee.Image(img)

img_0=ee.Image(ee.Number(-100))
img_16000=ee.Image(ee.Number(16000))

img=img.min(img_16000)
img=img.max(img_0)

for loc1, loc2, lat, lon in locations.values:
    fname = '{}_{}'.format(int(loc1), int(loc2))

    scale  = 500
    crs='EPSG:4326'

    # filter for a county (NOTE: VERIFY THESE COLUMN NAMES ARE IN YOUR FEATURE COLLECTION)
    region = ee.Feature(county_region
        .filterMetadata('StateFips', 'equals', int(loc1))
        .filterMetadata('CntyFips', 'equals', int(loc2))
        .first())

    while True:
        try:
            export_oneimage(img.clip(region), 'data_temperature', fname, scale, crs)
        except Exception as ex:
            print("Unexpected error:", ex)
            print("Failed on:", fname)
        break
