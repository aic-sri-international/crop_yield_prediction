# Downloading Data

The data source for the models in this project comes from the MODIS information available in the Google Earth Engine.  In order to work with the data a Google Account is required along with a sufficently large (recommend 1TB) Google Drive to accomidate saving imagery from earth engine export commands.



1. Create a Kubernetes instance of the Data Lab environment for executing the download scripts.  A local docker instance would be minimally sufficent and could be adapted from `google-datalab.yaml`. Comand: `kubectl create -f google-datalab.yaml`

2. Request access to the Google Earth Engine service for your Google account.  At present this service is available as an invite only beta.  There is a request process to have access granted to your account.  Expect a day or two for this approval based on recent experience (Dec 2017).

3. Open a shell agaist the running instance. Command: `kubectl exec -it YOUR-K8S-POD-IDENTIFIER -- bash`

4. Authenticate with Google Earth Engine using the command: `earthengine auth`

5. Execute the scripts within the pod to generate data into your Google Drive.  It is recommended that to verify the install is working by importing `ee` into a python REPL and executing `ee.Initialize()` and verifying that no errors are reported.

## Data Sets Used

The datasets downloaded are primarily from [MODIS](https://en.wikipedia.org/wiki/Moderate-resolution_imaging_spectroradiometer) satelite coverage and are:

* MODIS reflectivity hyperspectral imagery (layers 1-7) http://dx.doi.org/10.5067/MODIS/MOD09A1.006 - 
https://explorer.earthengine.google.com/#detail/MODIS%2F006%2FMOD09A1

* Surface temperature (day and night) at 1km/pixel http://dx.doi.org/10.5067/MODIS/MYD11A2.006 - 
https://explorer.earthengine.google.com/#detail/MODIS%2F006%2FMYD11A2 Layers 1 and 5 (LST_Day_1km, and LST_Night_1km)

* Global Landcover Classification data - MCD12Q1.051 Land Cover Type Yearly Global 500m layer 1 (International Geosphere-Biosphere Programme (IGBP) global vegetation classification scheme 0-255 scheme) https://explorer.earthengine.google.com/#detail/MODIS%2F051%2FMCD12Q1. This classification data covers Jan 1, 2000 - Jan 1, 2013. 

### Notes

*  The temperature and reflectivity layers are world wide coverages in two different versions.  Version 005 is available from Jul 4, 2002 - Mar 30, 2017 and Version 006 is available from Mar 5, 2000 - Mar 14, 2018 (roughly current).  Each sample is computed from an 8 day average

* The limiting factor on temporal coverage is the MCD12Q1.051 coverage dataset that spans from Jan 2000-2013.  Additionally this coverage is not a very high resolution set.  More background infomation [here](https://gisgeography.com/free-global-land-cover-land-use-data/).

* The download process needs to use a tile based approach to keep image sizes down to avoid API rate limits.  The county shape clip approach provides one option as does the buffered lat/long location series.  These approaches are controlled by the location CSV files and the scripts chosen.

## Configuration

There are two types of imagery exported in the data download process.  Imagery in tiles based on buffered lat/long locations and imagery clipped to shape boundaries.  For each of these location types there are several layers of information downloaded; temperature, land cover in multiple spectral bands, and a shape mask.

1. A copy of the county shape file and related data is required for running the exports of data that are clipped to counties.  This feature is a publicly available [feature collection](https://fusiontables.google.com/data?docid=1S4EB6319wWW2sWQDPhDvmSBIVrD3iEmCLYB7nMM).  If using the public identifier of `ft:1S4EB6319wWW2sWQDPhDvmSBIVrD3iEmCLYB7nMM` Then the region filters should match the column names which are `CntyFips` and `StateFips` for selecting the county regions.  The contents of the various locations_XXX.csv files are simply extracts of `StateFips,CntyFips,Long,Lat` from this feature collection.

2. Data products should be time aligned based on the date range of the land coverage usage type (Jan 2010-2013)

3. This climate data coverage is made up of a grid.  What would make the most sense (not currently in place) would be to download the coverages in the appropriate grids they are generated in and build from this complete worldwide coverage.

## Helpful Hints

* When each script runs it will output a series of task ready events.  These events do not indicate the Google Earth engine task was successfully completed.  It is useful to run `earthengine task list > status` followed by `less status` to review the recent generated task output.  As the Earth engine executes each queued job their status will be set to success or failure from `READY`.
* The GRIVE software package for Ubuntu can be used to pull data from Google Drive directly to the instances on a cluster that will process the imagery.

