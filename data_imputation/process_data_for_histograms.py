import numpy as np
import os
import gdal
import shutil
from os import listdir
from os.path import isfile, join

# from joblib import Parallel, delayed
# import multiprocessing
from multiprocessing import Pool

"""Data preprocessing pipeline

Combines and filters geotiff files to numpy array objects and saves them to disk.

Original Doc:
################
# Data range
# MODIS: 2003-2016, 14 years
# MODIS_landcover: 2003-2013, 12 years
# MODIS_temperature: 2003_2015, 13 years
# Intersection: 2003-2013, 11 years
################

Example
-------
Run this pipeline:

    $ python final_clean_data.py

"""

# Global Variables
img_output_dir = "/content/ee-data/img_full_output/"
img_zoom_output_dir = "/content/ee-data/img_zoom_full_output/"
count = 1


def divide_image_by_year(img, current_year, samples_per_year, total_years):
    """Divide state-county image stack by crop yield years.

    Parameters
    ----------
    img
        input image stack
    current_year
        starting year and currently iterated year
    samples_per_year
        number of total samples taken in a given year times number of bands
    total_years
        total years in sample set

    Returns
    -------
    list of images split by year ranges

    """
    image_list = []
    for i in range(0, total_years - 1):
        image_list.append(img[:, :, current_year:current_year + samples_per_year])
        current_year += samples_per_year
    image_list.append(img[:, :, current_year:])
    return image_list


def extend_mask(img, columns):
    """Extends image mask data by the specified number of columns with duplicates of the last column.

    Parameters
    ----------
    img
        input image
    columns
        number of columns to be added
    Returns
    -------
    new image mask with padding at the end columns

    """
    for i in range(0, columns):
        img = np.concatenate((img, img[:, :, -2:-1]), axis=2)
    return img


# very dirty... but should work
def merge_image(MODIS_img_list, MODIS_temperature_img_list):
    """Merges the 7 bands from the original image with the 2 bands from the temperature image

    Parameters
    ----------
    MODIS_img_list
        list of 7 band images
    MODIS_temperature_img_list
        list of 2 band images

    Returns
    -------
    list of 9 band images

    """
    MODIS_list = []
    for i in range(0, len(MODIS_img_list)):
        img_shape = MODIS_img_list[i].shape
        img_temperature_shape = MODIS_temperature_img_list[i].shape
        img_shape_new = (img_shape[0], img_shape[1], img_shape[2] + img_temperature_shape[2])
        merge = np.empty(img_shape_new)
        for j in range(0, int(img_shape[2] / 7)):
            img = MODIS_img_list[i][:, :, (j * 7):(j * 7 + 7)]
            temperature = MODIS_temperature_img_list[i][:, :, (j * 2):(j * 2 + 2)]
            merge[:, :, (j * 9):(j * 9 + 9)] = np.concatenate((img, temperature), axis=2)
        MODIS_list.append(merge)
    return MODIS_list


def mask_image(MODIS_list, MODIS_mask_img_list):
    """Apply mask to image file to filter according to specified mask. Elements are either 0 or original value depending

    Parameters
    ----------
    MODIS_list
        list of 9 band images
    MODIS_mask_img_list
        list of masks

    Returns
    -------
    list of masked 9 band images

    """
    MODIS_list_masked = []
    for i in range(0, len(MODIS_list)):
        mask = np.tile(MODIS_mask_img_list[i], (1, 1, MODIS_list[i].shape[2]))
        print('Mask NaN Error: ' + np.isnan(mask).any())
        print('Image NaN Error: ' + np.isnan(MODIS_list[i]).any())
        masked_img = np.nan_to_num(MODIS_list[i]) * np.nan_to_num(mask)
        MODIS_list_masked.append(masked_img)
    return MODIS_list_masked


def create_gdal_array(file_path):
    """Rasterize a file.

    Parameters
    ----------
    file_path
        location of file

    Returns
    -------
    array with pixel values of the rasterized file
    """
    raster = gdal.Open(file_path)
    arr = raster.ReadAsArray()
    if not raster or not arr.size:
        print("ERROR GDAL raster failed! raster_array: {}, raster: {} Skipping file: {}".format(file_path, arr, raster))
        raster = None
        del raster
        return
    raster = None
    del raster
    return arr


def preprocess_save_data(file_tuple):
    """Modify the file to be 9 bands, masked for target crop, and converted into proper spectrum and dimensions

    Parameters
    ----------
    file_tuple
        the index (unused) and tif file

    Returns
    -------
    None if an error was encountered or if the file is one of the black listed images

    """
    MODIS_dir = "/content/ira-gdrive/Data2/"
    MODIS_mask_dir = "/content/ira-gdrive/data_mask/"
    MODIS_temperature_dir = "/content/ira-gdrive/data_temperature/"
    data_yield = np.genfromtxt('/content/datalab/notebooks/crop_yield_prediction/2 clean data/yield_final.csv',
                               delimiter=',', dtype=float)

    index, tif_file = file_tuple
    print("File: {}, Index: {}, PID: {}".format(tif_file, index, os.getpid()))

    # # TODO: Need to figure out why these files are broken. Can they be fixed or filtered better?
    # broken_images = ['17_97.tif', '26_141.tif', '1_3.tif', '29_3.tif', '38_101.tif', '51_159.tif', '46_93.tif',
    #                  '17_187.tif', '31_159.tif', '20_115.tif', '22_23.tif', '17_177.tif', '27_39.tif', '51_135.tif',
    #                  '1_33.tif', '26_69.tif', '51_41.tif', '46_95.tfi', '27_77.tif', '31_87.tif', '27_35.tif',
    #                  '45_3.tif', '27_143.tif', '22_19.tif', '13_37.tif', '13_119.tif', '13_43.tif', '17_139.tif',
    #                  '29_157.tif', '18_17.tif', '20_207.tif', '21_69.tif', '26_9.tif']
    # broken_images = ['22_23.tif', '26_69.tif', '27_71.tif', '46_23.tif', '1_3.tif', '26_141.tif', '46_93.tif']

    if tif_file.endswith(".tif"):
        # if tif_file in broken_images:
        #     print("skipping {} due to numpy multiple errors".format(tif_file))
        #     return

        MODIS_path = os.path.join(MODIS_dir, tif_file)
        MODIS_temperature_path = os.path.join(MODIS_temperature_dir, tif_file)
        MODIS_mask_path = os.path.join(MODIS_mask_dir, tif_file)

        # get geo location
        raw = tif_file.replace('_', ' ').replace('.', ' ').split()
        loc1 = int(raw[0])
        loc2 = int(raw[1])
        # read image
        try:
            # TODO: why the shift, scale, clean on temperature only? Also what wavelengths are all of these?
            # Are they the same scale?
            MODIS_raster_arr = create_gdal_array(file_path=MODIS_path)
            MODIS_img = np.transpose(np.array(MODIS_raster_arr, dtype='uint16'), axes=(1, 2, 0))
            # read temperature
            MODIS_raster_temp_arr = create_gdal_array(MODIS_temperature_path)
            MODIS_temperature_img = np.transpose(np.array(MODIS_raster_temp_arr, dtype='uint16'), axes=(1, 2, 0))
            # shift
            MODIS_temperature_img = MODIS_temperature_img - 12000
            # scale
            MODIS_temperature_img = MODIS_temperature_img * 1.25
            # clean
            MODIS_temperature_img[MODIS_temperature_img < 0] = 0
            MODIS_temperature_img[MODIS_temperature_img > 5000] = 5000
            # read mask
            MODIS_raster_mask_arr = create_gdal_array(MODIS_mask_path)
            MODIS_mask_img = np.transpose(np.array(MODIS_raster_mask_arr, dtype='uint16'), axes=(1, 2, 0))
            # Non-crop = 0, crop = 1
            MODIS_mask_img[MODIS_mask_img != 12] = 0
            MODIS_mask_img[MODIS_mask_img == 12] = 1
        except ValueError as msg:
            print("Exception: {}".format(msg))
            return

        # Extend mask img to accomidate new bands
        MODIS_mask_img = extend_mask(img=MODIS_mask_img, columns=3)

        # Divide image into years in range: 2002-12-31 to 2016-8-4
        MODIS_img_list = divide_image_by_year(img=MODIS_img, current_year=0, samples_per_year=46 * 7, total_years=14)
        MODIS_temperature_img_list = divide_image_by_year(img=MODIS_temperature_img, current_year=0,
                                                          samples_per_year=46 * 2, total_years=14)
        MODIS_mask_img_list = divide_image_by_year(img=MODIS_mask_img, current_year=0, samples_per_year=1,
                                                   total_years=14)

        # Merge image and temperature
        MODIS_list = merge_image(MODIS_img_list=MODIS_img_list, MODIS_temperature_img_list=MODIS_temperature_img_list)

        # Do the mask job
        MODIS_list_masked = mask_image(MODIS_list=MODIS_list, MODIS_mask_img_list=MODIS_mask_img_list)

        # check if the result is in the list
        year_start = 2003
        for i in range(0, 14):
            year = i + year_start
            key = np.array([year, loc1, loc2])
            if np.sum(np.all(data_yield[:, 0:3] == key, axis=1)) > 0:
                # save as .npy
                filename = img_output_dir + str(year) + '_' + str(loc1) + '_' + str(loc2) + '.npy'
                np.save(filename, MODIS_list_masked[i])

        # Dealloc the C level pointers
        del MODIS_raster_arr
        del MODIS_raster_temp_arr
        del MODIS_raster_mask_arr

