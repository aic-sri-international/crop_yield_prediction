import numpy as np
import scipy.io as io
import math
import os
import skimage.io
import pandas as pd
from osgeo import gdal
from scipy.ndimage.interpolation import zoom
import shutil
from os import listdir
from os.path import isfile, join
from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Pool


img_output_dir="/content/ee-data/img_full_output/"
img_zoom_output_dir="/content/ee-data/img_zoom_full_output/"
count=1

################
# Data range
# MODIS: 2003-2016, 14 years
# MODIS_landcover: 2003-2013, 12 years
# MODIS_temperature: 2003_2015, 13 years

# Intersection: 2003-2013, 11 years

################


def check_data_integrity_del():
    data = np.genfromtxt('yield_final_highquality.csv', delimiter=',')
    # check if they have related files
    dir = "/atlas/u/jiaxuan/data/google_drive/img_zoom_output/"
    list_del = []
    for i in range(data.shape[0]):
        year = data[i,0]
        loc1 = data[i,1]
        loc2 = data[i,2]
        filename = str(int(year)) + '_' + str(int(loc1)) + '_' + str(int(loc2)) + '.npy'
        if os.path.isfile(dir + filename)==False:
            print 'del'
            list_del.append(i)

    list_del = np.array(list_del)
    data_clean=np.delete(data, list_del, axis=0)
    np.savetxt("yield_final_highquality.csv", data_clean, delimiter=",")

def check_data_integrity():
    print 'begin'
    data = np.genfromtxt('yield_final_highquality.csv', delimiter=',')
    # check if they have related files
    dir = "/atlas/u/jiaxuan/data/google_drive/img_output/"
    for i in range(data.shape[0]):
        year = data[i,0]
        loc1 = data[i,1]
        loc2 = data[i,2]
        filename = str(int(year)) + '_' + str(int(loc1)) + '_' + str(int(loc2)) + '.npy'
        if os.path.isfile(dir + filename)==False:
            print filename
    print 'end'

def divide_image(img,first,step,num):
    image_list=[]
    for i in range(0,num-1):
        image_list.append(img[:, :, first:first+step])
        first+=step
    image_list.append(img[:, :, first:])
    return image_list

def extend_mask(img,num):
    for i in range(0,num):
        img = np.concatenate((img, img[:,:,-2:-1]),axis=2)
    return img

# very dirty... but should work
def merge_image(MODIS_img_list,MODIS_temperature_img_list):
    MODIS_list=[]
    for i in range(0,len(MODIS_img_list)):
        img_shape=MODIS_img_list[i].shape
        img_temperature_shape=MODIS_temperature_img_list[i].shape
        img_shape_new=(img_shape[0],img_shape[1],img_shape[2]+img_temperature_shape[2])
        merge=np.empty(img_shape_new)
        for j in range(0,img_shape[2]/7):
            img=MODIS_img_list[i][:,:,(j*7):(j*7+7)]
            temperature=MODIS_temperature_img_list[i][:,:,(j*2):(j*2+2)]
            merge[:,:,(j*9):(j*9+9)]=np.concatenate((img,temperature),axis=2)
        MODIS_list.append(merge)
    return MODIS_list


def mask_image(MODIS_list,MODIS_mask_img_list):
    MODIS_list_masked = []
    for i in range(0, len(MODIS_list)):
        mask = np.tile(MODIS_mask_img_list[i],(1,1,MODIS_list[i].shape[2]))
        masked_img = MODIS_list[i]*mask
        MODIS_list_masked.append(masked_img)
    return MODIS_list_masked

def quality_dector(image_temp):
        filter_0=image_temp>0
        filter_5000=image_temp<5000
        filter=filter_0*filter_5000
        return float(np.count_nonzero(filter))/image_temp.size


def create_gdal_array(file_path):
    raster = gdal.Open(file_path)
    arr = raster.ReadAsArray()
    if not raster or not arr.size:
        print("ERROR GDAL raster failed! raster_array: {}, raster: {} Skipping file: {}"
              .format(file,
               raster_array,
               raster))
        return
    raster = None
    return arr


def preprocess_save_data(file_tuple):  # img_output_dir, img_zoom_output_dir):
#     MODIS_dir="/atlas/u/jiaxuan/data/google_drive/data_image"
#     MODIS_temperature_dir="/atlas/u/jiaxuan/data/google_drive/data_temperature"
#     MODIS_mask_dir="/atlas/u/jiaxuan/data/google_drive/data_mask"

#     img_output_dir="/atlas/u/jiaxuan/data/google_drive/img_output/"

    # MODIS_processed_dir="C:/360Downloads/6_Data_county_processed_scaled/"

    # MODIS_dir="/atlas/u/jiaxuan/data/MODIS_data_county/3_Data_county"
    # MODIS_temperature_dir="/atlas/u/jiaxuan/data/MODIS_data_county_temperature"
    # MODIS_mask_dir="/atlas/u/jiaxuan/data/MODIS_data_county_mask"
    # MODIS_processed_dir="/atlas/u/jiaxuan/data/MODIS_data_county_processed_compressed/"
    
#     data_yield = np.genfromtxt('yield_final.csv', delimiter=',', dtype=float)
    MODIS_dir="/content/ira-gdrive/Data2/"
    MODIS_mask_dir="/content/ira-gdrive/data_mask/"
    MODIS_temperature_dir="/content/ira-gdrive/data_temperature/"
    data_yield = np.genfromtxt('/content/datalab/notebooks/crop_yield_prediction/2 clean data/yield_final.csv', delimiter=',', dtype=float)

    index, tif_file = file_tuple
    #     print("File, index, PID: {}, {}, {} ".format(tif_file, index, multiprocessing.current_process().name))
    print ("File: {}".format(tif_file))
    # 1_3 check what it is...?
    broken_images = ['17_97.tif', '26_141.tif', '1_3.tif', '29_3.tif', '38_101.tif', '51_159.tif', '46_93.tif', 
                     '17_187.tif', '31_159.tif', '20_115.tif', '22_23.tif', '17_177.tif', '27_39.tif', '51_135.tif', 
                     '1_33.tif', '26_69.tif', '51_41.tif', '46_95.tfi', '27_77.tif', '31_87.tif', '27_35.tif', '45_3.tif', 
                     '27_143.tif', '22_19.tif', '13_37.tif', '13_119.tif', '13_43.tif', '17_139.tif', '29_157.tif',
                     '18_17.tif', '20_207.tif', '21_69.tif', '26_9.tif']
#     for root, dirs, files in os.walk(MODIS_dir):
#     continue_preprocess = False

#     for tif_file in files:
#         if file not in '1_33.tif' and not continue_preprocess:
#             continue
#         elif file in '1_33.tif':
#             continue_preprocess = True
#             continue
    if tif_file.endswith(".tif"):
        if tif_file in broken_images:
            print("skipping {} due to numpy multiple errors".format(tif_file))
            return
        MODIS_path=os.path.join(MODIS_dir, tif_file)
#             check file size to see if it's broken
#         if os.path.getsize(MODIS_path) < 10000000:
#             print 'file broken, continue'
#             return
        MODIS_temperature_path=os.path.join(MODIS_temperature_dir,tif_file)
        MODIS_mask_path=os.path.join(MODIS_mask_dir,tif_file)

        # get geo location
        raw = tif_file.replace('_',' ').replace('.',' ').split()
        loc1 = int(raw[0])
        loc2 = int(raw[1])
        # read image
        try:
            # TODO why the shift, scale, clean on temperature only? Also what wavelengths are all of these?
            # Are they the same scale?
            MODIS_raster_arr = create_gdal_array(MODIS_path)
#             print (MODIS_raster_arr)
            MODIS_img = np.transpose(np.array(MODIS_raster_arr, dtype='uint16'),axes=(1,2,0))
            # read temperature
            MODIS_raster_temp_arr = create_gdal_array(MODIS_temperature_path)
            MODIS_temperature_img = np.transpose(np.array(MODIS_raster_temp_arr, dtype='uint16'),axes=(1,2,0))
            # shift
            MODIS_temperature_img = MODIS_temperature_img-12000
            # scale
            MODIS_temperature_img = MODIS_temperature_img*1.25
            # clean
            MODIS_temperature_img[MODIS_temperature_img<0]=0
            MODIS_temperature_img[MODIS_temperature_img>5000]=5000
            # read mask
            MODIS_raster_mask_arr = create_gdal_array(MODIS_mask_path)
            MODIS_mask_img = np.transpose(np.array(MODIS_raster_mask_arr, dtype='uint16'),axes=(1,2,0))
            # Non-crop = 0, crop = 1
            MODIS_mask_img[MODIS_mask_img != 12] = 0
            MODIS_mask_img[MODIS_mask_img == 12] = 1
            

        except ValueError as msg:
            print msg
            return

        # Divide image into years in range: 2002-12-31 to 2016-8-4
        MODIS_img_list=divide_image(MODIS_img, 0, 46 * 7, 14)
        MODIS_temperature_img_list = divide_image(MODIS_temperature_img, 0, 46 * 2, 14)
        MODIS_mask_img = extend_mask(MODIS_mask_img, 3)
        MODIS_mask_img_list = divide_image(MODIS_mask_img, 0, 1, 14)

        # Merge image and temperature
        MODIS_list = merge_image(MODIS_img_list,MODIS_temperature_img_list)

        # Do the mask job
        MODIS_list_masked = mask_image(MODIS_list,MODIS_mask_img_list)

        # check if the result is in the list
        year_start = 2003
        for i in range(0, 14):
            year = i+year_start
            key = np.array([year,loc1,loc2])
            if np.sum(np.all(data_yield[:,0:3] == key, axis=1))>0:
                # save as .npy
                filename=img_output_dir+str(year)+'_'+str(loc1)+'_'+str(loc2)+'.npy'
                np.save(filename,MODIS_list_masked[i])
#                 print filename,':written '  # ,str(count)
        del MODIS_raster_arr
        del MODIS_raster_temp_arr
        del MODIS_raster_mask_arr

                

if __name__ == "__main__":
    # # save data
#     MODIS_dir="/content/ee-data/Data/"
    MODIS_dir="/content/ira-gdrive/Data2/"
        # img_output_dir="/atlas/u/jiaxuan/data/google_drive/img_output/"
    if (os.path.isdir(img_output_dir)):
        shutil.rmtree(img_output_dir)
    os.mkdir(img_output_dir)
    
    #img_zoom_output_dir="/atlas/u/jiaxuan/data/google_drive/img_zoom_full_output/"
    if (os.path.isdir(img_zoom_output_dir)):
        shutil.rmtree(img_zoom_output_dir)
    os.mkdir(img_zoom_output_dir)


#     for _, _, files in os.walk(MODIS_dir):
#         Parallel(n_jobs=12)(delayed(preprocess_save_data_parallel)(file) for file in files)
#     for _, _, files in os.walk(MODIS_dir):
#         print (len(files))
#         file_count = 0
#         for file in files:
#             file_count+=1
#             print("File: {}, file_count: {}".format(file, file_count))
#             preprocess_save_data_parallel(file, img_output_dir, img_zoom_output_dir)

    print ("STARTING CLEAN...")
    files = [f for f in listdir(MODIS_dir) if isfile(join(MODIS_dir, f))]
    for f in enumerate(files):
        preprocess_save_data(f)
#    try:
#        p = Pool(60)
#        files = [f for f in listdir(MODIS_dir) if isfile(join(MODIS_dir, f))]
#        print("Number of files: ", len(files))
#        p.map(preprocess_save_data, enumerate(files))
#    except KeyboardInterrupt:
#        p.terminate()
#        p.join()
#    preprocess_save_data(img_output_dir, img_zoom_output_dir)
            
    print("DONE")

    # # clean yield (low quality)
    # check_data_integrity_del()
    # # check integrity
    # check_data_integrity()

