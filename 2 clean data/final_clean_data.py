import numpy as np
import scipy.io as io
import math
import os
import skimage.io
import pandas as pd
import gdal
from scipy.ndimage import zoom
import shutil

from joblib import Parallel, delayed
import multiprocessing


################
# Data range
# MODIS: 2003-2016, 14 years
# MODIS_landcover: 2003-2013, 12 years
# MODIS_temperature: 2003_2015, 13 years

# Intersection: 2003-2013, 11 years

################


def check_data_integrity_del():
    data = np.genfromtxt('/content/datalab/notebooks/crop_yield_prediction/2 clean data/yield_final_highquality.csv', delimiter=',')
    # check if they have related files
    dir = "/content/ee-data/img_zoom_output/"
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
    np.savetxt("/content/datalab/notebooks/crop_yield_prediction/2 clean data/yield_final_highquality.csv", data_clean, delimiter=",")
    #np.savetxt("yield_final_highquality.csv", data_clean, delimiter=",")

def check_data_integrity():
    print 'begin'
    data = np.genfromtxt('/content/datalab/notebooks/crop_yield_prediction/2 clean data/yield_final_highquality.csv', delimiter=',')
    # check if they have related files
    dir = "/content/ee-data/img_output/"
    for i in range(data.shape[0]):
        year = data[i,0]
        loc1 = data[i,1]
        loc2 = data[i,2]
        filename = str(int(year)) + '_' + str(int(loc1)) + '_' + str(int(loc2)) + '.npy'
        if os.path.isfile(dir + filename)==False:
            print filename
    print 'end'

# def check_data_integrity():
#     data = pd.read_csv('locations_final.csv',header=None)
#     # check if they have related files
#     idx=0
#     dir = "/atlas/u/jiaxuan/data/google_drive/data_image/"
#     for loc1, loc2,_,_ in data.values:
#         # filename = str(int(year)) + '_' + str(int(loc1)) + '_' + str(int(loc2)) + '.npz'
#         filename = str(int(loc1)) + '_' + str(int(loc2)) + '.tif'
#         if os.path.isfile(dir + filename)==False:
#             print filename,idx
#         idx+=1
#     print 'done',idx
#     print data.values.shape[0]

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
            # print "img.shape: ", img.shape
            # print "temperature.shape: ", temperature.shape
            if temperature.shape[2] != 2 or img.shape[2] != 7:
                # print 'continue merge'
                continue
            merge[:,:,(j*9):(j*9+9)]=np.concatenate((img,temperature),axis=2)
        MODIS_list.append(merge)
    return MODIS_list


def mask_image(MODIS_list,MODIS_mask_img_list):
    MODIS_list_masked = []
    # print 'modis list length: ', len(MODIS_list)
    for i in range(0, len(MODIS_list)):
        mask = np.tile(MODIS_mask_img_list[i],(1,1,MODIS_list[i].shape[2]))
        # print 'mask shape: ', mask.shape
        # print 'modis list shape: ', MODIS_list[i].shape
        if mask.shape[2] != MODIS_list[i].shape[2]:
            # print 'unmatched shapes, continuing without image'
            continue
        masked_img = MODIS_list[i]*mask
        MODIS_list_masked.append(masked_img)
    return MODIS_list_masked

def quality_dector(image_temp):
        filter_0=image_temp>0
        filter_5000=image_temp<5000
        filter=filter_0*filter_5000
        return float(np.count_nonzero(filter))/image_temp.size

def preprocess_save_data():

    # MODIS_dir="/atlas/u/jiaxuan/data/google_drive/data_image"
    MODIS_dir="/content/ee-data/Data_county"
    # MODIS_temperature_dir="/atlas/u/jiaxuan/data/google_drive/data_temperature"
    MODIS_temperature_dir="/content/ee-data/Data_county_temperature"
    # MODIS_mask_dir="/atlas/u/jiaxuan/data/google_drive/data_mask"
    MODIS_mask_dir="/content/ee-data/Data_county_mask"

    # img_output_dir="/atlas/u/jiaxuan/data/google_drive/img_output/"
    img_output_dir="/content/ee-data/img_output/"
    if (os.path.isdir(img_output_dir)):
        shutil.rmtree(img_output_dir)
    os.mkdir(img_output_dir)

    # MODIS_processed_dir="C:/360Downloads/6_Data_county_processed_scaled/"

    # MODIS_dir="/atlas/u/jiaxuan/data/MODIS_data_county/3_Data_county"
    # MODIS_temperature_dir="/atlas/u/jiaxuan/data/MODIS_data_county_temperature"
    # MODIS_mask_dir="/atlas/u/jiaxuan/data/MODIS_data_county_mask"
    # MODIS_processed_dir="/atlas/u/jiaxuan/data/MODIS_data_county_processed_compressed/"

    data_yield = np.genfromtxt('/content/datalab/notebooks/crop_yield_prediction/2 clean data/yield_final.csv', delimiter=',', dtype=float)
    count=1
    for root, dirs, files in os.walk(MODIS_dir):
        print 'root, dirs: ', root, dirs
#         print "number of files: ", len(files)
        for file in files:
#             print 'file: ', file
            if file.endswith(".tif"):
                MODIS_path=os.path.join(MODIS_dir, file)
                # check file size to see if it's broken
                # if os.path.getsize(MODIS_path) < 10000000:
                #     print 'file broken, continue'
                #     continue
                MODIS_temperature_path=os.path.join(MODIS_temperature_dir,file)
                MODIS_mask_path=os.path.join(MODIS_mask_dir,file)

                # get geo location
                raw = file.replace('_',' ').replace('.',' ').split()
                loc1 = int(raw[0])
                loc2 = int(raw[1])
                # read image
                try:
                    MODIS_img = np.transpose(np.array(gdal.Open(MODIS_path).ReadAsArray(), dtype='uint16'),axes=(1,2,0))
                except ValueError as msg:
                    print msg
                    continue
                # read temperature
                MODIS_temperature_img = np.transpose(np.array(gdal.Open(MODIS_temperature_path).ReadAsArray(), dtype='uint16'),axes=(1,2,0))
                # shift
                MODIS_temperature_img = MODIS_temperature_img-12000
                # scale
                MODIS_temperature_img = MODIS_temperature_img*1.25
                # clean
                MODIS_temperature_img[MODIS_temperature_img<0]=0
                MODIS_temperature_img[MODIS_temperature_img>5000]=5000
                # read mask
                MODIS_mask_img = np.transpose(np.array(gdal.Open(MODIS_mask_path).ReadAsArray(), dtype='uint16'),axes=(1,2,0))
                # Non-crop = 0, crop = 1
                MODIS_mask_img[MODIS_mask_img != 12] = 0
                MODIS_mask_img[MODIS_mask_img == 12] = 1

                # Divide image into years
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
#                 print len(MODIS_list_masked)
                for i in range(0, len(MODIS_list_masked)):
#                     print "file: ", file, " at position: ", i
                    year = i+year_start
                    # print 'year: ', year
                    key = np.array([year,loc1,loc2])
                    # print 'key: ', key
                    if np.sum(np.all(data_yield[:,0:3] == key, axis=1))>0:
#                         print 'after np.sum'
                        # save as .npy
                        filename=str(year)+'_'+str(loc1)+'_'+str(loc2)+'.npy'
#                         print 'got filename'
#                         if "38_51" in file and i == 6:
#                             print MODIS_list_masked[6]
#                         if "31_159" in file and i == 11:
#                             from IPython.core.debugger import Tracer; Tracer()()
                        np.save(img_output_dir + filename, MODIS_list_masked[i])
                        # print filename,':written ',str(count)
#                         print 'moving on...'
                        count+=1
#                         print 'in loop'
#                 print "end of loop"
                if "19_57" in file:
                    break

def preprocess_save_data_parallel(file):
    # MODIS_dir="/atlas/u/jiaxuan/data/google_drive/data_image_full"
    MODIS_dir="/content/ee-data/Data_county"
    # MODIS_temperature_dir="/atlas/u/jiaxuan/data/google_drive/data_temperature"
    MODIS_temperature_dir="/content/ee-data/Data_county_temperature"
    # MODIS_mask_dir="/atlas/u/jiaxuan/data/google_drive/data_mask"
    MODIS_mask_dir="/content/ee-data/Data_county_mask"

    # img_output_dir="/atlas/u/jiaxuan/data/google_drive/img_output/"
    img_output_dir="/content/ee-data/img_full_output/"
    os.mkdir(img_output_dir)
    
    #img_zoom_output_dir="/atlas/u/jiaxuan/data/google_drive/img_zoom_full_output/"
    img_zoom_output_dir="/content/ee-data/img_zoom_full_output/"

    # MODIS_processed_dir="C:/360Downloads/6_Data_county_processed_scaled/"

    # MODIS_dir="/atlas/u/jiaxuan/data/MODIS_data_county/3_Data_county"
    # MODIS_temperature_dir="/atlas/u/jiaxuan/data/MODIS_data_county_temperature"
    # MODIS_mask_dir="/atlas/u/jiaxuan/data/MODIS_data_county_mask"
    # MODIS_processed_dir="/atlas/u/jiaxuan/data/MODIS_data_county_processed_compressed/"

    data_yield = np.genfromtxt('/content/datalab/notebooks/crop_yield_prediction/2 clean data/yield_final.csv', delimiter=',', dtype=float)
    if file.endswith(".tif"):
        MODIS_path=os.path.join(MODIS_dir, file)
        # check file size to see if it's broken
        # if os.path.getsize(MODIS_path) < 10000000:
        #     print 'file broken, continue'
        #     continue
        MODIS_temperature_path=os.path.join(MODIS_temperature_dir,file)
        MODIS_mask_path=os.path.join(MODIS_mask_dir,file)

        # get geo location
        raw = file.replace('_',' ').replace('.',' ').split()
        loc1 = int(raw[0])
        loc2 = int(raw[1])
        # read image
        try:
            MODIS_img = np.transpose(np.array(gdal.Open(MODIS_path).ReadAsArray(), dtype='uint16'),axes=(1,2,0))
        except ValueError as msg:
            print msg
        # read temperature
        MODIS_temperature_img = np.transpose(np.array(gdal.Open(MODIS_temperature_path).ReadAsArray(), dtype='uint16'),axes=(1,2,0))
        # shift
        # MODIS_temperature_img = MODIS_temperature_img-12000
        # scale
        # MODIS_temperature_img = MODIS_temperature_img*1.25
        # clean
        # MODIS_temperature_img[MODIS_temperature_img<0]=0
        # MODIS_temperature_img[MODIS_temperature_img>5000]=5000
        # read mask
        MODIS_mask_img = np.transpose(np.array(gdal.Open(MODIS_mask_path).ReadAsArray(), dtype='uint16'),axes=(1,2,0))
        # Non-crop = 0, crop = 1
        MODIS_mask_img[MODIS_mask_img != 12] = 0
        MODIS_mask_img[MODIS_mask_img == 12] = 1

        # Divide image into years
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
                # # detect quality
                # quality = quality_dector(MODIS_list_masked[i])
                # if quality < 0.01:
                #     print 'omitted'
                #     print year,loc1,loc2,quality

                    # # delete
                    # yield_all = np.genfromtxt('yield_final_highquality.csv', delimiter=',')
                    # key = np.array([year,loc1,loc2])
                    # index = np.where(np.all(yield_all[:,0:3] == key, axis=1))
                    # yield_all=np.delete(yield_all, index, axis=0)
                    # np.savetxt("yield_final_highquality.csv", yield_all, delimiter=",")

                    # continue

                ## 1 save original file
                filename=img_output_dir+str(year)+'_'+str(loc1)+'_'+str(loc2)+'.npy'
                np.save(filename,MODIS_list_masked[i])
                print filename,':written '

                ## 2 save zoomed file (48*48)
                zoom0 = float(48) / MODIS_list_masked[i].shape[0]
                zoom1 = float(48) / MODIS_list_masked[i].shape[1]
                output_image = zoom(MODIS_list_masked[i], (zoom0, zoom1, 1))

                filename=img_zoom_output_dir+str(year)+'_'+str(loc1)+'_'+str(loc2)+'.npy'
                np.save(filename,output_image)
                print filename,':written '


                

if __name__ == "__main__":
    # # save data
    # MODIS_dir="/atlas/u/jiaxuan/data/google_drive/data_image_full"
    MODIS_dir="/content/ee-data/Data_county"
    for _, _, files in os.walk(MODIS_dir):
        Parallel(n_jobs=12)(delayed(preprocess_save_data_parallel)(file) for file in files)

    # # clean yield (low quality)
    # check_data_integrity_del()
    # # check integrity
    # check_data_integrity()


