import numpy as np
import scipy.io as io
import math
import os
from os import listdir
from os.path import basename, isfile, join
import skimage.io
import pandas as pd
# from clean_data_county import *
from joblib import Parallel, delayed
import multiprocessing
import final_clean_data as clean
import shutil
import csv


"""histogram creation module for crop_yield_prediction

This module reads in numpy files that represent geotiff images and builds histograms
based on the file contents.

Histogram: (time, num_bins, num_bands) = (32, 32, 9)
    time: the y-axis, the time within a given year at which a data point was taken
    num_bins: the y-axis, total number of bins, where a bin is a group of pixels in a range of 256 different values
    num_bands: spectral bands, each band has 32 bins, where each bin has up to 256 different values.


Example
-------
Run this module:

    $ python final_save_histogram.py

"""

class CropHistograms():
    """Read in numpy data and convert to histograms

    Parameters
    ----------
    output_dir : str
        file path of directory with files of name <year>_<state_num>_<county_num>.npy, containaing the numpy arrays
        outputted by final_clean_data
        Also the directory where histograms will be saved
    data_yield_file : str
        file path containing the csv info corresponding to the data yield for each location and year
    data_yield_subset_file : str
        file path where the new csv yields corresponding to the output_dir files will be stored
    locations_file : str
        file path that contains the latitude and longitude values corresponding to the proper output_dir file

        """
    def __init__(self, output_dir, data_yield_file, data_yield_subset_file, locations_file):
        """
        Parameters
        ----------
        output_dir : str
            file path of directory with files of name <year>_<state_num>_<county_num>.npy, containaing the numpy arrays
            outputted by final_clean_data
            Also the directory where histograms will be saved
        data_yield_file : str
            file path containing the csv info corresponding to the data yield for each location and year
        data_yield_subset_file : str
            file path where the new csv yields corresponding to the output_dir files will be stored
        locations_file : str
            file path that contains the latitude and longitude values corresponding to the proper output_dir file
        """
        self.dir = output_dir
        self.data_yield_file = data_yield_file
        self.data_yield_subset_file = data_yield_subset_file
        self.locations = np.genfromtxt(locations_file, delimiter=',')
        # fill in subset yield
        self.create_yield_csv()
        self.data_yield = np.genfromtxt(data_yield_subset_file, delimiter=',')
                
        # generate index for all data
        length = self.data_yield.shape[0]
        self.index_all = np.arange(length)

        self.year = 2012
        # divide data by year
        self.index_train=[]
        self.index_val=[]
        for i in range(self.data_yield.shape[0]):
            if self.data_yield[i,0]==self.year:
                self.index_val.append(i)
            else:
                self.index_train.append(i)
        self.index_train=np.array(self.index_train)
        self.index_val=np.array(self.index_val)

        
    def create_yield_csv(self):
        """Build the csv of yields only for the files that exist in output_dir"""
        files = [f for f in listdir(self.dir) if isfile(join(self.dir, f))]  # output files created in preprocess_data
        print ("number of files: {}".format(len(files)))
        with open(self.data_yield_subset_file, 'wb') as yield_csv:
            for f in files:
                # for each file, if there is a corresponding yield, write to a new subset file
                file_str = os.path.splitext(basename(f))[0]
                yield_str = file_str.split("_")
                with open(self.data_yield_file) as og_yield:  # open the original data yield file
                    crop_yield = None
                    for l in og_yield.readlines():
                        # for each line in the original data_yield, check if the corresponding yield has been saved
                        if ','.join(yield_str) in l:
                            # if the corresponding yield has been saved in preprocess_data, fill crop_yield and break
                            crop_yield = l.split(",")[3][:-2]
                            break
                    if crop_yield:
                        # if yield was found for file f, write crop_yield to data_yield_subset_file
                        yield_str.append(crop_yield)
                        wr = csv.writer(yield_csv)
                        wr.writerow(yield_str)

    def filter_timespan(self,image_temp,start_day,end_day,bands):
        """Filter the current image by a given time frame

        Parameters
        ----------
        image_temp
            current image to build histogram off of
        start_day
            day of the year the histogram will start at
        end_day
            day of the year the histogram will end at
        bands
            number of bands for the histogram

        Returns
        -------
        numpy array
            the filtered image
        """
        start_index=int(math.floor(start_day/8))*bands
        end_index=int(math.floor(end_day/8))*bands
        if end_index>image_temp.shape[2]:
            image_temp = np.concatenate((image_temp, 
                np.zeros((image_temp.shape[0],image_temp.shape[1],end_index-image_temp.shape[2]))),axis=2)
        return image_temp[:,:,start_index:end_index]


    def calc_histogram(self,image_temp,bin_seq,bins,times,bands):
        """

        Parameters
        ----------
        image_temp
            current image being processed for the histogram
        bin_seq
            numpy linspace object: evenly spaced numbers over an interval
        bins
            int: number of bins
        times
            int: number of data points taken over a given year
        bands
            int: number of bands

        Returns
        -------
        hist
            numpy histogram object
        """
        hist=np.zeros([bins,times,bands])  # build zeroed numpy array of shape (bins, times, bands)
        for i in range(image_temp.shape[2]):
            # get the densities for the provided sequence and fill the histogram with the mean density
            density, _ = np.histogram(image_temp[:, :, i], bin_seq, density=False)
            hist[:, i / bands, i % bands] = density / float(density.sum())
        return hist

    # save supervised data
    def build_and_save_histogram(self):
        """Build and save the histogram based on the file corresponding to the provided yield."""
        output_image = np.zeros([self.index_all.shape[0], 32, 32, 9])
        output_yield = np.zeros([self.index_all.shape[0]])
        output_year = np.zeros([self.index_all.shape[0]])
        output_locations = np.zeros([self.index_all.shape[0],2])
        output_index = np.zeros([self.index_all.shape[0],2])

        for i in self.index_all:
            year = str(int(self.data_yield[i, 0]))
            loc1 = str(int(self.data_yield[i, 1]))
            loc2 = str(int(self.data_yield[i, 2]))

            location = None
            for l in self.locations:
                if int(l[0]) == int(loc1) and int(l[1]) == int(loc2):
                    location = l
                    break

            key = np.array([int(loc1),int(loc2)])
            longitude = location[2]
            latitude = location[3]

            filename = year + '_' + loc1 + '_' + loc2 + '.npy'
            image_temp = np.load(self.dir + filename)
            # Paper uses 281 for end_day, code uses 305 for end_day in the following line:
            image_temp = self.filter_timespan(image_temp, 49, 305, 9)
            image_temp[np.isnan(image_temp)] = 0

            # 33 objects, evenly spaced between 1 and 4999 - approximately 151 step size between
            # color spectrum in temperature images adjusted to be 0-5000 for reasons unknown
            # possibly due to crop spectrum defaults which we don't know...
            bin_seq = np.linspace(1, 4999, 33)
            # calc_histogram inputs: img, bin_seq, bins, times, bands
            # in the paper, times = 30, here times = 32 possibly due to using 305 days vs 281 in the paper
            image_temp = self.calc_histogram(image_temp, bin_seq ,32, 32, 9)
            image_temp[np.isnan(image_temp)] = 0

            # i is matched with data_yield at beginning of for loop
            output_image[i, :] = image_temp
            output_yield[i] = self.data_yield[i, 3]
            output_year[i] = int(year)
            output_locations[i, 0] = longitude
            output_locations[i, 1] = latitude
            output_index[i,:] = np.array([int(loc1),int(loc2)])

            print i,np.sum(image_temp),year,loc1,loc2
        np.savez(self.dir+'histogram_all_full.npz',
                 output_image=output_image,output_yield=output_yield,
                 output_year=output_year,output_locations=output_locations,output_index=output_index)
        print 'save done'

        
if __name__ == '__main__':

    output_dir='/content/ee-data/img_full_output/'
    data_yield_file = '/content/datalab/notebooks/crop_yield_prediction/2 clean data/yield_final.csv'
    data_yield_subset_file = '/content/datalab/notebooks/crop_yield_prediction/2 clean data/yield_final_subset.csv'
    locations_file = '/content/datalab/notebooks/crop_yield_prediction/2 clean data/locations_final.csv'        

    data=CropHistograms(output_dir, data_yield_file, data_yield_subset_file, locations_file)
    data.build_and_save_histogram()
