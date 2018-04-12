mport numpy as np
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


class fetch_data():
    def __init__(self, output_dir, data_yield_file, data_yield_subset_file, locations_file):
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

        # data dir
        # self.dir = '/atlas/u/jiaxuan/data/MODIS_data/MODIS_data'
        #.  self.dir = "/atlas/u/jiaxuan/data/google_drive/img_output/"
#         self.dir='/content/ee-data/img_full_output/'
        
        # self.dir = "/atlas/u/jiaxuan/data/google_drive/img_full_output/"
        # self.dir = 'C:\\0machine_learning\\MODIS_data\\'
        # self.dir = 'F:/0SummerIntern/4_data_input/6_Data_county_processed/'
        # self.dir = 'C:/360Downloads/6_Data_county_processed/'
        # self.dir = '/media/sf_360Downloads/6_Data_county_processed/'

        # output dir
        # self.dir_output = 'C:/360Downloads/6_Data_county_processed_scaled/'
        # self.dir_output = '/atlas/u/jiaxuan/data/MODIS_data_county_processed_scaled/'

        # load yield data
#         self.data_yield = np.genfromtxt('/content/datalab/notebooks/crop_yield_prediction/2 clean data/yield_final_highquality.csv', delimiter=',')
#         self.locations = np.genfromtxt('/content/datalab/notebooks/crop_yield_prediction/2 clean data/locations_final.csv', delimiter=',')
        # load soil and weather data
        # self.soil = np.genfromtxt(self.dir+'soil_output.csv', delimiter=',')
        # self.weather = np.genfromtxt(self.dir+'daymet_mean.csv', delimiter=',')
        
#         self.data_yield_og = '/content/datalab/notebooks/crop_yield_prediction/2 clean data/yield_final.csv'
#         self.data_yield_subset = '/content/datalab/notebooks/crop_yield_prediction/2 clean data/yield_final_subset.csv'

        
        # load random data, no need to shuffle
        # np.random.shuffle(self.index_all)

        # # divide all data into 3 groups
        # self.index_train = self.index_all[0:int(length * 0.8)]
        # self.index_val = self.index_all[int(length * 0.8):int(length * 0.9)]
        # self.index_test = self.index_all[int(length * 0.9):length]

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



        # load mean image
        # try:
        #     self.image_mean = np.load('image_mean_county.npy')
        #     print 'image mean loaded', self.image_mean.shape
        # except:
        #     print 'no mean image found'
        #     self.image_mean = self.calc_mean()
        #     # print self.image_mean
        
    def create_yield_csv(self):
        files = [f for f in listdir(self.dir) if isfile(join(self.dir, f))]
        print ("number of files: {}".format(len(files)))
        with open(self.data_yield_subset_file, 'wb') as yield_csv:
            for f in files:
                file_str = os.path.splitext(basename(f))[0]
                yield_str = file_str.split("_")
                with open(self.data_yield_file) as og_yield:
                    crop_yield = None
                    for l in og_yield.readlines():
                        if ','.join(yield_str) in l:
                            crop_yield = l.split(",")[3][:-2]
                            break
                    if crop_yield:
                        yield_str.append(crop_yield)
                        wr = csv.writer(yield_csv)
                        wr.writerow(yield_str)

    def calc_mean(self):
        # traverse training data
        image_mean = np.zeros([32, 32, 9], dtype=np.float32)

        print 'traversing training data to calc image_mean:'
        n = 0
        for i in self.index_train:
            year = str(int(self.data_yield[i, 0]))
            loc1 = str(int(self.data_yield[i, 1]))
            loc2 = str(int(self.data_yield[i, 2]))

            # filename = year + '_' + loc1 + '_' + loc2 + '.mat'
            filename = str(year)+'_'+str(loc1)+'_'+str(loc2)+ '.mat'
#             print "i: ", i
            with open(os.path.join(self.dir, filename), 'w'):
                pass
            content = io.loadmat(self.dir + filename)
            image_temp = content['image_divide']
            image_temp = self.filter_size(image_temp, 48)
            image_temp = self.filter_timespan(image_temp, 49, 305, 7)
            image_temp = self.filter_abnormal(image_temp, 0, 5000)
            image_temp = self.filter_crop_mask(image_temp, year, loc1, loc2)


            ratio = float(n) / float(n + 1)
            image_mean *= ratio
            ratio = float(n + 1)
            image_temp /= ratio
            image_mean += image_temp
            n += 1

            # print ratio
            # print np.mean(image_mean)
            if n % 1000 == 0:
                print n
        # print image_mean
        image_mean = image_mean.astype(dtype='float32', copy=False)
        np.save('image_mean_48by48.npy', image_mean)
        return image_mean

    def filter_size(self,image_temp,size):
        return image_temp[0:size, 0:size, :].astype(np.float32)

    def filter_abnormal(self,image_temp,min,max):
        image_temp[image_temp<min]=min
        image_temp[image_temp>max]=max
        return image_temp

    def filter_timespan(self,image_temp,start_day,end_day,bands):
        start_index=int(math.floor(start_day/8))*bands
        end_index=int(math.floor(end_day/8))*bands
        if end_index>image_temp.shape[2]:
            image_temp = np.concatenate((image_temp, 
                np.zeros((image_temp.shape[0],image_temp.shape[1],end_index-image_temp.shape[2]))),axis=2)
        return image_temp[:,:,start_index:end_index]


    def calc_histogram(self,image_temp,bin_seq,bins,times,bands):
        hist=np.zeros([bins,times,bands])
        for i in range(image_temp.shape[2]):
#             print ("shape of img tmp 3rd dimension: {}, {}".format(image_temp[:, :, i].shape, i))
            density, _ = np.histogram(image_temp[:, :, i], bin_seq, density=False)
            # if density.sum()==0:
            #     continue
            hist[:, i / bands, i % bands] = density / float(density.sum())
#         print image_temp.shape[2]
        return hist

    def calc_histogram_flat(self,image_temp,bin_seq,bins,times,bands):
        hist=np.zeros([bins,times,bands])
        for i in range(image_temp.shape[1]):
            density, _ = np.histogram(image_temp[:, i], bin_seq, density=False)
            hist[:, i / bands, i % bands] = density / float(density.sum())
        return hist

    # def calc_histogram(self,image_temp,bin_seq_img,bin_seq_temp,bins,times,bands):
    #     hist=np.zeros([bins,times,bands])
    #     for i in range(image_temp.shape[2]):
    #         if i%bands <=6:
    #             density, _ = np.histogram(image_temp[:, :, i], bin_seq_img, density=False)
    #             hist[:, i / bands, i % bands] = density / float(density.sum())
    #         else:
    #             density, _ = np.histogram(image_temp[:, :, i], bin_seq_temp, density=False)
    #             hist[:, i / bands, i % bands] = density / float(density.sum())
    #     return hist

    # def calc_histogram_flat(self,image_temp,bin_seq_img,bin_seq_temp,bins,times,bands):
    #     hist=np.zeros([bins,times,bands])
    #     for i in range(image_temp.shape[1]):
    #         if i%bands <=6:
    #             density, _ = np.histogram(image_temp[:, i], bin_seq_img, density=False)
    #             hist[:, i / bands, i % bands] = density / float(density.sum())
    #         else:
    #             density, _ = np.histogram(image_temp[:, i], bin_seq_temp, density=False)
    #             hist[:, i / bands, i % bands] = density / float(density.sum())
    #     return hist

    def quality_dector(self,image_temp):
        filter_0=image_temp>0
        filter_5000=image_temp<5000
        filter=filter_0*filter_5000
        return float(np.count_nonzero(filter))/image_temp.size

    def next_batch_hist(self,batch_size,type='train'):
        if type == 'train':
            index = self.index_train
        elif type == 'validate':
            index = self.index_val
        elif type == 'test':
            index = self.index_test
        index_batch = np.random.choice(index, batch_size)
        output_image = np.zeros([batch_size, 32, 32, 9])
        output_yield = np.zeros([batch_size])
        for n, i in enumerate(index_batch):
            year = str(int(self.data_yield[i, 0]))
            loc1 = str(int(self.data_yield[i, 1]))
            loc2 = str(int(self.data_yield[i, 2]))

            # filename = year + '_' + loc1 + '_' + loc2 + '.mat'
            filename = year + '_' + loc1 + '_' + loc2 + '.npz'
            content = np.load(self.dir + filename)
            image_temp = content['arr_0']
            image_temp = self.filter_timespan(image_temp, 49, 305, 9)

            bin_seq=np.linspace(1,4999,33)
            image_temp = self.calc_histogram(image_temp,bin_seq,32,32,9)
            image_temp[np.isnan(image_temp)] = 0
            if np.sum(image_temp) < 250:
                print 'broken image',filename
                print np.isnan(image_temp)

            # output_image[n, :] = image_temp - self.image_mean
            output_image[n,:] = image_temp
            output_yield[n] = self.data_yield[i, 3]
            # print image_temp.shape
            # print np.sum(image_temp)
        return (np.float32(output_yield), np.float32(output_image))

    ## save unsupervised data
#     def save_data(self):
#         count_max = 20000
#         count = 0
#         output_image = np.zeros([count_max, 32, 32, 9])
#         output_yield = np.zeros([count_max])
#         output_year = np.zeros([count_max])
#         output_locations = np.zeros([count_max,2])
#         output_index = np.zeros([count_max,2])
#         files_to_delete = []

#         for i in self.index_all:
#             year = str(int(self.data_yield[i, 0]))
#             loc1 = str(int(self.data_yield[i, 1]))
#             loc2 = str(int(self.data_yield[i, 2]))

#             key = np.array([int(loc1),int(loc2)])
#             index = np.where(np.all(self.locations[:,0:2].astype('int') == key, axis=1))
#             longitude = self.locations[index,2]
#             latitude = self.locations[index,3]

#             filename = year + '_' + loc1 + '_' + loc2 + '.npy'
#             try:
#                 image_temp = np.load(self.dir + filename)
#                 image_temp = self.filter_timespan(image_temp, 49, 305, 9)
#                 print self.dir + filename
#                 # print image_temp.shape,image_temp.mean()

#                 image_temp=np.reshape(image_temp,(image_temp.shape[0]*image_temp.shape[1],image_temp.shape[2]),order='C')
#                 # remove 0 and 5000
#                 image_temp[image_temp==5000]=0
#                 # image_temp = image_temp[np.all(image_temp, axis=1)]
#                 image_temp = image_temp[~np.all(image_temp == 0, axis=1)]
#                 # print image_temp.shape

#                 crop_pixel_count = 200
#                 j = 0
#                 while j < image_temp.shape[0]/crop_pixel_count:
#                     image_temp_part = image_temp[j*crop_pixel_count:(j+1)*crop_pixel_count,:]
#                     j += 1
#                     bin_seq = np.linspace(1, 4999, 33)
#                     image_temp_part = self.calc_histogram_flat(image_temp_part, bin_seq,32, 32, 9)
#                     image_temp_part[np.isnan(image_temp_part)] = 0
#                     # if np.sum(image_temp_part) < 288:
#                     #     print 'broken image', filename, np.sum(image_temp_part)
#                     #     continue

#                     epoch = count/count_max
#                     #saver
#                     if count%count_max == 0 and count!=0:
#                         # save
#                         np.savez(self.dir+'histogram_semi_rand_200_20000'+str(epoch)+'.npz',
#                              output_image=output_image,output_yield=output_yield,
#                              output_year=output_year,output_locations=output_locations,output_index=output_index)
#                         print 'save',self.dir+'histogram_semi_rand_200_20000'+str(epoch)+'.npz'
#                         # clear
#                         output_image = np.zeros([count_max, 32, 32, 9])
#                         output_yield = np.zeros([count_max])
#                         output_year = np.zeros([count_max])
#                         output_locations = np.zeros([count_max,2])
#                         output_index = np.zeros([count_max,2])

#                     output_image[count-epoch*count_max, :] = image_temp_part
#                     output_yield[count-epoch*count_max] = self.data_yield[i, 3]
#                     output_year[count-epoch*count_max] = int(year)
#                     output_locations[count-epoch*count_max, 0] = longitude
#                     output_locations[count-epoch*count_max, 1] = latitude
#                     output_index[count-epoch*count_max,:] = np.array([int(loc1),int(loc2)])
# #                     print epoch,i,j,count,np.sum(image_temp_part),year,loc1,loc2
#                     count += 1
#             except IOError as e:
#                 files_to_delete.append(i)
# #                 print("ERROR: {}".format(e))
#         output_image = [x for i,x in enumerate(output_image) if i not in files_to_delete]
#         output_yield = [x for i,x in enumerate(output_yield) if i not in files_to_delete]
#         output_year = [x for i,x in enumerate(output_year) if i not in files_to_delete]
#         output_locations = [x for i,x in enumerate(output_locations) if i not in files_to_delete]
#         output_index = [x for i,x in enumerate(output_index) if i not in files_to_delete]
#         np.savez(self.dir+'histogram_all.npz',
#                  output_image=output_image,output_yield=output_yield,
#                  output_year=output_year,output_locations=output_locations,output_index=output_index)
#         print 'save done'

    # save mean data
    def save_data_mean(self):
        path, dirs, files = os.walk(self.dir).next()
        file_count = len(files)
        output_image = np.zeros([file_count, 32*9])
        output_yield = np.zeros([file_count])
        output_year = np.zeros([file_count])
        output_locations = np.zeros([file_count,2])
        output_index = np.zeros([file_count,2])
#         output_image = np.zeros([self.index_all.shape[0], 32*9])
#         output_yield = np.zeros([self.index_all.shape[0]])
#         output_year = np.zeros([self.index_all.shape[0]])
#         output_locations = np.zeros([self.index_all.shape[0],2])
#         output_index = np.zeros([self.index_all.shape[0],2])

        files_to_delete = []
        print "file_count: ", file_count
        for file in files:
            file_index = files.index(file)
            year, loc1, loc2, ext = file.replace('_',' ').replace('.',' ').split()
#             print year, loc1, loc2
#             year = str(int(self.data_yield[i, 0]))
#             loc1 = str(int(self.data_yield[i, 1]))
#             loc2 = str(int(self.data_yield[i, 2]))

            key = np.array([int(loc1),int(loc2)])
            index = np.where(np.all(self.locations[:,0:2].astype('int') == key, axis=1))
            longitude = self.locations[index,2]
            latitude = self.locations[index,3]

#             filename = year + '_' + loc1 + '_' + loc2 + '.npy'
            image_temp = np.load(self.dir + file)
            image_temp = self.filter_timespan(image_temp, 49, 305, 9)

            image_temp = np.sum(image_temp,axis=(0,1))/np.count_nonzero(image_temp)*image_temp.shape[2]
            image_temp[np.isnan(image_temp)] = 0
            
#             file_index = files.index(file)
            output_image[file_index, :] = image_temp
            for i in range(len(self.data_yield)):
                out_yield = None
                if float(year) == self.data_yield[i, 0] and float(loc1) == self.data_yield[i, 1] and float(loc2) == self.data_yield[i, 2]:
                    out_yield = self.data_yield[i, 3]
                    break
            if not out_yield:
                files_to_delete.append(file_index)
                print "Error: no out_yield! ", year, loc1, loc2
                continue
            output_yield[file_index] = out_yield  # self.data_yield[i, 3]
            output_year[file_index] = int(year)
            output_locations[file_index, 0] = longitude
            output_locations[file_index, 1] = latitude
            output_index[file_index,:] = np.array([int(loc1),int(loc2)])
            # print image_temp.shape
#             print file_index,np.sum(image_temp),year,loc1,loc2
        output_image = [x for i,x in enumerate(output_image) if i not in files_to_delete]
        output_yield = [x for i,x in enumerate(output_yield) if i not in files_to_delete]
        output_year = [x for i,x in enumerate(output_year) if i not in files_to_delete]
        output_locations = [x for i,x in enumerate(output_locations) if i not in files_to_delete]
        output_index = [x for i,x in enumerate(output_index) if i not in files_to_delete]
        np.savez(self.dir+'histogram_all_mean.npz',
                 output_image=output_image,output_yield=output_yield,
                 output_year=output_year,output_locations=output_locations,output_index=output_index)
        print 'save done'
        
    # save supervised data
    def save_data(self):
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
            # LEAH: add locations intuitively, don't just pull from file in order
            # this could be improved to make for faster processing - create a new locations file based
            # on the subset data_yield
            for l in self.locations:
                if int(l[0]) == int(loc1) and int(l[1]) == int(loc2):
                    location = l
                    break

            key = np.array([int(loc1),int(loc2)])
#             index = np.where(np.all(self.locations[:,0:2].astype('int') == key, axis=1))
#             longitude = self.locations[index,2]
#             latitude = self.locations[index,3]
            # LEAH'S CHANGES:
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
            # if np.sum(image_temp) < 250:
            #     print 'broken image', filename
            #     print np.isnan(image_temp)

            # i is matched with data_yield at beginning of for loop
            output_image[i, :] = image_temp
            output_yield[i] = self.data_yield[i, 3]
            output_year[i] = int(year)
            output_locations[i, 0] = longitude
            output_locations[i, 1] = latitude
            output_index[i,:] = np.array([int(loc1),int(loc2)])
            # print image_temp.shape
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
    data=fetch_data(output_dir, data_yield_file, data_yield_subset_file, locations_file)
    # calculate mean
    # data.calc_mean()

#     output_yield, output_image=data.next_batch(8,'train')
#     print output_yield.shape,output_yield
#     print output_image.shape,output_image
#     clean.preprocess_save_data()
    # label,img=data.next_batch_hist(data.index_test.size,'test')
    # i=0
    # print img[i].shape, img[i].dtype, img[i][:,0,0].sum()

    # check_data_integrity()
    # data.calc_mean()
    # i,a=data.next_batch_hist(32,'train')
    data.save_data()
    print 'DONE'
    # data.save_data()
    # data.save_data_mean()
    # data.save_data('validate')
    # data.save_data('test')
