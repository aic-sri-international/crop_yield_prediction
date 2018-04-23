import argparse
import os
import shutil
from os import listdir
from os.path import isfile, join
from multiprocessing import Pool
from build_histograms import GenerateHists
from process_data_for_histograms import preprocess_save_data

"""
Data source: download data from Google Earth Engine
Data imputation: converts geotiffs to histograms
Model objects: trains and evaluates a cnn and gp
Results: collect error outputs from models

"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='<LEAH>')
    parser.add_argument('-d', action='store_true', default=False, dest='download_data', help='download data from Google Earth Engine')
    parser.add_argument('-c', action='store_true', default=False, dest='convert_data', help='convert geotiffs to histograms')
    parser.add_argument('-m', action='store_true', default=False, dest='train_model', help='train the model')
    parser.add_argument('-r', action='store_true', default=False, dest='test_model', help='test the model')

    args = parser.parse_args()

    if args.download_data:
        # No
        pass
    if args.convert_data:
        output_dir = '/content/ee-data/img_full_output/'
        data_yield_file = '/content/datalab/notebooks/crop_yield_prediction/2 clean data/yield_final.csv'
        data_yield_subset_file = '/content/datalab/notebooks/crop_yield_prediction/2 clean data/yield_final_subset.csv'
        locations_file = '/content/datalab/notebooks/crop_yield_prediction/2 clean data/locations_final.csv'

        MODIS_dir = "/content/ira-gdrive/Data2/"
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)

        files = [f for f in listdir(MODIS_dir) if isfile(join(MODIS_dir, f))]
        files = [x for x in files if os.path.getsize(join(MODIS_dir, x)) < 157286400]

        with Pool(20) as p:
            try:
                p.map(preprocess_save_data, enumerate(files))
            except KeyboardInterrupt:
                p.terminate()
                p.join()

        # Histograms
        # TODO: make this not a class (it doesn't need to be an object)
        data = GenerateHists(output_dir, data_yield_file, data_yield_subset_file, locations_file)
        data.build_and_save_histogram()
    if args.train_model:
        pass
    if args.test_model:
        pass