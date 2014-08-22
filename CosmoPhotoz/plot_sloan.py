#!/usr/local/bin/python2.7
from photoz import PhotoSample

# Data
#train_file = "../data/SDSS_train.csv"
#test_file = "../data/SDSS_test.csv"
filename = "../data/SDSS_small.csv"


# I will use a gamma family with log link
#SDSS = PhotoSample(filename_train=train_file, filename_test=test_file, family="Gamma", link="log")
SDSS = PhotoSample(filename=filename, family="Gamma", link="log")
SDSS.test_size = 10000
SDSS.color_palette = "MediumPurple"
SDSS.reduce_size = 5000

# I will take default values for now
SDSS.run_full()
