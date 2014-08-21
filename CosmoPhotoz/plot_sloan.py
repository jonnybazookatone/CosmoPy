#!/usr/local/bin/python2.7
from photoz import PhotoSample

# Data
train_file = "../data/SDSS_train.csv"
test_file = "../data/SDSS_test.csv"

# I will use a gamma family with log link
SDSS = PhotoSample(filename_train=train_file, filename_test=test_file, family="Gamma", link="log")
SDSS.color_palette = "MediumPurple"
SDSS.reduce_size = 5000

# I will take default values for now
SDSS.run_full()
