#!/usr/local/bin/python2.7
from photoz import PhotoSample

data_file = "../data/PHAT0_small.csv" # Called small, but this just means that it is ordered like the other files

# I will use a gamma family with log link
PHAT0 = PhotoSample(filename=data_file, family="Gamma", link="log")
PHAT0.num_components = 6

PHAT0.color_palette = "seagreen"
PHAT0.reduce_size = 5000

# Run
PHAT0.run_full()