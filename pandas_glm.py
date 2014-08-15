#!/usr/bin/env python

"""
A small example of how to fit photo-zs with Generalised Linear Models.
This has been adopted from Rafael's R scripts.

Usage:
 - pandas_glm.py -d/--dataset <DATASET>

Datasets:
 - SDSS:  SQL query taken from Krone-Martin et al. 2014
 - 2SLAQ: Same sample taken from Abdallah et al. 2011
 - PHAT0: This is from the photometric association

Details:
 - The response (redshift) is positive and continuous: Gamma family is used (log link)
 - Principle Component Analysis has been used to ensure each feature is independent from one another

Libraries used:

 - pandas:      Allows the use of DataFrames (alla R)
 - sklearn:     Easy implementation of PCA analysis
 - statsmodels: Easy implementation of GLM and fitting via IRLS (alla R)
 - seaborn:     Makes the fancy pandas plots fancier
 - matplotlib:  General plotting
 - time:        For timing
 - logging:     For logging
 - numpy:       For arrays
 - argparse:    Allow users to use it easily from the command line

"""

__author__ = "Jonathan Elliott"
__copyright__ = "Copyright 2014"
__version__ = "1.0"
__email__ = "jonnynelliott@googlemail.com"
__status__ = "Prototype"

from matplotlib.mlab import griddata
import matplotlib.pyplot as plt

import numpy as np

from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split

import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.genmod as smg

import pandas as pd

import seaborn as sns

import time, argparse, logging

# Setup the logger to the command line
# To a file can also be added fairly easily
logfmt = '%(levelname)s [%(asctime)s]:\t  %(message)s'
datefmt= '%m/%d/%Y %I:%M:%S %p'
formatter = logging.Formatter(fmt=logfmt,datefmt=datefmt)
logger = logging.getLogger('__main__')
logging.root.setLevel(logging.DEBUG)
ch = logging.StreamHandler() #console handler
ch.setFormatter(formatter)
logger.addHandler(ch)

def main(DATASET):

  t_start = time.time()

  DATADIR = "../data/"

  data_dictionary = {
  "SDSS": {"Filename": "{0}/gal.csv".format(DATADIR), "PCAHeader": ["u","g","r","i","z"], \
           "n_components": 5, "formula": "redshift ~ PC1*PC2*PC3",\
           "test_size": 4000  , "sample_size": -1, \
           "remove_header": "specObjID",
           "lims": {"x": [0., 1.5], "y": [0., 1.5]},
           },

  "PHAT0": {"Filename": "{0}/noise_training_big.csv".format(DATADIR), "PCAHeader": ["up","gp","rp","ip","zp","Y","J","H","K","IRAC_1","IRAC_2"], \
            "n_components": 9, "formula": "redshift ~ PC1*PC2*PC3*PC4*PC5",\
            "test_size": 5000, "sample_size": -1, \
            "remove_header": False, \
            "lims": {"x": [0., 1.5], "y": [0., 1.5]},
            },

  "2SLAQ": {"Filename": "{0}/2slaq.csv".format(DATADIR), "PCAHeader": ["mag_u","mag_g","mag_r","mag_i","mag_z"], \
            "n_components": 5, "formula": "redshift ~ PC1*PC2*PC3",\
            "test_size": 5482, "sample_size": -1, \
            "remove_header": False, \
            "lims": {"x": [0.3, 0.8], "y": [0.41, 0.71]},
            },
  }

  #DATASET = "SDSS"

  # Load the data into a pandas DataFrame - much like R data frame
  df_fit = pd.read_csv(data_dictionary[DATASET]["Filename"], encoding="utf-8")
  logger.info("Loaded DataFrame: {0}".format(data_dictionary[DATASET]["Filename"]))

  # Hack for removing a header
  if data_dictionary[DATASET]["remove_header"]:
    remove_header = data_dictionary[DATASET]["remove_header"]
    try:
      df_fit.drop(remove_header, axis=1, inplace=True)
      logger.info("Removing header: {0}".format(remove_header))
    except:
      logger.warning("FAILED TO: Remove header: {0}".format(remove_header))

  #df_fit = df[:data_dictionary[DATASET]["sample_size"]]
  
  # Load the PCA class
  logger.info("Carrying out Principle Component Analysis ({0} components)".format(data_dictionary[DATASET]["n_components"]))
  
  pca = PCA(n_components=data_dictionary[DATASET]["n_components"])
  pca.fit(df_fit[data_dictionary[DATASET]["PCAHeader"]])

  M_pca = pca.fit_transform(df_fit)
  
  M_df = {}
  M_df["redshift"] = df_fit["redshift"].values

  for i in range(data_dictionary[DATASET]["n_components"]):
    M_df["PC{0:d}".format(i+1)] = M_pca[:,i]

  

  df_pca = pd.DataFrame(M_df)

  # Cross Validation Section
  ##
  logger.info("Splitting into training/testing sets. Number of testing: {0}".format(data_dictionary[DATASET]["test_size"]))
  ## Split into train/test
  test_size = df_pca.shape[0] - data_dictionary[DATASET]["test_size"]
  train, test = train_test_split(df_pca, test_size=int(test_size), random_state=42)

  logger.info("Training set length: {0}".format(train.shape[0]))
  logger.info("Testing set length: {0}".format(test.shape[0]))

  ## Redefine some DataFrames, otherwise they are just numpy arrays
  col_train = {}
  col_test = {}
  col_train["redshift"] = train[:,-1]
  col_test["redshift"] = test[:,-1]

  for i in range(data_dictionary[DATASET]["n_components"]):
    col_train["PC{0:d}".format(i+1)] = train[:,i]
    col_test["PC{0:d}".format(i+1)] = test[:,i]

  
  df_test = pd.DataFrame(col_test)
  df_train = pd.DataFrame(col_train)

 

  # GLM
  ## Set the formula
  formula = data_dictionary[DATASET]["formula"]

  ## Load the Gamma family and the log-link
  
  log_link = smg.families.links.log
  family = sm.families.Gamma(link=log_link)

  logger.info("GLM with Gamma family,\tformula: {0}\tlink: log".format(formula))
  logger.info("Fitting...")
  model = smf.glm(formula=formula, data=df_train, family=family)
  results = model.fit()
  print(results.summary())


  # Plot the model with our test data
  ## Prediction
  measured = np.array(df_test["redshift"].values)
  predicted = results.predict(df_test)

  ## Outliers
  ## (z_phot - z_spec)/(1+z_spec)
  outliers = (predicted - measured) / (1.0 + measured)
  
  t_end = time.time()
  t_taken = (t_end - t_start) / 60. # minutes
  

  # R code
  # Out<-100*length(PHAT0.Pred$fit[(abs(PHAT0.test.PCA$redshift-PHAT0.Pred$fit))>0.15*(1+PHAT0.test.PCA$redshift)])/length(PHAT0.Pred$fit)
  catastrophic_error = 100.0*(abs(measured-predicted) > (0.15*(1+measured))).sum()/(1.0*measured.shape[0])
  logger.info("Catastrophic Error: {0}%".format(catastrophic_error))

  logger.info("Time taken: {0} minutes".format(t_taken))
  
  # Load figure canvas
  ##
  plot = True
  if plot:
    fig = plt.figure(0)
    
    # ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(111)

    x_straight = np.arange(0,1.6,0.1)
    # ax1.plot(x_straight, x_straight, color="black", lw=2)

    # sns.kdeplot(measured, predicted, bw="silverman", grid=50, cmap="BuGn_d", ax=ax1)
    # ax1.errorbar(measured, predicted, fmt="o",zorder=1, color="gray", label="{0}".format(DATASET))
    
    # ax1.set_xlabel(r"$z_{\rm spec}$", fontsize=20)
    # ax1.set_ylabel(r"$z_{\rm phot}$", fontsize=20)
    # ax1.set_xlim(data_dictionary[DATASET]["lims"]["x"])
    # ax1.set_ylim(data_dictionary[DATASET]["lims"]["y"])

    #          kde_kws={"color": "seagreen", "lw": 3, "label": "KDE"},
    sns.distplot(outliers,
             hist_kws={"histtype": "stepfilled", "color": "slategray"}, ax=ax2)
    
    ax2.set_xlabel(r"$(z_{\rm phot}-z_{\rm spec})/1+z_{\rm spec}$", fontsize=20)
    ax2.set_ylabel(r"$\rm Density$", fontsize=20)
    ax2.set_position([.15,.15,.75,.75])

    plt.savefig("KDE_PLOT_{0}.pdf".format(DATASET), format="pdf")

  plt.clf()
  g = sns.JointGrid(measured, predicted,size=4,space=0)
  g.plot_marginals(sns.distplot, kde=True, color="green")
  g.plot_joint(plt.scatter, color="silver", edgecolor="white")
  g.plot_joint(sns.kdeplot, kind="hex")
  g.ax_joint.set(xlim=data_dictionary[DATASET]["lims"]["x"], ylim=data_dictionary[DATASET]["lims"]["y"])  
  g.set_axis_labels(xlabel=r"$z_{\rm spec}$", ylabel=r"$z_{\rm phot}$")
  g.ax_joint.errorbar(x_straight, x_straight, lw=2)

  # Temp solution
  # http://stackoverflow.com/questions/21913671/subplots-adjust-with-seaborn-regplot
  axj, axx, axy = plt.gcf().axes
  axj.set_position([.15, .12, .7, .7])
  axx.set_position([.15, .85, .7, .13])
  axy.set_position([.88, .12, .13, .7])

  plt.savefig("KDE_2D_PLOT_{0}.pdf".format(DATASET), format="pdf")


if __name__=='__main__':

  parser = argparse.ArgumentParser(usage=__doc__)
  parser.add_argument('-d','--dataset',dest="dataset",default=None, required=True)

  args = parser.parse_args()

  main(args.dataset)
