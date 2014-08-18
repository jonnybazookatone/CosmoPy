#!/usr/bin/env python

"""
"""

__author__ = "Jonathan Elliott"
__copyright__ = "Copyright 2014"
__version__ = "1.0"
__email__ = "jonnynelliott@googlemail.com"
__status__ = "Prototype"

import numpy as np
import pandas as pd
import time, argparse, logging

class PhotoSample(object):

  def __init__(self, filename):

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

    self.logger = logger

    # Check they give a filename
    if not filename:
      self.logger.warning("You must give a filename")
      sys.exit(0)
      
    self.filename = filename

    # PCA analysis
    self.num_components = 5
    self.test_size = 10000
    self.data_frame = None
    self.data_frame_header = ["up","gp","rp","ip","zp"]
    self.PCA_data_frame = None
    self.data_frame_test = None
    self.data_frame_train = None

    # GLM
    ## Set the formula
    self.formula = "redshift ~ PC1*PC2*PC3"

    # Plots
    self.lims = {"x": [0.3, 0.8], "y": [0.41, 0.71]}

  def load_data_frame(self):
    try:
      self.data_frame = pd.read_csv(self.filename, encoding="utf-8")
    except:
      self.logger.info("Failed to open CSV file: {0}".format(sys.exc_info()[0]))

  def do_PCA(self):

    # TODO:
    # Do the following on the fly:
    #   1. determine num_components
    #   2. determine formula
    #   3. determine test size

    from sklearn.decomposition import PCA
    self.logger.info("Carrying out Principle Component Analysis ({0} components)".format(self.num_components))
    pca = PCA(self.num_components)
    pca.fit(self.data_frame[self.data_frame_header])
    M_pca = pca.fit_transform(self.data_frame)

    M_df = {}
    M_df["redshift"] = self.data_frame["redshift"].values

    for i in range(self.num_components):
      M_df["PC{0:d}".format(i+1)] = M_pca[:,i]

    self.PCA_data_frame = pd.DataFrame(M_df)

  def split_sample(self):

    from sklearn.cross_validation import train_test_split
    # Cross Validation Section
    ##
    self.logger.info("Splitting into training/testing sets. Number of testing: {0}".format(self.test_size))
    ## Split into train/test
      
    train, test = train_test_split(self.PCA_data_frame, test_size=int(self.test_size), random_state=42)

    self.logger.info("Training set length: {0}".format(train.shape[0]))
    self.logger.info("Testing set length: {0}".format(test.shape[0]))

    ## Redefine some DataFrames, otherwise they are just numpy arrays
    col_train = {}
    col_test = {}
    col_train["redshift"] = train[:,-1]
    col_test["redshift"] = test[:,-1]

    for i in range(self.num_components):
      col_train["PC{0:d}".format(i+1)] = train[:,i]
      col_test["PC{0:d}".format(i+1)] = test[:,i]

    self.data_frame_test = pd.DataFrame(col_test)
    self.data_frame_train = pd.DataFrame(col_train)

  def do_GLM(self):

    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    import statsmodels.genmod as smg

    ## Load the Gamma family and the log-link
    log_link = smg.families.links.log
    family = sm.families.Gamma(link=log_link)

    self.logger.info("GLM with Gamma family,\tformula: {0}\tlink: log".format(self.formula))
    self.logger.info("Fitting...")
    model = smf.glm(formula=self.formula, data=self.data_frame_train, family=family)
    results = model.fit()
    self.logger.info(results.summary())

    # Plot the model with our test data
    ## Prediction
    self.measured = np.array(self.data_frame_test["redshift"].values)
    self.predicted = results.predict(self.data_frame_test)

    ## Outliers
    ## (z_phot - z_spec)/(1+z_spec)
    self.outliers = (self.predicted - self.measured) / (1.0 + self.measured)

    # R code
    # Out<-100*length(PHAT0.Pred$fit[(abs(PHAT0.test.PCA$redshift-PHAT0.Pred$fit))>0.15*(1+PHAT0.test.PCA$redshift)])/length(PHAT0.Pred$fit)
    self.catastrophic_error = 100.0*(abs(self.measured-self.predicted) > (0.15*(1+self.measured))).sum()/(1.0*self.measured.shape[0])
    self.logger.info("Catastrophic Error: {0}%".format(self.catastrophic_error))

  def make_plot(self, show=False):

    from matplotlib.mlab import griddata
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Load figure canvas
    ##
    fig = plt.figure(0)

    ax = fig.add_subplot(111)
    x_straight = np.arange(0,1.6,0.1)
    
    sns.distplot(self.outliers, hist_kws={"histtype": "stepfilled", "color": "slategray"}, ax=ax)
  
    ax.set_xlabel(r"$(z_{\rm phot}-z_{\rm spec})/1+z_{\rm spec}$", fontsize=20)
    ax.set_ylabel(r"$\rm Density$", fontsize=20)
    ax.set_position([.15,.15,.75,.75])

    plt.savefig("KDE_PLOT_{0}.pdf".format(self.filename.split("/")[-1]), format="pdf")
    plt.clf()

    g = sns.JointGrid(self.measured, self.predicted,size=4,space=0)
    g.plot_marginals(sns.distplot, kde=True, color="green")
    g.plot_joint(plt.scatter, color="silver", edgecolor="white")
    g.plot_joint(sns.kdeplot, kind="hex")
    g.ax_joint.set(xlim=self.lims["x"], ylim=self.lims["x"])  
    g.set_axis_labels(xlabel=r"$z_{\rm spec}$", ylabel=r"$z_{\rm phot}$")
    g.ax_joint.errorbar(x_straight, x_straight, lw=2)

    # Temp solution
    # http://stackoverflow.com/questions/21913671/subplots-adjust-with-seaborn-regplot
    axj, axx, axy = plt.gcf().axes
    axj.set_position([.15, .12, .7, .7])
    axx.set_position([.15, .85, .7, .13])
    axy.set_position([.88, .12, .13, .7])

    plt.savefig("KDE_2D_PLOT_{0}.pdf".format(self.filename.split("/")[-1]), format="pdf")
    if show:
      plt.show()

  def print_info(self):
    self.logger.info("Everything is ok")

  def run_full(self, show=False):
    self.load_data_frame()
    self.do_PCA()
    self.split_sample()
    self.do_GLM()
    self.make_plot(show=show)



def main():

  TWOSLAQ = PhotoSample(filename="../data/2slaq.csv")
  TWOSLAQ.print_info()
  TWOSLAQ.run_full(show=True)


if __name__=='__main__':

  # parser = argparse.ArgumentParser(usage=__doc__)
  # parser.add_argument('-d','--dataset',dest="dataset",default=None, required=True)

  # args = parser.parse_args()

  main()
