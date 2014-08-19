#!/usr/bin/env python

"""
Photometric redshift library that implements Generalised Linear Models.
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

  def __init__(self, filename_train=False, filename_test=False, filename=False, family="Gamma", link=False, Testing=False):

    # Setup the logger to the command line
    # To a file can also be added fairly easily
    logfmt = '%(levelname)s [%(asctime)s]:\t  %(message)s'
    datefmt= '%m/%d/%Y %I:%M:%S %p'
    formatter = logging.Formatter(fmt=logfmt,datefmt=datefmt)
    logger = logging.getLogger('__main__')
    # logging.root.setLevel(logging.DEBUG)
    logging.root.setLevel(logging.WARNING)
    ch = logging.StreamHandler() #console handler
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    self.logger = logger

    # Book keeping of what the user entered
    self.logger.info("You gave the dataset: {0}".format(filename))
    self.filename_train = filename_train
    self.filename_test = filename_test
    self.filename = filename

    # GLM
    self.family_name = family
    self.link = link

    # Plots
    self.lims = {"x": [0.3, 0.8], "y": [0.41, 0.71]}


    # Testing
    self.Testing = Testing
    self.test_size = False
    self.num_components = False

    # This is for more test purposes
    if self.filename:
      self.data_frame = self.load_data_frame(filename)
      self.logger.info("You gave a complete file, seperating training sets")

    # This is for normal users
    elif self.filename_test and filename_train:
      self.data_frame_test = self.load_data_frame(filename_test)
      self.data_frame_train = self.load_data_frame(filename_train)


      self.data_frame = self.data_frame_test.copy()
      self.data_frame = self.data_frame.append(self.data_frame_train)

    else:
      self.logger.warning("You must give a training and test set or a complete file.")
      sys.exit(0)


  def load_data_frame(self, filename):
    try:
      data_frame = pd.read_csv(filename, encoding="utf-8")

      self.data_frame_header = [i for i in data_frame.columns if i not in ["redshift", "specObjID"]]

      if self.Testing:
        rows = np.random.choice(data_frame.index.values, 10000)
        sampled_df = data_frame.ix[rows]
        return sampled_df
      
      else:
        return data_frame

    except:
      self.logger.info("Failed to open CSV file: {0}".format(sys.exc_info()[0]))
      sys.exit(0)

  def do_PCA(self):
    # TODO:
    # Do the following on the fly:
    #   3. determine test size

    from sklearn.decomposition import PCA

    # Number of components
    if not self.num_components:
      self.num_components = len([i for i in self.data_frame.columns if i != "redshift"])
    

    self.logger.info("Carrying out Principle Component Analysis ({0} components)".format(self.num_components))
    pca = PCA(self.num_components)

    
    pca.fit(self.data_frame[self.data_frame_header])

    x = pca.explained_variance_ratio_
    x_cf = []
    for i in range(len(x)):
      if i>0: x_cf.append(x[i]+x_cf[i-1])
      else: x_cf.append(x[i])
    x_cf = x_cf

    j = False
    for i in range(len(x_cf)):
      if x_cf[i]>0.9995:
        j = i
        break
    if not j: j = len(x_cf)-1

    self.logger.info("explained variance: {0}".format(pca.explained_variance_ratio_))
    self.logger.info("CDF: {0}".format(x_cf))
    self.logger.info("99.5% variance reached with {0} components".format(j+1))

    self.num_components = j+1

    
    M_pca = pca.fit_transform(self.data_frame)

    M_df = {}
    M_df["redshift"] = self.data_frame["redshift"].values

    for i in range(self.num_components):
      M_df["PC{0:d}".format(i+1)] = M_pca[:,i]


    self.PCA_data_frame = pd.DataFrame(M_df)

  def split_sample(self, random):

    # Cross Validation Section
    ##
    if not self.test_size:
      self.test_size = int(self.data_frame.shape[0]*0.2)

    self.logger.info("Splitting into training/testing sets. Number of testing: {0}".format(self.test_size))
    ## Split into train/test
      
    if random:
      from sklearn.cross_validation import train_test_split
      test, train = train_test_split(self.PCA_data_frame, test_size=int(self.test_size), random_state=42)
    else:
      left = self.data_frame_train.shape[0]

      train = self.PCA_data_frame[:left]
      test = self.PCA_data_frame[left:]

    self.logger.info("Training set length: {0}".format(train.shape[0]))
    self.logger.info("Testing set length: {0}".format(test.shape[0]))

    ## Redefine some DataFrames, otherwise they are just numpy arrays
    col_train = {}
    col_test = {}
    try:
      col_train["redshift"] = train[:,-1]
      col_test["redshift"] = test[:,-1]
      for i in range(self.num_components):
        col_train["PC{0:d}".format(i+1)] = train[:,i]
        col_test["PC{0:d}".format(i+1)] = test[:,i]

    except:
      col_train["redshift"] = train["redshift"].values
      col_test["redshift"] = test["redshift"].values
      for i in range(self.num_components):
        col_train["PC{0:d}".format(i+1)] = train["PC{0:d}".format(i+1)]
        col_test["PC{0:d}".format(i+1)] = test["PC{0:d}".format(i+1)]

    self.data_frame_test = pd.DataFrame(col_test)
    self.data_frame_train = pd.DataFrame(col_train)


  def do_GLM(self, disp=1):

    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    import statsmodels.genmod as smg

    # Decide the family    
    if self.family_name == "Gamma":
      if self.link == "log":
        self.family = sm.families.Gamma(link=smg.families.links.log)
      else:
        self.family = sm.families.Gamma()
    elif self.family_name == "Quantile":
        self.family = self.family_name
        self.link = "None"
    else:
      logger.info("You can only pick the family: Gamma and Quantile")

    # Decide the formula
    formula = "redshift ~ "
    for i in range(self.num_components):
      if i<self.num_components-1:
        formula += "PC{0}*".format(i+1)
      else:
        formula += "PC{0}".format(i+1)
    self.formula = formula

    self.logger.info("Family: {0} with \tformula: {1}\tlink: {2}".format(self.family_name, self.formula, self.link))
    self.logger.info("Fitting...")
    
    t1 = time.time()
    if self.family == "Quantile":
      # Quantile regression
      model = smf.quantreg(formula=self.formula, data=self.data_frame_train)
      results = model.fit(q=.5)
      if verbose:
        self.logger.info(results.summary())
    else:
      model = smf.glm(formula=self.formula, data=self.data_frame_train, family=self.family)
      results = model.fit()
      self.logger.info(results.summary())
    t2 = time.time()

    dt = (t2 - t1)
    self.logger.info("Time taken: {0} seconds".format(dt))

    #Plot the model with our test data
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

  def make_1D_KDE(self):
    from matplotlib.mlab import griddata
    import matplotlib.pyplot as plt
    import seaborn as sns

    self.logger.info("Generating 1D KDE plot...")
    ind = range(len(self.outliers))
    rows = list(set(np.random.choice(ind,5000)))
    self.logger.info("Using a smaller size for space ({0} objects)".format(5000))

    outliers = self.outliers[rows]
    measured = self.measured[rows]
    predicted = self.predicted[rows]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    x_straight = np.arange(0,1.6,0.1)
    
    sns.distplot(outliers, hist_kws={"histtype": "stepfilled", "color": "slategray"}, ax=ax)
  
    ax.set_xlabel(r"$(z_{\rm phot}-z_{\rm spec})/1+z_{\rm spec}$", fontsize=20)
    ax.set_ylabel(r"$\rm Density$", fontsize=20)
    ax.set_position([.15,.15,.75,.75])

    plt.savefig("PHOTZ_KDE_1D_{0}.pdf".format(self.family_name), format="pdf")


  def make_2D_KDE(self):

    from matplotlib.mlab import griddata
    import matplotlib.pyplot as plt
    import seaborn as sns

    self.logger.info("Generating 2D KDE plot...")
    ind = range(len(self.outliers))
    rows = list(set(np.random.choice(ind,5000)))
    self.logger.info("Using a smaller size for space ({0} objects)".format(5000))

    outliers = self.outliers[rows]
    measured = self.measured[rows]
    predicted = self.predicted[rows]

    xmin, xmax = 0, measured.max()
    ymin, ymax = 0, xmax

    x_straight = np.arange(xmin, xmax+0.1, 0.1)
    plt.figure()

    g = sns.JointGrid(measured, predicted, size=10, space=0)
    g.plot_marginals(sns.distplot, kde=True, color="green")
    g.plot_joint(plt.scatter, color="silver", edgecolor="white")
    g.plot_joint(sns.kdeplot, kind="hex")
    g.ax_joint.set(xlim=[xmin, xmax], ylim=[ymin, ymax])  
    g.set_axis_labels(xlabel=r"$z_{\rm spec}$", ylabel=r"$z_{\rm phot}$")
    g.ax_joint.errorbar(x_straight, x_straight, lw=2)

    # Temp solution
    # http://stackoverflow.com/questions/21913671/subplots-adjust-with-seaborn-regplot
    axj, axx, axy = plt.gcf().axes
    axj.set_position([.15, .12, .7, .7])
    axx.set_position([.15, .85, .7, .13])
    axy.set_position([.88, .12, .13, .7])
    plt.savefig("PHOTZ_KDE_2D_{0}.pdf".format(self.family_name), format="pdf")

  def make_violin(self):

    from matplotlib.mlab import griddata
    import matplotlib.pyplot as plt
    import seaborn as sns

    self.logger.info("Generating violin plot...")
    ind = range(len(self.outliers))
    rows = list(set(np.random.choice(ind,5000)))
    self.logger.info("Using a smaller size for space ({0} objects)".format(5000))

    outliers = self.outliers[rows]
    measured = self.measured[rows]
    predicted = self.predicted[rows]

    plt.figure()

    bins = np.arange(0,1,0.1)
    text_bins = ["{0}".format(i) for i in bins]

    digitized = np.digitize(measured, bins)

    outliers2 = (measured - predicted)/(measured+1)

    violins = [outliers2[digitized == i] for i in range(1, len(bins))]
    dbin = (bins[1]-bins[0])/2.

    final_violin, final_names = [], []

    for i in range(len(violins)):

      if len(violins[i]) > 1:
        final_violin.append(violins[i])
        final_names.append(bins[i] + dbin)

    sns.offset_spines()
    ax = sns.violinplot(final_violin, names=final_names)
    sns.despine(trim=True)

    ax.set_ylabel(r"$(z_{\rm phot}-z_{\rm spec})/1+z_{\rm spec}$", fontsize=20)
    ax.set_xlabel(r"$z_{\rm spec}$", fontsize=20)
    ax.set_ylim([-1.0,1.0])

    plt.savefig("PHOTZ_VIOLIN_{0}.pdf".format(self.family_name), format="pdf")

  def run_full(self, show=False):
    self.do_PCA()

    if self.filename:
      random = True
    else:
      random = False

    self.split_sample(random=random)
    self.do_GLM()

    self.make_1D_KDE()
    self.make_2D_KDE()
    self.make_violin()

def main():

  # PHAT0 = PhotoSample(filename="../data/PHAT0_small.csv", family="Gamma", link="log")
  # PHAT0.num_components = 9
  # PHAT0.test_size = 4000
  # PHAT0.do_PCA()
  # PHAT0.formula = "redshift ~ PC1*PC2*PC3*PC4*PC5*PC6*PC7"
  # PHAT0.run_full(show=True)


  # TWOSLAQ = PhotoSample(filename="../data/2slaq_small.csv", family="Gamma", Testing=False, link="log")
  # TWOSLAQ.run_full(show=True)

  # SDSS = PhotoSample(filename_train="../data/SDSS_train.csv", filename_test="../data/SDSS_test.csv", family="Gamma", link="log")
  SDSS = PhotoSample(filename="../data/SDSS_nospec.csv", family="Gamma", link="log", Testing=False)
  SDSS.test_size = 100
  # SDSS.num_components = 3
  # SDSS.run_full(show=True)
  SDSS.do_PCA()
  SDSS.split_sample(random=True)
  SDSS.do_GLM()

  # SDSS.make_1D_KDE()
  # SDSS.make_2D_KDE()
  # SDSS.make_violin()



if __name__=='__main__':

  # parser = argparse.ArgumentParser(usage=__doc__)
  # parser.add_argument('-d','--dataset',dest="dataset",default=None, required=True)

  # args = parser.parse_args()

  main()
