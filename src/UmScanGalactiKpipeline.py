################################################################################
# UmScanGalactiK pipeline
# 
# Rosito et al. (2022), Application of dimensionality reduction and clustering 
# algorithms for the classification of kinematic morphologies of galaxies. 
# Publicated in Astronomy & Astrophysics:
# https://www.aanda.org/component/article?access=doi&doi=10.1051/0004-6361/202244707
# ArXiv: https://arxiv.org/abs/2212.03999
#
# UMAP library: https://github.com/lmcinnes/
# McInnes, L & Healy, J, UMAP: Uniform Manifold Approximation and Projection for 
# Dimension Reduction
# ArXiv: https://arxiv.org/abs/1802.03426
#
# HDBSCAN library: https://github.com/scikit-learn-contrib/hdbscan
# Campello, R. J., Moulavi, D., & Sander, J. (2013), Density-based clustering 
# based on hierarchical density estimates. 
# Link : https://link.springer.com/chapter/10.1007/978-3-642-37456-2_14
################################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
import umap
import os
import re
import hdbscan


################################################################################
# Function  definitions
################################################################################

# sorted_alphanumeric(data) ####################################################
# Description: Given a list of strings, sorts it alphanumerically
# Example: vLOS_gr_1i_60.dat before vLOS_gr_10i_60.dat
# 
# Input:
#   `data`: array of strings
# Output:
#   the array sorted alphanumerically
################################################################################
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)


# map_to_matrix(inclination, npix, input_dir, label) ###########################
# Description: Returns a numpy matrix, each row = galaxy flatten kinematic map
# NA is replaced by 0
# 
# Inputs:
# `inclination`: inclination used for the observation
# `npix`: number of pixels of each map
# `input_dir`: input directory
# `label`: type of kinematic map (string)
# Output:
# `X`: matrix containing the kinematic maps of the sample of galaxies   
################################################################################
def map_to_matrix(inclination, npix, input_dir, label):
    DIR = input_dir + "/i" + str(inclination) + "/" + label
    files = sorted_alphanumeric(os.listdir(DIR))
    nfiles = len(files)
    X = np.zeros((nfiles, npix))
    i = 0
    for i in range(nfiles):
        kmap = pd.read_table(DIR + "/" + files[i], sep = " ")        
        X[i] = (kmap.to_numpy(na_value = 0)).flatten()        
    return X


# read_and_preprocess_input_data(inclination, npix, input_dir) #################
# Description: Returns the normalized matrix concatenating the three types of 
# kinematic maps for all galaxies in the sample
#
# Inputs:
# `inclination`: inclination used for the observation
# `npix`: number of pixels of each map
# `input_dir`: input directory
# Output:
# `data`: matrix containing the data
################################################################################
def read_and_preprocess_input_data(inclination, npix, input_dir):    
    ## vLOS, dispersion, and flux maps
    datav = map_to_matrix(inclination, npix, input_dir, 'vLOS')
    datas = map_to_matrix(inclination, npix, input_dir, 'dispersion')
    dataf = map_to_matrix(inclination, npix, input_dir, 'flux')
    # Normalizations
    Max1 = abs(datav).max()
    Max2 = abs(datas).max()
    Max3 = abs(dataf).max()
    datav = datav / Max1
    datas = datas / Max2
    dataf = dataf / Max3        
    data = np.concatenate((datav, datas, dataf), axis = 1)
    return data


# projection_and_clustering(data, inclination, nn, ms, mcs, md, output_dir) ####
# Description: Given a matrix with high dimensonality rows, returns the UMAP
# bidimensional projection coordinates, the HDBSCAN cluster labels, the number  
# of clusters, and the number of clustered points. Plots the results. 
#
# Inputs:
# `data`: matrix with the data
# `inclination`: inclination used for the observation
# `nn`: n_neigbors parameter for the UMAP algorithm
# `md`: min_dist parameter for the UMAP algorithm
# `ms`: min_samples parameter for the HDBSCAN algorithm
# `mcs`: min_cluster_size parameter for the HDBSCAN algorithm
# `ouput_dir`: output directory
# Outputs:
# `x`: x component of the bidimensional projection
# `y`: y component of the bidimensional projection
# `l`: labels for the cluster of each sample, -1 depicts an outlier
# `Nc`: number of clusters
# `n_clustered`: number of clustered points
################################################################################
def projection_and_clustering(data, inclination, nn, md, ms, mcs, output_dir):
    # UMAP projection    
    clusterable_embedding = umap.UMAP(n_neighbors = nn, min_dist = md, n_components = 2,
    random_state = 42).fit_transform(data)
    x = clusterable_embedding[:, 0]
    y = clusterable_embedding[:, 1]
    # HDBSCAN clustering 
    clusters = hdbscan.HDBSCAN(min_samples = ms, min_cluster_size = mcs).fit(clusterable_embedding)
    Nc =  clusters.labels_.max() + 1
    l = clusters.labels_
    clustered = l >= 0
    non_clus = l < 0    
    lc = l[clustered]
    n_clustered = len(lc)    
    
    # Plot
    inc = str(int(inclination))
    cmapq = plt.cm.rainbow
    filename = output_dir + 'UmScanGalactiK_i' + inc + '_clusters.png' 
    boundaries = np.arange(-0.5, Nc + 0.5, 1) 
    norm = colors.BoundaryNorm(boundaries, cmapq.N)
    # Outliers are depicted in gray            
    plt.scatter(x[non_clus], y[non_clus], c = 'lightgray', s = 15)
    sc = plt.scatter(x[clustered], y[clustered], c = lc, s = 15, cmap = cmapq, 
    norm = norm, edgecolor = 'none')
    plt.colorbar(sc, ticks = np.arange(0., Nc + 1)).ax.tick_params(labelsize = 14)
    plt.figtext(0.45, 0.83, inc + ' deg',  fontsize = 16.5)
    plt.xlabel('UMAP 1', fontsize = 16.5)
    plt.ylabel('UMAP 2', fontsize = 16.5)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    figure = plt.gcf()
    figure.savefig(filename)
    plt.close()
        
    return x, y, l, Nc, n_clustered


# UmScanGalactiK(inclination, npix, input_dir, output_dir, nn = 70, ms = 1, ####
# mcs = 70, md = 0.)
# Description: UmScanGalactiK pipeline for each inclination 
# Results report and plot generation
#
# Inputs:
# `inclination`: inclination used for the observation
# `npix`: number of pixels of each map
# `input_dir`: input directory
# `output_dir`: output directory
# `nn`: n_neigbors parameter for the UMAP algorithm (default = 70)
# `md`: min_dist parameter for the UMAP algorithm (default = 0.)
# `ms`: min_samples parameter for the HDBSCAN algorithm (default = 1)
# `mcs`: min_cluster_size parameter for the HDBSCAN algorithm (default = 70)
# Default values are set according to the Experiment 2 in Rosito et al. (2022)
# Outputs:
# `x`: x component of the bidimensional projection
# `y`: y component of the bidimensional projection
# `l`: labels for the cluster of each galaxy, -1 depicts an outlier
# `Nc`: number of clusters
# `n_clustered`: number of clustered points
################################################################################
def UmScanGalactiK(inclination, npix, input_dir, output_dir, nn = 70, md = 0., 
    ms = 1, mcs = 70):
    # read and preprocess the kinematic maps
    data = read_and_preprocess_input_data(inclination, npix, input_dir)
    # perform dimensionality reduction and clustering
    x, y, l, Nc, n_clustered = projection_and_clustering(data, inclination, nn, 
    md, ms, mcs, output_dir)
    # report results
    print('Inclination ', inclination, 'degrees')
    print('Number of clusters:', Nc)
    print('Number of clustered galaxies:', n_clustered)
    for i in range(Nc):
        print('Number of galaxies in cluster', i, ':', len(l[l == i]))
    return x, y, l, Nc, n_clustered     
