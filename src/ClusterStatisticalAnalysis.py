################################################################################
# Tools for the statistical analysis of relevant parameters 
# (intra-cluster and inter-cluster)
# 
# Rosito et al. (2022), Application of dimensionality reduction and clustering 
# algorithms for the classification of kinematic morphologies of galaxies. 
# Publicated in Astronomy & Astrophysics:
# https://www.aanda.org/component/article?access=doi&doi=10.1051/0004-6361/202244707
# ArXiv: https://arxiv.org/abs/2212.03999
#
# LOESS library: https://pypi.org/project/loess/
# Cappellari et al. (2013), The ATLAS3D project - XX. Mass-size and mass-σ 
# distributions of early-type galaxies: bulge fraction drives kinematics, 
# mass-to-light ratio, molecular gas fraction and stellar initial mass function 
# Link: https://academic.oup.com/mnras/article/432/3/1862/1750208?searchresult=1
################################################################################

from loess.cap_loess_2d import loess_2d
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats
import pandas as pd


################################################################################
# Function  definitions
################################################################################


# distribution_on_projection(x, y, parameter, label, inclination, output_dir) ##
# Description: Plots the distribution of a certain physical or observational 
# parameter on a given bidimensional projection
#
# Inputs:
# `x`: x component of the bidimensional projection
# `y`: y component of the bidimensional projection
# `parameter`: values of the parameter 
# `label`: parameter name (string)
# `inclination`: inclination used for the observation
# `output_dir`: output directory
################################################################################

def distribution_on_projection(x, y, parameter, label, inclination, output_dir):
    inc = str(int(inclination))
    filename = output_dir + 'UmScanGalactiK_i' + inc + '_' +  label + '.png' 
    zout, wout = loess_2d(x, y, parameter)
    lim_inf = np.percentile(parameter, 10)
    lim_sup = np.percentile(parameter, 90)
    sc = plt.scatter(x, y, c = zout, cmap = plt.cm.jet.reversed(), s = 15, 
    edgecolor = 'none')
    plt.clim(lim_inf, lim_sup)
    col = plt.colorbar(sc)
    col.set_label(label = label ,size = 16.5)
    col.ax.tick_params(labelsize = 14)
    plt.figtext(0.45, 0.83, inc + ' deg', fontsize = 16.5)
    plt.xlabel('UMAP 1', fontsize = 16.5)
    plt.ylabel('UMAP 2', fontsize = 16.5)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    figure = plt.gcf()
    figure.savefig(filename)
    plt.close()
    
    
# density_plot(l, parameter, label, color, inclination, output_dir) ############
# Description: Plots the probability density functions of a given parameter for 
# each cluster
#
# Inputs:
# `l`: labels for the cluster of each sample, -1 depicts an outlier
# `parameter`: values of the parameter 
# `label`: parameter name (string)
# `color`: array of colors for the lines
# `patches`: patches for the legend
# `inclination`: inclination used for the observation
# `output_dir`: output directory  
################################################################################
def density_plot(l, parameter, label, color, patches, inclination, output_dir):
    Nc = max(l) + 1
    inc = str(int(inclination))
    filename = output_dir + 'PDFs_i' + inc + '_' +  label + '.png'      
    for i in range(Nc):
        lab = l == i
        sns.distplot(parameter[lab], hist = False, kde = True, 
        kde_kws = {'linewidth': 3, 'clip': (-0.05, 1.05)}, norm_hist = True, color = color[i])   
    plt.xlabel(label, fontsize = 16.5)
    plt.ylabel('PDF', fontsize = 16.5)      
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.legend(handles =  patches, fontsize = 14.85)
    plt.figtext(0.15, 0.83, inc + ' deg', fontsize = 16.5)
    figure1 = plt.gcf()
    figure1.savefig(filename)
    plt.close()
    

# compute_statistics(x) #########################################################
# Description: Returns descriptive statistics for a given array
#
# Input:
# `x`: numpy array
# Outputs:
# number of observations, minimum, maximum, median, first quartile, third quartile, 
# mean, unbiased variance, skewness, kurtosis
################################################################################ 
def compute_statistics(x):        
    summary = stats.describe(x)
    Q1, Q3 = np.percentile(x, 25), np.percentile(x, 75)
    return summary.nobs, summary.minmax[0], summary.minmax[1], np.median(x), Q1, \
     Q3, summary.mean, summary.variance, summary.skewness, summary.kurtosis
    

# stochastic_equality_brunnermunzel(l, cluster1, cluster2, parameter) ##########
# Description: Returns and reports the three p-values of Brunner-Munzel test for 
# the parameter values in two given clusters changing the alternative hypothesis
#
# Input:
# `l`: labels for the cluster of each sample, -1 depicts an outlier
# `cluster1`: label of the first cluster
# `cluster2`: label of the second cluster
# `parameter`: values of the parameter 
# `label`: parameter name (string)
# Outputs:
# p-values for the two-sided, less, and greater alternative hypothesis
################################################################################     
def stochastic_equality_brunnermunzel(l, cluster1, cluster2, parameter, label):
    alternative = ['two-sided', 'less', 'greater']
    parameter_cluster1 = parameter[l == cluster1]
    parameter_cluster2 = parameter[l == cluster2]
    pval = []
    for alt in alternative:
        bm, pvalue = stats.brunnermunzel(parameter_cluster1, parameter_cluster2, 
        alternative = alt)
        print('Test for', label, 'clusters', cluster1, 'and', cluster2, alt)
        print('p-value:', pvalue)        
        pval.append(pvalue)
    return pval[0], pval[1], pval[2]

    
# full_parameter_analysis(x, y, l, parameter, label, inclination, output_dir, ## 
# cluster1, cluster2)
# Description: Plots the distributions of a given parameter within each cluster 
# and reports descriptive statistics and stochastic equality test p-values for 
# two of them
#
# Inputs:
# `x`: x component of the bidimensional projection
# `y`: y component of the bidimensional projection
# `l`: labels for the cluster of each sample, -1 depicts an outlier
# `parameter`: values of the parameter 
# `label`: parameter name (string)
# `inclination`: inclination used for the observation
# `output_dir`: output directory
# `cluster1`: label of the first cluster
# `cluster2`: label of the second cluster
# `color`: array of colors for the lines
# `patches`: patches for the legend
#################################################################################
def full_parameter_analysis(x, y, l, parameter, label, inclination, output_dir, 
    cluster1, cluster2, color, patches):
    # plot of the parameter distribution on the UMAP projection
    distribution_on_projection(x, y, parameter, label, inclination, output_dir)
    # plot of the PDF of the parameter in each HDBSCAN cluster
    density_plot(l, parameter, label, color, patches, inclination, output_dir)
    # statistical description of the parameter values in cluster1 and cluster2
    for clus in [cluster1, cluster2]:
        nobs, parammin, parammax, parammed, paramQ1, paramQ3, parammean, paramvar, \
        paramskew, paramkurt = compute_statistics(parameter[l == clus])
        print('Summary of', label, 'cluster ', clus)
        print('Number of elements:', nobs)
        print('Minimum value:', parammin)
        print('First quartile:', paramQ1)
        print('Median:', parammed)
        print('Third quartile:', paramQ3)                          
        print('Maximum value:', parammax)               
        print('Mean:', parammean)
        print('Variance:', paramvar)
        print('Skewness:', paramskew)
        print('Kurtosis:', paramkurt)
    # Brunner-Munzel test
    pval2s, pvall, pvalg = stochastic_equality_brunnermunzel(l, cluster1, \
    cluster2, parameter, label)
