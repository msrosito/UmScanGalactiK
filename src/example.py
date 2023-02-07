################################################################################
# UmScanGalactiK example: (1) Application of UmScanGalactiK to a set of kinematic 
# maps as input. (2) Intra-cluster and inter-clusteranalysis of the triaxiality 
# parameter 
# 
# Rosito et al. (2022), Application of dimensionality reduction and clustering 
# algorithms for the classification of kinematic morphologies of galaxies. 
# Publicated in Astronomy & Astrophysics:
# https://www.aanda.org/component/article?access=doi&doi=10.1051/0004-6361/202244707
# ArXiv: https://arxiv.org/abs/2212.03999
################################################################################

from UmScanGalactiKpipeline import *
from ClusterStatisticalAnalysis import *

# Global variables
input_dir = '../data_example/'
output_dir = '../example/'
npix = 900

# Example
inclination = 90
T = np.fromfile(input_dir + 'Triaxiality-Parameter.txt', sep = '\n') # Triaxiality parameter
label = 'T'
cluster1 = 3
cluster2 = 4
# Color patches
violet_patch = mpatches.Patch(color ='blueviolet', label='C0')
blue_patch = mpatches.Patch(color = 'lightskyblue', label='C1')
green_patch = mpatches.Patch(color ='springgreen', label='C2')
orange_patch = mpatches.Patch(color ='orange', label='C3')
red_patch = mpatches.Patch(color = 'r', label='C4')
patches = [violet_patch, blue_patch, green_patch, orange_patch, red_patch]
color = ['blueviolet', 'lightskyblue', 'springgreen', 'orange', 'r']

# Step (1)
x, y, l, Nc, n_clustered = UmScanGalactiK(inclination, npix, input_dir, output_dir)

# Step (2)
full_parameter_analysis(x, y, l, T, label, inclination, output_dir, cluster1, 
cluster2, color, patches)
