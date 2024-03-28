import numpy as np
import time
from scipy.io import loadmat

# Load input data (1) for building tree metric
data1 = loadmat('Subset_200.mat')
XX = data1['XX']

# Parameters of tree metric
L = 5  # deepest level
KC = 4  # number of clusters for the farthest-point clustering

# Building tree metric by the farthest-point clustering
print('...Computing the tree metric from input data (1)')
start_time = time.time()
# Assuming BuildTreeMetric_HighDim_V2 is a function defined elsewhere to build the tree metric
# TM, TX = BuildTreeMetric_HighDim_V2(XX, L, KC)
runTime = time.time() - start_time
print(f'......running time: {runTime}')

# Load input data (2) for computing tree-Wasserstein distance matrix
# using the tree metric built from input data (1)
data2 = loadmat('Subset_1000.mat')
XX = data2['XX']
WW = data2['WW']  # Assuming WW is also part of the loaded data

print('...Computing tree representation for input data (2)')
start_time = time.time()
# Assuming TreeMapping is a function defined elsewhere for computing tree representation
# XX_TMWW = TreeMapping(XX, WW, TM)
runTime = time.time() - start_time
print(f'......running time: {runTime}')

print('...Computing l1-distance for tree representation data')
start_time = time.time()
# Compute TW distance matrix for XX
# L1 distance
DD_XX = np.zeros((len(XX), len(XX)))
for ii in range(len(XX)-1):
    # L1 distances between ii^th id and (ii+1 : len(XX))^th ids
    tmp = np.sum(np.abs(np.tile(XX_TMWW[ii, :], (len(XX) - ii, 1)) - XX_TMWW[(ii+1):len(XX), :]), axis=1)
    DD_XX[ii, (ii+1):len(XX)] = tmp
    DD_XX[(ii+1):len(XX), ii] = tmp
runTime = time.time() - start_time
print(f'......running time: {runTime}')
print('FINISH!')


