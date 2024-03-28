import numpy as np
import time
from scipy.io import loadmat

# Load data
data = loadmat('Subset_1000.mat')
XX = data['XX']
WW = data['WW']  # Assuming WW is part of the loaded data

# Parameters of tree metric
L = 5  # Deepest level
KC = 4  # Number of clusters for the farthest-point clustering

print('...Computing the tree metric from input data')
start_time = time.time()
# Assuming BuildTreeMetric_HighDim_V2 is replaced with a Python equivalent function
TM, TX = BuildTreeMetric_HighDim_V2(XX, L, KC)
runTime = time.time() - start_time
print(f'......running time: {runTime}')

print('...Computing tree representation for input data')
start_time = time.time()
# Mapping vector on tree
XX_TM = np.zeros((len(XX), len(TM['Edge_Weight'])))
for ii in range(len(XX)):
    XX_idVV = TX[ii]
    WW_idVV = WW[ii]
    for jj in range(len(XX_idVV)):
        XX_TM[ii, TM['Vertex_EdgeIdPath'][XX_idVV[jj]]] += WW_idVV[jj]
# Weighted mapping
XX_TMWW = XX_TM * np.tile(TM['Edge_Weight'].reshape(1, -1), (len(XX), 1))
runTime = time.time() - start_time
print(f'......running time: {runTime}')

print('...Computing l1-distance for tree representation data')
start_time = time.time()
# Compute TW distance matrix for XX
DD_XX = np.zeros((len(XX), len(XX)))
for ii in range(len(XX)-1):
    tmp = np.sum(np.abs(np.tile(XX_TMWW[ii, :], (len(XX) - ii - 1, 1)) - XX_TMWW[(ii+1):, :]), axis=1)
    DD_XX[ii, (ii+1):] = tmp
    DD_XX[(ii+1):, ii] = tmp
runTime = time.time() - start_time
print(f'......running time: {runTime}')

print('FINISH!')


