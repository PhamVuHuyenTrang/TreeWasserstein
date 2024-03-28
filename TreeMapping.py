import numpy as np
from scipy.spatial import KDTree

def TreeMapping(XX, WW, TM):
    N = len(XX)
    dim = XX[0].shape[1]
    nSupports = sum(X.shape[0] for X in XX)
    sIDArray = np.zeros(N, dtype=int)
    eIDArray = np.zeros(N, dtype=int)
    
    allXX = np.zeros((nSupports, dim))
    allWW = np.zeros(nSupports)
    
    nSupports = 0
    for ii, X in enumerate(XX):
        sIDArray[ii] = nSupports
        nSupports += X.shape[0]
        eIDArray[ii] = nSupports
        allXX[sIDArray[ii]:eIDArray[ii], :] = X
        allWW[sIDArray[ii]:eIDArray[ii]] = WW[ii]
    
    allLeaves = TM['Vertex_Pos'][TM['LeavesIdArray'], :]
    tree = KDTree(allLeaves)
    _, idLeaves = tree.query(allXX)
    allIdVertices = TM['LeavesIdArray'][idLeaves]
    
    TX = np.zeros((N, TM['nVertices'] - 1))
    
    for ii in range(N):
        tmpVector = np.zeros(TM['nVertices'] - 1)
        for jj in range(sIDArray[ii], eIDArray[ii]):
            idEdges = TM['Vertex_EdgeIdPath'][allIdVertices[jj]]
            tmpVector[idEdges] += allWW[jj]
        TX[ii, :] = tmpVector * np.array(TM['Edge_Weight']).T
    
    return TX


