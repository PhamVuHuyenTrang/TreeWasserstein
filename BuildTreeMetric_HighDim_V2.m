import numpy as np

def BuildTreeMetric_HighDim_V2(XX, L, KC):
    MAXNUM_TREE = KC**(L+1)
    KCArray = KC**(np.arange(1, L+1))
    N = len(XX)
    dim = XX[0].shape[1]
    nSupports = 0
    sIDArray = np.zeros(N)
    eIDArray = np.zeros(N)
    
    for ii in range(N):
        sIDArray[ii] = nSupports + 1
        nSupports = nSupports + XX[ii].shape[0]
        eIDArray[ii] = nSupports
    
    allXX = np.zeros((nSupports, dim))
    for ii in range(N):
        allXX[sIDArray[ii]-1:eIDArray[ii], :] = XX[ii]
    
    nXX = nSupports
    kcCenterPP = np.mean(allXX, axis=0).reshape(-1, 1)
    numPP = 1
    idZZPP = np.zeros(N)
    
    TM = {}
    TM['nVertices'] = 0
    TM['Vertex_ParentId'] = np.zeros(MAXNUM_TREE)
    TM['Vertex_ChildId'] = [None] * MAXNUM_TREE
    TM['Vertex_Pos'] = np.zeros((MAXNUM_TREE, dim))
    TM['Vertex_EdgeIdPath'] = [None] * MAXNUM_TREE
    TM['Edge_LowNode'] = np.zeros(MAXNUM_TREE)
    TM['Edge_HighNode'] = np.zeros(MAXNUM_TREE)
    TM['Edge_Weight'] = np.zeros(MAXNUM_TREE)
    TM['Level_sID'] = np.zeros(L + 1)
    TM['Level_eID'] = np.zeros(L + 1)
    
    TM['nVertices'] = 1
    TM['Vertex_Pos'][0, :] = kcCenterPP.flatten()
    TM['Vertex_EdgeIdPath'][0] = []
    TM['Level_sID'][0] = 1
    TM['Level_eID'][0] = 1
    
    for idLL in range(L):
        idZZLL = np.zeros(nXX)
        kcCenterLL = np.zeros((dim, KCArray[idLL]))
        nkcCenterLL = 0
        TM['Level_sID'][idLL+1] = TM['Level_eID'][idLL] + 1
        TM['Level_eID'][idLL+1] = TM['Level_eID'][idLL]
        
        for idCCPP in range(numPP):
            idVertexPP = int(TM['Level_sID'][idLL] + idCCPP)
            
            if idLL == 0:
                idZZ_idCCPP = np.arange(1, nXX+1)
            else:
                idZZ_idCCPP = np.where(idZZPP == idCCPP)[0] + 1
            
            if len(idZZ_idCCPP) > 1:
                allZZ_idCCPP = allXX[idZZ_idCCPP-1, :]
                rKCLL_idCCPP, _, idZZLL_idCCPP, kcCenterLL_idCCPP, _, _ = figtreeKCenterClustering(dim, len(idZZ_idCCPP), allZZ_idCCPP.T, KC)
                ppMM = np.tile(kcCenterPP[:, idCCPP].reshape(-1, 1), (1, rKCLL_idCCPP))
                wLL_idCCPP = np.sqrt(np.sum((kcCenterLL_idCCPP - ppMM)**2, axis=0))
                setID_0 = np.where(wLL_idCCPP == 0)[0]
                
                if len(setID_0) > 0:
                    kcCenterLL_idCCPP = np.delete(kcCenterLL_idCCPP, setID_0, axis=1)
                    wLL_idCCPP = np.delete(wLL_idCCPP, setID_0)
                    clusterID_ZeroLength = setID_0 - 1
                    allID_Zero = []
                    
                    for iiCC_Zero in range(len(clusterID_ZeroLength)):
                        tmp = np.where(idZZLL_idCCPP == clusterID_ZeroLength[iiCC_Zero])[0]
                        allID_Zero = np.concatenate((allID_Zero, tmp))
                    
                    idZZLL_idCCPP[allID_Zero.astype(int)] = -1
                    clusterID_NonZero = np.arange(rKCLL_idCCPP)
                    clusterID_NonZero = np.delete(clusterID_NonZero, setID_0)
                    
                    for iiCC_NonZero in range(len(clusterID_NonZero)):
                        if clusterID_NonZero[iiCC_NonZero] != (iiCC_NonZero - 1):
                            idZZLL_idCCPP[idZZLL_idCCPP == clusterID_NonZero[iiCC_NonZero]] = iiCC_NonZero - 1
                    
                    rKCLL_idCCPP = rKCLL_idCCPP - len(setID_0)
                
                idZZLL[idZZ_idCCPP-1] = nkcCenterLL + idZZLL_idCCPP
                
                if len(setID_0) > 0:
                    idZZLL[idZZ_idCCPP[allID_Zero.astype(int)]-1] = -1
                
                if rKCLL_idCCPP > 0:
                    kcCenterLL[:, nkcCenterLL:nkcCenterLL+rKCLL_idCCPP] = kcCenterLL_idCCPP
                    TM['nVertices'] = TM['nVertices'] + rKCLL_idCCPP
                    idNewVertices = np.arange(TM['Level_eID'][idLL+1]+1, TM['Level_eID'][idLL+1]+rKCLL_idCCPP+1)
                    TM['Vertex_ParentId'][idNewVertices-1] = idVertexPP
                    TM['Vertex_ChildId'][idVertexPP] = idNewVertices.tolist()
                    TM['Vertex_Pos'][idNewVertices-1, :] = kcCenterLL_idCCPP.T
                    idNewEdges = idNewVertices - 1
                    TM['Edge_LowNode'][idNewEdges-1] = idVertexPP
                    TM['Edge_HighNode'][idNewEdges-1] = idNewVertices
                    TM['Edge_Weight'][idNewEdges-1] = wLL_idCCPP
                    TM['Vertex_EdgeIdPath'][idNewVertices-1] = [TM['Vertex_EdgeIdPath'][idVertexPP], idNewEdges.tolist()]
                    TM['Level_eID'][idLL+1] = TM['Level_eID'][idLL+1] + rKCLL_idCCPP
        
        idZZPP = idZZLL
        kcCenterPP = kcCenterLL[:, :nkcCenterLL]
        numPP = nkcCenterLL
    
    TM['LeavesIDArray'] = np.arange(TM['Level_sID'][L+1], TM['Level_eID'][L+1]+1)
    TM['Vertex_ParentId'] = TM['Vertex_ParentId'][:TM['nVertices']]
    TM['Vertex_ChildId'] = TM['Vertex_ChildId'][:TM['nVertices']]
    TM['Vertex_Pos'] = TM['Vertex_Pos'][:TM['nVertices'], :]
    TM['Vertex_EdgeIdPath'] = TM['Vertex_EdgeIdPath'][:TM['nVertices']]
    TM['Edge_LowNode'] = TM['Edge_LowNode'][:TM['nVertices']-1]
    TM['Edge_HighNode'] = TM['Edge_HighNode'][:TM['nVertices']-1]
    TM['Edge_Weight'] = TM['Edge_Weight'][:TM['nVertices']-1]
    
    XX_VertexID = []
    for ii in range(N):
        XX_VertexID.append(TM['Level_sID'][L+1] + idZZPP[int(sIDArray[ii])-1:int(eIDArray[ii])].tolist())
    
    return TM, XX_VertexID


