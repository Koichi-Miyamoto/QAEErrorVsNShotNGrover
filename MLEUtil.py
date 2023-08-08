import numpy as np

def LikelihoodQAE(theta, nGrovers, n1s, nShot):
    
    ret = 0.0
    
    for nGrover, n1 in zip(nGrovers, n1s):
        p = np.sin((2.0 * nGrover + 1) * np.pi * theta) ** 2
        ret += n1 * np.log(p) + (nShot - n1) * np.log(1.0 - p)
        
    return ret

def FindAmpSqMLE(nGrovers, n1s, nShot):
    numPoints = 1000
    thetas = 0.5 / numPoints * np.arange(1, numPoints)
    
    thetaMax = thetas[np.argmax([LikelihoodQAE(th, nGrovers, n1s, nShot) for th in thetas])]
    return np.sin(np.pi * thetaMax) ** 2