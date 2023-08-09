import numpy as np

def LikelihoodQAE(theta, nGrovers, n1s, nShot):
    
    ret = 0.0
    
    for nGrover, n1 in zip(nGrovers, n1s):
        p = np.sin((2.0 * nGrover + 1) * theta) ** 2
        ret += n1 * np.log(p) + (nShot - n1) * np.log(1.0 - p)
        
    return ret

def FindAmpSqMLE(nGrovers, n1s, nShot, thetas=None):
    if thetas is None:
        numPoints = 100000
        thetas = 0.5 * np.pi / numPoints * np.arange(1, numPoints)
    
    thetaMax = thetas[np.argmax([LikelihoodQAE(th, nGrovers, n1s, nShot) for th in thetas])]
    return np.sin(thetaMax) ** 2

def FindAmpSqMLERandom(nGrovers, n1s, nShot, thetaLB, thetaUB, numPoints):
    thetaMax = float('nan')
    likMax = -float('inf')
    for _ in range(numPoints):
        thetaTemp = np.random.uniform(thetaLB, thetaUB)
        likTemp = LikelihoodQAE(thetaTemp, nGrovers, n1s, nShot)
        if likTemp > likMax:
            thetaMax = thetaTemp
            likMax = likTemp
    return np.sin(thetaMax) ** 2