import numpy as np

def LikelihoodQAE(theta, nGrovers, n1s, nShot):
    
    ret = 0.0
    
    for nGrover, n1 in zip(nGrovers, n1s):
        p = np.sin((2.0 * nGrover + 1) * theta) ** 2
        if n1 == nShot:
            ret += n1 * np.log(p)
        elif n1 == 0:
            ret += nShot * np.log(1.0 - p)
        else:
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

def FindAmpSqMLESmart(nGrovers, n1s, nShot, numPoints):

    if nGrovers[0] != 0:
        raise ValueError("1st entry of nGrovers must be 0")
        
    sigmaNum = 5.0 # 5σ
        
    thetaMax = np.arcsin(np.sqrt(n1s[0] / nShot)) # MLE for nGrover=0
    
    for k in range(len(nGrovers)):
        sigma = 1.0 / (2.0 * (2.0 * nGrovers[k] + 1.0) * np.sqrt(nShot)) # 1/sqrt{Fisher info}
        thetaLB = max(thetaMax - sigmaNum * sigma, 0.0)
        thetaUB = min(thetaMax + sigmaNum * sigma, 0.5 * np.pi)
        
        # random search
        likMax = LikelihoodQAE(thetaMax, nGrovers[:(k + 1)], n1s[:(k + 1)], nShot)
        for _ in range(numPoints):
            thetaTemp = np.random.uniform(thetaLB, thetaUB)
            likTemp = LikelihoodQAE(thetaTemp, nGrovers[:(k + 1)], n1s[:(k + 1)], nShot)
            if likTemp > likMax:
                thetaMax = thetaTemp
                likMax = likTemp
    
    return np.sin(thetaMax) ** 2