import numpy as np


def IQAEBinom_OnlyFinalRound(ampSq, nGrover, nShot, nEstim=None):

    theta = np.arcsin(np.sqrt(ampSq))
    kRound = 2 * nGrover + 1
    prob1 = np.sin(kRound * theta) ** 2
    rRound = int(kRound * theta / (0.5 * np.pi))

    p1Obs = np.random.binomial(nShot, prob1, size=nEstim) / nShot

    if rRound % 2 == 0:
        gammaML = np.arcsin(np.sqrt(p1Obs))
    else:
        gammaML = 0.5 * np.pi - np.arcsin(np.sqrt(p1Obs))

    thetaML = (rRound * 0.5 * np.pi + gammaML) / kRound
    ampSqML = np.sin(thetaML) ** 2
    
    return ampSqML

