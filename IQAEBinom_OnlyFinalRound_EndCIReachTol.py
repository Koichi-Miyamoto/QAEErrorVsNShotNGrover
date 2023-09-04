import numpy as np

def IQAEBinom_OnlyFinalRound_EndCIReachTol(ampSq, nGrover, epsilon, alpha, nShotUnit):

    theta = np.arcsin(np.sqrt(ampSq))
    kRound = 2 * nGrover + 1
    prob1 = np.sin(kRound * theta) ** 2
    rRound = int(kRound * theta / (0.5 * np.pi))
    kmax = np.pi / 4 / epsilon
    alphaRound = 2 * alpha / 3 * kRound / kmax
    nShotRound = 0
    n1Round = 0

    while True:
        nShotRound += nShotUnit
        n1 = np.random.binomial(nShotUnit, prob1)
        n1Round += n1
        p1Obs = n1Round / nShotRound

        # confint_method = "chernoff":
        p1Width = np.sqrt(0.5 / nShotRound * np.log(2 / alphaRound))
        p1Max = min(p1Obs + p1Width, 1)
        p1Min = max(p1Obs - p1Width, 0)

        if rRound % 2 == 0:
            gammaMin = np.arcsin(np.sqrt(p1Min))
            gammaMax = np.arcsin(np.sqrt(p1Max))
            gammaML = np.arcsin(np.sqrt(p1Obs))
        else:
            gammaMin = 0.5 * np.pi - np.arcsin(np.sqrt(p1Max))
            gammaMax = 0.5 * np.pi - np.arcsin(np.sqrt(p1Min))
            gammaML = 0.5 * np.pi - np.arcsin(np.sqrt(p1Obs))
        
        theta_u = (rRound * 0.5 * np.pi + gammaMax) / kRound
        theta_l = (rRound * 0.5 * np.pi + gammaMin) / kRound
        thetaInterval = [theta_l, theta_u]
        thetaML = (rRound * 0.5 * np.pi + gammaML) / kRound

        ampSq_u = np.sin(theta_u) ** 2
        ampSq_l = np.sin(theta_l) ** 2
        ampSqInterval = [ampSq_l, ampSq_u]
        ampSqML = np.sin(thetaML) ** 2
        ampSqWidth = max(ampSq_u - ampSqML, ampSqML - ampSq_l)

        if ampSqWidth <= epsilon:
            break

    ret = {"Estimate":ampSqML,
           "TotalOracleCalls": (2 * nGrover + 1) *nShotRound,
           "nShots":nShotRound,
           "n1s":n1Round,
           "thetaIntervals":thetaInterval,
           "ampSqIntervals":ampSqInterval}
    return ret

