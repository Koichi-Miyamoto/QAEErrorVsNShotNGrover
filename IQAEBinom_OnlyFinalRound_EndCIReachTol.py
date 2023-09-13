import numpy as np

def IQAEBinom_OnlyFinalRound_EndCIReachTol(ampSq, nGrover, epsilon, alpha, nShotUnit, minRatio=None, endCriterion="Amplitude"):

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

        if endCriterion == "Amplitude":
            if ampSqWidth <= epsilon:
                break
        elif endCriterion == "Angle":
            if theta_u - theta_l <= 2 * epsilon:
                break
        else:
            raise Exception("Unknown end criterion:" + endCriterion)

        if minRatio is not None:
            if _find_next_k(nGrover, thetaInterval, minRatio) > nGrover:
                ampSqML = np.NaN
                break

    ret = {"Estimate":ampSqML,
           "TotalOracleCalls": (2 * nGrover + 1) *nShotRound,
           "nShots":nShotRound,
           "n1s":n1Round,
           "thetaIntervals":thetaInterval,
           "ampSqIntervals":ampSqInterval}
    return ret

def BiasFinalRound(ampSq, nGrover, epsilon, alpha, nShotUnit, nEstim, minRatio=None):

    errs = []
    for i in range(nEstim):
        result = IQAEBinom_OnlyFinalRound_EndCIReachTol(ampSq, nGrover, epsilon, alpha, nShotUnit, minRatio=minRatio)
        if result is None:
            errs.append(np.nan)
        else:
            errs.append(result["Estimate"] - ampSq)
    return np.nanmean(errs)

def _find_next_k(
    k_prev,
    theta_interval,
    minRatio
) -> int:

    # initialize variables
    theta_l, theta_u = theta_interval
    K_prev = 2 * k_prev + 1
    K = int(0.5 * np.pi / (theta_u-theta_l))
    K -= (K + 1) % 2 # subtract 1 if even
    
    while K >= minRatio * K_prev:
        R_u = np.ceil(K * theta_u / (0.5 * np.pi)) - 1
        R_l = int(K * theta_l / (0.5 * np.pi))
        
        # if (K * theta_u) - R_u < self._epsilon / 1000:
        #     R_u -= 1
        if R_u == R_l:
            return (K - 1) // 2 # integer is guaranteed, but cast to int
        K -= 2
    
    return k_prev