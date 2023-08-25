import numpy as np
from scipy.stats import beta

def IQAEFixedGroverBinom(ampSq, epsilon, alpha, nShotUnit, nGroverRatio, confint_method="chernoff"):

    theta = np.arcsin(np.sqrt(ampSq))
    nGrover = 0
    nGrovers = []
    nShots = []
    n1s = []
    ampSqIntervals = [[0, 1]]
    ampSqWidth = 1
    thetaInterval = [0, 0.5 * np.pi]
    thetaIntervals = [thetaInterval]
    kmax = np.pi / 4 / epsilon
    flgStop = False

    while True:
        nGrovers.append(nGrover)
        kRound = 2 * nGrover + 1
        prob1 = np.sin(kRound * theta) ** 2
        nShotRound = 0
        n1Round = 0
        alphaRound = 2 * alpha / 3 * kRound / kmax
        # nMaxRound = 2 / np.sin(np.pi / 21) ** 2 / np.sin(8 * np.pi / 21) ** 2 * np.log(2 / alphaRound)
        rRound = int(kRound * thetaInterval[0] / (0.5 * np.pi))
        
        # print(theta)
        # print(nGrover)
        # print(prob1)

        while True:
            nShotRound += nShotUnit
            n1 = np.random.binomial(nShotUnit, prob1)
            n1Round += n1
            p1Obs = n1Round / nShotRound

            if confint_method == "chernoff":
                p1Width = np.sqrt(0.5 / nShotRound * np.log(2 / alphaRound))
                p1Max = min(p1Obs + p1Width, 1)
                p1Min = max(p1Obs - p1Width, 0)
            elif confint_method == "beta":
                p1Min, p1Max = _clopper_pearson_confint(n1Round, nShotRound, alphaRound)
            else:
                raise Exception("unknown confint_method")

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

            nGroverNext = 1 if nGrover == 0 else nGrover * nGroverRatio
            if int((2 * nGroverNext + 1) * theta_u / (0.5 * np.pi)) == int((2 * nGroverNext + 1) * theta_l / (0.5 * np.pi)):
                nGrover = nGroverNext
                break

            if ampSqWidth <= epsilon:
                flgStop = True
                break

        thetaIntervals.append(thetaInterval)
        ampSqIntervals.append(ampSqInterval)
        nShots.append(nShotRound)
        n1s.append(n1Round)

        if flgStop:
            break

    ret = {"Estimate":ampSqML,
           "TotalOracleCalls": np.dot(2 * np.array(nGrovers) + 1, nShots),
           "nGrover":nGrovers,
           "nShots":nShots,
           "n1s":n1s,
           "thetaIntervals":thetaIntervals,
           "ampSqIntervals":ampSqIntervals}
    return ret

def _clopper_pearson_confint(counts, shots, alpha):
    """Compute the Clopper-Pearson confidence interval for `shots` i.i.d. Bernoulli trials.

    Args:
        counts: The number of positive counts.
        shots: The number of shots.
        alpha: The confidence level for the confidence interval.

    Returns:
        The Clopper-Pearson confidence interval.
    """
    lower, upper = 0, 1

    # if counts == 0, the beta quantile returns nan
    if counts != 0:
        lower = beta.ppf(alpha / 2, counts, shots - counts + 1)

    # if counts == shots, the beta quantile returns nan
    if counts != shots:
        upper = beta.ppf(1 - alpha / 2, counts + 1, shots - counts)

    return lower, upper