from uu import Error
import numpy as np
from scipy.stats import beta

def IQAEBinom(ampSq, epsilon, alpha, nShotUnit, confint_method="chernoff", minRatio=2):

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

    while ampSqWidth > epsilon:
        nGrovers.append(nGrover)
        nGroverPrev = nGrover
        kRound = 2 * nGrover + 1
        prob1 = np.sin(kRound * theta) ** 2
        nShotRound = 0
        n1Round = 0
        alphaRound = 2 * alpha / 3 * kRound / kmax
        # nMaxRound = 2 / np.sin(np.pi / 21) ** 2 / np.sin(8 * np.pi / 21) ** 2 * np.log(2 / alphaRound)
        rRound = int(kRound * thetaInterval[0] / (0.5 * np.pi))

        while nGrover == nGroverPrev:
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

            nGrover = _find_next_k(nGrover, thetaInterval, minRatio)

        thetaIntervals.append(thetaInterval)
        ampSqIntervals.append(ampSqInterval)
        nShots.append(nShotRound)
        n1s.append(n1Round)

    ret = {"Estimate":ampSqML,
           "nGrover":nGrovers,
           "nShots":nShots,
           "n1s":n1s,
           "thetaIntervals":thetaIntervals,
           "ampSqIntervals":ampSqIntervals}
    return ret

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