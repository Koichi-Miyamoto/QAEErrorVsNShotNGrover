import numpy as np

def Bias(ampSq, finalGroverNum, finalShotNum):

    theta = np.arcsin(np.sqrt(ampSq))
    k = 2 * finalGroverNum + 1   

    # 2nd derivative of ampSq w.r.t. theta
    sin2ktheta = np.sin(2 * k * theta)
    ampSqDerDerTheta = 2 * (np.cos(2 * theta) * sin2ktheta - k * np.cos(2 * k * theta) * np.sin(2 * theta)) / k ** 2 / sin2ktheta ** 3
    
    p1FinalRound = np.sin(k * theta) ** 2
    
    return 0.5 * ampSqDerDerTheta * p1FinalRound * (1 - p1FinalRound) / finalShotNum

