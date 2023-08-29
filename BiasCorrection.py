import numpy as np

def ExpectedLogLikDer2nd(ampSq, nGrover, nShot):
    
    theta = np.arcsin(np.sqrt(ampSq))
    k = 2 * nGrover + 1
    return -2 * nShot * k * k / np.sin(2 * theta) ** 2

def ExpectedLogLikDer3rd(ampSq, nGrover, nShot):
    
    theta = np.arcsin(np.sqrt(ampSq))
    k = 2 * nGrover + 1
    sin2theta = np.sin(2 * theta)
    tanktheta = np.tan(k * theta)

    return 4 * nShot * k * k * (k * (1/ tanktheta - tanktheta) * sin2theta + 3) / sin2theta ** 4

def DerExpectedLogLikDer2nd(ampSq, nGrover, nShot):
    
    theta = np.arcsin(np.sqrt(ampSq))
    k = 2 * nGrover + 1

    return 8 * nShot * k * k * np.cos(2 * theta) / np.sin(2 * theta) ** 4

def BiasCordeiroKlein(ampSq, nGrover, nShot):
    
    return (DerExpectedLogLikDer2nd(ampSq, nGrover, nShot) - 0.5 * ExpectedLogLikDer3rd(ampSq, nGrover, nShot)) / \
        ExpectedLogLikDer2nd(ampSq, nGrover, nShot) ** 2

def BiasTaylorExpTo2nd(ampSq, nGrover, nShot):

    theta = np.arcsin(np.sqrt(ampSq))
    k = 2 * nGrover + 1
    sin2ktheta = np.sin(2 * k * theta)
    cos2ktheta = np.cos(2 * k * theta)

    return (np.cos(2 * theta) * sin2ktheta - k * cos2ktheta * np.sin(2 * theta)) / 4 / k / k / nShot / sin2ktheta

def BiasTaylorExpTo3rd(ampSq, nGrover, nShot):

    theta = np.arcsin(np.sqrt(ampSq))
    k = 2 * nGrover + 1
    sin2ktheta = np.sin(2 * k * theta)
    cos2ktheta = np.cos(2 * k * theta)

    term2nd = -4 * (np.sin(2 * theta) * sin2ktheta ** 2 / k / k + 3 / k * np.cos(2 * theta) * cos2ktheta * sin2ktheta 
                    - np.sin(2 * theta) * (sin2ktheta ** 2 + 3 * cos2ktheta ** 2)) / sin2ktheta ** 3 \
                        / k / nShot / nShot * (1 - 2 * sin2ktheta * sin2ktheta) * cos2ktheta ** 2 / 6
    
    return BiasTaylorExpTo2nd(ampSq, nGrover, nShot) + term2nd

