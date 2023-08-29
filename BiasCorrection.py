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

    term3rd = -4 * (np.sin(2 * theta) * sin2ktheta ** 2 / k / k + 3 / k * np.cos(2 * theta) * cos2ktheta * sin2ktheta 
                - np.sin(2 * theta) * (sin2ktheta ** 2 + 3 * cos2ktheta ** 2)) / sin2ktheta ** 5 \
                / k / nShot / nShot * np.sin(k * theta) ** 2 * np.cos(k * theta) ** 2 * (1 - 2 * np.sin(k * theta) ** 2) / 6
    
    return BiasTaylorExpTo2nd(ampSq, nGrover, nShot) + term3rd

def BiasTaylorExpTo4th(ampSq, nGrover, nShot):

    theta = np.arcsin(np.sqrt(ampSq))
    k = 2 * nGrover + 1
    p1Final = np.sin(k * theta) ** 2
    sin2ktheta = np.sin(2 * k * theta)
    cos2ktheta = np.cos(2 * k * theta)
    sin2theta = np.sin(2 * theta)
    cos2theta = np.cos(2 * theta)

    thetaDeriv4th = \
        -8 * cos2theta / (k * sin2ktheta) ** 4 + 48 * sin2theta * cos2ktheta / k ** 3 / sin2ktheta ** 5 \
        +8 * cos2theta / k ** 2 * ((12 + 3 * cos2ktheta ** 2) / sin2ktheta ** 6 - 8 / sin2ktheta ** 4) \
        -24 * sin2theta * cos2ktheta / k * (5 / sin2ktheta ** 7 - 2 / sin2ktheta ** 5)
    moment4th = p1Final * (1 - p1Final) * (6 * p1Final ** 2 - 6 * p1Final + 1) / nShot ** 3 + \
       3 * p1Final ** 2 * (1 - p1Final) ** 2 / nShot ** 2
    
    return BiasTaylorExpTo3rd(ampSq, nGrover, nShot) + thetaDeriv4th * moment4th / 24
