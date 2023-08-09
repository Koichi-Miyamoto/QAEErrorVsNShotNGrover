import numpy as np
from qulacs import QuantumState, QuantumCircuit
from qulacs.gate import Z, RY, merge

def QAEQulacs(ampSq, nShot, nGroverMax):

    theta = 2.0 * np.arcsin(np.sqrt(ampSq))
    gateStateGen = RY(0, theta)
    grover = merge([Z(0), gateStateGen.get_inverse(), Z(0), gateStateGen])
    
    state = QuantumState(1)
    gateStateGen.update_quantum_state(state)
    nGrover = 0
    nGrovers = [nGrover]
    n1s = []
    
    while nGrover <= nGroverMax:
        prob1 = np.abs(state.get_amplitude(1)) ** 2
        n1s.append(np.random.binomial(nShot, prob1))
        
        nGroverNext = 1 if nGrover == 0 else nGrover * 2
        if nGroverNext <= nGroverMax:
            nGroverAdd = nGroverNext - nGrover # 追加の作用回数
            nGrovers.append(nGroverNext)
            for j in range(nGroverAdd): grover.update_quantum_state(state) # その回数だけGrover operatorを作用
                
        nGrover = nGroverNext
        
    return nGrovers, n1s

def QAEAnalyticDistFunc(ampSq, nShot, nGroverMax):

    theta = np.arcsin(np.sqrt(ampSq))
    nGrover = 0
    nGrovers = [nGrover]
    n1s = []
    
    while True:
        prob1 = np.sin((2 * nGrover + 1) * theta) ** 2
        n1s.append(np.random.binomial(nShot, prob1))
        
        nGrover = 1 if nGrover == 0 else nGrover * 2
        
        if nGrover <= nGroverMax:
            nGrovers.append(nGrover)
        else:
            break
        
    return nGrovers, n1s