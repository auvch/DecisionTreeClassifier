import numpy as np
import pandas as pd

def getEnt(col):
    prob = col.value_counts(normalize=True)
    return sum(np.log2(prob) * prob * (-1))

def getInfoGain(data, feature):
    e1 = data.groupby(feature).apply(lambda x: getEnt(x['class']))
    p1 = data[feature].value_counts(normalize=True)
    return getEnt(data['class']) - sum(e1 * p1)

def getInfoGainRatio(data, feat):
    return getInfoGain(data, feat) / getEnt(data[feat])

def getGini(data):
    prob = data.value_counts(normalize=True)
    return 1 - sum(prob * prob)