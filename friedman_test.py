# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import scipy.stats as stats

data = pd.read_csv("out.csv")
data.drop(['Unnamed: 0'], axis=1, inplace=True)

data_ranked = data.rank(axis=1)
data_ranked = data.as_matrix()
print stats.friedmanchisquare(*[data_ranked[x, :] for x in np.arange(data_ranked.shape[0])])