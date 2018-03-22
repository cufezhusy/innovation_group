import numpy as np
import pandas as pd

a = np.array([[0.0,1.630000e+01,1.990000e+01,1.840000e+01],
                 [1.0,1.630000e+01,1.990000e+01,1.840000e+01],
                 [2.0,1.630000e+01,1.990000e+01,1.840000e+01]])


dfobj = pd.DataFrame(data=a, index=None, columns=["a","b","c","d"])
print(dfobj.head())


dfobj.rename(columns={'0':'a','1':'b','2':'c','3':'d'},inplace=True)

print(dfobj.head())
