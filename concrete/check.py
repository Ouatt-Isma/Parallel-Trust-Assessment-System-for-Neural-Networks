import numpy as np
np.random.seed(42)
a = np.random.random(10)
b = np.where(a>0.2)
print(len(b[0]))