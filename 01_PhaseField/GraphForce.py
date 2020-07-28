import numpy as np
import matplotlib.pyplot as plt

file = np.loadtxt("ResultsDir1/ForcevsDisp.txt")

x, y = np.hsplit(file, 2)

plt.plot(x,y)

plt.show()