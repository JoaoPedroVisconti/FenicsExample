import numpy as np
import matplotlib.pyplot as plt

file = np.loadtxt("/mnt/c/Users/User/GoogleDrive/INEGI/05_Study/01_PhaseField/ResultsDir/ForcevsDisp.txt")

x, y = np.hsplit(file, 2)

plt.plot(x,y)

plt.show()