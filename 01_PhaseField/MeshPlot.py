from fenics import *
import matplotlib.pyplot as plt

mesh = Mesh('/mnt/c/Users/Visconti/GoogleDrive/INEGI/05_Study/01_PhaseField/mesh.xml')

plt.figure(1)
plot(mesh)

plt.show()