from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

from mshr import *
import sympy as sym
from ufl.operators import nabla_div, nabla_grad

x, y = sym.symbols('x[0], x[1]')            # needed by UFL
b = 1/6 
H=12 
dx=0.2 
dy=0.2
t = 0
f = 50*exp(-2*((x-t)**2 + (y-b*H)**2)/(dx*dx + dy*dy))                      # exact solution

# Collect variables
variables = [b, H, dx, dy]

# Turn into C/C++ code strings
variables = [sym.printing.ccode(var) for var in variables]

# Turn into FEniCS Expressions
variables = [Expression(var, degree=2) for var in variables]

# Extract variables
b, H, dx, dy = variables

T = 2
num_steps = 10
dt = T/num_steps

x0 = 0
y0 = 0
x1 = 20
y1 = 12

nx = 20
ny = 10

mesh = RectangleMesh(Point(x0, y0), Point(x1, y1), nx, ny)

# print("Plotting a RectangleMesh")
# plt.figure()
# plot(mesh, title="Rectangle")
# plt.show()

V = FunctionSpace(mesh, 'P', 1)

alpha = 3; beta = 1.2

# u_D = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t', degree=2, alpha=alpha, beta=beta, t=0)
u_D = Constant(20)

def boundary (x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

u_n = interpolate(u_D, V)


u = TrialFunction(V)
v = TestFunction(V)
# f = Expression('50*exp(-2*(pow((x[0]-t),2) + pow((x[1]-b*H),2))/(dx*dx + dy*dy))', degree=2, b=b, H=H, dx=dx, dy=dy, t = 0)

F = u*v*dx + dt*dot(grad(u), grad(v))*dx - (u_n + dt*f)*v*dx
a, L = lhs(F), rhs(F)

u = Function(V)
t = 0

for n in range (num_steps):

    t += dt
    f.t = t

    solve(a == L, u, bc)

    u_n.assign(u)

    plot(u)

    # Compute the error at vertices
    u_e = interpolate(u_D, V)
    error = np.abs(u_e.vector().get_local() - u.vector().get_local()).max()
    print('t = %.2f: error = %.3g' % (t, error))

plt.show()