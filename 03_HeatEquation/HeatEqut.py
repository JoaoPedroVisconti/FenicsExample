from fenics import *
import numpy as np
import matplotlib.pyplot as plt

from ufl.operators import nabla_div, nabla_grad
from sympy import symbols, diff


############################################################
###############         SUBDOMAIN      #####################

class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near (x[0], 0.0)

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near (x[0], 20.0)

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near (x[1], 0.0)

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near (x[1], 12.0)

left = Left()
top = Top()
right = Right()
bottom = Bottom()

### Another way to do it

# left = CompiledSubDomain("near(x[0], 0.0) && on_boundary")
# top = CompiledSubDomain("near(x[1], 12.0) && on_boundary")
# right = CompiledSubDomain("near(x[0], 20.0) && on_boundary")
# bottom = CompiledSubDomain("near(x[1], 0.0) && on_boundary")

############################################################

t = 0
T = 100
dt = 1
num_steps = int(T/dt)


hc = 0.0001
c = 10E6
k = 0.05
rho = 10E-6

############################################################
###############         MESH         #######################
x0 = 0
y0 = 0
x1 = 20
y1 = 12

nx = 20
ny = 12

mesh = RectangleMesh(Point(x0, y0), Point(x1, y1), nx, ny)
coordinates = mesh.coordinates()
############################################################
# print("Plotting a RectangleMesh")
# plt.figure()
# plot(mesh, title="Rectangle")
# plt.show()

############################################################
###########    INICIALIZING MESH FUNCTIONS     #############

boundaries = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
left.mark(boundaries, 1)
top.mark(boundaries, 2)
right.mark(boundaries, 3)
bottom.mark(boundaries, 4)

############################################################

V = FunctionSpace(mesh, 'P', 1)

b = 1/6; H=12; d_x=0.2; d_y=0.2

u_D = Constant(20)

tol = 1E-14

u_n = interpolate(u_D, V)


u = TrialFunction(V)
v = TestFunction(V)
f = Expression('50*exp(-2*(pow((x[0]-t),2) + pow((x[1]-b*H),2))/(d_x*d_x + d_y*d_y))', degree=1, b=b, H=H,d_x=d_x,d_y=d_y, t = 0)


g1 = Expression('-hc*(u_D - u_n)', degree=1, hc=hc, u_n=u_n, u_D=u_D)
g2 = Expression('-hc*(-u_D + u_n)', degree=1, hc=hc, u_n=u_n, u_D=u_D)
g3 = Expression('-hc*(-u_D + u_n)', degree=1, hc=hc, u_n=u_n, u_D=u_D)
g4 = Expression('-hc*(u_D - u_n)', degree=1, hc=hc, u_n=u_n, u_D=u_D)

F = (u*v*dx 
    + dt*dot(grad(u), grad(v))*dx 
    # + (u.dx(0))*v*dx  # Provides faster cooling of the plate
    - (u_n*v*dx) 
    - dt*f*v*dx 
    - dt*g1*v*ds(1)
    - dt*g2*v*ds(2)
    - dt*g3*v*ds(3)
    - dt*g4*v*ds(4))

a, L = lhs(F), rhs(F)

u = Function(V)

conc_f = File ("ResultsDir/phi.pvd")

t_plot = []
v_plot = []

for n in range (num_steps):

    t += dt
    f.t = t

    solve(a == L, u)

    # Compute the error at vertices
    # u_e = interpolate(u_D, V)
    # error = np.abs(u_e.vector().get_local() - u.vector().get_local()).max()
    # print('t = %.2f: error = %.3g' % (t, error))

    plt.figure(1)
    plot(u)

    nodal_values  = u.vector().get_local()
    vertex_values = u.compute_vertex_values()

    # print('t = %.2f:  Vertex_Value = %.2f' % (t, vertex_values[136]))
    # print(vertex_values[136])
    
    # for i, x in enumerate(coordinates):
    #     print('vertex %d: vertex_values[%d] = %g\tu(%s) = %g' % (i, i, vertex_values[i], x, u(x)))

    # print(type(t))
    # print(type(vertex_values[136]))

    t_plot.append(t)
    v_plot.append(vertex_values[136])

    u_n.assign(u)

    conc_f << u

plt.figure(2)
plt.plot(t_plot, v_plot)
plt.title('Vertex Value for (x=10, y=6)')
plt.grid(True)

plt.show()

    

