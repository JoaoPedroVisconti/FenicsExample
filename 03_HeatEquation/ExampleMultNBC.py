import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
#from IPython.display import clear_output
import numpy as np
import itertools
from dolfin import *

# Create classes for defining parts of the boundaries
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0)

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 1.0)

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.0)

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 1.0)

# Initialize sub-domain instances
left = Left()
top = Top()
right = Right()
bottom = Bottom()

nx = 64
ny = 64

# Define mesh
mesh = UnitSquareMesh(nx, ny)

t = 0.0 # initial time 
Time = 1.0         # final time
num_steps = 20     # number of time steps
dt = Time / num_steps # time step size

# Initialize mesh function for boundary domains
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
left.mark(boundaries, 1)
top.mark(boundaries, 2)
right.mark(boundaries, 3)
bottom.mark(boundaries, 4)

# Define input data
a0 = Constant(1.0)
a1 = Constant(0.1)
g = Constant("0.0001")
f = Constant("4.")

# Define function space and basis functions
V = FunctionSpace(mesh, "CG", 2)
u = TrialFunction(V)
v = TestFunction(V)

T_d = Expression('sin(pi*x[1])*exp( -pi*pi*t )*pow(sin(pi*x[0]) , 3)', degree=2, t=0)
T_old = interpolate(T_d, V)

# Define new measures associated with the exterior boundaries
ds = ds(subdomain_data = boundaries)

# Define variational form
F = (u*v*dx - T_old*v*dx
    + dt*inner(a1*grad(u), grad(v))*dx
    - dt*f*v*dx - dt*g*v*ds(1) - dt*g*v*ds(2) - dt*g*v*ds(3) - dt*g*v*ds(4)
    )

# Separate left and right hand sides of equation
a, L = lhs(F), rhs(F)
u = Function(V)

# Solve problem
for n in range(num_steps):

    # Update current time
    t += dt   
    # Compute solution
    solve(a == L, u)
    


    #3D Plot of the Solution  
    # matplotlib inline
    x = np.linspace(0.0, 1.0, nx + 1)
    y = np.linspace(0.0, 1.0, ny + 1)
    
    X, Y = np.meshgrid(x, y) 
    
    vertex_values_u = u.compute_vertex_values()
    Z_vert = vertex_values_u.reshape((nx + 1,ny + 1 ))
        
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot_surface(X, Y, Z_vert, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
    #ax.set_title('t = ' +str(round(t, 2)))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('T')
    #ax.set_zlim3d(-1.0 , 1.0)
    
    plt.show()
    
    print('t = %.2f' % (t)) 
     
    clear_output(wait=True)
    
    T_old.assign(u)