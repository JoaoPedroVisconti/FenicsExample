from fenics import *

### MESH ###
# The mesh can be genearted by the GMSH program and then converted with
# dolfin-convert to xml

mesh = Mesh('mesh.xml') # Attention to run inside the folder

# Define the space
V = FunctionSpace(mesh, 'CG', 1)
W = VectorFunctionSpace(mesh, 'CG', 1)

WW = FunctionSpace(mesh, 'DG', 0)

p, q = TrialFunction(V), TestFunction(V)
u, v = TrialFunction(W), TestFunction(W)

n = FacetNormal(mesh)

# Material Parameters
Gc    = 2.7         # Critical energy release rate 2.7 MPa
l     = 0.015       # Phase Field length scale
lmbda = 121.1538E3  # Lame's first parameter MPa
mu    = 80.7692E3   # Lame's second parameter MPa

# Constitutive Functions
# Strain
def epsilon(u):
    return sym(grad(u))

# Stress
def sigma(u):
    return 2.0*mu*epsilon(u)+lmbda*tr(epsilon(u))*Identity(len(u))

# Energy strain
def psi(u):
    return 0.5*(lmbda+mu)*(0.5*(tr(epsilon(u))+abs(tr(epsilon(u)))))**2+\
           mu*inner(dev(epsilon(u)),dev(epsilon(u)))	

# History variable field
def H(uold,unew,Hold):
    return conditional(lt(psi(uold),psi(unew)),psi(unew),Hold)


# Subdomain
top    = CompiledSubDomain("near(x[1], 0.5) && on_boundary")
bottom = CompiledSubDomain("near(x[1], -0.5) && on_boundary")

def Crack(x):   # Define the initial crack to assign 1 on phase field variable
    return abs(x[1]) < 1E-3 and x[0] <= 0

# Boundary Conditions Displacement
load = Expression('t', t = 0, degree=1)  # Load expression

bcbot = DirichletBC(W, Constant((0,0)), bottom)
bctop = DirichletBC(W.sub(1), load, top)   # Applying the Force

bc_u = [bctop, bcbot]

# Boundary Conditions Phase Field
bc_phi = [DirichletBC(V, Constant(1), Crack)]

# Taging the boundaries
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)

boundaries.set_all(0)  # Tag all by 0

top.mark(boundaries, 1)
ds = Measure("ds")(subdomain_data=boundaries)

# Variational Form
unew, uold       = Function(W), Function(W)
pnew, pold, Hold = Function(V), Function(V), Function(V)

# Equations to be solved
# Strain Energy related with u
E_du = ((1.0-pold)**2)*inner(grad(v),sigma(u))*dx

# Strain Energy related with phase field
E_phi = (Gc*l*inner(grad(p),grad(q))+((Gc/l)+2.0*H(uold,unew,Hold))\
            *inner(p,q)-2.0*H(uold,unew,Hold)*q)*dx


# Defining the variational problems
# Displacement
p_disp = LinearVariationalProblem(lhs(E_du), rhs(E_du), unew, bc_u)

# Phase Field
p_phi = LinearVariationalProblem(lhs(E_phi), rhs(E_phi), pnew, bc_phi)

# Store the call for to the linear solver to obtain u and phi
solver_disp = LinearVariationalSolver(p_disp)
solver_phi  = LinearVariationalSolver(p_phi)


# Initialization parameter for interative procedure
t      = 0       # Initial time
u_r    = 0.007   # Applied displacement LOAD
deltaT = 0.1     # Increment in time
tol    = 1e-3    # Error tolerance

# Creating the files to store the output request
out_phi         = File("ResultsDir1/phi.pvd")
out_u           = File("ResultsDir1/disp.pvd")
forc_disp_fname = open("ResultsDir1/ForcevcDisp.txt", 'w')

# Staggered Scheme
while t <= 1.0:
    
    t += deltaT

    if t >= 0.7:
        deltaT = 0.05

    load.t = t*u_r  # The expression for the load is updated

    iter = 0  # Inicialize the interaction counter
    err  = 1  # Inicialize the error variable

    while err > tol:
        iter += 1

        solver_disp.solve()  # Solve phase field problem
        solver_phi.solve()   # Solve displacement problem

        # Compute the errors
        err_u   = errornorm(unew,uold,norm_type = 'l2',mesh = None)
        err_phi = errornorm(pnew,pold,norm_type = 'l2',mesh = None)
        err = max(err_u,err_phi)

        uold.assign(unew)
        pold.assign(pnew)
        Hold.assign(project(psi(unew), WW))  # Compute and project the phi+

        if err < tol:

            print('Iterations:', iter, ', Total time', t)

            if round(t*1e4) % 10 == 0:
                out_phi << pnew
                out_u   << unew

                Traction = dot(sigma(unew), n)

                fy = Traction[1]*ds(1)

                forc_disp_fname.write(str(t*u_r) + "\t")
                forc_disp_fname.write(str(assemble(fy)) + "\n")

forc_disp_fname.close()
print('Simulation completed')
