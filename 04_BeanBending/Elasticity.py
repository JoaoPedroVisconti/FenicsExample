from fenics import *

import matplotlib.pyplot as plt
from ufl import nabla_div, nabla_grad

#---- Gemetry -----#
L = 160
W = 2
H = 8
#------------------#

# #---- Boundary ----#
# class Left(SubDomain):
#     def inside(self, x, on_boundary):
#         return near (x[0], 0)

# class Right(SubDomain):
#     def inside(self, x, on_boundary):
#         return near (x[0], L)

# class Bottom(SubDomain):
#     def inside(self, x, on_boundary):
#         return near (x[1], 0)

# class Top(SubDomain):
#     def inside(self, x, on_boundary):
#         return near (x[1], W) 

# class Front(SubDomain):
#     def inside(self, x, on_boundary):
#         return near (x[2], 0)

# class Back(SubDomain):
#     def inside(self, x, on_boundary):
#         return near (x[2], H)

# left   = Left()
# top    = Top()
# bottom = Bottom()
# right  = Right()
# front  = Front()
# back   = Back()
# #------------------#

#------ Mesh ------#
mu = 80E3
lmbda = 120E3
rho = 1E-6
g = 500

mesh = BoxMesh(Point(0,0,0), Point(L, W, H), 20, 2, 8)
#------------------#

# #- Mesh Functions -#
# boundaries = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
# boundaries.set_all(0)

# left.mark(boundaries, 1)
# top.mark(boundaries, 2)
# right.mark(boundaries, 3)
# bottom.mark(boundaries, 4)
# front.mark(boundaries, 5)
# back.mark(boundaries, 6)
# #------------------#

V = VectorFunctionSpace(mesh, 'P', 1)

u_D = Constant((0,0,0))

tol = 1E-14

def clamped_boundary(x, on_boundary):
    return on_boundary and x[0] < tol

bc = DirichletBC(V, u_D, clamped_boundary)

def Solve_Lin_Problem(g):

    #- Variational Problem -#
    u = TrialFunction(V)
    v = TestFunction(V)

    d = u.geometric_dimension()

    f = Constant((0,0,-rho*g))

    #----- Strain -----#
    def epsilon(u):
        # return 0.5*(nabla_grad(u) + nabla_grad(u).T)
        return sym(nabla_grad(u))
    #------------------#

    #----- Stess ------#
    def sigma(u):
        return lmbda*nabla_div(u)*Identity(d) + 2*mu*epsilon(u)
    #------------------#

    a = inner(sigma(u), epsilon(v))*dx
    L = dot(f, v)*dx

    u = Function(V)
    solve (a == L, u, bc)

    file_L = File("Results/phi_l.pvd")
    file_L << u

    return -u(160, 1, 4)[2]


def Solve_NonLin_Problem(g):

    f = Constant((0,0,-g*rho))

    du = TrialFunction(V)
    v = TestFunction(V)
    u = Function(V)

    d = len(u)

    I = Identity(d)
    F = I + grad(u)
    C = F.T*F

    Ic = tr(C)
    J  = det(F)

    # Strain energy density (compressible neo-Hookean model)
    psi = (mu/2)*(Ic - 3) - mu*ln(J) + (lmbda/2)*(ln(J))**2

    # Potential Energy
    Pi = psi*dx - dot(f, u)*dx

    # First variation of PI (directional derivatice about u in the direction of v)
    L = derivative(Pi, u, v)

    # Jacobian of L
    J = derivative(L, u, du)

    # solve(L == 0, u, bc, J=J)

    # solve(L == 0, u, bc, J=J,
    #   solver_parameters={"maximum_iterations": 60})

    problem = NonlinearVariationalProblem(L, u, bc, J=J)
    solver  = NonlinearVariationalSolver(problem)
    prm = solver.parameters
    prm['newton_solver']['absolute_tolerance'] = 1E-8
    prm['newton_solver']['relative_tolerance'] = 1E-7
    prm['newton_solver']['maximum_iterations'] = 25
    prm['newton_solver']['relaxation_parameter'] = 1.0
    solver.solve()

    file_N = File("Results/phi_n.pvd")
    file_N << u

    return -u(160, 1, 4)[2]


L_u_end = []
N_u_end = []
g_list = []

while g <= 45E+5:
    g_list = g_list + [g/1E+4]

    L_u_end = L_u_end + [Solve_Lin_Problem(g)]
    N_u_end = N_u_end + [Solve_NonLin_Problem(g)]   

    g = g + 5E+5

    
plt.plot(L_u_end, g_list, N_u_end, g_list, '--')
plt.xlabel('Free end Displacement')
plt.ylabel('Body Force')

# plot(u)
plt.show()