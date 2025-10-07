from firedrake import *
from petsc4py import PETSc
"""
Computing eigenvalues and eigenvectors of the Stokes operator
using SLEPc4Py and Scott-Vogelius elements.
"""
opt = PETSc.Options()
n = opt.getInt('n', 10) 


# Create mesh
mesh = UnitSquareMesh(n, n)
# Define function space
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "DG", 1)
W = V * Q
# Define trial and test functions
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
# Define bilinear forms
a = (inner(grad(u), grad(v)) - inner(p,div(v)) + inner(div(u),q))*dx
m = (inner(u, v) + inner(p,q))*dx
bc = DirichletBC(W.sub(0), as_vector([0, 0]), [1,2,3,4])
eigenproblem = LinearEigenproblem(A=a, M=m, bcs=bc, restrict=False, bc_shift=1e8)
opts = {"eps_gen_non_hermitian": None}
eigensolver = LinearEigensolver(eigenproblem, n_evals=10, solver_parameters=opts, options_prefix="")
nconv = eigensolver.solve()
for i in range(nconv):
    print(eigensolver.eigenvalue(i))
