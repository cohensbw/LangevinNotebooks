using Profile

using Langevin
using Langevin.Geometries: Geometry
using Langevin.Lattices: Lattice
using Langevin.QuantumLattices: view_by_site, view_by_τ
using Langevin.HolsteinModels: HolsteinModel
using Langevin.HolsteinModels: assign_μ!, assign_ω!, assign_λ!
using Langevin.HolsteinModels: assign_tij!, assign_ωij!, assign_λij!
using Langevin.HolsteinModels: setup_checkerboard!
using Langevin.InitializePhonons: init_phonons_single_site!

# functions to be timed
using Langevin.HolsteinModels: setup_checkerboard!, construct_expnΔτV!
using Langevin.PhononAction: calc_Sbose, calc_dSbose!
using LinearAlgebra: mul!

# number of dimensions
ndim = 3

# number of orbitals per unit cell
norbits = 1

# lattice vectors
lvecs = [[1.0,0.0,0.0],
         [0.0,1.0,0.0],
         [0.0,0.0,1.0]]

# basis vectors
bvecs = [[0.0,0.0,0.0]]

# defining square lattice geometry
geom = Geometry(ndim, norbits, lvecs, bvecs)

# defining lattice size
L = 10

# constructing finite square lattice
lattice = Lattice(geom,L)

# discretization
Δτ = 0.1

# setting temperature
β = 5.0

println("Constructing Holstein Model")
print('\n')

# constructing holstein model
holstein = HolsteinModel(geom,lattice,β,Δτ)

# assigning phonon frequency of 1 to each site
assign_ω!(holstein, 1.0, 0.0)

# assigning electron-phonon couplong of 1 to each site
assign_λ!(holstein, 1.0, 0.0)

# assigning chemical potential for half-filling (μ=-λ²/ω²) to each site
assign_μ!(holstein, -1.0, 0.0)

# adding hopping parameters in x direction
assign_tij!(holstein, 1.0, 0.0, 1, 1, [1,0,0])

# adding hopping parameters in y direction
assign_tij!(holstein, 1.0, 0.0, 1, 1, [0,1,0])

# adding hopping parameters in z direction
assign_tij!(holstein, 1.0, 0.0, 1, 1, [0,0,1])

# organize electron hoppings for checkerboard decomposition
setup_checkerboard!(holstein)

# intialize phonon field
init_phonons_single_site!(holstein)

###########################################################
## TIME DIFFERENT METHODS WHERE PERFORMANCE IS IMPORTANT ##
###########################################################

println("Timing `construct_expnΔτV!`")
for i in 1:10
    @time construct_expnΔτV!(holstein)
end
print('\n')

println("Timing `calc_dSbose!`")
dSbose = zeros(Complex{Float64},size(holstein,1))
for i in 1:10
    @time calc_dSbose!(dSbose,holstein)
end
print('\n')

println("Timing `mul!(y,holstein,v)`")
y = ones(Complex{Float64},size(holstein,1))
v = ones(Complex{Float64},size(holstein,1))
for i in 1:10
    @time mul!(y,holstein,v)
end
# @profile mul!(y,holstein,v)
# Profile.print()