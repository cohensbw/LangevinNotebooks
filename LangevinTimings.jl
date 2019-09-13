# NOTE: For improved performence run this script from the terminal as:
# >julia --optimize=3 --math-mode=fast --check-bounds=no LangevinTimings.jl

using Profile
using BenchmarkTools
using IterativeSolvers
using SparseArrays

using Langevin
using Langevin.Geometries: Geometry
using Langevin.Lattices: Lattice
using Langevin.HolsteinModels: HolsteinModel
using Langevin.HolsteinModels: assign_μ!, assign_ω!, assign_λ!
using Langevin.HolsteinModels: assign_tij!, assign_ωij!
using Langevin.HolsteinModels: setup_checkerboard!
using Langevin.HolsteinModels: mulM!, mulMᵀ!, mulMᵀM!
using Langevin.InitializePhonons: init_phonons_single_site!
using Langevin.Checkerboard: checkerboard_matrix, checkerboard_mul!
using Langevin.LangevinDynamics: calc_dSdϕ!

# functions to be timed
using Langevin.HolsteinModels: setup_checkerboard!, construct_expnΔτV!
using Langevin.PhononAction: calc_dSbosedϕ!
import LinearAlgebra: mul!

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
β = 12.0

println("Constructing Holstein Model")
print('\n')

# constructing holstein model
holstein = HolsteinModel(geom,lattice,β,Δτ)

# assigning phonon frequency of 1 to each site
ω = 1.0
assign_ω!(holstein, ω, 0.0)

# assigning electron-phonon couplong of 1 to each site
λ = 1.0
assign_λ!(holstein, λ, 0.0)

# assigning chemical potential for half-filling (μ=-λ²/ω²) to each site
μ = -λ^2/ω^2
assign_μ!(holstein, μ, 0.0)

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

#################################################
## PRE-ALLOCATING ARRAYS NEEDED FOR SIMULATION ##
#################################################

dϕdt     = zeros(Float64,          length(holstein))
fft_dϕdt = zeros(Complex{Float64}, length(holstein))

dSdϕ     = zeros(Float64,          length(holstein))
fft_dSdϕ = zeros(Complex{Float64}, length(holstein))
dSdϕ2    = zeros(Float64,          length(holstein))

g    = zeros(Float64, length(holstein))
Mᵀg  = zeros(Float64, length(holstein))
M⁻¹g = zeros(Float64, length(holstein))

η     = zeros(Float64,          length(holstein))
fft_η = zeros(Complex{Float64}, length(holstein))

###########################################################
## TIME DIFFERENT METHODS WHERE PERFORMANCE IS IMPORTANT ##
###########################################################

println("Timing `mul!`")
y = ones(Float64,length(holstein))
v = ones(Float64,length(holstein))
t = @benchmark mul!($y,$holstein,$v)
display(t)
print('\n')
print('\n')

# println("Timing calc_dSdϕ")
# tol = 1e-4
# @time iters = calc_dSdϕ!(dSdϕ2, g, Mᵀg, M⁻¹g, holstein, tol)
# println(iters)

println("Timing `construct_expnΔτV!`")
t = @benchmark construct_expnΔτV!($holstein)
display(t)
print('\n')
print('\n')

println("Timing `calc_dSbosedϕ!`")
dSbose = zeros(Float64,size(holstein,1))
t = @benchmark calc_dSbosedϕ!($dSbose, $holstein)
display(t)
print('\n')
print('\n')

# println("Timing `cg` algorithms for solving M*v=b")
# state = CGStateVariables(zeros(Float64,length(holstein)),
#                          zeros(Float64,length(holstein)),
#                          zeros(Float64,length(holstein)))
# Mᵀb = zeros(Float64,length(holstein))
# for i in 1:10
#     b = randn(length(holstein))
#     mulMᵀ!(Mᵀb,holstein,b)
#     # b = rand(-1.0:2.0:1.0,length(holstein))
#     v = zeros(Float64,length(holstein))
#     @time r = cg!(v,holstein,Mᵀb,tol=1e-4,statevars=state,log=true)[2]
#     println(r)
# end
# print('\n')

# println("Trying `minres` for solving M*v=b")
# b = randn(length(holstein))
# Mᵀb = zeros(Float64,length(holstein))
# v = zeros(Float64,length(holstein))
# iterable = IterativeSolvers.minres_iterable!(v,holstein,Mᵀb,tol=1e-4)

# for i in 1:10

#     b = randn(length(holstein))
#     mulMᵀ!(Mᵀb,holstein,b)
#     v = zeros(Float64,length(holstein))

#     @time iteration = minres!(v,holstein,Mᵀb,tol=1e-4,log=true)[2]

#     println(iterations)
# end
# print('\n')