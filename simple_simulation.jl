# NOTE: For improved performence run this script from the terminal as:
# >julia simple_simulation.jl --optimize=3 --math-mode=fast --check-bounds=no

using Random
using IterativeSolvers
using SparseArrays

using Langevin
using Langevin.Geometries: Geometry
using Langevin.Lattices: Lattice
using Langevin.HolsteinModels: HolsteinModel
using Langevin.HolsteinModels: assign_μ!, assign_ω!, assign_λ!
using Langevin.HolsteinModels: assign_tij!, assign_ωij!
using Langevin.HolsteinModels: setup_checkerboard!, construct_expnΔτV!, mulMᵀ!,construct_M
using Langevin.InitializePhonons: init_phonons_single_site!
using Langevin.FourierAcceleration: FourierAccelerator
using Langevin.FourierAcceleration: forward_fft!, inverse_fft!, accelerate!, accelerate_noise!
using Langevin.LangevinDynamics: update_euler!, update_rk!, update_euler_fa!, update_rk_fa!


"""
Function for estimating electron density.
"""
function estimate_density(holstein::HolsteinModel{T1,T2},
                          g::Vector{T1}, Mᵀg::Vector{T1}, M⁻¹g::Vector{T1},
                          tol::T1=1e-4,ntime::Int=1)::T1 where {T1<:AbstractFloat,T2<:Number}
    
    total = 0.0
    
    for n in 1:ntimes
        
        rand!(g,-1:2:1)

        # getting Mᵀg
        mulMᵀ!( Mᵀg , holstein , g )

        # solve MᵀM⋅v=Mᵀg ==> M⁻¹g
        minres!( M⁻¹g , holstein , Mᵀg , tol=tol)

        # density
        total += sum(1.0 .- g.*M⁻¹g)/length(holstein)
    end
    
    return total/ntimes
end

#############################
## DEFINING HOLSTEIN MODEL ##
#############################

# NOTE: Currently Defining Square Lattice Geometry

# number of dimensions
ndim = 2

# number of orbitals per unit cell
norbits = 1

# lattice vectors
lvecs = [[1.0,0.0],
         [0.0,1.0]]

# basis vectors
bvecs = [[0.0,0.0]]

# defining square lattice geometry
geom = Geometry(ndim, norbits, lvecs, bvecs)

# defining lattice size
L = 4

# constructing finite square lattice
lattice = Lattice(geom,L)

# discretization
Δτ = 0.1

# setting temperature
β = 3.0

# constructing holstein model
holstein = HolsteinModel(geom,lattice,β,Δτ)

# defining nearest neighbor hopping on square lattice
t = 1.0
assign_tij!(holstein, t, 0.0, 1, 1, [1,0,0]) # x direction hopping
assign_tij!(holstein, t, 0.0, 1, 1, [0,1,0]) # y direction hopping

# hamiltonian parameter values
ω = 1.0
λ = 1.0
μ = -(λ/ω)^2 # for half-filling

assign_ω!(holstein, ω, 0.0)
assign_λ!(holstein, λ, 0.0)
assign_μ!(holstein, μ, 0.0)

# organize electron hoppings for checkerboard decomposition
setup_checkerboard!(holstein)

# intialize phonon field
holstein.ϕ .= -λ/ω^2

# construct exponentiated interaction matrix
construct_expnΔτV!(holstein)

####################################
## DEFINING SIMULATION PARAMETERS ##
####################################

# langevin time step
Δt = 2.5e-3

# tolerace of IterativeSolvers
tol = 1e-4

# mass for fourier acceleration: increasing the mass reduces the
# amount of acceleration
mass = 1.0

# number of thermalization steps
ntherm = 10000

# number of langevin steps after thermalization
nsteps = 50000

# frequncy with which to measure electron density
meas_freq = 100

# number of measurements made of electron density
nmeas = div(nsteps,meas_freq)

# number of stochastic estimates of the electron density to make
ntimes = 1

# defining FourierAccelerator type
fa = FourierAccelerator(holstein,mass,Δt)


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

density_history = zeros(Float64, nmeas)

########################
## RUNNING SIMULATION ##
########################

# to store the number of IterativeSolver steps
iters = 0

# first do thermalization sweeps
for i in 1:ntherm

    # Runge-Kutta/Huen's Update with Fourier Acceleration.
    iters = update_rk_fa!(holstein, fa, dϕdt, fft_dϕdt, dSdϕ2, dSdϕ, fft_dSdϕ, g, Mᵀg, M⁻¹g, η, fft_η, Δt, tol)

    if i%1000==0
        println("Therm Step: ",i)
    end
end

# now do the measurement steps
for i in 1:nsteps

    # Runge-Kutta/Huen's Update with Fourier Acceleration.
    iters = update_rk_fa!(holstein, fa, dϕdt, fft_dϕdt, dSdϕ2, dSdϕ, fft_dSdϕ, g, Mᵀg, M⁻¹g, η, fft_η, Δt, tol)

    # Measureing electron density
    if i%meas_freq==0
        density_history[div(i,meas_freq)] = estimate_density(holstein, g, Mᵀg, M⁻¹g, tol, ntimes)
    end

    if i%1000==0
        println("Meas Step: ",i)
    end
end

##############################################################################
## CONSTRUCTING SPARSE MATRIX M CORRESPONDING TO FINAL PHONON CONFIGURATION ##
##############################################################################

M = construct_M(holstein)