# NOTE: For improved performence run this script from the terminal as:
# >julia --optimize=3 --math-mode=fast --check-bounds=no simple_simulation.jl

# using Profile
# Profile.clear()
# using ProfileView

using Random
using IterativeSolvers
using SparseArrays
using Statistics

using Langevin
using Langevin.Geometries: Geometry
using Langevin.Lattices: Lattice
using Langevin.HolsteinModels: HolsteinModel
using Langevin.HolsteinModels: assign_μ!, assign_ω!, assign_λ!
using Langevin.HolsteinModels: assign_tij!, assign_ωij!
using Langevin.HolsteinModels: setup_checkerboard!
using Langevin.InitializePhonons: init_phonons_single_site!
using Langevin.FourierAcceleration: FourierAccelerator
using Langevin.LangevinSimulationParameters: SimulationParameters
using Langevin.RunSimulation: run_simulation!

#############################
## DEFINING HOLSTEIN MODEL ##
#############################

# NOTE: Currently Defining Cubic Lattice Geometry

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
β = 2.0

# constructing holstein model
holstein = HolsteinModel(geom,lattice,β,Δτ)

# defining nearest neighbor hopping on square lattice
t = 1.0
assign_tij!(holstein, t, 0.0, 1, 1, [1,0,0]) # x direction hopping
assign_tij!(holstein, t, 0.0, 1, 1, [0,1,0]) # y direction hopping

# hamiltonian parameter values
ω = 1.0
g = 0.0
λ = sqrt(2.0*ω)*g
μ = -0.75 # for half-filling

assign_ω!(holstein, ω, 0.0)
assign_λ!(holstein, λ, 0.0)
assign_μ!(holstein, μ, 0.0)

# organize electron hoppings for checkerboard decomposition
setup_checkerboard!(holstein)

# intialize phonon field
init_phonons_single_site!(holstein)

###################################
## DEFINING FOURIER ACCELERATION ##
###################################

# langevin time step
Δt = 1e-3

# mass for fourier acceleration: increasing the mass reduces the
# amount of acceleration
mass = 0.5

# defining FourierAccelerator type
fa = FourierAccelerator(holstein,mass,Δt)

####################################
## DEFINING SIMULATION PARAMETERS ##
####################################

# tolerace of IterativeSolvers
tol = 1e-4

# number of thermalization steps
burnin = 250000

# total number of steps
nsteps = 1000000

# measurement frequency
meas_freq = 100

# number of bins
num_bins = 1

# euler or runge-kutta updates
euler = false

# filepath to where to write data
filepath = "."

# name of folder for data to get dumped into
foldername = "test"

sim_params = SimulationParameters(Δt,euler,tol,burnin,nsteps,meas_freq,num_bins,filepath,foldername)

########################
## RUNNING SIMULATION ##
########################

simulation_time, measurement_time, write_time, iters = run_simulation!(holstein, sim_params, fa)

println("Simulation Time (min) = ",  simulation_time)
println("Measurement Time (min) = ", measurement_time)
println("Write Time (min) = ", write_time)
println("Average Iterative Solver Iterations = ", iters)