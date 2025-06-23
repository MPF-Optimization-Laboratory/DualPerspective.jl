module DualPerspective

using LinearAlgebra
using Printf
using UnPack
using DataFrames
import Roots
using NLPModels
using LinearOperators
using SolverCore
using Pkg
using LineSearches: BackTracking

import Optim
using Krylov: cg, krylov_workspace, krylov_solve!, issolved, iteration_count
using LinearOperators: LinearOperator

export DPModel, SSModel, OTModel, LPModel
export SSTrunkLS, SequentialSolve
export solve!, scale!, scale, regularize!, histogram, reset!, update_y0!
export randDPModel
export DensityEstimation
export fixed_point

DEFAULT_PRECISION(T) = (eps(T))^(1/3)

include("logexp.jl")
include("dualperspective-model.jl")
include("ss-model.jl")
include("newton.jl")
include("newtoncg.jl")
include("newtonls.jl")
include("selfscale.jl")
include("sequential-scale.jl")
include("optimal-transport.jl")
include("precon.jl")
include("linear-programming.jl")
include("utils.jl")
include("DensityEstimation.jl")
include("fixed-point.jl")

end 