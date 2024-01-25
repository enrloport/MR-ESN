# Load external libraries

using Augmentor
using BenchmarkTools
using CSV
using CUDA
using DataFrames
using Dates
using DelimitedFiles
using Distributions
# using GraphPlot
using Graphs
using Images
using LinearAlgebra
# using MatrixDepot
using Metaheuristics
using MLDatasets
using NetCDF
using Plots
using Random
using SimpleWeightedGraphs
using SparseArrays
using StableRNGs
using StatsBase
using Suppressor

# Types definition
Mtx = Union{Matrix,SparseMatrixCSC, Array, CuArray}
Data= Union{DataFrame, Mtx, Vector, Array, CuArray}

# Includes
include.( filter(contains(r".jl$"), readdir("./files/"; join=true)))
include.( filter(contains(r".jl$"), readdir("./files/batch_functions"; join=true)))
include.( filter(contains(r".jl$"), readdir("./files/sigmoid_functions"; join=true)))
include.( filter(contains(r".jl$"), readdir("./files/reservoir_generators"; join=true)))
include.( filter(contains(r".jl$"), readdir("./files/test_functions"; join=true)))
include.( filter(contains(r".jl$"), readdir("./files/train_functions"; join=true)))
include.( filter(contains(r".jl$"), readdir("./files/update_functions"; join=true)))
include.( filter(contains(r".jl$"), readdir("./files/wandb_functions"; join=true)))
