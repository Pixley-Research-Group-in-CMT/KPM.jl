# Docs build script for Documenter.jl
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using KPMru
using Documenter

makedocs(
    sitename = "KPMru",
    modules  = [KPMru],
    pages    = ["Home" => "index.md"],
    build    = joinpath("build", "dev"),
)
