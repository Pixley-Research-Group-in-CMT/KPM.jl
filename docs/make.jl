# Docs build script for Documenter.jl
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using KPM
using Documenter

makedocs(
    sitename = "KPM",
    modules  = [KPM],
    pages    = ["Home" => "index.md"],
    build    = joinpath("build", "dev"),
)
