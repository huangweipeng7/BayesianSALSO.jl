using BayesianSALSO
using Documenter

DocMeta.setdocmeta!(BayesianSALSO, :DocTestSetup, :(using BayesianSALSO); recursive = true)

makedocs(
    sitename = "BayesianSALSO",
    authors = "Weipeng Huang",
    modules = [BayesianSALSO],
    repo = Documenter.Remotes.GitHub("huangweipeng7", "BayesianSALSO.jl"),
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
    ],
    checkdocs = :exports,
)
