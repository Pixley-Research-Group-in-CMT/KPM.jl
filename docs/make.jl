using KPM
using Documenter

makedocs(;
    modules=[KPM],
    authors="Yixing Fu",
    repo="https://github.com/yixingfu/KPM.jl/blob/{commit}{path}#L{line}",
    sitename="KPM.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://yixingfu.github.io/KPM.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/yixingfu/KPM.jl",
)
