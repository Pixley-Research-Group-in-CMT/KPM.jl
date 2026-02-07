using KPM
using Documenter

makedocs(;
    modules=[KPM],
    authors="Yixing Fu",
    repo="https://github.com/yixingfu/KPM.jl/blob/{commit}{path}#L{line}",
    sitename="KPM.jl",
    checkdocs=:exports,
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://yixingfu.github.io/KPM.jl",
        edit_link="main",
        repolink="https://github.com/yixingfu/KPM.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/yixingfu/KPM.jl",
    devbranch="main",
)
