using Revise
Revise.revise()

using Documenter
using Documenter.Remotes
using DualPerspective

# Define macros here. Provides `KATEX_MACROS`.
include("katex_macros.jl") 

makedocs(
    sitename = "DualPerspective.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://MPF-Optimization-Laboratory.github.io/DualPerspective.jl/stable",
        repolink = "https://github.com/MPF-Optimization-Laboratory/DualPerspective.jl",
        mathengine = Documenter.KaTeX(KATEX_MACROS)
    ),
    modules = [DualPerspective],
    authors = "Michael P. Friedlander and contributors",
    pages = [
        "Home" => "index.md",
        "User Guide" => "guide.md",
        "Examples" => "examples.md",
        "API Reference" => "api.md",
    ],
    repo = Remotes.GitHub("MPF-Optimization-Laboratory", "DualPerspective.jl"),
    warnonly = [:docs_block, :missing_docs],  # Don't fail on missing docstrings
)

deploydocs(
    repo = "github.com/MPF-Optimization-Laboratory/DualPerspective.jl.git",
    devbranch = "main",
    push_preview = true,
) 