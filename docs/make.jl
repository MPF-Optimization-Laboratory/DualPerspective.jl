using Pkg
Pkg.activate(@__DIR__) # Activate the docs environment
Pkg.develop(PackageSpec(path=joinpath(@__DIR__, ".."))) # Develop the main package

using Revise
using Documenter
using Documenter.Remotes
using DocumenterCitations
using Distributions
using DualPerspective

# Define macros here.
mathjax3_macros = Dict(
    :tex => Dict(
        :inlineMath => [["\$","\$"], ["\\(","\\)"]],
        :tagSide => "left",
        :tags       => "ams",
        :packages   => ["base", "ams", "autoload", "configmacros"],
        :processEscapes => true,
        :macros     => Dict(
            :Diag => [raw"\mathop{\mathrm{\bf Diag}}", 0],
            :ip    => [raw"\langle #1 \rangle", 1],
            :R => [raw"\mathbb{R}", 0],
            :eR => [raw"\overline\R", 0],
            :rent => [raw"\mathop{\mathrm{\bf rent}}", 0],
            :KL => [raw"\mathop{\mathrm{\bf KL}}", 0],
            :interior => [raw"\mathop{\mathrm{\bf int}}", 0],
            :relint => [raw"\mathop{\mathrm{\bf relint}}", 0],
            :logexp => [raw"\mathop{\mathrm{\bf logexp}}", 0],
            :dom => [raw"\mathop{\mathrm{\bf dom}}", 0],
            :xbar => [raw"\bar{x}", 0],
        ),
    ),
)

bib = CitationBibliography(
    joinpath(@__DIR__, "src", "refs.bib"),
    style = :authoryear,
)

makedocs(
    sitename = "DualPerspective.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://MPF-Optimization-Laboratory.github.io/DualPerspective.jl/stable",
        repolink = "https://github.com/MPF-Optimization-Laboratory/DualPerspective.jl",
        # mathengine = Documenter.KaTeX(KATEX_MACROS)
        mathengine = Documenter.MathJax3(mathjax3_macros)
    ),
    modules = [DualPerspective],
    authors = "Michael P. Friedlander and contributors",
    pages = [
        "Home" => "index.md",
        # "User Guide" => "guide.md",
        "Theory" => "theory.md",
        "Density Estimation" => "density.md",
        "Examples" => "examples.md",
        "API Reference" => "api.md",
        "Development" => "dev.md",
        "References" => "refs.md",
    ],
    repo = Remotes.GitHub("MPF-Optimization-Laboratory", "DualPerspective.jl"),
    warnonly = [:docs_block, :missing_docs],  # Don't fail on missing docstrings
    plugins = [bib],
)

deploydocs(
    repo = "github.com/MPF-Optimization-Laboratory/DualPerspective.jl.git",
    devbranch = "main",
    push_preview = true,
) 