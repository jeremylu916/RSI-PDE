"""Write docs/pde-rsi-benchmarks.tex: every PDE benchmark of the project in one LaTeX document.

Contents: the protocols and scoring; pde_benchmarks.tex Problems 1-16 (taken from pde_benchmarks.tex itself) with the
numbers of every public case; the PDE Data Inventory suite; the manufactured code-protocol suites (from
scripts/write_problem_latex.py). Numbers come from the case generators, so the document matches the code. Sealed
test cases are counted, but their numbers are omitted. Usage (after sourcing scripts/env.sh):
    python scripts/write_benchmarks_latex.py
    module load texlive && cd docs && pdflatex pde-rsi-benchmarks.tex && pdflatex pde-rsi-benchmarks.tex
"""

import importlib.util
import json
import math

from pde_rsi import inventory as inv
from pde_rsi import pdebench as pb
from pde_rsi import pdebench_ext as px
from pde_rsi.storage import PROJECT_ROOT

PUBLIC = {"evolution": "training", "development": "validation"}


def n(value):
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, int):
        return str(value)
    if abs(value-math.pi) < 1e-12:
        return r"$\pi$"
    if abs(value-2*math.pi) < 1e-12:
        return r"$2\pi$"
    if abs(value-32*math.pi) < 1e-12:
        return r"$32\pi$"
    return f"{value:.4g}"


NAMES = (("Navier-Stokes", "Navier--Stokes"), ("Allen-Cahn", "Allen--Cahn"), ("convection-diffusion", "convection--diffusion"),
         ("Convection-diffusion", "Convection--diffusion"), ("advection-diffusion-reaction", "advection--diffusion--reaction"),
         ("Wang-Wang-Perdikaris", "Wang--Wang--Perdikaris"))


def dashes(text):
    """En dashes in two-name terms (Navier--Stokes); hyphens elsewhere (1-D, multi-term) stay."""
    for old, new in NAMES:
        text = text.replace(old, new)
    return text


def tabular(header, rows, align=None, size=r"\small"):
    align = align or "l"*len(header)
    out = [r"\begin{center}", size, rf"\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(header)+r" \\",
           r"\midrule"]
    out += [" & ".join(r)+r" \\" for r in rows]
    out += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    return out


def tex_body():
    """The problem definitions of pde_benchmarks.tex (preliminaries through Problem 16) and its bibliography."""
    source = (PROJECT_ROOT/"pde_benchmarks.tex").read_text()
    start = source.index(r"\section{Preliminaries}")
    bib = source.index(r"\begin{thebibliography}")
    end = source.index(r"\end{thebibliography}")+len(r"\end{thebibliography}")
    return source[start:bib], source[bib:end]


def tex_split_tables():
    out = [r"\subsection*{Numbers of the public cases}",
           "Training (evolution) and validation (development) cases. The literature cases use the values below. "
           "The manufactured cases (1a, 2a, 3a, 4a, 5, 7, 8) draw $L,T,A,D,\\beta$ (and $v$, $\\alpha$, $w_j$) per "
           "case and split from the ranges in \\texttt{pdebench.manufactured}. The sealed test splits (test: 17 "
           "cases, test2 and test3: 20 cases each) use further values that are withheld here."]
    keys = {"1b": ["D"], "1c": ["D", "sharpness", "x0"], "2b": ["b", "D"], "2c": ["T", "D", "gamma"],
            "3b": ["v", "D"], "3c": ["v", "D"], "3d": ["b", "v", "D", "x0", "sigma0"], "4b": ["D"],
            "4c": ["b", "ul", "ur", "D"], "6": ["alpha", "p"], "9": ["alpha", "p"], "10": ["c", "p"], "11": ["c", "p"]}
    names = {"sharpness": r"$\alpha$", "gamma": r"$\gamma$", "sigma0": r"$\sigma_0$", "x0": r"$x_0$", "b": "$L$",
             "ul": r"$u_\ell$", "ur": "$u_r$", "alpha": r"$\alpha$", "D": "$D$", "T": "$T$", "v": "$v$", "c": "$c$",
             "p": "$p$"}
    rows = []
    for case, fields in keys.items():
        cells = []
        for split in PUBLIC:
            entry = next((v for p, v in pb.FIXED[split] if p == case), None)
            cells.append("---" if entry is None else
                         r"$D=0.01/\pi$" if case == "4b" else
                         ", ".join(f"{names.get(k, k)} $=$ {n(entry[k])}" for k in fields))
        rows.append([case, dashes(pb.TITLES[case].split(" (")[0]), *cells])
    out += tabular(["Case", "Title", "Training", "Validation"], rows, "lp{3.6cm}p{4.6cm}p{4.6cm}", r"\footnotesize")
    out += [r"\subsection*{Numbers of the public cases, Problems 12--16}"]
    fields = {"12a": ["b", "A", "beta", "T"], "12b": ["b", "T"], "13a": ["k", "nu", "T"], "13b": ["nu"],
              "14a": ["b", "A", "beta", "M", "eps", "T"], "14b": ["eps", "M", "T"], "15a": ["b", "d", "m", "n", "c", "T"],
              "15b": ["b", "d", "A", "beta", "c", "T"], "16a": ["A", "B", "p", "n", "m", "q", "k"],
              "16b": ["A", "B", "p", "n", "m", "q", "k"]}
    labels = {"b": "$L$", "d": "$b$", "beta": r"$\beta$", "nu": r"$\nu$", "eps": r"$\varepsilon$", "A": "$A$",
              "B": "$B$", "M": "$M$", "T": "$T$", "k": "$k$", "m": "$m$", "n": "$n$", "p": "$p$", "q": "$q$", "c": "$c$"}
    rows = []
    for case, names_ in fields.items():
        cells = []
        for split in PUBLIC:
            spec = px.make_spec(case, split)
            shown = dict(spec)
            if case.startswith("15"):
                shown["b"], shown["d"] = spec["b"], spec["d"]
            parts = []
            for key in names_:
                label = labels[key]
                if case.startswith("15") and key == "b":
                    label = "$a$"
                parts.append(f"{label} $=$ {n(shown[key])}")
            cells.append(", ".join(parts))
        rows.append([case, px.CASES[case][3].replace("Kuramoto-Sivashinsky", "Kuramoto--Sivashinsky")
                     .replace("Navier-Stokes", "Navier--Stokes").replace("Taylor-Green", "Taylor--Green")
                     .replace("Cahn-Hilliard", "Cahn--Hilliard").replace("Kassam-Trefethen", "Kassam--Trefethen")
                     .replace("TE_z", r"TE$_z$"), *cells])
    out += tabular(["Case", "Title", "Training", "Validation"], rows, "lp{3.4cm}p{4.7cm}p{4.7cm}", r"\footnotesize")
    out.append("The sealed test split of Problems 12--16 has one case per row above (10 cases); its numbers are withheld.")
    return out


def protocol_section():
    out = [r"\section{Protocols, data and scoring}", r"""
\paragraph{Student protocol.} For the \texttt{pde\_benchmarks.tex} suites and the inventory suite, an agent receives,
for each case, 256 noisy samples of the solution (Gaussian noise with standard deviation 1\,\% of the samples' RMS,
per component), the domain, and black-box forcing, initial, boundary and (where present) coefficient data. It must
identify the equation class and its coefficients from a library of candidate equations and return the solution over
the whole domain. Case identifiers are hash-ordered and reveal neither the class nor the problem.

\paragraph{Code protocol.} For the manufactured suites (Section~\ref{sec:manufactured}) the equation is given and the
agent writes one solver per family.

\paragraph{Scoring.} The returned solution is linearly interpolated onto a uniform evaluation grid and compared with
the exact or a high-accuracy reference solution using the trapezoidal rule. A case passes when the space--time and
the final-time relative $L^2$ errors are both at most the tolerance, the initial and boundary conditions hold to
$10^{-10}$, and the solve fits the time limit (wall and CPU time, including start-up) and the memory limit. The
reported class and coefficients are scored separately and never gate a case.

\paragraph{Splits.} Training (evolution) cases come with labelled feedback; validation (development) cases drive the
acceptance gate; sealed test cases are never shown to any agent and are scored once, after the solver is frozen."""]
    out += tabular(["Suite", "Cases per split", "Tolerance", "Limits", "Evaluation grid"], [
        [r"\texttt{pde\_benchmarks.tex} 1--11", "16 / 16 / 17, 20, 20", r"$2\times10^{-3}$", "90 s, 16 GB",
         r"1-D $512\times500$; 2-D $128^2\times100$"],
        [r"\texttt{pde\_benchmarks.tex} 12--16", "10 / 10 / 10", r"$2\times10^{-3}$", "90 s (1-D), 300 s (2-D), 16 GB",
         r"1-D $512\times501$; 2-D $129^2\times21$"],
        ["PDE Data Inventory", "12 / 12 / 12, 12", r"$2\times10^{-3}$ (4-D: $5\times10^{-2}$)", "90 s / 300 s, 16 GB",
         r"1-D $512\times500$; 2-D $256^2$ or $128^2\times11$"],
        ["Manufactured (code protocol)", "24 / 24 / 24 (1-D)", r"$2\times10^{-3}$", "15 s, 1 GB",
         r"1-D $512\times500$; wave $128^2\times100$"],
    ], "p{3.3cm}p{2.3cm}p{2.1cm}p{2.4cm}p{3.6cm}")
    out.append(r"Vector solutions (Problems 13, 15, 16) are scored with all components together; the pressure is "
               r"compared after removing its spatial mean.")
    return out


def inventory_section():
    lib = inv.LIBRARY_ADDITIONS
    tex_eq = {
        "darcy_2d": r"-\nabla\cdot\bigl(a(x,y)\nabla u\bigr)=f(x,y)",
        "poisson_2d": r"u_{xx}+u_{yy}=f(x,y)",
        "helmholtz_2d": r"u_{xx}+u_{yy}+k^2u=f(x,y)",
        "vorticity_ns": r"\omega_t+\mathbf{v}\cdot\nabla\omega=\nu\Delta\omega-\mu\omega+q,\quad \mathbf{v}=(\psi_y,-\psi_x),\ -\Delta\psi=\omega",
        "allen_cahn_2d": r"u_t=\nabla\cdot(\kappa\nabla u)+\lambda(u-u^3)+f",
        "burgers_2d": r"u_t+c_xuu_x+c_yuu_y=\nabla\cdot(\kappa\nabla u)+f",
        "adr_2d": r"u_t+c_xuu_x+c_yuu_y=\nabla\cdot(\kappa\nabla u)+\lambda(u-u^3)+f",
        "elliptic_4d": r"-\nabla\cdot(a\nabla u)=f \text{ in } B^3\times(-1,1),\quad u=0 \text{ on the boundary}",
    }
    out = [r"\section{PDE Data Inventory suite}", r"""
Twelve families from \emph{PDE Data Inventory} (X.~Kong, 19 October 2026), with inputs from the DiffusionPDE test
set (Huang et al.\ 2024) where the inventory leaves gaps. The library adds the following classes to the ten classes
of Problems 1--11:""", r"\begin{align*}"]
    out += [rf"&\text{{{name.replace('_', r'\_')}}}:\quad {tex_eq[name]}\\" for name in lib]
    out[-1] = out[-1][:-2]
    out += [r"\end{align*}"]
    rows = []
    for family, (klass, _, layout, seconds) in inv.FAMILIES.items():
        spec = {split: inv.make_spec(family, split) for split in PUBLIC}

        def values(s):
            k = s["kind"]
            if k == "burgers_gauss":
                return f"$D={n(s['D'])}$"
            if k == "convdiff":
                return f"$x_0={n(s['x0'])}$, $\\sigma_0={n(s['sigma0'])}$"
            if k in ("burgers_random", "darcy", "poisson", "helmholtz", "ns"):
                return f"instance {inv.INSTANCE[s['split']] if 'split' in s else ''}"
            if k == "allen_cahn":
                return f"$\\lambda={s['lam']:.3f}$"
            if k == "burgers2d":
                return f"$c_x={s['cx']:.3f}$, $c_y={s['cy']:.3f}$"
            if k == "adr":
                return f"$c_x={s['cx']:.3f}$, $c_y={s['cy']:.3f}$, $\\lambda={s['lam']:.3f}$"
            return "own draw"
        cells = []
        for split, s in spec.items():
            text = values(s)
            cells.append(f"instance {inv.INSTANCE[split]}" if text.startswith("instance") else text)
        tol = inv.TOLERANCES.get(family, inv.TOLERANCE)
        rows.append([dashes(inv.TITLES[family].split(" (")[0]), klass.replace("_", r"\_"),
                     layout.replace("_", r"\_"), f"{seconds:g} s", n(tol), *cells])
    out += tabular(["Family", "Class", "Layout", "Limit", "Tol.", "Training", "Validation"], rows,
                   "p{2.4cm}p{2.9cm}p{2.45cm}p{0.8cm}p{0.9cm}p{2.4cm}p{2.4cm}", r"\footnotesize\setlength{\tabcolsep}{3pt}")
    out.append(r"""\noindent The sealed splits \texttt{inventory\_test} and \texttt{inventory\_test2} (DiffusionPDE
instances 2 and 3, and further draws) are withheld. References are computed at high resolution from the same
continuous inputs: pseudospectral ETDRK4 (1-D Burgers), finite volumes on an $8\times$ refined grid (Darcy), sine
series (Poisson, Helmholtz), integrating-factor RK4 (Navier--Stokes, Kolmogorov), pseudospectral RK4 (Allen--Cahn,
2-D Burgers, ADR), $63^4$ finite volumes (4-D elliptic) and the closed form (linear convection--diffusion). Each is
checked against a coarser solve; the tolerance is at least five times the reference error.""")
    return out


def main():
    spec = importlib.util.spec_from_file_location("write_problem_latex", PROJECT_ROOT/"scripts/write_problem_latex.py")
    manufactured = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(manufactured)
    body, bibliography = tex_body()
    out = [r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,booktabs}
\usepackage[hidelinks]{hyperref}
\newcommand{\Caputo}[1]{{}^{C}_{0}D_t^{#1}}
\title{PDE-RSI Benchmark Problems}
\author{Generated by \texttt{scripts/write\_benchmarks\_latex.py}}
\date{\today}
\begin{document}
\maketitle
\tableofcontents
"""]
    out += protocol_section()
    out += [r"\section{\texttt{pde\_benchmarks.tex}: Problems 1--16}",
            r"Problems 1--11 form the \texttt{pdebench} suite and Problems 12--16 the \texttt{pdebench-ext} suite. "
            r"The definitions below are those of \texttt{pde\_benchmarks.tex}.",
            r"\setcounter{subsection}{0}", r"\renewcommand{\thesubsection}{\arabic{subsection}}",
            body.replace(r"\section{", r"\subsection{").replace(r"\subsection*{Problem", r"\subsubsection*{Problem")
                .replace(r"\subsection{Spatial", r"\subsubsection*{Spatial").replace(r"\subsection{Temporal", r"\subsubsection*{Temporal")
                .replace(r"\subsection{Caputo", r"\subsubsection*{Caputo").replace(r"\subsection{Scalar", r"\subsubsection*{Scalar")
                .replace(r"\subsection{Constant", r"\subsubsection*{Constant").replace(r"\subsection{Variable", r"\subsubsection*{Variable")]
    out += tex_split_tables()
    out += inventory_section()
    out += [r"\section{Manufactured suites (code protocol)}\label{sec:manufactured}",
            r"\renewcommand{\thesubsection}{\thesection.\arabic{subsection}}",
            r"Generated from \texttt{src/pde\_rsi/tasks.py} by \texttt{scripts/write\_problem\_latex.py}. Every "
            r"problem has a manufactured exact solution; agents see the forcing and initial data only as black boxes."]
    out += [part.replace(r"\section{", r"\subsection{").replace(r"\subsection{Poisson", r"\subsubsection{Poisson")
            for part in manufactured.sections()]
    out += [bibliography, r"\end{document}"]
    text = "\n".join(out)+"\n"
    # Second-level headings of the manufactured part sit under one section; keep their family headings one level down.
    path = PROJECT_ROOT/"docs/pde-rsi-benchmarks.tex"
    path.write_text(text)
    print(path, len(text))


if __name__ == "__main__":
    main()
