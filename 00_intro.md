# A miniature molecular-dynamics simulation, end to end
### Trp-cage (TC5b, PDB 1L2Y) in explicit water — build it, run it, watch it, analyze it

This tutorial is a *deliberately small* MD workflow you can run on a laptop or free Colab — no HPC or
datacenter GPU required, just the modest GPU in a laptop or Colab's free tier. (It *will* run CPU-only, but
explicit-solvent MD on a CPU is painfully slow, so don't attempt that unless you specifically mean to.) The
point is not a publication-grade result — it is to make the moving parts of an MD simulation **visible**:
how a structure is prepared, what "solvation" and "minimization" actually do, and what the force field is
doing to the atoms once dynamics start.

This notebook set is the hands-on companion to [*tutorial article — citation to come*]. It assumes only
that you can run a Jupyter notebook and know what a protein is — no prior MD experience — and by the end
you will have built a solvated system from a PDB entry, run reproducible dynamics, watched the force field
move the atoms one step at a time, and judged whether the numbers you get out are actually trustworthy.

**The system.** We use the Trp-cage miniprotein *TC5b* (PDB **1L2Y**; Neidigh, Fesinmeyer & Andersen 2002),
a 20-residue engineered construct distilled from the C-terminal fold of **exendin-4 / exenatide** (the
Gila-monster GLP-1 receptor agonist). It is a small, cooperatively folding miniprotein that has been widely
used as a test system in the protein-folding simulation literature (Simmerling 2002; Zhou 2003;
Lindorff-Larsen *et al.* *Science* 2011) — well-characterized and well suited to a tutorial, with a body of
published behavior to compare a short run against.

**Two tiers.** In these notebooks the *live tier* samples only picoseconds to nanoseconds — thermal motion, terminal fraying,
dihedral flips — **not** de-novo folding, which takes µs (Lindorff-Larsen *et al.* 2011). Where a figure needs longer sampling it loads a
precomputed multi-ns **reference** trajectory instead. The live tier teaches the *machinery*; the reference
tier is long enough to be honest about its own uncertainty. That distinction is the point: a too-short run
reports a *falsely* short correlation time and looks converged, while the reference reaches a timescale
where you can at least see that it is **not** yet (§2.6). Neither tier is truly converged — the reference is
just honest about it.

## How the tutorial is organized

The tutorial is split into short notebooks — one per stage of the workflow, run in order — backed by shipped
modules (`mdtutorial`, `mdtviz`) the notebooks import. Run them in order:

- **`01_build_system`** — repair → box → water → minimize (saves the prepared system)
- **`02_dynamics`** — three trajectories, the synchronized molecule/observable player, and a
  timescale/scalogram view of where each motion's fluctuations live (§2.6)
- **`03_enhanced_sampling`** — reaching past the timescale wall: when a transition is too rare to wait for,
  *bias* a chosen coordinate to drive it, then remove the bias to recover the true thermodynamics. We pry the
  cage open by steering its poly-proline lid off Trp6 (steered MD) — Trp6 stays largely put, the lid peels
  back — tile the path with umbrella windows, and stitch them with MBAR into a free-energy profile

Companions:

- **`minimal`** — the conventional MD pipeline end to end (prep → run → a first RMSD look), flat in one self-contained file; no enhanced sampling or deeper analysis
- **`sandbox`** — turn the knobs: protein, temperature, force field, water model, ensemble, thermostat, timestep
- **`determinism`** — measure what is, and isn't, bit-for-bit reproducible on *your* hardware, and what it takes to get there

## Where the code and the math live

To keep each cell readable, the notebooks **import** the actual implementation from shipped Python
modules — open any of them to read the real code and equations (they are plain text files in the repo,
meant to be read, not hidden):

- **`mdtutorial.py`** — the fixed pipeline and all the analysis math: structure repair, solvation, building
  the `System`, running a trajectory, and the collective-variable and convergence/uncertainty calculations.
- **`mdtviz.py`** — the visualization helpers (ensemble and solvated-box viewers, cartoons, the synchronized
  molecule/observable player, and the 03 before/after molecular views).
- **`mdtsandbox.py`** — the "turn the knobs" machinery behind `sandbox`.
- **`md_scalogram.py`** — the timescale/wavelet analysis behind 02's §2.6 scalograms.
- **`pull_screen.py`** — the 03 enhanced-sampling engine: steered pulls, umbrella windows, and the MBAR
  free-energy reconstruction (with `pymbar` for the overlap diagnostic; the notebook's PMF error bars are bootstrapped, not analytical — see §3.5c).

*(The notebooks import these as `mdt`, `mdtviz`, `mdtsandbox`, `md_scalogram`, and `pull_screen` (aliased
`steer`) — so a cell that calls `mdt.repair` is running `repair` from `mdtutorial.py`.)*

Many of these are **thin, purpose-built wrappers around OpenMM, PDBFixer, and MDTraj** — gathered so a cell
reads as one clear step instead of a dozen library calls. The function *name* tells you what the step does;
the module *body* shows you exactly which library calls (and which equations) it makes, so you can see where
our code ends and the underlying package begins. Where a section leans on one of these, its prose points you
to the specific function to open. And if you would rather read the whole *conventional* pipeline **inline in
one flat file** with nothing imported, that is what **`minimal`** is for.

> **Built on** open-source tools — **OpenMM** (Eastman *et al.* 2017) for simulation, **PDBFixer** for
> structure repair, **MDTraj** (McGibbon *et al.* 2015) for trajectory analysis, **py3Dmol** (Rego & Koes
> 2015) and open-source **PyMOL** for molecular views, **pymbar** (Shirts & Chodera 2008) for the MBAR
> overlap check, and **NumPy** (Harris *et al.* 2020) / **SciPy** (Virtanen *et al.* 2020) / **Matplotlib**
> (Hunter 2007) for numerics and figures.
>
> Setup and environment instructions are in the `README`; full references (system, methods,
> reproducibility, and software) are in `references.bib`. The structure itself is Neidigh, Fesinmeyer &
> Andersen 2002 (PDB 1L2Y).
