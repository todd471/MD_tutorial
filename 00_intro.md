# A miniature molecular-dynamics simulation, end to end
### Trp-cage (TC5b, PDB 1L2Y) in explicit water — build it, run it, watch it, analyze it

This tutorial is a *deliberately small* MD workflow you can run on a laptop or free Colab — no HPC or
datacenter GPU required, just the modest GPU either one already has. (It *will* run CPU-only, but
explicit-solvent MD on a CPU is painfully slow — don't, unless you specifically mean to.) The
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
tier is long enough to show its own uncertainty. That distinction is the point: a too-short run
reports a *falsely* short correlation time and looks converged, while the reference reaches a timescale
where you can at least see that it is **not** (§2.6). Neither tier is truly converged; only the reference is
long enough to **show** it.

## How the tutorial is organized

The tutorial is split into short notebooks — one per stage of the workflow, run in order — backed by shipped
modules (`mdtutorial`, `mdtviz`) the notebooks import. Run them in order:

- **`01_build_system`** — repair → box → water → minimize → equilibrate (NVT, then NPT; saves the prepared system)
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

<!-- glossary:start (generated by glossary.py; edit TERMS there, not here) -->
## Glossary

Every MD term the notebooks use, in plain words, listed under the notebook that first uses it. Each core notebook also opens with its own short list.


### 01 Build the system

- **PDB entry, model** · The Protein Data Bank file holding the atom coordinates of a solved structure. An NMR entry contains several *models*, alternative coordinate sets that all fit the data; a crystal entry contains one.
- **Residue notation (ILE4)** · Three-letter amino-acid code plus the residue's position in the sequence: ILE4 is the isoleucine at position 4. Numbering follows the PDB entry (1 to 20 for Trp-cage).
- **Force field** · The function that assigns a potential energy to any arrangement of the atoms, together with its parameters. The force on each atom is the slope of that function. **CHARMM36** is one widely used protein force field; AMBER ff14SB is another.
- **Water model** · The force field's description of one water molecule. **TIP3P** (three rigid sites) is used here.
- **Periodic box** · The simulation cell. Anything that leaves through one face re-enters through the opposite face, so a small box behaves like bulk solution with no walls.
- **Solvation** · Filling the box around the protein with water molecules, plus ions to neutralize the total charge.
- **Energy minimization** · Adjusting atomic positions to relieve bad contacts and lower the potential energy before dynamics starts. Not a simulation in time: no velocities, no temperature.
- **Equilibration** · Short dynamics after minimization and before production, so the packed water and box relax. Here two stages: at fixed volume (NVT) the velocities are drawn and the water loosens; at fixed pressure (NPT) the box resizes to the force field's liquid density. Production then runs at that relaxed box.
- **System (OpenMM sense)** · The assembled object holding every atom, its force-field parameters, and the box: the thing the simulation advances in time.
- **Ensemble (two meanings)** · In an NMR entry, the set of models. In simulation, the statistical-mechanics meaning: which bulk quantities are held fixed while the atoms move (see NVT in `02_dynamics`). Each notebook says which meaning it intends.

### 02 Dynamics

- **Trajectory, frame** · The saved sequence of coordinate snapshots from a run. One snapshot is a frame (every 1 ps here).
- **Integrator, timestep** · The algorithm that advances every atom by one small time step (2 fs here) using the current forces.
- **Seed** · The number that fixes the random draws (initial velocities, thermostat noise). Same seed on the same hardware repeats a run; a different seed gives an independent repeat.
- **NVE, NVT, NPT** · Ensemble labels. N = number of atoms, V = volume, E = total energy, T = temperature, P = pressure; the letters name what is held fixed. NVE: energy and volume constant (plain Newton). NVT: temperature and volume constant (what these notebooks run). NPT: temperature and pressure constant, so the box resizes.
- **Thermostat, barostat** · The algorithm that holds the *average* temperature (thermostat) or pressure, by resizing the box (barostat). Langevin is the thermostat used here.
- **Observable** · Any number computed from a frame: a distance, an angle, an energy, a surface area. Observables are the measurements of a simulation.
- **Collective variable (CV)** · An observable chosen to summarize one motion of interest in a single number, for example the distance between two groups of atoms. Used to track a motion here, and to push on it in `03_enhanced_sampling`.
- **RMSD, Cα RMSD** · Root-mean-square deviation: after overlaying a frame on a reference structure, the average distance between matching atoms. Using only the backbone Cα atoms gives a fold-level measure. Small means still close to the reference fold.
- **Radius of gyration (Rg)** · The root-mean-square distance of the atoms from the molecule's center. A compact fold gives a small Rg, an unfolded chain a large one.
- **Dihedral, rotamer, well** · A dihedral (torsion) is the rotation angle about a bond: χ1 for a side chain, φ and ψ for the backbone. Side chains prefer a few torsion ranges called rotamers. A *well* is a region of low energy the system settles into; a rotamer is one such well, and the *folded well* is the set of near-native conformations.
- **Helix fraction** · The fraction of residues assigned as α-helix in a frame (by the DSSP algorithm).
- **Salt bridge** · A close contact between an acidic and a basic side chain, here ASP9 and ARG16.
- **Integrated autocorrelation time (τ)** · How long an observable takes to forget its earlier value. Frames closer together than about 2τ are not independent samples, so a run of length L holds only about L / 2τ independent measurements of that observable. It is the number that decides whether a run is long enough.
- **Block averaging** · Splitting a run into blocks and comparing the block means. The spread between blocks stops growing with block size only once blocks are longer than τ, which is how τ is estimated.
- **Scalogram** · A map of how strongly an observable fluctuates at each timescale, across the run (§2.6b).

### 03 Enhanced sampling

- **Enhanced sampling** · Any method that makes a rare event happen in feasible simulation time. The family used here adds a bias along one coordinate and then corrects for it; other families (replica exchange, metadynamics) work differently.
- **Bias** · An extra energy term added on top of the force field to push the system along a chosen coordinate. Here it is a spring attached to a collective variable.
- **Steered MD** · Dragging the chosen coordinate with a spring whose target moves, to force a transition and watch the path it takes.
- **Reaction coordinate** · The coordinate that truly describes a transition. A chosen collective variable is only an approximation to it.
- **Umbrella sampling, window** · A set of separate runs, each restrained by a spring to a different target value of the coordinate (one *window* each), so together they visit the whole path, including the high-energy parts a plain run never reaches.
- **Overlap** · How much neighboring windows visit the same coordinate values. Without overlap the windows cannot be joined into one profile.
- **PMF (potential of mean force)** · The free-energy profile along a coordinate: G(x) = −k_B T ln P(x), where P(x) is how often the coordinate takes the value x at equilibrium. A deep point on the profile is a frequently visited value.
- **WHAM, MBAR** · Methods that combine the umbrella windows into one PMF by removing each window's bias and weighting each by its statistical quality. MBAR is the more general estimator, provided by the `pymbar` library.
- **SASA** · Solvent-accessible surface area. Here that of the Trp6 side chain: large when the cage is open and the side chain is exposed, small when it is buried.
- **Unbiased reference** · The long plain-MD runs from `02_dynamics`, never pushed on. They check the reconstructed profile wherever plain MD samples the coordinate on its own.
- **Bootstrap** · Estimating an error bar by recomputing a result on many resampled copies of the data.
<!-- glossary:end -->
