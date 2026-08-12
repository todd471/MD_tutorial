<!-- NOTE: the Colab/raw links below point at the PRACTICE repo `todd471/MD_tutorial`.
     Swap the slug to the final repo when it lands (tracked as task #30). The on-ramp cells read the
     same slug from $MDTUTORIAL_BASE, so update both together. -->

# A miniature molecular-dynamics simulation, end to end

**Trp-cage (TC5b, PDB 1L2Y) in explicit water — build it, run it, watch it, analyze it.**

A deliberately small, license-free MD workflow you can run on a laptop or free Colab — no HPC or datacenter
GPU required. The point isn't a publication-grade result; it's to make the moving parts of an MD study
**visible and reproducible**: how a structure is prepared, what *solvation* and *minimization* actually do,
what the force field does to the atoms once dynamics start, and how to tell whether the numbers you get out
are trustworthy. It assumes only that you can run a Jupyter notebook and know what a protein is.

📖 **New here? Read [`00_intro.md`](00_intro.md) first** — it's the narrative walkthrough (the science, the
two-tier live/reference design, what each figure teaches). This README is the *operational* front door: how
to get it running.

---

## Run it — two paths

### Path 1 · Colab (zero install)

Each notebook opens and runs on its own free Colab VM. Cell 0 provisions the dependencies and fetches the
shipped modules; large reference trajectories are pulled **only when a cell needs them**. Each notebook is
**self-sufficient** — there's no shared filesystem between Colab VMs, and none is needed. **Start with 01.**

| Notebook | Open in Colab |
|---|---|
| `01_build_system` — prepare, solvate, minimize | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/todd471/MD_tutorial/blob/main/01_build_system.ipynb) |
| `02_dynamics` — three trajectories, player, timescales | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/todd471/MD_tutorial/blob/main/02_dynamics.ipynb) |
| `03_enhanced_sampling` — steered MD + umbrella + MBAR | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/todd471/MD_tutorial/blob/main/03_enhanced_sampling.ipynb) |
| `minimal` — the conventional MD pipeline, flat, one file | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/todd471/MD_tutorial/blob/main/minimal.ipynb) |
| `sandbox` — turn the knobs | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/todd471/MD_tutorial/blob/main/sandbox.ipynb) |
| `determinism` — what's reproducible on *your* hardware | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/todd471/MD_tutorial/blob/main/determinism.ipynb) |

> On Colab, set **Runtime → Change runtime type → GPU** (the free T4 is plenty). CPU works but explicit-solvent
> MD on a CPU is painfully slow. Google Drive is **not** required — it's an opt-in convenience only (see below).

### Path 2 · Local (conda — the reliable path)

`openmm` + `pdbfixer` resolve cleanly on **conda-forge**; pip's resolver conflicts on that pair. So conda is
the recommended local/HPC route:

```bash
conda env create -f environment.yml          # or: mamba env create -f environment.yml
conda activate mdtutorial
python -m ipykernel install --user --name mdtutorial --display-name "MD tutorial (mdtutorial)"
jupyter lab                                   # then pick the "MD tutorial (mdtutorial)" kernel
```

With that kernel selected, each notebook's **Cell 0 finds everything and installs nothing** (it only
provisions on Colab). `requirements.txt` is a pip fallback for environments without conda — note PyMOL is
omitted there (pip has no working headless PyMOL; the cartoon panels skip gracefully without it).

---

## The notebooks (run order)

Core tutorial — run in order:

1. **`01_build_system`** — repair → box → water → minimize, and **save** the prepared system.
2. **`02_dynamics`** — three trajectories, the synchronized molecule/observable player, and a
   timescale/scalogram view of where each motion's fluctuations live.
3. **`03_enhanced_sampling`** — reaching past the timescale wall: *bias* a coordinate to drive a rare
   transition (steered MD), tile the path with umbrella windows, and stitch them with MBAR into a
   free-energy profile.

Companions (any order):

- **`minimal`** — the conventional MD pipeline (prep → run → a first RMSD look) flat in one self-contained
  file, no imports — the "read it all in one place" reference; no enhanced sampling or deeper analysis.
- **`sandbox`** — turn the knobs: protein, temperature, force field, water model, ensemble, thermostat, timestep.
- **`determinism`** — measure what is, and isn't, bit-for-bit reproducible on *your* hardware.

**On run order:** 01 saves the prepared system (`system.xml` + `stage4_minimized.pdb`) and 02/03 reuse
it — a **convenience, not a requirement**. If the prep is absent (02 run standalone, or on Colab where each
notebook is a separate VM), the notebook does a loud, explicit **re-prep** via `mdt.load_or_prepare(...)`. So
any notebook runs on its own; running in order just saves you a redundant solvation.

---

## What's in the repo

| | |
|---|---|
| **Notebooks** | `01_build_system.ipynb`, `02_dynamics.ipynb`, `03_enhanced_sampling.ipynb`, `minimal.ipynb`, `sandbox.ipynb`, `determinism.ipynb` |
| **Modules** (imported at runtime — plain-text, meant to be read) | `mdtutorial.py` (pipeline + analysis math), `mdtviz.py` (viewers/cartoons/player), `mdtsandbox.py` (the knobs), `pull_screen.py` (steered MD + umbrella + MBAR), `md_scalogram.py` (timescale analysis) |
| **Builders** | `build_figures.py`, `build_enhanced.py`, `build_sandbox.py`, `build_minimal.py`, `build_determinism.py` — these *generate* the notebooks (see Maintainers) |
| **Reference data** | `reference_10ns/` (shipped) and `reference_10ns_fullres/` (see below) |
| **Environments** | `environment.yml` (conda, preferred), `requirements.txt` (pip fallback) |

---

## Reference data

The figures that need longer sampling than the live tier can afford load a **precomputed reference**
trajectory set (Trp-cage, 6 seeds × 10 ns, CHARMM36):

- **`reference_10ns/`** — the **shipped** set, strided to 2 ps/frame. This is what the notebook figures
  actually reproduce, so *what you see is what a reader regenerates*. Includes `MANIFEST.sha256` and each
  run's `run_meta.json`.
- **`reference_10ns_fullres/`** — the 1 ps/frame full-resolution archive (same runs, finer stride).

Both are **paper-frozen** — a repo update should never change them. `mdt.load_reference()` reads
`reference_stride_ps` from `run_meta.json`, so the analysis adapts to whichever stride it's handed.

---

## Reproducibility — the honest version

- **Dynamics** reproduce bit-for-bit from the seed **on a given GPU model** (CUDA + deterministic forces),
  only with a **fresh Context per run** (re-using and re-seeding a Context diverged in our tests — the
  mechanism is untraced, so we state the recipe, not the cause).
- **Minimization** is bit-reproducible only with deterministic forces (CUDA); we ship the **minimized start**
  so it stops mattering.
- **Preparation** is deterministic via seeded RNGs + hydrogen placement on the Reference platform.
- Headline: **archive the prepared system, not just the seed** — solvation is stochastic down to the water
  *count*. Don't expect cross-machine bit-identity. `determinism` demonstrates all three directly.

---

## Hardware

A modest laptop GPU (including Apple Silicon via OpenCL) or Colab's free GPU tier is enough. CPU-only works
but explicit-solvent MD is painfully slow on a CPU — don't unless you mean to.

---

## Maintainers

**The notebooks are generated — do not hand-edit the `.ipynb` files.** Each is emitted by a `build_*.py`
script via `nbformat`. To change a notebook, edit the shipped module *or* the builder, then re-run the
builder:

```bash
python build_figures.py && python build_sandbox.py && python build_minimal.py && python build_determinism.py && python build_enhanced.py
```

See [`CLAUDE.md`](CLAUDE.md) for the full architecture contract (the module/builder invariant, the on-ramp
completeness rule, the porting/run-order logistics, and the reproducibility claims not to weaken).

---

## Citation & license

This notebook set is the hands-on companion to *[tutorial article — citation to come]*. Built entirely on
open-source, license-free tools (OpenMM, PDBFixer, MDTraj, py3Dmol, PyMOL open-source, pymbar). License to
be finalized with the manuscript.
