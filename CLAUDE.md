# CLAUDE.md — working on the Trp-cage MD tutorial (v2)

Guidance for AI agents (and humans) editing this repo. `README.md` is the user-facing front door; this
file is the maintainer's contract — the invariants that keep the tutorial correct and reproducible.

## What this repo is
A deliberately small, license-free, end-to-end protein MD workflow on Trp-cage (**1L2Y**), built to make
the moving parts of an MD study *visible and reproducible* for a review/tutorial manuscript. Not a
production result — a teaching artifact. This is **v2**: the tutorial split into short per-stage notebooks
(build → dynamics → enhanced sampling), run in order, backed by shipped importable modules. The v1 monolith lives in `../tutorial_peptides/` (frozen, known-good
fallback; do not develop there).

## THE architecture invariant (read this first)
- **Two shipped, importable modules are the source of truth for the simulation code:**
  - **`mdtutorial.py`** — the fixed tutorial pipeline + analysis (`fetch_pdb`, `repair`, `solvate`,
    `build_system`, `pick_platform`, `minimize`, `prepare_system`, `run_repeat`, `compute_cvs`, the
    `trust_report` convergence toolkit) and the notebook-to-notebook **porting** handoff
    (`save_prepared` / `load_prepared`, and `load_or_prepare` = load-else-reprep so each notebook is
    self-sufficient). State travels in a `PreparedSystem` bundle — no module globals.
  - **`mdtsandbox.py`** — the "turn the knobs" machinery (force-field/water presets, ensemble/thermostat/
    timestep, an in-memory run loop, generic observables). It **reuses** `mdtutorial` (`fetch_pdb`,
    `pick_platform`) and adds only what the fixed pipeline can't do.
  These are **imported at notebook runtime** (unlike v1's `core_cells.py`, which was build-time-only string
  inlining — that model is gone).
- **Builders emit the notebooks** (`nbformat`): `build_figures.py`, `build_sandbox.py`, `build_minimal.py`,
  `build_determinism.py`. **Do NOT hand-edit the `.ipynb` files** — edit the module or the builder, then
  re-run the builder. After changing a module, re-run any builder whose notebook depends on it.
- **`minimal.ipynb` is the exception: it does NOT import** — the whole pipeline is inline, flat,
  self-contained (the "read the entire skeleton in one place" reference). Its logic must stay **consistent
  with `mdtutorial`** (same repair order, seeded + Reference-platform prep, fresh-context dynamics).

## Notebooks & how they connect
| notebook | imports | role |
|---|---|---|
| `01_build_system.ipynb` | `mdtutorial` | prepare + minimize, **save** the prepared system |
| `02_dynamics.ipynb` | `mdtutorial` | **load** the system, 3 repeats, dynamics figure |
| `03_enhanced_sampling.ipynb` | `mdtutorial` | steered MD + umbrella + MBAR (enhanced sampling) |
| `sandbox.ipynb` | `mdtsandbox` | interactive physics playground (in-memory) |
| `minimal.ipynb` | — (self-contained) | bare-bones flat pipeline |
| `determinism.ipynb` | `mdtutorial` | prep/minimize/dynamics reproducibility experiments |

**Porting (the run-order logistics):** the figure notebooks share **one output root** (`OUT="trpcage_out"`,
with `PREP=OUT` the prep location) and run in a **prescribed order** — 01 `save_prepared()`s the system
(`system.xml` + `stage4_minimized.pdb`), 02+ `load_prepared(PREP)` it instead of re-prepping. **This is a
default, not a hard requirement** (reversed 2026-07-28): if the prep is absent — 02 run standalone, or on
Colab where each notebook is a *separate VM* with no shared filesystem — 02/03 fall back to an **EXPLICIT,
loud re-prep** via the shared **`mdt.load_or_prepare(PREP, OUT)`** helper (load-else-`prepare_system`+
`save_prepared`, printing which path ran) — the logic lives ONCE in `mdtutorial.py`, not inline per notebook. The earlier "no re-prep" rule guarded against a *silent* fallback that blurred 01/02 and
mis-seeded a shadow solvation; the loud version fixes that without breaking the Colab / standalone case. The
fallback prep is an INDEPENDENT solvation (default seed), not 01's exact one — fine for dynamics, not for
bit-reproducibility. **Colab is self-sufficient-notebooks by design; the file handoff is a local-only
optimization.** Drive persistence of `PREP` is an opt-in (README), NOT a default code path (drive.mount is
flaky: full-Drive-access prompt + `credential propagation` cookie errors). `SEED` is 01's solvation seed.
**Trajectory reuse is fingerprinted:** each `traj_<seed>.dcd` is stamped (`.prepid`) with a hash of the
prep's minimized coords; `run_repeat` reuses it only if the current prep's fingerprint AND the frame count
match, so changing the solvation (even at the same atom count) forces regeneration instead of silently
analyzing a stale run. Document this order in the README. Generated output roots are gitignored.

## Reproducibility claims — DO NOT weaken these
- **Dynamics** reproduce bit-for-bit from the seed **on a given GPU model**, only with a **fresh Context
  per run** (re-using and re-seeding a Context silently diverged in our tests — mechanism untraced, so
  state the recipe, not the cause). Observed on OpenCL/Apple Silicon.
- **Minimization** is bit-reproducible only with `DeterministicForces` (CUDA); we ship the **minimized
  start** so it stops mattering.
- **Preparation** is deterministic via seeded RNGs + hydrogen placement on the **Reference** platform;
  `mdtutorial.repair` bakes this in. The determinism notebook demonstrates all three directly.
- Headline: **archive the prepared system, not just the seed** — solvation is stochastic down to the water
  *count*. **Never** reintroduce an unqualified "bit-reproducible" claim (always per-GPU-model,
  fresh-context, fixed-prepared-system).

## Environment
- **conda-forge is the reliable path** (`environment.yml`); `pip`'s resolver conflicts on `openmm`+`pdbfixer`.
  `requirements.txt` is the Colab/pip fallback. On HPC, **select your conda kernel** so the on-ramp installs
  nothing. Notebooks carry a Cell-0 on-ramp that provisions deps + fetches the module(s) on Colab and
  no-ops locally.
- **On-ramp completeness invariant (do NOT let these drift apart):** for EVERY notebook,
  `detect list == Colab-install list == the notebook's pip-provisionable import closure` — where the closure
  includes deps pulled in transitively by the shipped modules it imports (`mdtviz`→`py3Dmol`, `pull_screen`
  →`pymbar`+`scipy`, etc.), not just the cell-level imports. If the detect list is narrower than the closure,
  a locally-missing dep skips the clean "build/refresh your env" SystemExit and dies later with an opaque
  `ImportError` inside a compute/viz cell — the exact trap a novice can't debug. Concretely: 01/02/sandbox
  = `openmm,pdbfixer,mdtraj,py3Dmol`; enhanced = `+pymbar`; minimal/determinism = `openmm,pdbfixer,mdtraj`
  (no viz). **`numpy`/`scipy`/`matplotlib` are baseline** (guaranteed by both `environment.yml` and Colab)
  and are deliberately NOT pip-installed on Colab — reinstalling `numpy` there can break openmm's ABI.
  `environment.yml` must remain the UNION of every notebook's closure. After adding any import to a module or
  cell, re-audit both lists in every affected builder, then rebuild.

## Conventions
- **Residue numbering:** user-facing **labels use canonical PDB numbering** (1L2Y = 1–20; TRP6 the caged
  Trp, ILE4 the tracked χ1 rotamer (02_dynamics / paper Fig 4 dihedral CV; PRO19 ψ was the prior choice), GLN5/ASP9 the
  helix H-bond donors). mdtraj `resid`/`residue.index` is
  **0-based** (`= resSeq − 1`); notebook labels do the +1 correctly. `md_scalogram.py --resid` is 0-based.
- **N→C coloring** runs **blue→red**; use the `turbo` colormap.
- **Dihedrals are circular** — never `np.unwrap` for a scalogram; use cos/sin (`md_scalogram` handles it).
- **repair order:** `removeHeterogens` **before** `findMissingAtoms` (else terminal atoms like OXT aren't
  added). Both `mdtutorial` and `mdtsandbox` do this.

## Analysis tools (standalone; live with v1, port over as needed)
`md_scalogram.py` (timescale×time; blocking=Haar + Morlet CWT; circular-aware), `cross_scale.py`,
`make_reference.py` (strip a full run to a protein-only strided reference), `psi_screen.py`.

## Open work (see the task list)
- **viz** cells (01 PyMOL cartoons, 02 synchronized player) —
  likely a `mdtviz.py` companion; a **v2 `README`** (run order); the **second pass: hide non-essential
  cells** (`hide-input` tags) once the partition settles; a **statistics/label review** (every observable's
  calc + every textual reference reconciled — e.g. "backbone RMSD" labels vs the all-atom calc); and the
  **enhanced-sampling** notebook (steered/pulling MD + MBAR).

## Regenerating
```bash
python build_figures.py && python build_sandbox.py && python build_minimal.py && python build_determinism.py
```
