"""mdtutorial.py — the shipped, importable simulation + analysis core for the Trp-cage MD tutorial.

Purpose-built companion module (see CLAUDE.md). The polished per-figure notebooks and the determinism
test `import mdtutorial as mdt` and call these functions; `md_minimal.ipynb` inlines the same bodies
verbatim (so a reader can see the whole skeleton in one place). This is a FAITHFUL extraction of the
pipeline that used to be inlined from `core_cells.py`'s string constants — same behavior, now with an
explicit functional API and no shared module globals: state travels in a `PreparedSystem` bundle.

Pipeline:  fetch_pdb -> repair -> solvate -> build_system -> minimize   (or prepare_system() for all of it)
           run_repeat(prep, seed) -> trajectory   ;   compute_cvs / trust_report -> analysis

Reproducibility (do not weaken — see CLAUDE.md): dynamics reproduce bit-for-bit from the seed on a given
GPU model with a FRESH Context per run; minimization needs CUDA+DeterministicForces; prep is made
deterministic by seeding RNGs and running hydrogen placement on the Reference platform.
"""
import os
import re
import glob
import json
import gc
import random
import numpy as np
import openmm as mm
from openmm import app, unit
import mdtraj as md

SEEDS = [2024, 2025, 2026]           # the tutorial's three independent repeats


# ============================================================ output routing
def outp(name, out_root="."):
    """Route an output file (under out_root) into a subfolder by type -- PDBs -> structures/, PNGs ->
    figures/, trajectories & logs -> md_output/ -- created on first use. Each notebook sets its own
    out_root so outputs can't collide."""
    e = name.rsplit(".", 1)[-1].lower()
    sub = "figures" if e == "png" else "structures" if e in ("pdb", "cif", "ent") else "md_output"
    d = os.path.join(out_root, sub)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def default_forcefield():
    """The tutorial's force field: CHARMM36m protein + CHARMM-modified TIP3P water."""
    return app.ForceField("charmm36.xml", "charmm36/water.xml")


# ============================================================ prepared-system bundle
class PreparedSystem:
    """Everything a production run needs, carried between pipeline steps instead of module globals:
    the topology, the OpenMM System, the chosen platform (+ its properties), the minimized starting
    positions, and the hardware report. Returned by prepare_system(); consumed by run_repeat()."""
    def __init__(self, topology, system, platform, plat_props, min_positions, hardware,
                 modeller=None, energy_curve=None):
        self.topology = topology
        self.system = system
        self.platform = platform
        self.plat_props = plat_props
        self.min_positions = min_positions
        self.hardware = hardware
        self.modeller = modeller
        self.energy_curve = energy_curve      # PE vs minimization step, if recorded


# ============================================================ fetch / repair / solvate
def fetch_pdb(pdb_id, out_root="."):
    """Download a PDB (RCSB) to structures/ if not already present; return its path."""
    import urllib.request
    path = outp(f"{pdb_id}.pdb", out_root)
    if not os.path.exists(path):
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb_id}.pdb", path)
    return path


def repair(pdb_path, seed, out_root=".", pH=7.0, verbose=True):
    """PDBFixer repair + force-field-consistent hydrogens. addHydrogens is non-deterministic in two ways
    (random H placement, then a fix-up minimizer that defaults to a fast non-deterministic platform); we
    seed the RNGs AND pin the fix-up to the Reference platform -> bit-deterministic prep. Returns the
    Modeller (also writes stage2_repaired.pdb)."""
    from pdbfixer import PDBFixer
    ff = default_forcefield()
    fixer = PDBFixer(filename=pdb_path)                    # PDBFixer reads model 1 only
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)                # BEFORE findMissingAtoms: doing it between find and
    fixer.findMissingAtoms()                               # add invalidates residue refs, so terminal atoms
    fixer.addMissingAtoms()                                # (e.g. C-terminal OXT) never get added. No-op for 1L2Y.
    if verbose:
        print("missing residues:", fixer.missingResidues)
        print("missing atoms  :", fixer.missingAtoms, "| terminals:", fixer.missingTerminals)
    modeller = app.Modeller(fixer.topology, fixer.positions)
    modeller.delete([a for a in modeller.topology.atoms() if a.element == app.element.hydrogen])
    random.seed(seed); np.random.seed(seed)               # seed BOTH RNGs the H-placement uses
    modeller.addHydrogens(ff, pH=pH, platform=mm.Platform.getPlatformByName("Reference"))
    app.PDBFile.writeFile(modeller.topology, modeller.positions,
                          open(outp("stage2_repaired.pdb", out_root), "w"))
    if verbose:
        print("repaired peptide atoms:", modeller.topology.getNumAtoms())
    return modeller


def repair_report(pdb_path, label, save=None):
    """Repair a structure and PRINT the contrast stats -- deposited H, crystallographic waters removed,
    nonstandard residues, H built by the force field -- for showing why an X-ray/cryo-EM structure needs
    more repair than a clean NMR ensemble. Optionally write the repaired PDB (for a before/after figure)."""
    from pdbfixer import PDBFixer
    ff = default_forcefield()
    fx = PDBFixer(filename=pdb_path)
    h0 = sum(a.element == app.element.hydrogen for a in fx.topology.atoms())
    w0 = sum(r.name in ("HOH", "WAT") for r in fx.topology.residues())
    fx.findMissingResidues(); fx.findNonstandardResidues()
    fx.removeHeterogens(keepWater=False); fx.findMissingAtoms(); fx.addMissingAtoms()
    m = app.Modeller(fx.topology, fx.positions)
    m.delete([a for a in m.topology.atoms() if a.element == app.element.hydrogen])
    m.addHydrogens(ff, pH=7.0)
    nH = sum(a.element == app.element.hydrogen for a in m.topology.atoms())
    print(f"{label:12s}: deposited H={h0:4d} | cryst. waters removed={w0:3d} | "
          f"nonstandard={len(fx.nonstandardResidues)} | H built by repair={nH}")
    if save:
        app.PDBFile.writeFile(m.topology, m.positions, open(save, "w"))


def solvate(modeller, seed, out_root=".", padding_nm=1.0, verbose=True):
    """Add a periodic box of TIP3P water + neutralizing ions. addSolvent's water packing / ion placement
    vary ~nm run-to-run unless the RNGs are seeded (seeded -> max|delta| = 0). Returns the Modeller (also
    writes stage3_solvated.pdb)."""
    ff = default_forcefield()
    random.seed(seed); np.random.seed(seed)
    modeller.addSolvent(ff, model="tip3p", padding=padding_nm * unit.nanometer, neutralize=True)
    if verbose:
        nwat = sum(r.name in ("HOH", "WAT") for r in modeller.topology.residues())
        nion = sum(r.name in ("NA", "CL", "K") for r in modeller.topology.residues())
        boxL = modeller.topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer)[0][0]
        print(f"{modeller.topology.getNumAtoms()} atoms  ·  {nwat} waters  ·  {nion} ion(s)  ·  {boxL:.2f} nm box")
    app.PDBFile.writeFile(modeller.topology, modeller.positions,
                          open(outp("stage3_solvated.pdb", out_root), "w"))
    return modeller


# ============================================================ system + platform + minimize
def pick_platform(topology, system, positions, seed, temp_K=300.0):
    """Return (simulation, platform, plat_props): the fastest platform that ACTUALLY builds a Context on
    this machine. A platform can be registered yet have no usable device (a CUDA-enabled OpenMM on a CPU
    node lists "CUDA" but raises CUDA_ERROR_NO_DEVICE at Context creation), so TRY each in preference order
    and fall back on failure. Shared by build_system() and load_prepared() so a run ports across hardware."""
    temp = temp_K * unit.kelvin
    names = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]

    def _try(pname):
        plat = mm.Platform.getPlatformByName(pname)
        props = {"Precision": "mixed", "DeterministicForces": "true"} if pname == "CUDA" else {}
        integ = mm.LangevinMiddleIntegrator(temp, 1 / unit.picosecond, 0.002 * unit.picoseconds)
        integ.setRandomNumberSeed(seed)
        s = app.Simulation(topology, system, integ, plat, props)   # device error (if any) happens HERE
        s.context.setPositions(positions)
        return s, plat, props

    for pname in ("CUDA", "OpenCL", "CPU", "Reference"):
        if pname not in names:
            continue
        try:
            return _try(pname)
        except Exception as e:
            print(f"platform {pname}: registered but unusable ({str(e).splitlines()[0][:60]}) -> falling back")
    raise RuntimeError("no usable OpenMM platform (tried CUDA/OpenCL/CPU/Reference)")


def build_system(modeller, seed, temp_K=300.0):
    """Create the OpenMM System and pick a working platform. Returns (system, simulation, platform, plat_props)."""
    ff = default_forcefield()
    system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME,
                             nonbondedCutoff=1.0 * unit.nanometer, constraints=app.HBonds)
    sim, platform, props = pick_platform(modeller.topology, system, modeller.positions, seed, temp_K)
    return system, sim, platform, props


def hardware_report(context):
    """Full hardware/environment provenance -- printed on every run and archived in canonical mode. If two
    runs disagree when they 'should' be deterministic, compare these to see where they differ (GPU model,
    precision, driver, OpenMM version)."""
    import platform as _pl
    import subprocess as _sp
    p = context.getPlatform()
    hw = {"openmm": mm.version.version, "platform": p.getName(), "python": _pl.python_version(),
          "os": _pl.platform(), "cpu": _pl.processor() or _pl.machine()}
    for k in ("DeviceName", "Precision", "DeterministicForces", "DeviceIndex"):
        try:
            hw[k] = p.getPropertyValue(context, k)          # not all keys exist on all platforms
        except Exception:
            pass
    try:                                                    # NVIDIA driver/model, only if a GPU is usable
        r = _sp.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=10)
        hw["nvidia_smi"] = r.stdout.strip() if (r.returncode == 0 and r.stdout.strip()) else None
    except Exception:                                       # a FAILED nvidia-smi prints its error to stdout -> check returncode
        hw["nvidia_smi"] = None
    return hw


def print_hardware_report(hw):
    """Pretty-print a hardware_report() dict with the reproducibility scope and exactly ONE situational
    note chosen by (platform, is a GPU visible at all)."""
    print("hardware / environment")
    for k, v in hw.items():
        if v is not None:
            print(f"  {k:16s}: {v}")
    print("  reproducibility : " + ("bit-for-bit from a SEED on this GPU model (CUDA + DeterministicForces)"
                                     if hw["platform"] == "CUDA" else
                                     "STATISTICAL only -- not bit-for-bit on this platform"))
    print("                    whole pipeline reproduces from the seed: prep is made deterministic (seeded + H")
    print("                    fix-up on the Reference platform); dynamics/minimization are deterministic on CUDA+DetForces.")
    print("                    Still PER-GPU-MODEL: a different NVIDIA model diverges via floating-point non-associativity.")
    if hw["platform"] == "OpenCL" and hw.get("nvidia_smi"):
        print( "  NOTE            : an NVIDIA GPU is present but OpenMM is using OpenCL, not CUDA -- fast, but")
        print( "                    minimization is NOT bit-reproducible here. Install a CUDA-enabled OpenMM")
        print( "                    (conda-forge openmm + cudatoolkit, e.g. condacolab) for the DeterministicForces path.")
    elif hw["platform"] in ("CPU", "Reference") and hw.get("nvidia_smi"):
        print(f"  NOTE            : an NVIDIA GPU is visible on this node but OpenMM could not use it and fell back to")
        print(f"                    {hw['platform']} -- correct but MUCH slower (GPU not allocated to this job, or a driver/")
        print( "                    toolkit mismatch). ~35 s / 200 ps on a GPU vs ~20-60 MIN on CPU; keep n_prod_ps small.")
    elif hw["platform"] in ("CPU", "Reference"):
        print(f"  NOTE            : no GPU device found -- running on {hw['platform']} (correct, just SLOW). The default")
        print( "                    200 ps run is ~35 s on a GPU but ~20-60 MINUTES on CPU; keep n_prod_ps small, GPU for more.")


def minimize(simulation, record_curve=False, n_chunks=25, chunk_iters=10):
    """Energy-minimize the current configuration (a downhill walk on the PES, NOT dynamics). If
    record_curve, minimize in n_chunks of chunk_iters iterations and return (min_positions, pe_curve) so
    the notebook can plot the descent; otherwise run to convergence and return min_positions."""
    def _pe():
        return simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    if record_curve:
        pe = [_pe()]
        for _ in range(n_chunks):
            simulation.minimizeEnergy(maxIterations=chunk_iters)
            pe.append(_pe())
        return simulation.context.getState(getPositions=True).getPositions(), pe
    simulation.minimizeEnergy()
    return simulation.context.getState(getPositions=True).getPositions()


def prepare_system(pdb_id="1L2Y", seed=2024, out_root=".", temp_K=300.0,
                   minimize_curve=False, verbose=True):
    """Convenience: fetch -> repair -> solvate -> build_system -> minimize, returned as a PreparedSystem.
    Use the granular functions instead when a notebook wants to SHOW each step (the build-system figure)."""
    path = fetch_pdb(pdb_id, out_root)
    modeller = repair(path, seed, out_root, verbose=verbose)
    modeller = solvate(modeller, seed, out_root, verbose=verbose)
    system, sim, platform, props = build_system(modeller, seed, temp_K)
    hw = hardware_report(sim.context)
    if verbose:
        print_hardware_report(hw)
    mn = minimize(sim, record_curve=minimize_curve)
    min_positions, curve = mn if minimize_curve else (mn, None)
    return PreparedSystem(sim.topology, system, platform, props, min_positions, hw,
                          modeller=modeller, energy_curve=curve)


# ============================================================ porting between notebooks
# The tutorial is split into one notebook per figure. A downstream notebook can reconstruct the prepared,
# minimized system produced upstream INSTEAD of re-prepping -- run the notebooks in order, or load the
# shipped reference. save_prepared() writes the three files load_prepared() needs.
def save_prepared(prep, out_root="."):
    """Serialize a PreparedSystem so a later notebook can load it: the System (system.xml), the solvated
    topology (stage3_solvated.pdb, already written by solvate), and the MINIMIZED coordinates
    (stage4_minimized.pdb, box preserved). Returns the dict of paths."""
    paths = {"system": outp("system.xml", out_root),
             "topology": outp("stage3_solvated.pdb", out_root),
             "minimized": outp("stage4_minimized.pdb", out_root)}
    open(paths["system"], "w").write(mm.XmlSerializer.serialize(prep.system))
    app.PDBFile.writeFile(prep.topology, prep.min_positions, open(paths["minimized"], "w"))
    return paths


def load_prepared(out_root=".", seed=2024, temp_K=300.0):
    """Reconstruct a PreparedSystem from save_prepared()'s files (system.xml + stage4_minimized.pdb),
    re-picking the platform on THIS machine so the run ports across hardware. Skips prep + minimize
    entirely -- the downstream figure notebooks call this instead of prepare_system(). Errors clearly if
    fig1's output is absent: fig2+ REQUIRE it and there is no silent re-prep."""
    min_pdb, sys_xml = outp("stage4_minimized.pdb", out_root), outp("system.xml", out_root)
    if not (os.path.exists(min_pdb) and os.path.exists(sys_xml)):
        raise FileNotFoundError(
            f"No prepared system in {out_root}/ (need system.xml + stage4_minimized.pdb). Run the Figure 1 "
            f"notebook (fig1_build_system) first -- it prepares and saves the system this notebook loads.")
    pdb = app.PDBFile(min_pdb)
    system = mm.XmlSerializer.deserialize(open(sys_xml).read())
    sim, platform, props = pick_platform(pdb.topology, system, pdb.positions, seed, temp_K)
    return PreparedSystem(pdb.topology, system, platform, props, pdb.positions,
                          hardware_report(sim.context))


def load_or_prepare(prep_root, out_root=None, seed=2024, temp_K=300.0, pdb_id="1L2Y", verbose=True):
    """Return a PreparedSystem, degrading gracefully so every notebook is SELF-SUFFICIENT: LOAD the system
    fig1 saved (load_prepared) if it's there, else PREPARE a fresh one here (prepare_system + save_prepared).
    The fig1 -> fig2 -> fig3 order sharing one prep needs a shared filesystem -- true locally, but NOT on Colab
    (each notebook runs in its own VM) or when a notebook is opened standalone. This is the ONE place that
    logic lives; fig2/fig3 call it instead of re-implementing the try/except. The fresh fallback is an
    INDEPENDENT solvation (its own water) -- fine for dynamics, not for bit-for-bit reproducibility -- and the
    print states which path ran. `out_root` (where a fresh prep + its stage PDBs are written) defaults to
    `prep_root`. To share fig1's EXACT prep across VMs on Colab, point `prep_root` at a persisted location."""
    if out_root is None:
        out_root = prep_root
    try:
        prep = load_prepared(prep_root, seed=seed, temp_K=temp_K)
        if verbose:
            print("loaded the prepared system saved by Figure 1.")
        return prep
    except FileNotFoundError:
        if verbose:
            print("No saved prep found -- preparing a FRESH system here (its own solvation, not fig1's exact")
            print("  one; fine for dynamics, use fig1's saved prep for bit-for-bit reproducibility).")
        prep = prepare_system(pdb_id=pdb_id, seed=seed, out_root=out_root, temp_K=temp_K, verbose=verbose)
        save_prepared(prep, out_root)
        return prep


def _prep_fingerprint(prep):
    """Short content hash of the prepared system's minimized coordinates -- identifies THIS solvation so a
    trajectory is reused only if it came from the same prepared system (two solvation seeds can share an
    atom count but never share these coordinates)."""
    import hashlib
    xyz = np.asarray(prep.min_positions.value_in_unit(unit.nanometer), dtype=np.float64)
    return hashlib.md5(np.round(xyz, 5).tobytes()).hexdigest()[:12]


# ============================================================ production runner
def run_repeat(prep, seed, n_prod_ps=200, run_mode="interactive", out_root=".",
               force_rerun=False, load_reference=False, ref_root="reference_run", temp_K=300.0):
    """Run one independent production trajectory from the prepared, minimized system. Uses a FRESH OpenMM
    Context (the integrator seed is applied at Context creation, so the Langevin stream is reproducible;
    re-seeding a reused Context does NOT reset the GPU RNG and silently breaks reproducibility). Writes
    traj_<seed>.dcd + the scalar state log; in run_mode='canonical' also the restart/reproduction slate.
    Returns the trajectory path.

    Guards: load_reference returns the shipped reference trajectory instead of simulating; an existing
    right-length traj_<seed>.dcd is REUSED (not overwritten) unless force_rerun (protects real runs)."""
    if load_reference:                                    # no-GPU path: analyze the shipped reference run
        ref = os.path.join(ref_root, "md_output", f"traj_{seed}.dcd")
        if os.path.exists(ref):
            print(f"[reference] using shipped {ref} (not simulating)")
            return ref
        print(f"[reference] {ref} missing -- simulating this seed instead")
    dcd = outp(f"traj_{seed}.dcd", out_root)
    stamp_path = dcd + ".prepid"                          # provenance: which prepared system this traj came from
    want_fp = _prep_fingerprint(prep)                     # fingerprint of the CURRENT prep's minimized coords
    if os.path.exists(dcd) and not force_rerun:           # OVERWRITE GUARD: protect an existing run
        try:
            n = len(md.open(dcd))                         # header-only frame count (~1 ms, no full load)
        except Exception:
            n = None
        stamp = open(stamp_path).read().strip() if os.path.exists(stamp_path) else None
        if n == n_prod_ps and stamp == want_fp:           # right LENGTH *and* made from THIS exact prep
            print(f"[keep] {dcd} matches this prep ({n} frames, prep {want_fp}) -- REUSING it, NOT re-simulating "
                  f"(force_rerun=True, or delete the file, to force a fresh run).")
            return dcd
        why = (f"has {n} frames != n_prod_ps={n_prod_ps} (length changed)" if n != n_prod_ps
               else "was made from a DIFFERENT prepared system (solvation/prep changed)" if stamp and stamp != want_fp
               else "has no prep fingerprint (older run) -- regenerating to stamp it")
        print(f"[re-run] {dcd} {why} -- regenerating.")

    temp = temp_K * unit.kelvin
    integ_r = mm.LangevinMiddleIntegrator(temp, 1 / unit.picosecond, 0.002 * unit.picoseconds)
    integ_r.setRandomNumberSeed(seed)
    sim_r = app.Simulation(prep.topology, prep.system, integ_r, prep.platform, prep.plat_props)
    sim_r.context.setPositions(prep.min_positions)
    sim_r.context.setVelocitiesToTemperature(temp, seed)
    sim_r.step(10000)                                     # 20 ps NVT equilibration
    dcd_rep = app.DCDReporter(dcd, 500); sim_r.reporters.append(dcd_rep)   # TRAJECTORY: every figure derives from it
    sim_r.reporters.append(app.StateDataReporter(outp(f"state_{seed}.csv", out_root), 500,   # scalar STATE LOG (always)
        step=True, time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
        temperature=True, volume=True, density=True, speed=True))
    if run_mode == "canonical":
        sim_r.reporters.append(app.CheckpointReporter(outp(f"checkpoint_{seed}.chk", out_root), 5000))
        open(outp("system.xml", out_root), "w").write(mm.XmlSerializer.serialize(prep.system))
        open(outp(f"integrator_{seed}.xml", out_root), "w").write(mm.XmlSerializer.serialize(integ_r))
    sim_r.step(n_prod_ps * 500)                           # production
    sim_r.reporters.clear()
    if run_mode == "canonical":
        sim_r.saveState(outp(f"final_state_{seed}.xml", out_root))
        meta = {**hardware_report(sim_r.context), "seed": seed, "temperature_K": temp_K,
                "timestep_fs": 2, "prod_ps": n_prod_ps, "forcefield": "charmm36 + charmm36/water"}
        json.dump(meta, open(outp(f"run_meta_{seed}.json", out_root), "w"), indent=2)
    # Flush to STABLE STORAGE before anything reads it back: DCDReporter closes only on GC, so on a
    # networked FS an unflushed DCD can read back as ZERO frames (the intermittent compute_cvs IndexError).
    try:
        dcd_rep._out.flush(); os.fsync(dcd_rep._out.fileno()); dcd_rep._out.close()
    except Exception:
        pass
    dcd_rep = None; gc.collect()
    open(stamp_path, "w").write(want_fp)                  # stamp provenance beside the trajectory (prep fingerprint)
    return dcd


# ============================================================ observables / collective variables
def circmean_deg(A):
    """Circular mean of an angle series (degrees)."""
    r = np.radians(A)
    return np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean()))


def recenter_deg(A, center):
    """Recenter an angle series's +-180 branch cut on `center` so tiny wiggles across the seam stop
    looking like ~360 deg cliffs."""
    return center + (np.asarray(A) - center + 180) % 360 - 180


def compute_cvs(dcd, top):
    """Load a trajectory (topology path `top`), slice to protein, superpose, and return the tutorial's
    collective variables: **Ca RMSD** (fold stability), helix fraction (simplified DSSP), the two
    alpha-helix backbone H-bond distances (GLN5 N-H..ASN1 O=C, ASP9 N-H..GLN5 O=C; donor i to carbonyl
    i-4), the **ASP9-ARG16 salt bridge** (min carboxylate-guanidinium distance; the canonical Trp-cage
    stabilizer, slow), and two backbone dihedrals -- **PRO19 psi** (rare slow basin hop) and **SER13 psi**
    (3-10 helix; fast, multi-scale). Labels canonical (PDB 1-20); mdtraj `resid`/index is 0-based (index
    4 = GLN5, 8 = ASP9, 12 = SER13, 15 = ARG16, 18 = PRO19)."""
    full = md.load(dcd, top=top)
    if full.n_frames == 0:                                # networked-FS flush lag or a clobbered file
        raise RuntimeError(f"{dcd} read back with 0 frames -- the trajectory was not yet durable on disk "
                           "(shared-filesystem lag) or was overwritten by a concurrent run. Re-run this "
                           "step; if it recurs, give each notebook its own working directory.")
    t = full.atom_slice(full.topology.select("protein")); t.superpose(t, 0)
    x = t.xyz * 10
    a = lambda resid, name: t.topology.select(f"resid {resid} and name {name}")[0]
    dist = lambda i, j: np.linalg.norm(x[:, i] - x[:, j], axis=1)
    pidx, psi = md.compute_psi(t)
    _psi = lambda ridx: next(np.degrees(psi[:, k]) for k, idx in enumerate(pidx)
                             if t.topology.atom(idx[1]).residue.index == ridx)   # 0-based resid
    p19, ser13 = _psi(18), _psi(12)                           # PRO19 (rare slow basin hop) · SER13 (3-10 helix, fast jitter)
    ca = t.topology.select("name CA")
    _o, _n = t.topology.select("resid 8 and name OD1 OD2"), t.topology.select("resid 15 and name NH1 NH2 NE")
    d9r16 = md.compute_distances(t, [[i, j] for i in _o for j in _n]).min(1) * 10   # ASP9 carboxylate <-> ARG16 guanidinium
    return dict(t=t, ps=np.arange(t.n_frames), rmsd=md.rmsd(t, t, 0, atom_indices=ca) * 10,   # Ca RMSD (fold stability)
                helix=(md.compute_dssp(t, simplified=True) == 'H').mean(1),
                gln5=dist(a(4, "H"), a(0, "O")), asp9=dist(a(8, "H"), a(4, "O")),
                p19=p19, ser13=ser13, d9r16=d9r16)


def load_reference(ref_root="reference_run", dt_ps=None):
    """Load a shipped protein-only reference bundle (make_reference.py output) as a list of CV dicts -- the
    SAME shape compute_cvs returns, so downstream analysis is identical to a live run. The point is an
    HONEST autocorrelation time: a short live run underestimates tau by ~50x because it never samples the
    ns-scale motions, so convergence (2.6) and any timescale analysis should be quoted from a multi-ns
    reference. Frame spacing comes from the bundle's run_meta (`reference_stride_ps`); each returned dict
    carries its `seed`, and `solvation_seed` when the bundle recorded one (so a seed cross-check can span
    both dynamics seeds and water configurations). Returns (cvs, seeds, dt_ps). Seed-count-agnostic:
    works for the interim 3-seed set or a future 6-seed / 2-solvation-seed bundle without changes."""
    top = os.path.join(ref_root, "structures", "protein.pdb")
    if not os.path.exists(top):
        raise FileNotFoundError(f"reference topology not found: {top} -- drop a reference_run/ bundle "
                                "(make_reference.py output) beside the notebook, or point REF_ROOT at one")
    dcds = sorted(glob.glob(os.path.join(ref_root, "md_output", "traj_*.dcd")))
    if not dcds:
        raise FileNotFoundError(f"no traj_*.dcd found in {ref_root}/md_output")
    if dt_ps is None:                                        # frame spacing (ps) from bundle meta, else 1
        meta = {}
        for mp in [os.path.join(ref_root, "run_meta.json")] + \
                  sorted(glob.glob(os.path.join(ref_root, "md_output", "run_meta_*.json"))):
            if os.path.exists(mp):
                meta = json.load(open(mp)); break
        dt_ps = float(meta.get("reference_stride_ps", 1))
    cvs, seeds = [], []
    for d in dcds:
        m = re.search(r"traj_(\d+)", os.path.basename(d)); seed = int(m.group(1)) if m else None
        c = compute_cvs(d, top=top)
        c["ps"] = np.arange(c["t"].n_frames) * dt_ps         # honor the reference stride, not 1 frame = 1 ps
        c["seed"] = seed
        pm = os.path.join(ref_root, "md_output", f"run_meta_{seed}.json")
        if os.path.exists(pm):                               # record water config if the bundle tracked it
            jm = json.load(open(pm))
            for key in ("solvation_batch", "solvation_seed"):
                if key in jm:
                    c["solvation"] = jm[key]; break
        cvs.append(c); seeds.append(seed)
    return cvs, seeds, dt_ps


# ============================================================ convergence / uncertainty toolkit
# Pure numpy on purpose: for a teaching notebook it's better to SEE the estimator than to import pymbar.
def integrated_autocorr_time(x):
    """Integrated autocorrelation time tau = 1/2 + sum_k rho(k), in frames (Sokal 1997 convention; the 1/2 is
    the zero-lag self-term; for a single exponential tau equals the relaxation time). Truncated by Geyer's
    (1992) initial-positive-sequence rule: for a reversible process the consecutive-PAIR autocorrelations
    rho(2m+1)+rho(2m+2) are positive, so we sum pairs and stop at the first non-positive one -- a
    conservative estimate. A run of N frames holds N_eff = N/(2 tau) independent samples (white noise ->
    tau=1/2 -> N_eff=N)."""
    x = np.asarray(x, float) - np.mean(x); n = len(x); v = np.dot(x, x) / n
    if v == 0:
        return 0.5
    rho = lambda k: np.dot(x[:-k], x[k:]) / (n * v)      # normalized autocorrelation at lag k (biased, 1/n)
    tau = 0.5
    for k in range(1, n - 1, 2):
        pair = rho(k) + rho(k + 1)                       # Geyer pair sum: positive for a reversible process
        if pair <= 0:
            break                                        # initial-positive-sequence truncation (first non-positive pair)
        tau += pair
    return tau


def block_curve(x, min_blocks=8, n_sizes=48):
    """SEM as a function of block size (log-spaced), with the Flyvbjerg-Petersen uncertainty ON each SEM.
    As the block size passes the correlation time the blocks become independent and SEM(b) rises toward a
    plateau -- the honest error of the mean; the naive b=1 value underestimates it. But an SEM estimated
    from M blocks has relative error 1/sqrt(2(M-1)), so the large-b tail (few blocks) is noisy and rarely
    plateaus cleanly -- that is what `errs` quantifies. Returns (block_sizes, sems, errs)."""
    x = np.asarray(x, float); N = len(x)
    bmax = max(2, N // min_blocks)
    sizes = np.unique(np.geomspace(1, bmax, n_sizes).astype(int))
    bs, sems, errs = [], [], []
    for b in sizes:
        M = N // b
        if M < 2:
            continue
        means = x[:M * b].reshape(M, b).mean(axis=1)
        sem = means.std(ddof=1) / np.sqrt(M)
        bs.append(int(b)); sems.append(sem); errs.append(sem / np.sqrt(2 * (M - 1)))   # FP error on SEM(b)
    return np.array(bs), np.array(sems), np.array(errs)


def trust_report(series, dt_ps=1.0, label="observable"):
    """Print mean +/- autocorrelation-corrected SEM (sigma*sqrt(2 tau / N)), tau, and N_eff for a 1-D
    series. That SEM stays valid even when the run is too short for block averaging to plateau;
    block_curve() is the picture (its curve should climb up to this value). Returns a dict of the stats."""
    x = np.asarray(series, float); N = len(x)
    tau = integrated_autocorr_time(x); neff = N / (2.0 * tau)
    bsize, sems, berr = block_curve(x)
    sem = float(np.std(x, ddof=1) / np.sqrt(neff))
    print(f"{label}: mean {x.mean():.3f} +/- {sem:.3f}   tau ~ {tau * dt_ps:.1f} ps   "
          f"N_eff ~ {neff:.1f}  (of {N} frames)")
    return dict(mean=float(x.mean()), sem=sem, tau=tau, neff=neff, bsize=bsize, sems=sems, berr=berr)
