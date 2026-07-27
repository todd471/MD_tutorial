"""mdtsandbox.py -- machinery for the "turn the knobs" sandbox notebook, externalized to keep the notebook
tidy. It REUSES mdtutorial for the generic pieces (fetch, the working platform picker) and adds what the
sandbox needs and the fixed tutorial pipeline does not: force-field/water PRESETS, ensemble/thermostat/
timestep parameterization, and an IN-MEMORY run loop (no trajectory files) with GENERIC observables that
work for any fold. (Whether any of these stay exposed in the notebook for teaching is a second-pass call.)
"""
import random
import numpy as np
import openmm as mm
from openmm import app, unit
from pdbfixer import PDBFixer
import mdtraj as md
import mdtutorial as mdt

# tested menu: each entry preps + runs cleanly
PDB_MENU = {
    "1UAO": "chignolin — β-hairpin, 10 res         · NMR   (~2.3k atoms, fastest)",
    "1L2Y": "Trp-cage — α/loop cage, 20 res        · NMR   (~4.9k atoms)",
    "1FME": "FSD-1 — designed ββα, 28 res          · NMR   (~9.6k atoms)",
    "1VII": "villin HP36 — 3-helix bundle, 36 res  · NMR   (~8.6k atoms)",
    "1CRN": "crambin — 46 res, 3 disulfides        · X-ray (~7.7k atoms)",
    "1UBQ": "ubiquitin — α/β, 76 res               · X-ray (~17k atoms, slowest)",
}

# force field + water come as a MATCHED SET -> tested presets, not free choices
FF_MENU = {
    "amber14 + TIP3P":        (["amber14-all.xml", "amber14/tip3p.xml"], "tip3p"),
    "amber14 + OPC (4-site)": (["amber14-all.xml", "amber14/opc.xml"],   "tip4pew"),
    "amber14 + SPC/E":        (["amber14-all.xml", "amber14/spce.xml"],  "spce"),
    "amber99SB-ILDN + TIP3P": (["amber99sbildn.xml", "tip3p.xml"],       "tip3p"),
    "CHARMM36 + TIP3P":       (["charmm36.xml", "charmm36/water.xml"],   "tip3p"),
}
_FF_CACHE = {}
def forcefield(preset):
    """Return (ForceField, addSolvent water-model name) for a tested preset."""
    if preset not in _FF_CACHE:
        _FF_CACHE[preset] = app.ForceField(*FF_MENU[preset][0])
    return _FF_CACHE[preset], FF_MENU[preset][1]


_PREP_CACHE = {}
def prep(pdb, seed, ff_preset="amber14 + TIP3P"):
    """fetch -> repair (seeded + Reference-platform H) -> solvate (seeded), IN-MEMORY (no stage files);
    the System is built in run() so an NPT barostat never mutates a cached object. Deterministic + cached.
    Reuses mdtutorial.fetch_pdb; the repair order (removeHeterogens BEFORE findMissingAtoms) matches
    mdtutorial and keeps terminal refs valid for off-menu structures."""
    key = (pdb, seed, ff_preset)
    if key in _PREP_CACHE:
        return _PREP_CACHE[key]
    ff, water_model = forcefield(ff_preset)
    path = mdt.fetch_pdb(pdb, ".")                         # -> structures/<pdb>.pdb (reused)
    fx = PDBFixer(filename=path)
    fx.findMissingResidues(); fx.findNonstandardResidues()
    fx.removeHeterogens(keepWater=False)                  # before findMissingAtoms so terminal refs stay valid
    fx.findMissingAtoms(); fx.addMissingAtoms()
    m = app.Modeller(fx.topology, fx.positions)
    m.delete([a for a in m.topology.atoms() if a.element == app.element.hydrogen])
    random.seed(seed); np.random.seed(seed)
    m.addHydrogens(ff, pH=7.0, platform=mm.Platform.getPlatformByName("Reference"))
    random.seed(seed); np.random.seed(seed)
    m.addSolvent(ff, model=water_model, padding=1.0 * unit.nanometer, neutralize=True)
    _PREP_CACHE[key] = (m, ff)
    return m, ff


def make_system(ff, topology, ensemble, temp_k):
    """Build the System; add a Monte Carlo barostat only for NPT (constant pressure)."""
    system = ff.createSystem(topology, nonbondedMethod=app.PME,
                             nonbondedCutoff=1.0 * unit.nanometer, constraints=app.HBonds)
    if ensemble == "NPT":
        system.addForce(mm.MonteCarloBarostat(1.0 * unit.bar, temp_k * unit.kelvin, 25))
    return system


def integrator(ensemble, thermostat, temp_k, dt):
    """NVE -> Verlet (no thermostat); NVT/NPT -> Nose-Hoover (deterministic) or LangevinMiddle (default)."""
    T = temp_k * unit.kelvin
    if ensemble == "NVE":
        return mm.VerletIntegrator(dt)
    if thermostat == "Nose-Hoover":
        return mm.NoseHooverIntegrator(T, 1 / unit.picosecond, dt)
    return mm.LangevinMiddleIntegrator(T, 1 / unit.picosecond, dt)


def run(pdb, temp_k, seed, prod_ps, equil_ps=20,
        ff_preset="amber14 + TIP3P", ensemble="NVT", thermostat="Langevin", timestep_fs=2):
    """One run, IN MEMORY (no files). Returns (trajectory, cvs). cvs always has rmsd/rg/helix/ete; also
    volume/density (NPT) and total_energy (NVE). Defaults reproduce Section 1's physics (amber14+TIP3P,
    NVT, Langevin, 2 fs). Uses mdtutorial.pick_platform for the working-platform fallback."""
    T = temp_k * unit.kelvin; dt = timestep_fs * unit.femtoseconds
    spp = int(round(1000 / timestep_fs))                  # MD steps per ps
    m, ff = prep(pdb, seed, ff_preset)
    # minimize on an NVT system (no barostat), picking a platform that actually works on this machine
    sim0, plat, props = mdt.pick_platform(m.topology, make_system(ff, m.topology, "NVT", temp_k),
                                          m.positions, seed, temp_k)
    sim0.minimizeEnergy()
    minpos = sim0.context.getState(getPositions=True).getPositions()
    system = make_system(ff, m.topology, ensemble, temp_k)     # production system (barostat if NPT)
    integ = integrator(ensemble, thermostat, temp_k, dt)
    try:
        integ.setRandomNumberSeed(seed)                   # Verlet has no seed
    except Exception:
        pass
    sim = app.Simulation(m.topology, system, integ, plat, props)   # FRESH context
    sim.context.setPositions(minpos); sim.context.setVelocitiesToTemperature(T, seed)
    sim.step(equil_ps * spp)
    prot = md.Topology.from_openmm(m.topology).select("protein")
    ptop = md.Topology.from_openmm(m.topology).subset(prot)
    tmass = sum(system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(system.getNumParticles()))
    xyz = np.empty((prod_ps, len(prot), 3)); vol = np.empty(prod_ps); etot = np.empty(prod_ps)
    for f in range(prod_ps):
        sim.step(spp)
        st = sim.context.getState(getPositions=True, getEnergy=True)
        xyz[f] = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)[prot]
        bv = st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
        vol[f] = abs(float(np.linalg.det(bv)))
        etot[f] = (st.getPotentialEnergy() + st.getKineticEnergy()).value_in_unit(unit.kilojoule_per_mole)
    t = md.Trajectory(xyz, ptop); t.superpose(t, 0)
    x = t.xyz * 10; ca = t.topology.select("name CA")
    cvs = dict(ps=np.arange(t.n_frames),
               rmsd=md.rmsd(t, t, 0) * 10, rg=md.compute_rg(t) * 10,
               helix=(md.compute_dssp(t, simplified=True) == "H").mean(1),
               ete=np.linalg.norm(x[:, ca[0]] - x[:, ca[-1]], axis=1),
               volume=vol, density=tmass * 1.66053907e-3 / vol, total_energy=etot)
    return t, cvs
