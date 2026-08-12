#!/usr/bin/env python
"""pull_screen.py -- Nb3 reaction-coordinate bake-off. Steered MD (moving harmonic restraint) along each of 5
candidate CVs, one at a time, at BOTH 300 K (matches the unbiased reference) and 315 K (near the experimental TC5b Tm ~42 C; Neidigh & Andersen 2002),
in implicit solvent (fast, runs on M3/Colab). Bias ONE CV; judge success by INDEPENDENT referees never used
as a restraint -- Trp6 side-chain SASA (cage exposure), the Asp9-Arg16 salt bridge, and Cα RMSD. The CV that
raises SASA most (cage opens) while distorting the rest least is the production coordinate (the shipped nb03 found no 300-330 K trend and screens at a single temperature). Deliberately NOT measuring the biased CV as the success metric (avoids the
self-referential skew).

    python pull_screen.py --smoke              # 1 CV, 2 ps, Reference platform (assembly check)
    python pull_screen.py --ps 100 --platform OpenCL     # full screen
"""
import argparse
import warnings
import numpy as np
import mdtraj as md
import openmm
import openmm.app as app
from openmm import unit

PDB = "reference_10ns/structures/protein.pdb"
OUT = "scalograms/pull_screen.png"
DTPS = 0.002                                   # ps per step (2 fs)


def build_system(pdb, solvent="explicit", T=300.0, padding_nm=1.0):
    """Build the OpenMM System for the steering harness. Returns (system, topology, positions, n_protein).
    solvent='explicit': TIP3P water in a periodic box + PME + a MonteCarloBarostat (NPT) -- the honest model.
    solvent='implicit': GBn2 (no water, no box); ~10-20x faster, for pushing a CV far past what explicit can
    afford in a notebook. n_protein = the count of leading (protein) atoms; addSolvent appends water AFTER
    them, so protein atom indices (hence every CV/referee selection) are preserved and callers can slice
    [:n_protein] to analyze protein-only frames."""
    n_protein = pdb.topology.getNumAtoms()
    if solvent == "implicit":
        ff = app.ForceField("charmm36.xml", "implicit/gbn2.xml")
        system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
        return system, pdb.topology, pdb.positions, n_protein
    ff = app.ForceField("charmm36.xml", "charmm36/water.xml")
    mod = app.Modeller(pdb.topology, pdb.positions)
    mod.addSolvent(ff, model="tip3p", padding=padding_nm * unit.nanometer, neutralize=True)
    system = ff.createSystem(mod.topology, nonbondedMethod=app.PME,
                             nonbondedCutoff=1.0 * unit.nanometer, constraints=app.HBonds)
    system.addForce(openmm.MonteCarloBarostat(1.0 * unit.bar, T * unit.kelvin, 25))
    return system, mod.topology, mod.positions, n_protein


def _cv_force(cv_factory, solvent):
    """Instantiate a CV force and, in explicit (periodic) solvent, make it minimum-image aware."""
    cvf = cv_factory()
    if solvent == "explicit":
        try:
            cvf.setUsesPeriodicBoundaryConditions(True)
        except Exception:
            pass                                        # RMSDForce etc. may not support it; the protein stays whole
    return cvf


def dist_cv(g1, g2):
    f = openmm.CustomCentroidBondForce(2, "distance(g1,g2)")
    f.addGroup([int(i) for i in g1]); f.addGroup([int(i) for i in g2]); f.addBond([0, 1], [])
    return f


def referees(traj, sel):
    sasa = md.shrake_rupley(traj, mode="atom")[:, sel["trpsc"]].sum(1) * 100.0
    sb = md.compute_distances(traj, [(a, b) for a in sel["aspO"] for b in sel["argN"]]).min(1) * 10.0
    rmsd = md.rmsd(traj, sel["native"], 0, atom_indices=sel["ca"]) * 10.0
    return sasa, sb, rmsd


def selections(mdtop, native):
    """Atom-index selections for the referees (Trp6 SASA, Asp9-Arg16 salt bridge, Cα RMSD to `native`)."""
    S = lambda q: mdtop.select(q)
    return dict(trpsc=S("resSeq 6 and sidechain"), aspO=S("resSeq 9 and name OD1 OD2"),
                argN=S("resSeq 16 and name NH1 NH2 NE"), ca=S("name CA"), native=native)


def cage_cvs(mdtop, pdb_positions):
    """The 5 candidate biasing CVs (name -> (force factory, delta_nm)). indole->polyPro lid is the data-chosen
    production coordinate; salt bridge is the not-sufficient foil (see cage_cv_screen.py / pull_screen results)."""
    S = lambda q: [int(i) for i in mdtop.select(q)]
    ring = S("resSeq 6 and name CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2")
    return {
        "Trp6→helix core": (lambda: dist_cv(ring, S("resSeq 2 3 4 5 6 7 8 and name CA")), 0.6),
        "indole→polyPro lid": (lambda: dist_cv(ring, S("resSeq 17 18 19 and name CA")), 0.5),
        "Trp6 NE1→Pro17": (lambda: dist_cv(S("resSeq 6 and name NE1"), S("resSeq 17 and name CA")), 0.5),   # crisp single contact; no floppy poly-Pro centroid
        "indole→Pro18": (lambda: dist_cv(ring, S("resSeq 18 and name CB CG CD")), 0.5),   # the CORRECT indole stacker (Pro18, not Pro17); ring→ring, r(SASA)=+0.27
        "indole→Pro12": (lambda: dist_cv(ring, S("resSeq 12 and name CB CG CD")), 0.5),   # 3-10-helix proline, ANTI-correlated w/ opening (r=-0.35) -> outward ramp = negative control
        "salt bridge": (lambda: dist_cv(S("resSeq 9 and name OD1 OD2"), S("resSeq 16 and name NH1 NH2 NE")), 0.6),
        "helix span (Cα2-8)": (lambda: dist_cv(S("resSeq 2 and name CA"), S("resSeq 8 and name CA")), 0.5),
        "RMSD global": (lambda: openmm.RMSDForce(pdb_positions, S("name CA")), 0.35),
    }


def run_umbrella(pdb, cv_factory, centers, T, platform, k=2000.0, equil_ps=10.0, sample_ps=40.0, save_ps=0.5,
                 solvent="explicit", independent=False):
    """Umbrella sampling: harmonic restraint 0.5 k (cv - r0)^2 at each center r0 (nm, folded -> open); each
    window equilibrates then samples the CV. Returns per-window CV-sample arrays (nm) -> feed to mbar_pmf.
    independent=False (default): SEQUENTIAL -- one restraint dragged through `centers` in a single run (cheap,
    but the drag leaves each window out of equilibrium, biasing the PMF UPWARD and accumulating with distance).
    independent=True: reset to the folded start and re-relax in place at EACH center (no drag carry-over) --
    removes that hysteresis at ~the same cost. solvent: 'explicit' (TIP3P+PME+NPT) or 'implicit' (GB)."""
    system, topo, pos, _ = build_system(pdb, solvent, T)
    bias = openmm.CustomCVForce("0.5*k*(cv-r0)^2")
    bias.addCollectiveVariable("cv", _cv_force(cv_factory, solvent))
    bias.addGlobalParameter("k", k); bias.addGlobalParameter("r0", float(centers[0]))
    system.addForce(bias)
    integ = openmm.LangevinMiddleIntegrator(T * unit.kelvin, 1.0 / unit.picosecond, DTPS * unit.picoseconds)
    sim = app.Simulation(topo, system, integ, openmm.Platform.getPlatformByName(platform))
    sim.context.setPositions(pos); sim.minimizeEnergy(maxIterations=500 if solvent == "explicit" else 200)
    sim.context.setVelocitiesToTemperature(T * unit.kelvin)
    if solvent == "explicit":
        sim.step(int(10.0 / DTPS))                                 # ~10 ps NPT settle before the first window
    _seed = sim.context.getState(getPositions=True).getPositions()  # folded, settled start to reseed from (independent mode)
    windows = []
    for r0 in centers:
        sim.context.setParameter("r0", float(r0))
        if independent:                                            # reset to folded + relax in place -> no sequential-drag carry-over
            sim.context.setPositions(_seed)
            sim.minimizeEnergy(maxIterations=200)
            sim.context.setVelocitiesToTemperature(T * unit.kelvin)
        sim.step(int(equil_ps / DTPS))
        cvs = []
        for _ in range(int(sample_ps / save_ps)):
            sim.step(int(save_ps / DTPS))
            cvs.append(float(bias.getCollectiveVariableValues(sim.context)[0]))
        windows.append(np.array(cvs))
    return windows


def mbar_pmf(windows, centers, k, T, nbins=40):
    """Self-contained MBAR (binless free-energy reweighting) on umbrella data -> PMF. Solves the MBAR
    self-consistent equations for the per-window free energies f_k with logsumexp for stability, then
    histograms the unbiased sample weights. windows: list of CV-sample arrays (nm); centers: window centers
    (nm); k: restraint constant (kJ/mol/nm^2); T: kelvin. Returns (x_Å, pmf_kJ/mol). pymbar is the production
    drop-in; this is the transparent tutorial version."""
    from scipy.special import logsumexp
    kT = 0.00831446261815324 * T                                   # kJ/mol
    x = np.concatenate(windows); N = np.array([len(w) for w in windows], float)
    u = (0.5 * k * (x[None, :] - np.asarray(centers, float)[:, None]) ** 2) / kT   # reduced bias (K, Ntot)
    logN, f = np.log(N), np.zeros(len(centers))
    for _ in range(5000):                                          # MBAR fixed point
        logD = logsumexp((logN + f)[:, None] - u, axis=0)          # log Σ_l N_l exp(f_l - u_l,i)
        fn = -logsumexp(-u - logD[None, :], axis=1); fn -= fn[0]
        if np.max(np.abs(fn - f)) < 1e-9:
            f = fn; break
        f = fn
    logw = -logsumexp((logN + f)[:, None] - u, axis=0)             # unnormalized log unbiased weights
    edges = np.linspace(x.min(), x.max(), nbins + 1); mids = 0.5 * (edges[:-1] + edges[1:])
    ib = np.clip(np.digitize(x, edges) - 1, 0, nbins - 1)
    logP = np.array([logsumexp(logw[ib == b]) if np.any(ib == b) else -np.inf for b in range(nbins)])
    pmf = -kT * logP; pmf -= np.nanmin(pmf[np.isfinite(pmf)])
    pmf[~np.isfinite(pmf)] = np.nan                                # unsampled bins -> nan (not +inf) so plots gap cleanly
    return mids * 10.0, pmf


def mbar_pmf_bootstrap(windows, centers, k, T, nbins=40, n_boot=20, seed=0):
    """PMF (mbar_pmf) plus a BOOTSTRAP error bar: resample each window's frames with replacement n_boot
    times, recompute the PMF, take the per-bin std. A transparent, overlap-honest uncertainty -- preferred
    over any solver's built-in ASYMPTOTIC error bar, which assumes good overlap + uncorrelated samples and
    goes optimistic exactly when a run is short/poorly-overlapped (Klimovich, Shirts & Mobley 2015).
    Returns (x_Å, pmf, err) in kJ/mol."""
    x, pmf = mbar_pmf(windows, centers, k, T, nbins)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        rw = [w[rng.integers(0, len(w), len(w))] for w in windows]
        boots.append(mbar_pmf(rw, centers, k, T, nbins)[1])
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)           # unsampled edge bins are all-nan columns -> err=nan there
        err = np.nanstd(np.array(boots), axis=0)
    return x, pmf, err


def mbar_pmf_pymbar(windows, centers, k, T, nbins=40):
    """PMF via pymbar's FES (histogram method) with pymbar's ANALYTICAL uncertainties -- the production path
    now that pymbar's FES works (4.2.x; the histogram-KeyError / no-uncertainty bugs that forced the
    hand-rolled fallback were 4.0.3). Returns (x_Å, pmf_kJ/mol, err_kJ/mol). windows: per-window CV arrays
    (nm); centers (nm); k (kJ/mol/nm^2); T (K). Unsampled bins come back nan so plots gap cleanly. The
    transparent mbar_pmf above is kept as a readable reference implementation (validated equal to pymbar)."""
    import io, contextlib
    kT = 0.00831446261815324 * T
    x = np.concatenate(windows); Nk = np.array([len(w) for w in windows])
    u_kn = (0.5 * k * (x[None, :] - np.asarray(centers, float)[:, None]) ** 2) / kT
    edges = np.linspace(x.min(), x.max(), nbins + 1); mids = 0.5 * (edges[:-1] + edges[1:])
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")                            # mute pymbar's JAX banner + timeseries note
        from pymbar import FES                                     # import INSIDE the redirect (banner prints on import)
        fes = FES(u_kn, Nk, verbose=False)
        fes.generate_fes(np.zeros(len(x)), x, fes_type="histogram", histogram_parameters={"bin_edges": edges})
        res = fes.get_fes(mids, reference_point="from-lowest", uncertainty_method="analytical")
    f = np.asarray(res["f_i"], float); df = np.asarray(res.get("df_i", np.full_like(f, np.nan)), float)
    f = np.where(np.isfinite(f), f, np.nan)
    return mids * 10.0, f * kT, df * kT


def mbar_overlap(windows, centers, k, T):
    """pymbar's window OVERLAP matrix -- whether adjacent umbrella windows sampled enough common CV range to
    be connected by MBAR. Returns (matrix, min_adjacent_overlap); a min above ~0.03 means the windows chain
    folded->open. pymbar is a tutorial-env dependency (environment.yml); the notebook on-ramp provisions it
    exactly like openmm/mdtraj, so it's present in the 'MD tutorial' kernel. pymbar dumps a JAX 64-bit banner
    and a timeseries note to stdout on first use; we mute that noise here (real exceptions still propagate)."""
    import io, contextlib
    kT = 0.00831446261815324 * T
    x = np.concatenate(windows); Nk = np.array([len(w) for w in windows])
    u_kn = (0.5 * k * (x[None, :] - np.asarray(centers, float)[:, None]) ** 2) / kT
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from pymbar import MBAR
        O = MBAR(u_kn, Nk).compute_overlap()["matrix"]
    adj = np.array([O[i, i + 1] for i in range(len(centers) - 1)])
    return O, float(adj.min())


def reference_pmf(cv_per_seed, T, nbins=30):
    """Honest 1-D free-energy profile  -k_BT ln P(CV)  from UNBIASED reference trajectories -- the single
    source for BOTH the 3.0 motivation curve and the 3.5 cross-check, so the two can never drift.
    cv_per_seed: list of per-seed CV arrays (Å). Pools all seeds for the central curve, then builds an
    uncertainty BAND that (a) combines seed-to-seed disagreement with the per-bin counting error k_BT/sqrt(N),
    (b) FLARES where few of the seeds reach a bin (no consensus -> low confidence), and (c) is floored in the
    1-2 seed tail so it can never look tight where only one seed strays; finally masks the undersampled tails
    CONTIGUOUSLY (walk out from the densest bin) so the sampled curve has no interior gaps -- which is what
    stops a lone undersampled bin from posing as a phantom metastable basin. Returns
    (mids_Å, G_kJ, band_kJ, wall_idx): G zeroed at the folded minimum, NaN outside the sampled range;
    wall_idx = index of the right edge of the sampled range (the 'wall')."""
    from scipy.ndimage import gaussian_filter1d
    kT = 0.00831446261815324 * T
    allcv = np.concatenate(cv_per_seed)
    edges = np.linspace(allcv.min(), allcv.max(), nbins + 1); mids = 0.5 * (edges[:-1] + edges[1:])
    cnt = np.histogram(allcv, bins=edges)[0]
    G = -kT * np.log(np.histogram(allcv, bins=edges, density=True)[0] + 1e-12); G -= np.nanmin(G[cnt >= 5])
    FEs = []                                                        # per-seed -k_BT lnP, each aligned to its own well
    for v in cv_per_seed:
        gg = -kT * np.log(np.histogram(v, bins=edges, density=True)[0] + 1e-12)
        gg[np.histogram(v, bins=edges)[0] < 5] = np.nan
        if np.isfinite(gg).any(): gg -= np.nanmin(gg)
        FEs.append(gg)
    FEa = np.array(FEs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)            # bins with <2 finite seeds -> DOF<=0
        std = np.nanstd(FEa, axis=0)
    nseed = np.sum(np.isfinite(FEa), axis=0); ntot = len(cv_per_seed)
    cerr = kT / np.sqrt(np.maximum(cnt, 1))                         # counting error on -k_BT lnP (kJ/mol)
    band = np.sqrt(np.nan_to_num(std) ** 2 + cerr ** 2) * np.sqrt(ntot / np.maximum(nseed, 1))  # flare where seeds thin
    band[nseed <= 2] = np.maximum(band[nseed <= 2], 1.3 * kT)       # unvalidated tail: never let it look tight
    band = gaussian_filter1d(band, 1.2)                             # smooth -> a continuous band, not chunky spikes
    thr = max(20, len(allcv) // 600); i0 = int(np.argmax(cnt)); il = ir = i0
    while ir + 1 < len(cnt) and cnt[ir + 1] >= thr: ir += 1         # contiguous sampled range (no interior gaps)
    while il - 1 >= 0 and cnt[il - 1] >= thr: il -= 1
    G[:il] = np.nan; G[ir + 1:] = np.nan
    return mids, G, band, ir


def integrated_autocorr_time(x):
    """Integrated autocorrelation time tau = 1/2 + sum_k rho(k) (frames), Geyer (1992) initial-positive-
    sequence PAIR rule -- the SAME estimator as mdtutorial / md_scalogram.integrated_autocorr_time, so nb03's
    §3.6 salt-bridge aside quotes the same quantity nb02 uses for N_eff. Keep the three in sync if any is
    changed. Linear observable (a distance); the circular/vector variant lives in md_scalogram."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x); v = np.dot(x, x) / n
    if v == 0:
        return 0.5
    rho = lambda k: np.dot(x[:-k], x[k:]) / (n * v)
    tau = 0.5
    for k in range(1, n - 1, 2):
        pair = rho(k) + rho(k + 1)
        if pair <= 0:
            break
        tau += pair
    return tau


def run_pull(pdb, mdtop, cv_factory, delta, T, n_ps, platform, sel, k=2000.0, save_ps=2.0, solvent="explicit"):
    system, topo, pos0, n_protein = build_system(pdb, solvent, T)
    bias = openmm.CustomCVForce("0.5*k*(cv-r0)^2")
    bias.addCollectiveVariable("cv", _cv_force(cv_factory, solvent))
    bias.addGlobalParameter("k", k); bias.addGlobalParameter("r0", 0.0)
    system.addForce(bias)
    integ = openmm.LangevinMiddleIntegrator(T * unit.kelvin, 1.0 / unit.picosecond, DTPS * unit.picoseconds)
    sim = app.Simulation(topo, system, integ, openmm.Platform.getPlatformByName(platform))
    sim.context.setPositions(pos0)
    sim.minimizeEnergy(maxIterations=500 if solvent == "explicit" else 200)
    r0 = float(bias.getCollectiveVariableValues(sim.context)[0])           # folded CV value (nm)
    sim.context.setParameter("r0", r0); sim.context.setVelocitiesToTemperature(T * unit.kelvin)
    if solvent == "explicit":
        sim.step(int(10.0 / DTPS))                                         # NPT water settle at the folded restraint
    nchunk = max(1, int(n_ps / save_ps)); per = max(1, int(save_ps / DTPS))
    pos = []
    for c in range(nchunk):
        sim.context.setParameter("r0", r0 + (c + 1) / nchunk * delta)      # ramp the restraint center out
        sim.step(per)
        allpos = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        pos.append(allpos[:n_protein])                                     # protein-only frame (water dropped for referees/viewer)
    traj = md.Trajectory(np.array(pos), mdtop)
    return (*referees(traj, sel), traj)


def run_unbiased(pdb, mdtop, T, n_ps, platform, save_ps=2.0, solvent="explicit"):
    """A short UNBIASED run from the folded structure -> md.Trajectory. The 'nothing happens' / cage-stays-
    shut panel for the side-by-side, in the SAME solvent setup as the steered pull (explicit or implicit)."""
    system, topo, pos0, n_protein = build_system(pdb, solvent, T)
    integ = openmm.LangevinMiddleIntegrator(T * unit.kelvin, 1.0 / unit.picosecond, DTPS * unit.picoseconds)
    sim = app.Simulation(topo, system, integ, openmm.Platform.getPlatformByName(platform))
    sim.context.setPositions(pos0); sim.minimizeEnergy(maxIterations=500 if solvent == "explicit" else 200)
    sim.context.setVelocitiesToTemperature(T * unit.kelvin)
    if solvent == "explicit":
        sim.step(int(10.0 / DTPS))                                         # NPT water settle
    pos = []
    for _ in range(int(n_ps / save_ps)):
        sim.step(int(save_ps / DTPS))
        allpos = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        pos.append(allpos[:n_protein])
    return md.Trajectory(np.array(pos), mdtop)


def save_aligned(traj, native, path, ref_sel="name CA and resSeq 2 to 8"):
    """Superpose a trajectory onto `native` using a STABLE reference selection (default: the alpha-helix Cα,
    residues 2-8) so the folded scaffold stays fixed and only the cage-opening motion -- the Trp6 sidechain
    swinging out -- moves. Superposing on ALL Cα lets the opening wobble the whole body (least-squares fit
    compromises across the moving cage), so the ghost/animation reads as the molecule drifting rather than
    Trp6 flipping out. Writes a multi-frame PDB for an animated py3Dmol view."""
    t = traj[:]
    idx = native.topology.select(ref_sel)
    t.superpose(native, 0, atom_indices=idx, ref_atom_indices=idx)
    t.save_pdb(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ps", type=float, default=100.0); ap.add_argument("--platform", default="OpenCL")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    pdb = app.PDBFile(PDB)
    mdtop = md.load(PDB).topology
    sel = selections(mdtop, md.load(PDB))
    CVS = cage_cvs(mdtop, pdb.positions)
    temps = [300, 315]
    if a.smoke:
        CVS = {"Trp6→helix core": CVS["Trp6→helix core"]}; temps = [300]; a.ps = 2.0; a.platform = "Reference"

    results = {}
    for name, (fac, delta) in CVS.items():
        for T in temps:
            sasa, sb, rmsd, traj = run_pull(pdb, mdtop, fac, delta, T, a.ps, a.platform, sel, solvent="implicit")
            results[(name, T)] = (sasa, sb, rmsd, traj)
            print(f"{name:20s} {T}K:  Trp6 SASA {sasa[0]:.0f}→{sasa.max():.0f} Å²   "
                  f"salt bridge →{sb[-1]:.1f} Å   RMSD →{rmsd[-1]:.1f} Å")
    if a.smoke:
        print("smoke OK"); return

    OPEN_CUT, BROKEN = 67.0, 5.0                        # cage "open" = past the unbiased ceiling; salt bridge "broken"
    print("\ncoupling under biasing — P(salt bridge broken | cage open) and Pearson r(SASA, salt bridge):")
    for name in CVS:
        sasa = np.concatenate([results[(name, T)][0] for T in temps])
        sb = np.concatenate([results[(name, T)][1] for T in temps])
        op = sasa > OPEN_CUT
        frac = (sb[op] > BROKEN).mean() * 100 if op.any() else float("nan")
        print(f"  {name:20s} open {op.mean()*100:3.0f}% of frames | P(sb broken | open)={frac:3.0f}% | r(SASA,sb)={np.corrcoef(sasa,sb)[0,1]:+.2f}")
    import os
    os.makedirs("es_structures", exist_ok=True)                     # for the notebook's py3Dmol side-by-side
    sel["native"].save("es_structures/trp6_folded.pdb")
    win = "indole→polyPro lid"; bT = max(temps, key=lambda T: results[(win, T)][0].max())
    jw = int(np.argmax(results[(win, bT)][0])); results[(win, bT)][3][jw].save("es_structures/trp6_open_polyPro.pdb")
    print(f"saved es_structures/{{trp6_folded, trp6_open_polyPro}}.pdb  (open SASA {results[(win,bT)][0][jw]:.0f} Å², {win} @ {bT}K)")

    fig, ax = plt.subplots(len(CVS), 1, figsize=(7, 2.2 * len(CVS)), sharex=True, layout="constrained")
    for i, name in enumerate(CVS):
        for T, ls in [(300, "-"), (315, "--")]:
            sasa = results[(name, T)][0]
            ax[i].plot(np.linspace(0, 1, len(sasa)), sasa, ls, label=f"{T} K")
        ax[i].axhline(67, color="0.6", ls=":", lw=1)      # the unbiased 10 ns max exposure
        ax[i].set_ylabel("Trp6 SASA (Å²)", fontsize=8); ax[i].set_title(name, fontsize=9, loc="left")
        ax[i].legend(fontsize=7)
    ax[-1].set_xlabel("pull progress (fraction)")
    fig.suptitle("Pull screen — which CV opens the cage (Trp6 SASA, independent of the biased CV)? "
                 "dotted = unbiased 10 ns ceiling (67 Å²)", fontsize=10)
    fig.savefig(OUT, dpi=120, bbox_inches="tight"); print(f"\nwrote {OUT}")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    main()
