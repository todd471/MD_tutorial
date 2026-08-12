"""mdtviz.py -- molecular visualization for the tutorial figures: interactive py3Dmol views + headless,
license-free, ray-traced open-source PyMOL panels. Ported from the v1 tutorial so the figure notebooks
stay short. Everything degrades gracefully: py3Dmol needs only pip; the PyMOL panels SKIP (never crash) if
PyMOL isn't available, and the simulation + py3Dmol views are unaffected.

Typical use in a figure notebook:
    import mdtviz
    PYMOL = mdtviz.setup_pymol()                       # find (or on Colab provision) a working PyMOL
    mdtviz.view_ensemble("structures/1L2Y.pdb")        # interactive NMR ensemble
    b, a = mdtviz.hydrogen_sticks(before, after, out)  # before/after H-addition sticks (PNG paths, or None)
    panels = mdtviz.cartoon_panels(stage2, stage3, out)
"""
import os
import sys
import glob
import shutil
import subprocess


# ============================================================ PyMOL discovery / provisioning
def _pymol_works(binpath):
    """A candidate PyMOL counts only if it actually LAUNCHES (some HPC /usr/bin/pymol stubs do not)."""
    try:
        r = subprocess.run([binpath, "-cq", "-d", "print('pymol-ok')"],
                           capture_output=True, text=True, timeout=90)
        return r.returncode == 0 and "pymol-ok" in (r.stdout or "")
    except Exception:
        return False


def find_pymol():
    """Return the path to a working headless PyMOL, or None. Prefers the active env, then common conda
    env locations, then PATH (a possibly-broken /usr/bin/pymol is tried LAST and validated)."""
    cands = []
    if os.environ.get("PYMOL_BIN"):
        cands.append(os.path.expanduser(os.environ["PYMOL_BIN"]))
    cands.append(os.path.join(sys.prefix, "bin", "pymol"))         # the ACTIVE env first
    for pat in ("~/.conda/envs/*/bin/pymol", "~/anaconda3/envs/*/bin/pymol", "~/miniconda3/envs/*/bin/pymol",
                "~/mambaforge/envs/*/bin/pymol", "~/opt/anaconda3/envs/*/bin/pymol"):
        cands += glob.glob(os.path.expanduser(pat))
    w = shutil.which("pymol")
    if w:
        cands.append(w)
    cands += ["/opt/homebrew/bin/pymol", "/usr/local/bin/pymol"]
    for c in cands:
        if c and os.path.exists(c) and _pymol_works(c):
            return c
    return None


def _is_colab():
    return "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ or os.path.isdir("/content")


def _install_pymol_micromamba():
    """Colab: a self-contained pymol-open-source (its OWN Python) via micromamba -> we invoke the BINARY
    only, never touching the notebook's Python/OpenMM. (apt PyMOL 2.5 is dead on Colab's Python 3.12.)"""
    prefix, binp = "/content/pmenv", "/content/pmenv/bin/pymol"
    if not os.path.exists(binp):
        print("installing pymol-open-source via micromamba (~1-2 min, one-time this session)...")
        subprocess.run("curl -Ls --max-time 180 https://micro.mamba.pm/api/micromamba/linux-64/latest "
                       "| tar -xj bin/micromamba", shell=True)
        subprocess.run("MAMBA_ROOT_PREFIX=/content/mamba ./bin/micromamba create -y -p " + prefix +
                       " -c conda-forge pymol-open-source", shell=True)
    return binp if os.path.exists(binp) else None


def setup_pymol(verbose=True):
    """Find a working headless PyMOL; on Colab, auto-provision one. Returns the binary path (also stored in
    $PYMOL_BIN) or "" if none is available. When "", the cartoon panels skip gracefully. Local/HPC are NOT
    auto-installed (avoids offline-node hangs) -- install pymol-open-source (conda) or `module load pymol`."""
    pymol = find_pymol()
    if pymol is None and _is_colab():
        if sys.version_info < (3, 12) and shutil.which("apt-get"):
            subprocess.run("apt-get -qq update && apt-get -qq install -y pymol", shell=True); pymol = find_pymol()
        if pymol is None:
            c = _install_pymol_micromamba()
            if c:
                os.environ["PYMOL_BIN"] = c; pymol = find_pymol()
    os.environ["PYMOL_BIN"] = pymol or ""
    if verbose:
        if pymol:
            print("pymol:", pymol)
        else:
            print("pymol: NOT AVAILABLE -- cartoon panels SKIPPED (simulation + interactive py3Dmol views unaffected).")
            print("       enable:  conda install -c conda-forge pymol-open-source  (local)  |  module load pymol  (HPC)")
    return pymol or ""


def render_pymol(script, pymol=None):
    """Run a headless PyMOL script; True on success, else print a short note and return False (never
    crashes). `pymol` defaults to $PYMOL_BIN (set by setup_pymol)."""
    pymol = pymol or os.environ.get("PYMOL_BIN") or ""
    if not pymol:
        print(f"[skip] PyMOL unavailable -- skipping {script}. Install pymol-open-source to enable this panel.")
        return False
    r = subprocess.run([pymol, "-cq", script], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[skip] PyMOL failed on {script} (exit {r.returncode}) -- skipping this panel. Last output:")
        for ln in (r.stderr or r.stdout or "").strip().splitlines()[-3:]:
            print("    ", ln)
        return False
    return True


# ============================================================ interactive py3Dmol views
def view_ensemble(pdb_path, width=480, height=360):
    """Animated py3Dmol view of an NMR ensemble. The spread across models reflects both genuine conformational
    heterogeneity AND how tightly the data restrain each region (sparsely-restrained regions look floppy
    regardless) -- a hint at flexibility, not a direct measurement. Returns the view (call in a cell to display)."""
    import py3Dmol
    view = py3Dmol.view(width=width, height=height)
    view.addModelsAsFrames(open(pdb_path).read(), "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})              # N(blue) -> C(red)
    view.animate({"loop": "forward"}); view.zoomTo()
    return view


def view_solvated(pdb_path, width=480, height=360):
    """py3Dmol view of a finished solvated system: protein cartoon + faint waters + ion spheres."""
    import py3Dmol
    view = py3Dmol.view(width=width, height=height)
    view.addModel(open(pdb_path).read(), "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.addStyle({"resn": ["HOH", "WAT"], "atom": "O"},                # water oxygens as a faint haze. py3Dmol
                  {"sphere": {"scale": 0.15, "opacity": 0.45, "color": "0x9ecae1"}})    # sphere opacity is unreliable
                  # (WebGL transparency sorting != PyMOL), so SMALL scale + light color carry the "faint" look
    view.addStyle({"resn": ["NA", "CL", "K"]}, {"sphere": {"scale": 0.5, "color": "green"}})
    view.zoomTo()
    return view


def _first_model_pdb(pdb_text):
    """Return just the first MODEL...ENDMDL block (with any leading header) of a multi-frame PDB -- a static
    single-frame structure to use as a 'ghost' of the starting conformation."""
    out, started = [], False
    for ln in pdb_text.splitlines():
        if ln.startswith("MODEL"):
            if started:
                break                                       # reached the 2nd frame -> stop
            started = True
        out.append(ln)
        if ln.startswith("ENDMDL"):
            break
    return "\n".join(out) + "\n"


def _cv_frames_pdb(traj, cv_pairs, n_dots=13):
    """Build a multi-frame PDB (one MODEL per frame) that renders each CV as an animated DOTTED line: for every
    cv_pair, sample n_dots+2 points along the centroid->centroid segment as pseudo-atoms (endpoints tagged
    resn CVE, the interior dots CVL). Added to a py3Dmol view as its OWN frame-model it animates in lockstep
    with the trajectory. We draw the line AS dots rather than a bond because py3Dmol will NOT render a CONECT
    stick across a multi-frame model (the endpoints show but the connector doesn't) -- and dots give the dashed
    look for free. cv_pairs: list of (sel_a_mdtraj, sel_b_mdtraj, color_hex) -- same format as cv_groups_view."""
    sels = [(traj.topology.select(a), traj.topology.select(b)) for a, b, _ in cv_pairs]
    out = []
    for f in range(traj.n_frames):
        out.append(f"MODEL {f + 1:>8}")
        serial = 1
        for ia, ib in sels:
            ca = traj.xyz[f, ia].mean(0) * 10.0                     # nm -> Å
            cb = traj.xyz[f, ib].mean(0) * 10.0
            for k in range(n_dots + 2):
                t = k / (n_dots + 1)
                p = ca * (1 - t) + cb * t
                resn = "CVE" if k in (0, n_dots + 1) else "CVL"     # endpoints vs line dots
                out.append(f"HETATM{serial:>5} {'X':<4} {resn} X{serial:>4}    "
                           f"{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00  0.00           C")
                serial += 1
        out.append("ENDMDL")
    return "\n".join(out) + "\n"


def dual_view(pdb_left, pdb_right, highlight_resi=6, width=780, height=420, animate=True, ghost=True,
              cv_pairs=None, cv_dots=7):
    """Two py3Dmol panels side by side. If the PDBs are multi-frame trajectories (e.g. from
    pull_screen.save_aligned, superposed so both start in the folded orientation) they ANIMATE -- watch the
    cage stay shut (left) vs crack open (right), Trp6 drawn as orange sticks so the opening is obvious. NO
    in-viewer labels (they float over the side chain and obscure it); caption the two panels in plain text
    above the cell instead.

    ghost=True leaves a faint STATIC copy of each panel's first frame underneath the animation (added as a
    separate single-frame model, so it never animates), so you can see how far the moving structure has
    drifted from where it started. py3Dmol/WebGL transparency is unreliable, so the ghost reads faint mainly
    via a light-grey color (opacity is a bonus, not load-bearing). Each model is styled with {"model": -1}
    right after it's added, so '-1' always means 'the model just added' regardless of grid indexing."""
    import py3Dmol
    view = py3Dmol.view(width=width, height=height, viewergrid=(1, 2))
    for col, path in enumerate([pdb_left, pdb_right]):
        text = open(path).read()
        view.addModelsAsFrames(text, "pdb", viewer=(0, col))                              # animated trajectory
        view.setStyle({"model": -1}, {"cartoon": {"color": "spectrum"}}, viewer=(0, col))  # N(blue)->C(red)
        view.addStyle({"model": -1, "resi": str(highlight_resi)},
                      {"stick": {"colorscheme": "orangeCarbon", "radius": 0.25}}, viewer=(0, col))
        if ghost:
            view.addModel(_first_model_pdb(text), "pdb", viewer=(0, col))                  # static frame-0 ghost
            view.setStyle({"model": -1}, {"cartoon": {"color": "0xd9d9d9", "opacity": 0.55}}, viewer=(0, col))
            view.addStyle({"model": -1, "resi": str(highlight_resi)},
                          {"stick": {"color": "0xb0b0b0", "radius": 0.16, "opacity": 0.6}}, viewer=(0, col))
        if cv_pairs:                                                                    # animated CV overlay
            import mdtraj as _md
            cvtext = _cv_frames_pdb(_md.load(path), cv_pairs, n_dots=cv_dots)
            view.addModelsAsFrames(cvtext, "pdb", viewer=(0, col))
            view.setStyle({"model": -1, "resn": "CVL"},                                  # dashed line = row of dots
                          {"sphere": {"radius": 0.12, "color": "0x111111"}}, viewer=(0, col))
            view.setStyle({"model": -1, "resn": "CVE"},                                  # the two centroid endpoints
                          {"sphere": {"radius": 0.30, "color": "0x111111"}}, viewer=(0, col))
    view.zoomTo()
    if animate:
        view.animate({"loop": "forward", "interval": 100})
    return view


def overlay_view(pdb_a, pdb_b, highlight_resi=6, width=520, height=420):
    """Superimpose two SINGLE-frame structures (e.g. the first and last frame of a long unbiased run, already
    aligned) in ONE py3Dmol viewer: both backbones as faint grey cartoon, residue `highlight_resi` drawn as
    sticks in two colors (blue = A / first, orange = B / last). If the residue barely moved, the two stick
    sets nearly coincide -- that's the '50x longer, still folded' point, seen rather than quoted. Caption
    A/B in plain text above the cell (no in-viewer labels -- they float over the side chain)."""
    import py3Dmol
    view = py3Dmol.view(width=width, height=height)
    view.addModel(open(pdb_a).read(), "pdb")                       # model 0 = A (first)
    view.addModel(open(pdb_b).read(), "pdb")                       # model 1 = B (last)
    view.setStyle({"model": 0}, {"cartoon": {"color": "0xcfcfcf", "opacity": 0.55}})
    view.setStyle({"model": 1}, {"cartoon": {"color": "0xcfcfcf", "opacity": 0.55}})
    view.addStyle({"model": 0, "resi": str(highlight_resi)}, {"stick": {"colorscheme": "blueCarbon", "radius": 0.3}})
    view.addStyle({"model": 1, "resi": str(highlight_resi)}, {"stick": {"colorscheme": "orangeCarbon", "radius": 0.3}})
    view.zoomTo()
    return view


def cv_groups_view(pdb_path, groups, cv_pairs=None, width=520, height=420):
    """Show ONE folded structure (faint grey cartoon) with named residue selections as colored sticks -- WHERE
    each collective variable's atom groups sit -- and, if `cv_pairs` is given, the CVs THEMSELVES as
    centroid-to-centroid **dashed distances**: a dot at each group's center + a dashed line between them. That
    makes 'the distance we push on' literal AND shows a reader the CV endpoints are *computed centers* (the
    biasing force is spread over the whole group by mass), not single atoms. `groups` is a list of
    (resi_selection, colorscheme), e.g. [('6', 'orangeCarbon'), ('17-19', 'blueCarbon')]. `cv_pairs` is a list
    of (mdtraj_sel_a, mdtraj_sel_b, color_hex): each end is the centroid of that atom selection, e.g.
    ('resSeq 6 and name CG CD1 ...', 'resSeq 17 18 19 and name CA', '0x333333'). Caption colors in text above."""
    import py3Dmol
    view = py3Dmol.view(width=width, height=height)
    view.addModel(open(pdb_path).read(), "pdb")
    view.setStyle({"cartoon": {"color": "0xdcdcdc"}})
    for resi, colorscheme in groups:
        view.addStyle({"resi": str(resi)}, {"stick": {"colorscheme": colorscheme, "radius": 0.22}})
    if cv_pairs:
        import mdtraj as md
        t = md.load(pdb_path)
        for sel_a, sel_b, color in cv_pairs:
            ca = t.xyz[0, t.topology.select(sel_a)].mean(0) * 10.0     # group centroid, nm -> Å (PDB/py3Dmol units)
            cb = t.xyz[0, t.topology.select(sel_b)].mean(0) * 10.0
            for c in (ca, cb):
                view.addSphere({"center": {"x": float(c[0]), "y": float(c[1]), "z": float(c[2])},
                                "radius": 0.35, "color": color})
            view.addLine({"start": {"x": float(ca[0]), "y": float(ca[1]), "z": float(ca[2])},
                          "end":   {"x": float(cb[0]), "y": float(cb[1]), "z": float(cb[2])},
                          "color": color, "dashed": True})
    view.zoomTo()
    return view


# ============================================================ headless PyMOL rendered panels
def hydrogen_sticks(before_pdb, after_pdb, out_dir, sel="resi 4-6",
                    ring_sel="resi 4 and name CG+CD1+CD2+CE1+CE2+CZ", pymol=None):
    """Before/after stick close-up of the hydrogens the repair added: heavy atoms only vs. + hydrogens
    (teal). Renders two PNGs from the SAME camera. Returns (before_png, after_png) or (None, None) if
    PyMOL is unavailable. Default selection targets ubiquitin residues 4-6 with the PHE4 ring face-on."""
    os.makedirs(out_dir, exist_ok=True)
    before_png = os.path.join(out_dir, "sticks_before.png")
    after_png = os.path.join(out_dir, "sticks_after.png")
    script = os.path.join(out_dir, "_sticks.py")
    with open(script, "w") as fh:
        fh.write(f"""from pymol import cmd
cmd.bg_color("white"); cmd.set("ray_opaque_background", 0); cmd.set("ray_shadows", 0); cmd.set("antialias", 2)
cmd.set("stick_radius", 0.14); cmd.set("valence", 0)
SEL = {sel!r}
cmd.load({before_pdb!r}, "before"); cmd.load({after_pdb!r}, "after")
cmd.hide("everything")
for o in ("before", "after"):
    cmd.show("sticks", o + " and " + SEL)
    cmd.color("grey60", o + " and " + SEL + " and elem C")
    cmd.color("blue",   o + " and " + SEL + " and elem N")
    cmd.color("red",    o + " and " + SEL + " and elem O")
    cmd.color("0x4DB6AC", o + " and " + SEL + " and elem H")   # teal H: visible + colorblind-safe vs red O
cmd.orient("after and {ring_sel}")
cmd.turn("x", 8); cmd.zoom("after and " + SEL, buffer=1.5)
V = cmd.get_view()
cmd.disable("after"); cmd.set_view(V); cmd.ray(700, 700); cmd.png({before_png!r}, dpi=150)
cmd.enable("after"); cmd.disable("before"); cmd.set_view(V); cmd.ray(700, 700); cmd.png({after_png!r}, dpi=150)
""")
    if render_pymol(script, pymol) and os.path.exists(after_png):
        return before_png, after_png
    return None, None


def cartoon_panels(stage2_pdb, stage3_pdb, out_dir, pymol=None):
    """The three build-stage panels: (1) repaired peptide in SCHEME 7 (rainbow cartoon + ball-and-stick +
    light translucent surface); (2) & (3) the
    SAME scheme-7 rainbow rendering (rainbow cartoon + rainbow ball-and-stick) of the solvated peptide + translucent grey surface inside the FULL periodic box --
    panel 2 empty box, panel 3 filled with water + ion(s). Panels 2 & 3 share one camera, zoomed out so the
    whole box is in frame (shows how small the solute is relative to the solvent). Returns {name: png} for
    whichever rendered (empty dict if PyMOL absent)."""
    os.makedirs(out_dir, exist_ok=True)
    out = {"panel1_repair": os.path.join(out_dir, "panel1_repair.png"),
           "panel2_box": os.path.join(out_dir, "panel2_box.png"),
           "panel3_solvated": os.path.join(out_dir, "panel3_solvated.png")}
    script = os.path.join(out_dir, "_cartoons.py")
    with open(script, "w") as fh:
        fh.write(f"""from pymol import cmd
from pymol.cgo import LINEWIDTH, BEGIN, LINES, COLOR, VERTEX, END
W = 1000
cmd.bg_color("white"); cmd.set("ray_opaque_background", 0)
cmd.set("cartoon_fancy_helices", 1); cmd.set("cartoon_highlight_color", "grey50")
cmd.set("ray_shadows", 0); cmd.set("antialias", 2); cmd.set("orthoscopic", 1); cmd.set("stick_radius", 0.14)
cmd.set("transparency_mode", 1); cmd.set("surface_quality", 1)
def spec(sel): cmd.spectrum("count", "rainbow", sel + " and name CA")   # N(blue)->C(red)
def add_surface(sel):
    cmd.show("surface", sel); cmd.set("transparency", 0.85, sel); cmd.set("surface_color", "grey70", sel)
def box(mn, mx, name, color=(0, 0, 0), lw=2.5):
    x0, y0, z0 = mn; x1, y1, z1 = mx
    c = [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0),(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)]
    e = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    o = [LINEWIDTH, lw, BEGIN, LINES, COLOR, *color]
    for a, b in e: o += [VERTEX, *c[a], VERTEX, *c[b]]
    o += [END]; cmd.load_cgo(o, name)
# Panel 1: SCHEME 7 -- rainbow cartoon + ball-and-stick (all atoms) + light grey translucent surface (0.85)
cmd.load({stage2_pdb!r}, "pep"); cmd.hide("everything")
cmd.show("cartoon", "pep"); cmd.show("sticks", "pep"); cmd.set("stick_radius", 0.09, "pep")
cmd.show("spheres", "pep"); cmd.set("sphere_scale", 0.2, "pep")
cmd.spectrum("count", "rainbow", "pep")                       # everything rainbow (cartoon + atoms), N->C
cmd.show("surface", "pep"); cmd.set("surface_color", "grey70", "pep"); cmd.set("transparency", 0.85, "pep")
cmd.orient("pep"); cmd.turn("y", 20); cmd.turn("x", 20)
cmd.zoom("pep", buffer=2.5, complete=1)   # +buffer + complete=1 so the whole translucent SURFACE fits in frame
cmd.ray(W, W); cmd.png({out['panel1_repair']!r}, dpi=150)
# Panels 2 & 3 share the SAME solvated system, box, and zoom -> identical molecule size
cmd.load({stage3_pdb!r}, "sol"); cmd.hide("everything", "sol"); cmd.disable("pep")
cmd.show("cartoon", "sol and polymer"); cmd.show("sticks", "sol and polymer"); cmd.set("stick_radius", 0.09, "sol and polymer")
cmd.show("spheres", "sol and polymer"); cmd.set("sphere_scale", 0.2, "sol and polymer")
cmd.spectrum("count", "rainbow", "sol and polymer"); add_surface("sol and polymer")   # SCHEME 7: rainbow ribbon + rainbow all-atom (matches panel 1)
cmd.orient("sol and polymer"); cmd.turn("y", 20); cmd.turn("x", 20)
es = cmd.get_extent("sol"); box(es[0], es[1], "cellbox", color=(0.2, 0.2, 0.2), lw=2.0)
cmd.zoom("cellbox", buffer=6, complete=1); cmd.clip("slab", 300)
cmd.ray(W, W); cmd.png({out['panel2_box']!r}, dpi=150)
# Panel 3: reveal water + ion (identical view)
cmd.show("spheres", "sol and resn HOH+WAT and name O")
cmd.set("sphere_scale", 0.28, "sol and resn HOH+WAT"); cmd.set("sphere_transparency", 0.72, "sol and resn HOH+WAT")
cmd.color("marine", "sol and resn HOH+WAT")
cmd.show("spheres", "sol and resn CL+NA+K"); cmd.color("green", "sol and resn CL")
cmd.set("sphere_scale", 0.6, "sol and resn CL+NA+K")
cmd.ray(W, W); cmd.png({out['panel3_solvated']!r}, dpi=150)
""")
    render_pymol(script, pymol)
    return {k: v for k, v in out.items() if os.path.exists(v)}


def _circmean_deg(A):
    import numpy as np
    r = np.radians(A); return np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean()))


def _recenter_deg(A, center):
    import numpy as np
    return center + (np.asarray(A) - center + 180) % 360 - 180


def player(c, label="repeat 1"):
    """Synchronized molecule + collective-variable animation -- one play button, one timeline. `c` is a
    compute_cvs() dict (needs t, ps, rmsd, gln5, asp9, ile4). Backbone coloured N->C; the two alpha-helix
    H-bonds (GLN5 N-H..ASN1 O=C orange, ASP9 N-H..GLN5 O=C blue) are dashed lines that go faint-dotted when
    they break (>2.5 A); a shared cursor marks the current moment on every trace. Long runs are subsampled
    to <=200 embedded frames so the in-browser player fits in memory. Returns an IPython HTML to display."""
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML
    plt.rcParams["lines.scale_dashes"] = False    # dash lengths = literal points (else a short H-bond collapses to one dash)
    t = c["t"]; ps = c["ps"]; x = t.xyz * 10
    ca = t.topology.select("name CA")
    # canonical orientation: PCA-align on frame-0 CA (long axis -> x), then pin both terminal landmarks
    _ref = x[0, ca] - x[0, ca].mean(0); _R = np.linalg.svd(_ref.T @ _ref)[0]
    if np.linalg.det(_R) < 0: _R[:, -1] *= -1                  # proper rotation (no chirality flip)
    x = (x - x[0, ca].mean(0)) @ _R
    if x[0, ca[0], 0] > 0: x[:, :, [0, 2]] *= -1               # N-terminus to the left
    if x[0, ca[-1], 1] < 0: x[:, :, [1, 2]] *= -1              # C-terminus up -> orientation fully pinned
    CA = x[:, ca, :]
    a = lambda resid, name: t.topology.select(f"resid {resid} and name {name}")[0]
    gH, gO, aH, aO = a(4, "H"), a(0, "O"), a(8, "H"), a(4, "O")   # index 4=GLN5, 0=ASN1, 8=ASP9 (canonical labels)
    ile4c = _recenter_deg(c["ile4"], _circmean_deg(c["ile4"]))
    ile4lo, ile4hi = ile4c.min() - 10, ile4c.max() + 10
    step = max(3, int(np.ceil(t.n_frames / 200)))             # cap embedded frames; subsample across the WHOLE run
    F = list(range(0, t.n_frames, step))
    if step > 3:
        _dps = float(ps[step] - ps[0])                        # TIME between kept frames (ps) = step * frame-spacing
        print(f"player: subsampled to {len(F)} of {t.n_frames} frames (every {step} frames = {_dps:.0f} ps) so the "
              f"embedded video fits in browser memory; the video's last frame is at {ps[F[-1]]:.0f} ps.")
    figA = plt.figure(figsize=(13.5, 6.2))
    gs = figA.add_gridspec(3, 2, width_ratios=[1.15, 1], hspace=0.55, wspace=0.22)
    axM = figA.add_subplot(gs[:, 0], projection="3d")
    axR, axH, axP = figA.add_subplot(gs[0, 1]), figA.add_subplot(gs[1, 1]), figA.add_subplot(gs[2, 1])
    allc = CA.reshape(-1, 3); ct = (allc.min(0) + allc.max(0)) / 2; rr = (allc.max(0) - allc.min(0)).max() / 2 * 0.9
    axM.set_xlim(ct[0]-rr, ct[0]+rr); axM.set_ylim(ct[1]-rr, ct[1]+rr); axM.set_zlim(ct[2]-rr, ct[2]+rr)
    axM.set_axis_off(); axM.set_title("structure (N→C); H-bonds dashed", fontsize=11)
    tube = Line3DCollection(np.stack([CA[0, :-1], CA[0, 1:]], axis=1), linewidths=5); axM.add_collection3d(tube)
    cols = plt.get_cmap("turbo")(np.linspace(0, 1, len(ca) - 1))
    hbG, = axM.plot([], [], [], ls="--", lw=2, color="darkorange")
    hbA, = axM.plot([], [], [], ls="--", lw=2, color="dodgerblue")
    ile4dot, = axM.plot([], [], [], "o", color="magenta", ms=8)
    _hlo = min(c["gln5"].min(), c["asp9"].min()) - 0.3
    _hhi = max(c["gln5"].max(), c["asp9"].max()) + 0.3
    for ax, ttl, yl, ylim in [(axR, "Cα RMSD", "Å", (0, c["rmsd"].max() * 1.1)),
                              (axH, "backbone amide N–H···O=C", "Å", (_hlo, _hhi)),
                              (axP, "ILE4 χ1 (re-centered)", "deg", (ile4lo, ile4hi))]:
        ax.set(title=ttl, ylabel=yl, xlim=(0, ps[F[-1]])); ax.set_ylim(*ylim)
    axH.axhline(2.5, ls=":", c="k", lw=1, label="2.5 Å cutoff"); axP.set_xlabel("time (ps)")
    lR, = axR.plot([], [], color="navy"); lG, = axH.plot([], [], color="darkorange", label="GLN5···ASN1")
    lAx, = axH.plot([], [], color="dodgerblue", label="ASP9···GLN5"); axH.legend(fontsize=7, loc="upper right")
    lP, = axP.plot([], [], color="purple")
    cursors = [ax.axvline(ps[0], color="0.55", lw=1, ls="--") for ax in (axR, axH, axP)]

    def upd(fi):
        f = F[fi]
        tube.set_segments(np.stack([CA[f, :-1], CA[f, 1:]], axis=1)); tube.set_color(cols)
        for hb, H, O, dd in [(hbG, gH, gO, c["gln5"][f]), (hbA, aH, aO, c["asp9"][f])]:
            hb.set_data_3d([x[f, H, 0], x[f, O, 0]], [x[f, H, 1], x[f, O, 1]], [x[f, H, 2], x[f, O, 2]])
            sat = dd <= 2.5
            hb.set_alpha(1.0 if sat else 0.7); hb.set_linewidth(2.8 if sat else 2.0)
            if sat: hb.set_dashes([3, 2])
            else: hb.set_linestyle(":")
        ile4dot.set_data_3d([CA[f, 3, 0]], [CA[f, 3, 1]], [CA[f, 3, 2]])
        axM.view_init(elev=40, azim=-60)
        i = f + 1
        lR.set_data(ps[:i], c["rmsd"][:i]); lG.set_data(ps[:i], c["gln5"][:i])
        lAx.set_data(ps[:i], c["asp9"][:i]); lP.set_data(ps[:i], ile4c[:i])
        for cur in cursors: cur.set_xdata([ps[f], ps[f]])
        figA.suptitle(f"{label} — t = {ps[f]} ps", fontsize=13)
        return tube, hbG, hbA, ile4dot, lR, lG, lAx, lP, *cursors

    anim = FuncAnimation(figA, upd, frames=len(F), interval=160, blit=False)
    plt.close(figA)
    return HTML(anim.to_jshtml(fps=6))


def filmstrip(traj, out_dir, fracs=(0.25, 0.5, 0.75, 1.0), pymol=None):
    """Ray-traced PyMOL filmstrip: cartoon snapshots at the given trajectory fractions, all aligned to
    frame 0, with ILE4 highlighted in magenta sticks. Returns (png_paths, frame_indices); png_paths is
    empty if PyMOL is unavailable."""
    os.makedirs(out_dir, exist_ok=True)
    n = traj.n_frames
    frames = [min(n - 1, int(round(fr * (n - 1)))) for fr in fracs]
    pdbs = [os.path.join(out_dir, f"_fs_{j}.pdb") for j in range(len(frames))]
    pngs = [os.path.join(out_dir, f"fs_render_{j}.png") for j in range(len(frames))]
    for p, fr in zip(pdbs, frames):
        traj[fr].save_pdb(p)
    script = os.path.join(out_dir, "_filmstrip.py")
    with open(script, "w") as fh:
        fh.write(f"""from pymol import cmd
W = 800
cmd.bg_color("white"); cmd.set("ray_opaque_background", 0)
cmd.set("cartoon_fancy_helices", 1); cmd.set("ray_shadows", 0); cmd.set("antialias", 2); cmd.set("orthoscopic", 1)
PDBS = {pdbs!r}
PNGS = {pngs!r}
objs = [f"f{{j}}" for j in range(len(PDBS))]
for o, p in zip(objs, PDBS): cmd.load(p, o)
cmd.hide("everything")
for o in objs[1:]: cmd.align(o, objs[0])
cmd.show("cartoon", objs[0]); cmd.spectrum("count", "rainbow", objs[0] + " and name CA")
cmd.orient(objs[0]); cmd.turn("y", 20); cmd.turn("x", 20); VIEW = cmd.get_view()
for o, png in zip(objs, PNGS):
    cmd.disable("all"); cmd.enable(o); cmd.show("cartoon", o)
    cmd.spectrum("count", "rainbow", o + " and name CA")
    cmd.show("sticks", o + " and resi 4"); cmd.color("magenta", o + " and resi 4 and elem C")
    cmd.set_view(VIEW); cmd.ray(W, W); cmd.png(png, dpi=150)
""")
    if render_pymol(script, pymol):
        return [p for p in pngs if os.path.exists(p)], frames
    return [], frames


def sqcrop(img):
    """Center-crop an image array to a square (for tidy figure panels)."""
    h, w = img.shape[:2]; s = min(h, w); y0 = (h - s) // 2; x0 = (w - s) // 2
    return img[y0:y0 + s, x0:x0 + s]


def convergence_figure(series, dt_ps=1.0, out_png=None, label="observable"):
    """Two-panel block-averaging figure on ONE shared SEM scale: LEFT an idealized well-sampled AR(1)
    reference whose SEM climbs to a clean plateau (the honest error of the mean); RIGHT the real `series`
    block curve with Flyvbjerg-Petersen error bars -- its tail is noisy and usually does NOT plateau -- plus
    the tau-based SEM floor sigma*sqrt(2 tau / N). Teaches why we quote the tau-based SEM rather than trying
    to read a plateau off a too-short run. Uses mdtutorial.block_curve / integrated_autocorr_time. Returns
    the figure (also saves to out_png if given)."""
    import numpy as np
    import matplotlib.pyplot as plt
    import mdtutorial as mdt
    y = np.asarray(series, float); N = len(y)
    br, sr, er = mdt.block_curve(y)
    tau = mdt.integrated_autocorr_time(y); floor = y.std(ddof=1) / np.sqrt(N / (2 * tau))
    rng = np.random.default_rng(0)                       # idealized: short-tau AR(1) with MANY samples -> plateaus
    xi = np.empty(6000); xi[0] = rng.standard_normal()
    for i in range(1, 6000):
        xi[i] = 0.8 * xi[i - 1] + np.sqrt(1 - 0.64) * rng.standard_normal()
    ti = mdt.integrated_autocorr_time(xi)
    xi *= floor / (xi.std(ddof=1) / np.sqrt(len(xi) / (2 * ti)))   # scale so its plateau matches the real SEM range
    bi, si, _ = mdt.block_curve(xi)
    naive_i = xi.std(ddof=1) / np.sqrt(len(xi))
    plateau_i = xi.std(ddof=1) / np.sqrt(len(xi) / (2 * ti))
    BOX = dict(boxstyle="round", fc="white", ec="0.6", alpha=0.95)
    YL = (0, max(plateau_i, floor, float((sr + er).max())) * 1.25)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True, facecolor="white")
    ax[0].semilogx(bi, si, "o-", color="navy", ms=3)
    ax[0].axhline(naive_i, ls="--", color="0.55", lw=1); ax[0].axhline(plateau_i, ls="--", color="firebrick", lw=1)
    ax[0].text(bi[-1], naive_i, "naïve σ/√N\n(independent-frames floor)", va="center", ha="right", fontsize=8, color="0.4", bbox=BOX)
    ax[0].text(bi[0], plateau_i, "honest SEM\n(the plateau)", va="center", ha="left", fontsize=8.5, color="firebrick", bbox=BOX)
    ax[0].set(title="Well-sampled reference: SEM plateaus", xlabel="block size (frames)",
              ylabel="standard error of the mean (Å)"); ax[0].set_ylim(*YL)
    naive_r = y.std(ddof=1) / np.sqrt(N)                  # what a SHUFFLED (independent-frames) run would give
    ax[1].errorbar(br, sr, yerr=er, fmt="o-", color="darkorange", ms=3, lw=1, capsize=2, ecolor="0.6")
    ax[1].axhline(floor, ls="--", color="firebrick", lw=1); ax[1].axhline(naive_r, ls="--", color="0.55", lw=1)
    ax[1].text(br[0], floor, "τ-based SEM (the floor)\nσ·√(2τ/N)", va="center", ha="left", fontsize=8.5, color="firebrick", bbox=BOX)
    ax[1].text(br[-1], naive_r, "naïve σ/√N — the independent-frames\n(shuffled) floor; the climb above it is √(2τ)",
               va="bottom", ha="right", fontsize=8, color="0.4", bbox=BOX)
    ax[1].set(title=f"{label} ({N} frames, τ≈{tau * dt_ps:.0f} ps): no clean plateau", xlabel="block size (frames)")
    ax[1].set_ylabel("standard error of the mean (Å)"); ax[1].yaxis.set_label_position("right"); ax[1].set_ylim(*YL)
    fig.suptitle("Block-averaging error bar — idealized (left) vs. real (right), same SEM scale", fontsize=11)
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=130)
    return fig


def cv_locator_map(pdb_path):
    """§3.2 schematic LOCATOR (pairs with the py3Dmol 3D render of the same groups). A 2D PCA projection of
    the Cα trace as a grey tube; the four observable groups colored and each tethered to its backbone Cα by a
    color-coded bond; the biased CV (indole→poly-Pro) and the Asp9–Arg16 salt bridge drawn as dashed lines ON
    the molecule; and four margin pictograms (biased CV, Trp6 SASA, salt bridge, global Cα-RMSD) connected to
    their location by dotted leader lines. Pure matplotlib -> reproducible, shows inline."""
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch, Circle, Ellipse
    import matplotlib.patheffects as pe
    from scipy.interpolate import splprep, splev
    import mdtraj as md

    t = md.load(pdb_path); top = t.topology
    ca = top.select("name CA"); X = t.xyz[0, ca] * 10
    Xc = X - X.mean(0); _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Xc @ Vt[:2].T
    cen = lambda q: (t.xyz[0, top.select(q)].mean(0) * 10 - X.mean(0)) @ Vt[:2].T
    IND = cen("resSeq 6 and name CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2")
    LID = cen("resSeq 17 18 19 and name CA")
    ASP = cen("resSeq 9 and name OD1 OD2"); ARG = cen("resSeq 16 and name NH1 NH2 NE")
    C_IND, C_LID, C_ASP, C_ARG = "#e8820c", "#2166ac", "#c0392b", "#27ae60"

    fig = plt.figure(figsize=(10.5, 8.2))
    ax = fig.add_axes([0.26, 0.10, 0.50, 0.82]); ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-12.5, 10.5); ax.set_ylim(-8.5, 11.5)
    (tck, _) = splprep([P[:, 0], P[:, 1]], s=0, k=3); uu = np.linspace(0, 1, 400)
    bx, by = splev(uu, tck)
    ax.plot(bx, by, color="0.82", lw=11, solid_capstyle="round", zorder=1)             # tube body
    ax.plot(bx, by, color="0.5", lw=1.6, solid_capstyle="round", zorder=1.5)           # centerline
    ecx, ecy = P.mean(0); ea = np.ptp(P[:, 0]) + 6; eb = np.ptp(P[:, 1]) + 6            # "whole molecule" (RMSD)
    ax.add_patch(Ellipse((ecx, ecy), ea, eb, angle=0, fill=False, ls=(0, (4, 3)), lw=1.0, ec="0.75", zorder=0))
    _th = np.radians(140)                                                               # upper-left arc, ~halfway N<->C
    E_RMSD = (ecx + 0.5 * ea * np.cos(_th), ecy + 0.5 * eb * np.sin(_th))               # terminate the RMSD leader ON the arc
    ax.plot([IND[0], LID[0]], [IND[1], LID[1]], color="k", lw=1.8, ls=(0, (5, 2)), zorder=4)     # biased CV
    ax.plot([ASP[0], ARG[0]], [ASP[1], ARG[1]], color="0.2", lw=1.4, ls=(0, (2, 2)), zorder=4)   # salt bridge

    def marker(xy, col, r=0.95, lab=None, dx=0.0, dy=0.0, anchor=None):
        if anchor is not None:                                                          # side-chain group -> backbone Cα
            ax.plot([anchor[0], xy[0]], [anchor[1], xy[1]], color=col, lw=2.6, solid_capstyle="round", zorder=4.5)
            ax.add_patch(Circle(anchor, 0.34, fc=col, ec="white", lw=0.7, zorder=4.6))
        ax.add_patch(Circle(xy, r, fc=col, ec="white", lw=1.5, zorder=5))
        if lab:
            ax.annotate(lab, xy, xytext=(xy[0] + dx, xy[1] + dy), fontsize=8.5, weight="bold", color=col,
                        ha="center", va="center", path_effects=[pe.withStroke(linewidth=2.4, foreground="white")], zorder=6)
    marker(IND, C_IND, r=1.15, lab="Trp6", dx=-2.0, dy=-0.2, anchor=P[5])
    marker(LID, C_LID, r=1.05, lab="Pro17–19\nlid", dx=0.0, dy=2.3, anchor=P[17])       # attaches at Pro18 Cα (the stacker)
    marker(ASP, C_ASP, r=0.8, lab="Asp9", dx=-1.8, dy=-0.9, anchor=P[8])
    marker(ARG, C_ARG, r=0.8, lab="Arg16", dx=2.0, dy=0.4, anchor=P[15])
    ax.annotate("N", P[0], xytext=(P[0, 0] - 1.6, P[0, 1] - 0.4), fontsize=9, color="0.4", ha="center", va="center")
    ax.annotate("C", P[-1], xytext=(P[-1, 0] - 1.4, P[-1, 1] + 0.6), fontsize=9, color="0.4", ha="center", va="center")

    def picto_ax(rect):
        a = fig.add_axes(rect); a.set_xlim(0, 1); a.set_ylim(0, 1); a.axis("off"); return a

    def ptitle(a, s, col): a.text(0.5, -0.16, s, transform=a.transAxes, ha="center", va="top", fontsize=8, color=col, weight="bold")

    def water(a, cx, cy, s=1.0, rot=0.0):                                               # 3-atom water: red O + 2 white H
        ro, rh, d = 0.030 * s, 0.018 * s, 0.056 * s
        for sign in (-1, 1):
            ang = rot + sign * np.radians(52.25)
            hx, hy = cx + d * np.cos(ang), cy + d * np.sin(ang)
            a.plot([cx, hx], [cy, hy], color="0.55", lw=1.0, zorder=1, solid_capstyle="round")
            a.add_patch(Circle((hx, hy), rh, fc="white", ec="0.55", lw=0.7, zorder=2))
        a.add_patch(Circle((cx, cy), ro, fc="#d43a2f", ec="white", lw=0.5, zorder=3))

    a1 = picto_ax([0.775, 0.60, 0.205, 0.26])                                           # biased CV
    a1.add_patch(Circle((0.22, 0.55), 0.11, fc=C_IND, ec="white", lw=1.2))
    a1.add_patch(Circle((0.80, 0.55), 0.10, fc=C_LID, ec="white", lw=1.2))
    _sx = np.linspace(0.33, 0.69, 60); _sy = 0.55 + 0.06 * np.sin((_sx - 0.33) * 46)
    a1.plot(_sx, _sy, color="0.3", lw=1.4)
    a1.annotate("", (0.95, 0.55), (0.80, 0.55), arrowprops=dict(arrowstyle="->", color="firebrick", lw=1.6))
    a1.text(0.5, 0.90, "we ramp r₀ outward", ha="center", fontsize=6.8, color="firebrick", style="italic")
    ptitle(a1, "biased CV\nindole→poly-Pro", "k")

    a2 = picto_ax([0.775, 0.10, 0.205, 0.24])                                           # salt bridge
    a2.add_patch(Circle((0.26, 0.55), 0.13, fc=C_ASP, ec="white", lw=1.2))
    a2.add_patch(Circle((0.74, 0.55), 0.13, fc=C_ARG, ec="white", lw=1.2))
    a2.text(0.26, 0.55, "–", ha="center", va="center", fontsize=15, color="white", weight="bold")
    a2.text(0.74, 0.55, "+", ha="center", va="center", fontsize=13, color="white", weight="bold")
    a2.plot([0.39, 0.61], [0.55, 0.55], color="0.3", lw=1.3, ls=(0, (2, 2)))
    ptitle(a2, "referee: Asp9–Arg16\nsalt bridge", "0.25")

    a3 = picto_ax([0.02, 0.10, 0.19, 0.24])                                             # Trp6 SASA
    a3.add_patch(Circle((0.5, 0.52), 0.16, fc=C_IND, ec="white", lw=1.2))
    a3.add_patch(Circle((0.5, 0.52), 0.30, fill=False, ec="0.45", lw=1.1, ls=(0, (3, 2))))
    for _i, _a in enumerate(np.linspace(0, 2 * np.pi, 6, endpoint=False)):
        water(a3, 0.5 + 0.40 * np.cos(_a), 0.52 + 0.40 * np.sin(_a), s=0.95, rot=_a + 0.9 * _i)
    ptitle(a3, "referee: Trp6 SASA\n(cage exposure)", C_IND)

    a4 = picto_ax([0.02, 0.60, 0.19, 0.24])                                             # global Cα-RMSD
    _zx = np.linspace(0.12, 0.88, 7); _zy = 0.5 + 0.13 * np.array([0, 1, -1, 1, -1, 1, 0])
    a4.plot(_zx, _zy + 0.06, color="0.7", lw=2.0, solid_capstyle="round")
    a4.plot(_zx, _zy - 0.06, color="0.2", lw=2.0, solid_capstyle="round")
    a4.annotate("", (0.5, 0.5 + 0.13 - 0.06), (0.5, 0.5 + 0.13 + 0.06), arrowprops=dict(arrowstyle="<->", color="firebrick", lw=1.2))
    ptitle(a4, "referee: global\nCα-RMSD", "0.3")

    def leader(a, xyA, xyB, col="0.5"):
        fig.add_artist(ConnectionPatch(xyA=xyA, coordsA=a.transAxes, xyB=xyB, coordsB=ax.transData,
                                       ls=(0, (1, 2)), lw=1.1, color=col, zorder=3))
    leader(a1, (0.05, 0.45), (0.5 * (IND[0] + LID[0]) + 0.4, 0.5 * (IND[1] + LID[1])), "k")
    leader(a2, (0.10, 0.75), (0.5 * (ASP[0] + ARG[0]), 0.5 * (ASP[1] + ARG[1])), "0.4")
    leader(a3, (0.85, 0.75), (IND[0] - 0.6, IND[1]), C_IND)
    leader(a4, (0.88, 0.40), E_RMSD, "0.4")                                             # -> lands ON the ellipse arc

    fig.suptitle("Where we push, and what watches — one biased coordinate, three independent referees",
                 y=0.975, fontsize=11.5)
    plt.close(fig)          # drop from pyplot's registry so inline doesn't auto-show a 2nd copy;
    return fig              # the notebook displays the returned Figure exactly once (script savefig()s it)
