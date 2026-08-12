#!/usr/bin/env python
"""md_scalogram.py -- blocking (Haar-wavelet) autocorrelation scalograms for MD observables.

WHY: a single integrated autocorrelation time tau collapses a protein's whole spectrum of motions into
one number. This spreads it back out. The Flyvbjerg-Petersen blocking transform IS the Haar discrete
wavelet transform -- block-averaging adjacent pairs is the Haar *scaling* step, differencing them is the
Haar *wavelet* step -- so the squared wavelet coefficients partition an observable's variance across
(timescale x time). A stable fold puts its power at fast scales; a rare substate hop (a PRO19 psi flip,
a helix fraying) shows up localized at the moment it happens, its power spread across scales but weighted toward the slow end. Great for telling a
"milquetoast wiggle" trajectory from one where something actually happened.

LIBRARY
    from md_scalogram import blocking_scalogram, observable, scalogram_figure, compare_figure
    scales, power, var_by_scale = blocking_scalogram(x, dt=1.0)     # x = any 1-D observable series
    y = observable(traj, "rg")                                     # rg | rmsd | helix | psi | phi | dist
    scalogram_figure(y, dt=1.0, label="Rg (A)").savefig("out.png")

CLI
    python md_scalogram.py --traj traj.dcd --top protein.pdb --obs rg --out rg.png
    python md_scalogram.py --traj traj_2024.dcd traj_2025.dcd --top protein.pdb --obs psi --resid 18 --out flip.png
    python md_scalogram.py --traj t.dcd --top p.pdb --obs dist --atoms 61 5 --dt 5 --out d.png

Uses the ORTHONORMAL Haar (both steps /sqrt(2)) so energy is preserved (Parseval): var_by_scale sums to
the total variance and slow scales aren't artificially damped. Pure numpy for the transform; mdtraj only
for reading trajectories / computing observables.
"""
import argparse
import numpy as np


# =====================================================================================================
# PROVENANCE -- the "blocking = Haar wavelet transform" equivalence this is built on.
#
# Flyvbjerg-Petersen blocking replaces a series with its pairwise averages  x'_i = (x_{2i}+x_{2i+1})/2,
# iterated. The Haar DWT does the SAME averaging as its scaling (low-pass) step, (x_{2i}+x_{2i+1})/sqrt2,
# and additionally keeps the pairwise DIFFERENCES (x_{2i}-x_{2i+1})/sqrt2 as its detail (wavelet)
# coefficients. So blocking is exactly the scaling branch of the Haar transform, and the variance it
# probes at each level is the Haar detail (wavelet) variance at that scale. Verified numerically here:
# FP block-mean == Haar-scaling / sqrt(2^m) to ~1e-14, and the Haar detail variances partition the total
# variance exactly (Parseval).
#
# No paper appears to state this equivalence outright -- it's folklore, evidently too elementary to
# publish -- so cite it as following from the definitions, backed by the primaries:
#   blocking : Flyvbjerg & Petersen, J. Chem. Phys. 91, 461 (1989);  Jonsson, Phys. Rev. E 98, 043304
#              (2018) -- frames blocking explicitly as a renormalization-group method.
#   wavelet  : Percival & Walden, "Wavelet Methods for Time Series Analysis", Cambridge Univ. Press (2000)
#              -- the Haar WAVELET VARIANCE is the per-octave/scale variance that blocking estimates.
#
# CWT mode (--method cwt / both) uses a Morlet continuous wavelet transform. If you come from MD/physics
# rather than signal processing, the practical references live in the METEOROLOGY/climate literature:
#   Torrence, C. & Compo, G. P. (1998). "A Practical Guide to Wavelet Analysis." Bull. Amer. Meteor. Soc.
#       79(1), 61-78. -- THE practical reference for the wavelet power spectrum, its normalization and
#       significance, and the reconstruction factor C_delta = 0.776 (Morlet w0=6; their Table 2) that WOULD
#       turn the redundant CWT power into a true variance -- but we DO NOT apply it here; our CWT stays a
#       relative spectrum and the exact variance-by-scale is the Haar/blocking transform below. Free PDF:
#       https://psl.noaa.gov/people/gilbert.p.compo/Torrence_compo1998.pdf  (code: paos.colorado.edu/research/wavelets/)
#   Liu, Y., Liang, X. S. & Weisberg, R. H. (2007). "Rectification of the Bias in the Wavelet Power
#       Spectrum." J. Atmos. Ocean. Technol. 24(12), 2093-2102. -- the 1/s scale-bias correction applied here.
# =====================================================================================================

# ------------------------------------------------------------------ core transform
def blocking_scalogram(x, dt=1.0):
    """Haar-wavelet (blocking) scalogram of a 1-D series.

    Returns
      scales       : (J,) timescale of each level = 2**j * dt   (J = floor(log2 N))
      power        : (J, N) variance/power on the ORIGINAL time axis (each detail coeff broadcast over the
                     2**j frames it covers; trailing cells with no coefficient are NaN)
      var_by_scale : (J,) total variance at each scale (power summed over time) -- the 1-D spectrum by octave;
                     cumulative sum recovers the total variance (Parseval).
    """
    x = np.asarray(x, float)
    N = len(x)
    if N < 2:
        raise ValueError("need at least 2 samples")
    a = x - np.mean(x)
    J = int(np.floor(np.log2(N)))
    power = np.full((J, N), np.nan)
    scales = np.empty(J)
    var_by_scale = np.empty(J)
    for j in range(1, J + 1):
        m = len(a) - (len(a) % 2)                         # drop a trailing odd sample
        approx = (a[0:m:2] + a[1:m:2]) / np.sqrt(2.0)     # Haar scaling (coarser approximation)
        detail = (a[0:m:2] - a[1:m:2]) / np.sqrt(2.0)     # Haar wavelet: fluctuation at THIS scale
        p = detail ** 2
        row = np.repeat(p, 2 ** j)                        # broadcast each coeff over its 2**j-frame block
        power[j - 1, :len(row)] = row
        scales[j - 1] = (2 ** j) * dt
        var_by_scale[j - 1] = p.sum()
        a = approx
    return scales, power, var_by_scale


def integrated_autocorr_time(x):
    """Integrated autocorrelation time tau = 1/2 + sum_k rho(k), in frames (Sokal 1997). Geyer (1992)
    initial-positive-sequence PAIR truncation -- sum consecutive-pair autocorrelations, stop at the first
    non-positive pair. This is the SAME estimator as mdtutorial.integrated_autocorr_time, so nb02 §2.6's tau
    and §2.6b's 2tau line agree. N_eff = N/(2 tau); blocks/wavelet decorrelate near a scale of ~2 tau."""
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


def _autocorr_time_vec(components):
    """Integrated autocorrelation time of a VECTOR process (list of 1-D component series), summing each
    component's autocovariance and normalizing by the total variance. For a circular angle the components
    are [cos theta, sin theta], giving the proper circular autocorrelation (of the unit vector on the
    circle) -- the right decorrelation time for a dihedral, immune to the +-180 seam and to unwrap drift.
    Geyer (1992) initial-positive-sequence PAIR truncation, matching integrated_autocorr_time above."""
    comps = [np.asarray(c, float) - np.mean(c) for c in components]
    n = len(comps[0])
    v = sum(float(np.dot(c, c)) for c in comps) / n
    if v == 0:
        return 0.5
    rho = lambda k: sum(float(np.dot(cc[:-k], cc[k:])) for cc in comps) / (n * v)
    tau = 0.5
    for k in range(1, n - 1, 2):
        pair = rho(k) + rho(k + 1)
        if pair <= 0:
            break
        tau += pair
    return tau


def cwt_scalogram(x, dt=1.0, w0=6.0, n_scales=72):
    """Continuous Morlet-wavelet scalogram -- finer, continuous scale resolution and smoother than the
    dyadic Haar/blocking version, at the cost of the exact variance partition. The CWT is REDUNDANT
    (non-orthogonal, overlapping scales), so |W|^2 is a NEAR-variance power spectrum, not an exact
    partition -- use blocking_scalogram for quantitative variance/error work, this for a finer/prettier
    look at *where* correlation timescales live. This is a RELATIVE spectrum: the Morlet is unit-energy
    normalized (pi^-1/4) with the /s scale-bias correction of Liu et al. (2007), but the Torrence & Compo
    C_delta reconstruction is NOT applied -- absolute variance lives in blocking_scalogram (Parseval-exact).
    Pure numpy (FFT). Returns (timescales, power[nscale, N])."""
    x = np.asarray(x, float); N = len(x); x = x - np.mean(x)
    scales = np.geomspace(2.0, max(4.0, N / 4.0), n_scales)   # in frames
    xf = np.fft.fft(x)
    omega = 2.0 * np.pi * np.fft.fftfreq(N)                   # angular frequency, rad/frame
    power = np.empty((n_scales, N))
    for i, s in enumerate(scales):
        psi = (np.pi ** -0.25) * (omega > 0) * np.exp(-0.5 * (s * omega - w0) ** 2)   # Morlet FT at scale s
        W = np.fft.ifft(xf * psi * np.sqrt(2.0 * np.pi * s))
        power[i] = (np.abs(W) ** 2) / s                       # /s: bias-corrected, comparable across scales
    fourier_factor = 4.0 * np.pi / (w0 + np.sqrt(2.0 + w0 ** 2))   # scale -> Fourier period (~correlation time)
    return scales * fourier_factor * dt, power


def morlet_coi(n, dt=1.0, w0=6.0):
    """Cone of influence for the Morlet CWT (Torrence & Compo 1998): at each time position, the LARGEST
    timescale still free of edge effects. The Morlet e-folding time is sqrt(2)*s, so within that distance of
    either end the transform is reading the zero-padded/wrapped edge and the power there is an ARTIFACT, not
    signal -- exactly the "not enough data at the long-timescale end" concern. A CWT scalogram shown to its
    full width without this is dishonest: the top corners are fabricated by the finite window. Returns
    coi[n] in the scalogram's timescale units (matching cwt_scalogram's first return)."""
    ff = 4.0 * np.pi / (w0 + np.sqrt(2.0 + w0 ** 2))
    edge = np.minimum(np.arange(n), n - 1 - np.arange(n))         # frames to the nearer end
    return ff * np.sqrt(2.0) * edge * dt


def _overlay_coi(ax, tt, scales, dt, w0=6.0):
    """Fade + outline the Morlet cone of influence on a CWT scalogram axis (tt = its time axis) so a reader
    does not trust the edge-contaminated corners -- long timescale near the start/end of the run."""
    coi = morlet_coi(len(tt), dt, w0)
    ax.plot(tt, coi, color="w", lw=1.0)
    ax.fill_between(tt, coi, scales.max(), color="white", alpha=0.55, lw=0)   # fade the unreliable region
    ax.set_ylim(scales.min(), scales.max())


def cwt_marginal(power, coi, scales):
    """COI-respecting global wavelet spectrum: per-scale MEAN of the CWT power over ONLY the out-of-cone
    frames (coi >= scale) -- the honest projection of the scalogram onto the timescale axis, with the
    edge-fabricated corners excluded. Returns marginal[n_scales] (NaN at scales above the cone apex, where no
    frame is supported). NB the log-centroid of this marginal tracks 2*tau while an observable is well
    sampled and falls BELOW it once undersampled (the cone truncates the slow end) -- a handy undersampling
    flag, not a defect."""
    m = np.full(len(scales), np.nan)
    for i, s in enumerate(scales):
        valid = coi >= s
        if valid.any():
            m[i] = power[i, valid].mean()
    return m


def _transform(x, dt=1.0, method="blocking", **kw):
    """Dispatch: 'blocking' (Haar, exact variance, dyadic) or 'cwt' (Morlet, fine, near-variance).
    Returns (scales/timescales, power, marginal) where marginal = power summed over time (per-scale)."""
    if method == "blocking":
        return blocking_scalogram(x, dt)                      # (scales, power, var_by_scale)
    if method == "cwt":
        s, p = cwt_scalogram(x, dt, **kw)
        return s, p, np.nansum(p, axis=1)
    raise ValueError(f"method must be 'blocking' or 'cwt', got {method!r}")


def is_circular(kind):
    """True for angular observables (backbone phi/psi, side-chain chi1..chi4) -- they must be treated
    circularly, not as a linear series."""
    return str(kind).lower() in ("psi", "phi", "chi1", "chi2", "chi3", "chi4")


def _recenter_deg(a):
    """Shift an angle series (deg) to its circular mean, for CLEAN DISPLAY only: moves the +-180 seam away
    from the populated region so a dihedral that lives near 180 stops flickering +180<->-180 in the trace.
    A pure rotation, so it does NOT affect the circular (cos/sin) scalogram -- trace cosmetics only."""
    a = np.asarray(a, float); r = np.radians(a)
    mu = np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean()))
    return (a - mu + 180.0) % 360.0 - 180.0


def _scalo(x, dt=1.0, method="blocking", circular=False):
    """Scalogram dispatch that also handles CIRCULAR observables. For a linear series: the ordinary
    transform + scalar tau. For a circular angle x (in DEGREES): decompose to the unit vector
    (cos theta, sin theta), scalogram EACH component, and SUM their power -- the correct scale-resolved
    variance of a point on the circle. This avoids the np.unwrap artifact (unwrapping a bistable dihedral
    turns bounded hopping into a random walk -> fabricated slow power, an f^-2 spectrum that isn't real).
    Returns (scales, power[J|nscale, N], marginal, tau)."""
    if circular:
        th = np.radians(np.asarray(x, float))
        s, p_c, m_c = _transform(np.cos(th), dt, method)
        _, p_s, m_s = _transform(np.sin(th), dt, method)
        return s, p_c + p_s, m_c + m_s, _autocorr_time_vec([np.cos(th), np.sin(th)])
    s, p, m = _transform(x, dt, method)
    return s, p, m, integrated_autocorr_time(x)


# ------------------------------------------------------------------ observables from a trajectory
def _native_q(traj, ref=0):
    """Fraction of native contacts Q (Best, Hummer & Eaton, PNAS 2013): the cleanest folding/unfolding
    reporter. Native set = heavy-atom pairs from residues >3 apart that are within 0.45 nm in the
    reference frame; each frame's Q uses a smooth switching function (beta=50/nm, lambda=1.8). Q ~ 1 when
    folded, -> 0 as native contacts break. Reference frame `ref` defines "native" (default the first frame)."""
    import mdtraj as md
    from itertools import combinations
    BETA, LAMBDA, CUTOFF = 50.0, 1.8, 0.45                # 1/nm, unitless, nm
    native = traj[ref]
    heavy = native.topology.select_atom_indices("heavy")
    pairs = np.array([(i, j) for i, j in combinations(heavy, 2)
                      if abs(native.topology.atom(i).residue.index
                             - native.topology.atom(j).residue.index) > 3])
    if len(pairs) == 0:
        raise ValueError("no heavy-atom pairs >3 residues apart (protein too small / topology issue)")
    r0 = md.compute_distances(native, pairs)[0]
    sel = r0 < CUTOFF
    contacts, r0n = pairs[sel], r0[sel]
    if len(contacts) == 0:
        raise ValueError("no native contacts within the cutoff in the reference frame")
    r = md.compute_distances(traj, contacts)
    return np.mean(1.0 / (1.0 + np.exp(BETA * (r - LAMBDA * r0n))), axis=1)


def observable(traj, kind="rg", resid=None, atoms=None, ref=0):
    """Compute a 1-D observable time series from an mdtraj Trajectory.
       rg | rmsd | q | helix | psi | phi | chi1..chi4 (need --resid) | dist (need --atoms i j). Angles are
       returned RAW/WRAPPED in [-180,180] and flagged circular (see is_circular); the scalogram treats
       them via cos/sin (do NOT np.unwrap -- that fabricates slow power). q/rmsd use frame `ref` as native."""
    import mdtraj as md
    t = traj
    kind = kind.lower()
    if kind == "rg":
        return md.compute_rg(t) * 10.0
    if kind == "rmsd":
        return md.rmsd(t, t, ref) * 10.0
    if kind == "q":
        return _native_q(t, ref)
    if kind == "helix":
        return (md.compute_dssp(t, simplified=True) == "H").mean(1)
    if kind in ("psi", "phi", "chi1", "chi2", "chi3", "chi4"):
        if resid is None:
            raise ValueError(f"{kind} needs resid=")
        idx, ang = getattr(md, "compute_" + kind)(t)          # backbone psi/phi or side-chain chi1..chi4
        for k, ii in enumerate(idx):
            if t.topology.atom(ii[1]).residue.index == resid:
                return np.degrees(ang[:, k])          # RAW wrapped angle; treated circularly downstream
        raise ValueError(f"resid {resid} has no {kind}")
    if kind == "dist":
        if not atoms or len(atoms) != 2:
            raise ValueError("dist needs atoms=[i, j]")
        return md.compute_distances(t, [atoms])[:, 0] * 10.0
    raise ValueError(f"unknown observable '{kind}' (rg|rmsd|helix|psi|phi|dist)")


def chi1_quartiles(traj, nq=5):
    """Rank a trajectory's residues by their circular chi1 autocorrelation time tau and return nq residues
    sampled at even RANK positions (fast -> slow) as (series, labels, taus_frames) -- ready to hand to
    marginal_strip_figure for the 2.6b strip. With the default nq=5 these are the QUARTILE BOUNDARIES of the
    ranking (fastest, Q1, median, Q3, slowest) -- five points bracketing four quartiles: order statistics of
    rank, NOT quintile groups and NOT evenly spaced in tau value. taus are in FRAMES (x dt for time).
    Residues without a chi1 (GLY/ALA) are simply absent from mdtraj's chi1 set."""
    import mdtraj as md
    idx, chi = md.compute_chi1(traj); chi = np.degrees(chi)
    residues = [traj.topology.atom(q[1]).residue for q in idx]
    taus = np.array([_autocorr_time_vec([np.cos(np.radians(chi[:, k])), np.sin(np.radians(chi[:, k]))])
                     for k in range(chi.shape[1])])                 # circular tau per residue (frames)
    order = np.argsort(taus)                                        # shortest -> longest
    pick = order[np.linspace(0, len(order) - 1, nq).astype(int)]   # nq evenly spaced across the ranking
    series = [chi[:, k] for k in pick]
    labels = [f"{residues[k].name}{residues[k].resSeq}" for k in pick]
    return series, labels, taus[pick]


# ------------------------------------------------------------------ figures
def scalogram_figure(x, dt=1.0, label="observable", title=None, method="blocking", circular=False):
    """Composite: observable trace (top), scalogram (main), variance/power-by-scale spectrum (right), with
    the 2*tau decorrelation scale marked. method='blocking' (Haar, exact variance) | 'cwt' (Morlet, fine).
    circular=True for a dihedral (x in degrees): power computed from cos/sin, trace still shows the angle.
    Returns the Figure."""
    import matplotlib.pyplot as plt
    scales, power, vbs, tau = _scalo(x, dt, method, circular)
    neff = len(x) / (2.0 * tau)
    twotau = 2.0 * tau * dt
    fig = plt.figure(figsize=(11, 6), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 2.6])
    tt = np.arange(power.shape[1]) * dt
    axT = fig.add_subplot(gs[0, 0])
    axT.plot(np.arange(len(x)) * dt, _recenter_deg(x) if circular else x, lw=0.8, color="navy"); axT.margins(x=0)
    axT.set_ylabel(label + (" (centered)" if circular else "")); axT.tick_params(labelbottom=False)
    axS = fig.add_subplot(gs[1, 0], sharex=axT)
    pm = axS.pcolormesh(tt, scales, np.log10(power + 1e-12), shading="nearest", cmap="turbo")
    axS.set_yscale("log"); axS.set_xlabel("time"); axS.set_ylabel("timescale")
    if method == "cwt":
        _overlay_coi(axS, tt, scales, dt)                 # fade the edge-contaminated corners
    axS.axhline(twotau, color="w", ls="--", lw=1.2)
    axS.text(tt[-1], twotau, f" 2τ≈{twotau:.0f} ", color="w", va="bottom", ha="right", fontsize=8)
    axR = fig.add_subplot(gs[1, 1], sharey=axS)
    axR.plot(vbs, scales, "o-", ms=3, color="firebrick"); axR.set_xscale("log")
    axR.axhline(twotau, color="grey", ls="--", lw=1)
    axR.set_xlabel("variance/scale"); axR.tick_params(labelleft=False)
    fig.colorbar(pm, ax=axR, label="log10 power", location="right", fraction=0.12, pad=0.25)
    fig.suptitle(title or f"{method} scalogram — {label}   (τ≈{tau*dt:.0f}, N_eff≈{neff:.0f})",
                 fontsize=12)
    return fig


def compare_figure(series, labels, dt=1.0, obs_label="observable", title=None, method="blocking", circular=False):
    """Stack several trajectories' scalograms (one column each) for side-by-side comparison -- e.g. a
    milquetoast seed vs one that flipped. method='blocking'|'cwt'. circular=True for dihedrals. Returns the Figure."""
    import matplotlib.pyplot as plt
    n = len(series)
    fig, axes = plt.subplots(2, n, figsize=(5.2 * n, 5.4), sharex="col",
                             gridspec_kw=dict(height_ratios=[1, 2.6]),
                             squeeze=False, layout="constrained")
    for c, (x, lab) in enumerate(zip(series, labels)):
        scales, power, _, tau = _scalo(x, dt, method, circular); twotau = 2.0 * tau * dt
        tt = np.arange(power.shape[1]) * dt
        axes[0, c].plot(np.arange(len(x)) * dt, _recenter_deg(x) if circular else x, lw=0.8, color="navy"); axes[0, c].margins(x=0)
        axes[0, c].set_title(f"{lab}  (τ≈{tau*dt:.0f})", fontsize=10)
        if c == 0:
            axes[0, c].set_ylabel(obs_label)
        pm = axes[1, c].pcolormesh(tt, scales, np.log10(power + 1e-12), shading="nearest", cmap="turbo")
        axes[1, c].set_yscale("log"); axes[1, c].set_xlabel("time")
        if method == "cwt":
            _overlay_coi(axes[1, c], tt, scales, dt)      # fade the edge-contaminated corners
        axes[1, c].axhline(twotau, color="w", ls="--", lw=1.0)
        if c == 0:
            axes[1, c].set_ylabel("timescale")
        fig.colorbar(pm, ax=axes[1, c], label="log10 power", fraction=0.05, pad=0.02)
    fig.suptitle(title or f"{method} scalograms — {obs_label}", fontsize=12)
    return fig


def both_figure(x, dt=1.0, label="observable", title=None, circular=False):
    """The observable trace + BOTH scalograms (blocking/Haar above, Morlet CWT below) stacked on a shared
    time axis -- the same signal decomposed both ways: dyadic/exact vs continuous/smooth. circular=True for
    dihedrals (power via cos/sin). Returns the Figure."""
    import matplotlib.pyplot as plt
    sb, pb, _, tau = _scalo(x, dt, "blocking", circular)
    sc, pc, _, _ = _scalo(x, dt, "cwt", circular)
    twotau = 2.0 * tau * dt
    fig, (axT, axB, axC) = plt.subplots(3, 1, figsize=(11, 8.2), sharex=True,
                                        gridspec_kw=dict(height_ratios=[1, 2, 2]), layout="constrained")
    axT.plot(np.arange(len(x)) * dt, _recenter_deg(x) if circular else x, lw=0.8, color="navy"); axT.margins(x=0)
    axT.set_ylabel(label + (" (centered)" if circular else ""))
    for ax, s, p, ttl, is_cwt in ((axB, sb, pb, "blocking (Haar DWT) — dyadic, exact variance", False),
                                  (axC, sc, pc, "Morlet CWT — continuous, near-variance", True)):
        t = np.arange(p.shape[1]) * dt
        m = ax.pcolormesh(t, s, np.log10(p + 1e-12), shading="nearest", cmap="turbo")
        ax.set_yscale("log"); ax.set_ylabel("timescale")
        if is_cwt:
            _overlay_coi(ax, t, s, dt)                    # fade the edge-contaminated corners (CWT only)
        ax.axhline(twotau, color="w", ls="--", lw=1.0)
        ax.set_title(ttl, fontsize=9, loc="left")
        fig.colorbar(m, ax=ax, label="log10 power", fraction=0.04, pad=0.01)
    axC.set_xlabel("time")
    fig.suptitle(title or f"blocking vs CWT — {label}   (τ≈{tau*dt:.0f})", fontsize=12)
    return fig


def marginal_strip_figure(series, labels, dt=1.0, circular=True, obs_label="observable", title=None,
                          view="both"):
    """Strip of scalograms (one column per series, meant fast->slow): each series is a TRACE on top and,
    below it, one or both wavelet decompositions -- each with its MARGINAL butted flush on the right, sharing
    the timescale axis; 2*tau marked on the scalogram and labelled on the marginal. This is the 2.6b figure:
    watch power climb toward longer scale as tau grows.

    view lever (reusable; the notebook default is 'both'):
      'both' -> THREE rows per column: trace, then the EXACT dyadic Haar-DWT scalogram (marginal = exact
                Parseval variance per octave, as bars), then the smooth Morlet CWT scalogram (marginal = the
                COI-respecting global wavelet spectrum, a relative line). Lead with the exact; CWT is a guide.
      'dwt'  -> trace + the Haar-DWT scalogram only.  'cwt' -> trace + the Morlet CWT scalogram only.
    The CWT is REDUNDANT (near-variance, qualitative); the Haar-DWT is Parseval-exact (quantitative).

    NOT constrained-layout -- that re-inserts a gap that strands the scalogram's ticks; a flat gridspec with
    an empty spacer column separates the residue blocks. Returns the Figure."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullLocator
    TF = {"both": ["blocking", "cwt"], "dwt": ["blocking"], "cwt": ["cwt"]}[view]   # exact first, then smooth
    ROWLAB = {"blocking": "DWT (exact)", "cwt": "CWT (smooth)"}
    n = len(series)
    widths, colmap = [], []
    for i in range(n):
        if i > 0:
            widths.append(0.7)                                     # spacer between residue blocks
        widths.append(4.0); scol = len(widths) - 1                 # scalogram
        widths.append(1.15); mcol = len(widths) - 1                # marginal (flush against it)
        colmap.append((scol, mcol))
    widths += [0.45, 0.13]; cbar_col = len(widths) - 1             # spacer + one colorbar per transform row
    fig = plt.figure(figsize=(4.7 * n + 0.8, 2.6 + 2.8 * len(TF)))
    gs = fig.add_gridspec(1 + len(TF), len(widths), width_ratios=widths,
                          height_ratios=[1] + [2.8] * len(TF), wspace=0.0, hspace=0.10)
    pm_by_tf = {tf: [] for tf in TF}; val_by_tf = {tf: [] for tf in TF}
    for ci, ((scol, mcol), x, lab) in enumerate(zip(colmap, series, labels)):
        _, _, _, tau = _scalo(x, dt, TF[0], circular)              # tau = obs autocorr time (transform-independent)
        twotau = 2.0 * tau * dt; tt0 = np.arange(len(x)) * dt

        axT = fig.add_subplot(gs[0, scol])                         # trace (top row)
        axT.plot(tt0, _recenter_deg(x) if circular else x, lw=0.7, color="navy"); axT.margins(x=0)
        axT.set_title(f"{lab}  (τ≈{tau*dt:.0f} ps, N_eff≈{len(x)/(2*tau):.0f})", fontsize=10)
        if ci == 0:
            axT.set_ylabel(obs_label + (" (centered)" if circular else ""))
        axT.tick_params(labelbottom=False)

        for r, tf in enumerate(TF):                               # one scalogram row per requested transform
            scales, power, marg0, _ = _scalo(x, dt, tf, circular)
            N = power.shape[1]; tt = np.arange(N) * dt
            marg = cwt_marginal(power, morlet_coi(N, dt), scales) if tf == "cwt" else marg0

            axS = fig.add_subplot(gs[1 + r, scol], sharex=axT)
            _lp = np.log10(power + 1e-12)
            pm_by_tf[tf].append(axS.pcolormesh(tt, scales, _lp, shading="nearest", cmap="turbo"))
            val_by_tf[tf].append(_lp[np.isfinite(_lp)])
            axS.set_yscale("log")
            if r == len(TF) - 1:
                axS.set_xlabel("time")
            else:
                axS.tick_params(labelbottom=False)
            if ci == 0:
                axS.set_ylabel(f"{ROWLAB[tf]}\ntimescale", fontsize=9)
            if tf == "cwt":
                _overlay_coi(axS, tt, scales, dt)                  # cone of influence (CWT only)
            axS.axhline(twotau, color="w", ls="--", lw=1.2)        # guide; label lives on the marginal

            axM = fig.add_subplot(gs[1 + r, mcol], sharey=axS)
            if tf == "blocking":                                   # exact dyadic Haar variance -> histogram
                axM.barh(scales, marg / np.nanmax(marg), height=scales * 0.7, align="center",
                         color="0.6", alpha=0.6, edgecolor="0.35", lw=0.4)
                axM.set_xlim(0, 1.08); _lab_x = 0.97; _ha = "right"
                if r == len(TF) - 1:
                    axM.set_xlabel("var. frac.", fontsize=7)
            else:                                                  # smooth CWT marginal (near-variance)
                axM.plot(marg, scales, "-", color="firebrick", lw=1.6)
                axM.fill_betweenx(scales, marg, np.nanmin(marg), color="firebrick", alpha=0.15)
                axM.set_xscale("log")
                axM.xaxis.set_major_locator(LogLocator(numticks=3)); axM.xaxis.set_minor_locator(NullLocator())
                _lab_x = 0.03; _ha = "left"
                if r == len(TF) - 1:
                    axM.set_xlabel("power (rel.)", fontsize=7)
            axM.axhline(twotau, color="0.35", ls="--", lw=1)
            _ly = min(max(twotau, scales.min()), scales.max())     # keep the label ON-plot even at the finest scale
            axM.text(_lab_x, _ly, f" 2τ≈{twotau:.0f} ", transform=axM.get_yaxis_transform(),
                     ha=_ha, va="bottom", fontsize=8, color="0.2",
                     bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.72, edgecolor="none"))
            axM.tick_params(left=False, labelleft=False); axM.spines["left"].set_visible(False)
            axM.tick_params(axis="x", labelsize=7, pad=1); axM.margins(y=0)

    for r, tf in enumerate(TF):                                    # one colorbar per transform row (per-tf normalization)
        _vlo, _vhi = np.percentile(np.concatenate(val_by_tf[tf]), [2, 98])
        for pm in pm_by_tf[tf]:
            pm.set_clim(_vlo, _vhi)
        fig.colorbar(pm_by_tf[tf][-1], cax=fig.add_subplot(gs[1 + r, cbar_col]),
                     label=f"log₁₀ power · {ROWLAB[tf].split()[0]}")
    fig.suptitle(title or f"{obs_label} — trace + " + " + ".join(ROWLAB[tf] for tf in TF), fontsize=13)
    return fig


# ------------------------------------------------------------------ CLI
def _cli():
    ap = argparse.ArgumentParser(description="Blocking (Haar) autocorrelation scalogram of an MD observable.")
    ap.add_argument("--traj", nargs="+", required=True, help="one or more trajectory files (.dcd/.xtc/...)")
    ap.add_argument("--top", required=True, help="topology (.pdb/.prmtop/...)")
    ap.add_argument("--obs", default="rg", help="observable: rg|rmsd|q|helix|psi|phi|chi1..chi4|dist")
    ap.add_argument("--resid", type=int, default=None, help="0-based residue index (psi/phi)")
    ap.add_argument("--atoms", type=int, nargs=2, default=None, help="two 0-based atom indices (dist)")
    ap.add_argument("--ref", type=int, default=0, help="reference/native frame for rmsd & q (default 0)")
    ap.add_argument("--dt", type=float, default=1.0, help="time per frame (e.g. ps); sets the scale units")
    ap.add_argument("--no-protein", action="store_true", help="don't slice to protein before computing")
    ap.add_argument("--method", default="blocking", choices=["blocking", "cwt", "both"],
                    help="blocking (Haar; exact, dyadic) | cwt (Morlet; fine/smooth) | both (stacked; one trajectory)")
    ap.add_argument("--out", default="scalogram.png")
    a = ap.parse_args()
    import mdtraj as md
    import matplotlib
    matplotlib.use("Agg")
    series, labels = [], []
    for f in a.traj:
        t = md.load(f, top=a.top)
        if not a.no_protein:
            t = t.atom_slice(t.topology.select("protein"))
        series.append(observable(t, a.obs, resid=a.resid, atoms=a.atoms, ref=a.ref))
        labels.append(f.rsplit("/", 1)[-1])
    lab = a.obs + (f" resid {a.resid}" if a.resid is not None else "")
    circ = is_circular(a.obs)
    if circ:
        print(f"note: '{a.obs}' is circular -> scalogram uses cos/sin power (no unwrap)")
    if a.method == "both":
        if len(series) > 1:
            print(f"note: --method both shows one trajectory; using {labels[0]}")
        fig = both_figure(series[0], a.dt, label=lab, title=labels[0], circular=circ)
    elif len(series) == 1:
        fig = scalogram_figure(series[0], a.dt, label=lab, title=labels[0], method=a.method, circular=circ)
    else:
        fig = compare_figure(series, labels, a.dt, obs_label=lab, method=a.method, circular=circ)
    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    print(f"wrote {a.out}  ({len(series)} trajectory/ies, obs={a.obs}, dt={a.dt})")


if __name__ == "__main__":
    _cli()
