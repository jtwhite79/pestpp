"""visualisation for the 2-D synthetic ies/enif test.

three views, each a standalone function so they can be called individually:

    plot_phi            - phi vs iteration, every realization drawn
    plot_obs_vs_sim     - simulated traces vs the obs+noise the run was fit to
    plot_property_maps  - truth / prior mean / posterior mean / error for hk and sy

everything ensemble-valued is drawn as individual realization traces, not as a
summary band: the spread of the members is the thing worth looking at.

colours: grey = prior, blue = posterior, red = obs+noise.  viridis for property
fields and RdBu_r centred on zero for the posterior-minus-truth error.
"""
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import pyemu

# --- palette -------------------------------------------------------------
CMAP_PROP = "viridis"
CMAP_ANOM = "RdBu_r"
C_PRIOR = "#a0aec0"
C_POST = "#2b6cb0"
C_TRUTH = "black"
C_METHOD = {"ies": "#c53030", "enif": "#2b6cb0",
            "enif_cov": "#2b6cb0", "enif_rook": "#2f855a", "enif_queen": "#d69e2e"}
LBL_METHOD = {"ies": "pestpp-ies", "enif": "EnIF",
              "enif_cov": "EnIF (dense parcov)",
              "enif_rook": "EnIF (rook graph)",
              "enif_queen": "EnIF (queen graph)"}


def _phi_file(master_d, case="synth2d"):
    return os.path.join(master_d, f"{case}.phi.actual.csv")


def plot_phi(master_dirs, case="synth2d", logy=True, figsize=None):
    """phi vs iteration, one panel per method, every realization drawn.

    thin lines are individual realizations, the heavy line is the mean.  the
    realization traces are the point: they show whether the ensemble is moving
    together or whether a few members are carrying the mean.
    """
    items = [(m, d) for m, d in master_dirs.items() if os.path.exists(_phi_file(d, case))]
    fig, axes = plt.subplots(1, len(items), figsize=figsize or (4.6 * len(items), 4.0),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (meth, d) in zip(axes, items):
        df = pd.read_csv(_phi_file(d, case))
        it = df["iteration"].values
        reals = [c for c in df.columns
                 if c not in ("iteration", "total_runs", "mean",
                              "standard_deviation", "min", "max")]
        c = C_METHOD.get(meth, C_POST)
        for r in reals:
            ax.plot(it, df[r].values, color=c, lw=0.5, alpha=0.18)
        ax.plot(it, df["mean"].values, color=c, lw=2.5, label="mean")
        ax.set_title(f"{LBL_METHOD.get(meth, meth)}  ({len(reals)} realizations)",
                     fontsize=10)
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
        if logy:
            ax.set_yscale("log")
    axes[0].set_ylabel("measurement phi")
    plt.tight_layout()
    return fig


def plot_phi_by_ensemble_size(model_d, sizes, prefix="nsweep", case="synth2d",
                              flag=None, figsize=None):
    """phi vs iteration, one panel per ensemble size.

    `flag` is an optional {N: "note"} of caveats to stamp on a panel - used here
    to mark the size where a model run failed, so the panel is not read as a
    clean comparison.
    """
    fig, axes = plt.subplots(1, len(sizes), figsize=figsize or (3.4 * len(sizes), 3.8),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, sizes):
        for meth in ("ies", "enif"):
            f = _phi_file(os.path.join(model_d, f"{prefix}_{meth}_{n}"), case)
            if not os.path.exists(f):
                continue
            df = pd.read_csv(f)
            ax.plot(df["iteration"], df["mean"], "o-", color=C_METHOD[meth],
                    lw=1.8, ms=4, label=LBL_METHOD[meth])
        ax.set_yscale("log")
        ax.set_title(f"N = {n}", fontsize=10)
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.3)
        if flag and n in flag:
            ax.text(0.5, 0.02, flag[n], transform=ax.transAxes, ha="center",
                    va="bottom", fontsize=7, color=C_METHOD["ies"])
    axes[0].set_ylabel("measurement phi")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("the EnIF gap narrows with ensemble size", y=1.02, fontsize=11)
    plt.tight_layout()
    return fig


def plot_obs_vs_sim(master_d, template_d, case="synth2d", iters=None, figsize=None):
    """simulated ensemble vs measured data, one row per observation group.

    grey band is the prior ensemble, blue band the final ensemble, black dots
    the measured values the run was conditioned on.
    """
    pst = pyemu.Pst(os.path.join(template_d, f"{case}.pst"))
    obs = pst.observation_data
    nz = obs.loc[obs.weight > 0]
    groups = sorted(nz.obgnme.unique())

    if iters is None:
        # must match BOTH .obs.csv and .obs.jcb - runs with save_binary write jcb,
        # and globbing only csv silently yields (0,0), i.e. the prior plotted twice
        its = sorted({int(f.split(".")[-3]) for f in os.listdir(master_d)
                      if f.startswith(case + ".")
                      and (f.endswith(".obs.csv") or f.endswith(".obs.jcb"))
                      and f.split(".")[-3].isdigit()})
        if len(its) < 2:
            raise FileNotFoundError(
                f"need at least two observation ensembles in {master_d}, found {its}")
        iters = (its[0], its[-1])

    def load(i):
        for ext in ("jcb", "csv"):
            f = os.path.join(master_d, f"{case}.{i}.obs.{ext}")
            if os.path.exists(f):
                if ext == "csv":
                    return pd.read_csv(f, index_col=0)
                return pyemu.ObservationEnsemble.from_binary(pst=pst, filename=f)._df
        return None

    oe0, oeN = load(iters[0]), load(iters[-1])
    if oe0 is None or oeN is None:
        raise FileNotFoundError(f"no observation ensembles found in {master_d}")

    # the observation+noise realizations are what the ensemble was conditioned to
    noise = None
    nf = os.path.join(master_d, f"{case}.obs+noise.jcb")
    if os.path.exists(nf):
        noise = pyemu.ObservationEnsemble.from_binary(pst=pst, filename=nf)._df

    # one column per observation site so the individual traces are legible
    sites = []
    for g in groups:
        sub = nz.loc[nz.obgnme == g].copy()
        for uc in sorted(sub["usecol"].astype(str).unique()):
            sites.append((g, uc, sub.loc[sub["usecol"].astype(str) == uc]))
    ncol = min(3, len(sites))
    nrow = int(np.ceil(len(sites) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize or (5.0 * ncol, 3.2 * nrow),
                             squeeze=False)
    flat = axes.ravel()

    for ax, (g, uc, sub) in zip(flat, sites):
        sub = sub.copy()
        sub["t"] = sub["time"].astype(float)
        sub = sub.sort_values("t")
        names, t = list(sub.index), sub["t"].values

        # every realization as its own trace: grey prior, blue posterior
        for oe, col, lab in ((oe0, C_PRIOR, f"prior (iter {iters[0]})"),
                             (oeN, C_POST, f"posterior (iter {iters[-1]})")):
            v = oe.loc[:, names].values
            ax.plot(t, v.T, color=col, lw=0.6, alpha=0.30)
            ax.plot([], [], color=col, lw=2, label=lab)   # legend proxy only

        # red: the obs+noise realizations the run was conditioned to
        if noise is not None:
            ax.plot(t, noise.loc[:, names].values.T, color=C_METHOD["ies"],
                    lw=0.0, marker=".", ms=2.0, alpha=0.30)
            ax.plot([], [], color=C_METHOD["ies"], lw=0.0, marker=".", ms=8,
                    label="obs + noise")
        ax.plot(t, sub.obsval.values, "-", color=C_METHOD["ies"], lw=1.4, alpha=0.9)

        ax.set_title(uc, fontsize=11)
        ax.set_xlabel("time (d)")
        ax.grid(alpha=0.3)
    for ax in flat[len(sites):]:
        ax.set_visible(False)
    flat[0].legend(frameon=False, fontsize=8)
    plt.tight_layout()
    return fig


def realize_arrays(template_d, parval, work_d=None, case="synth2d"):
    """turn one parameter vector into hk / sy arrays by running the same
    multiplier machinery the forward run uses"""
    work_d = work_d or os.path.join(os.path.dirname(template_d) or ".", "_viz_scratch")
    if os.path.exists(work_d):
        shutil.rmtree(work_d)
    shutil.copytree(template_d, work_d)
    pst = pyemu.Pst(os.path.join(work_d, f"{case}.pst"))
    pst.parameter_data.loc[parval.index, "parval1"] = parval.values
    pst.write_input_files(pst_path=work_d)
    cwd = os.getcwd()
    try:
        os.chdir(work_d)
        pyemu.geostats._try_import_ppu = lambda: (_ for _ in ()).throw(
            ImportError("ppu disabled"))
        # one big chunk keeps this on pyemu's serial path.  the multiprocessing
        # path uses spawn, which refuses to start unless the calling module is
        # guarded by if __name__ == "__main__" - too easy to trip from a
        # notebook or a driver script, and there is nothing to parallelise here.
        pyemu.helpers.apply_list_and_array_pars(arr_par_file="mult2model_info.csv",
                                                chunk_len=1_000_000)
        hk = np.loadtxt("hk.dat")
        sy = np.loadtxt("sy.dat")
    finally:
        os.chdir(cwd)
    shutil.rmtree(work_d, ignore_errors=True)
    return hk, sy


def plot_property_maps(master_d, template_d, truth_d, case="synth2d",
                       iters=None, figsize=None):
    """truth, prior mean, posterior mean and posterior error for hk and sy"""
    pst = pyemu.Pst(os.path.join(template_d, f"{case}.pst"))
    if iters is None:
        its = sorted(int(f.split(".")[-3]) for f in os.listdir(master_d)
                     if f.startswith(case + ".") and f.endswith(".par.jcb") and
                     f.split(".")[-3].isdigit())
        iters = (its[0], its[-1])

    def load_pe(i):
        for ext in ("jcb", "csv"):
            f = os.path.join(master_d, f"{case}.{i}.par.{ext}")
            if os.path.exists(f):
                if ext == "jcb":
                    return pyemu.ParameterEnsemble.from_binary(pst=pst, filename=f)._df
                return pd.read_csv(f, index_col=0)
        raise FileNotFoundError(f"no parameter ensemble for iteration {i}")

    pe0, peN = load_pe(iters[0]), load_pe(iters[-1])
    hk_t = np.loadtxt(os.path.join(truth_d, "truth_hk.dat"))
    sy_t = np.loadtxt(os.path.join(truth_d, "truth_sy.dat"))
    hk_p, sy_p = realize_arrays(template_d, pe0.mean())
    hk_n, sy_n = realize_arrays(template_d, peN.mean())

    # only show properties that are actually adjustable in this control file -
    # rendering a row for an unparameterised property reads as though it were
    # being estimated when it is simply held at its base value
    grps = set(pst.parameter_data.loc[pst.adj_par_names, "pargp"].astype(str))
    rows = [(nme, tt, pp, nn) for nme, tt, pp, nn in
            (("hk", hk_t, hk_p, hk_n), ("sy", sy_t, sy_p, sy_n))
            if any(g.startswith(nme) for g in grps)]
    if figsize is None:
        figsize = (13.0, 3.0 * len(rows))
    fig, axes = plt.subplots(len(rows), 4, figsize=figsize, squeeze=False)
    for r, (name, t, p, n) in enumerate(rows):
        vmin, vmax = min(t.min(), n.min()), max(t.max(), n.max())
        for c, (arr, lab) in enumerate(((t, "truth"), (p, "prior mean"),
                                        (n, "posterior mean"))):
            im = axes[r, c].imshow(arr, cmap=CMAP_PROP, vmin=vmin, vmax=vmax,
                                   interpolation="nearest")
            axes[r, c].set_title(f"{name} {lab}", fontsize=9)
            plt.colorbar(im, ax=axes[r, c], fraction=0.046)
        err = n - t
        v = float(np.abs(err).max())
        if v <= 0.0:
            # identically zero: this property was not adjusted in this variant,
            # so TwoSlopeNorm would reject vmin == vcenter == vmax
            im = axes[r, 3].imshow(err, cmap=CMAP_ANOM, vmin=-1.0, vmax=1.0,
                                   interpolation="nearest")
            axes[r, 3].set_title(f"{name} posterior - truth\n(not adjusted)", fontsize=9)
        else:
            im = axes[r, 3].imshow(err, cmap=CMAP_ANOM,
                                   norm=TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v),
                                   interpolation="nearest")
            axes[r, 3].set_title(f"{name} posterior - truth", fontsize=9)
        plt.colorbar(im, ax=axes[r, 3], fraction=0.046)
    for a in axes.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(_variant_label(pst, template_d), y=1.01, fontsize=11)
    plt.tight_layout()
    return fig


def plot_all(master_dirs, template_d, truth_d, case="synth2d", out_d="synth2d_figs"):
    """every view, written to png.  master_dirs is {method: master directory}"""
    matplotlib.use("Agg")
    os.makedirs(out_d, exist_ok=True)
    written = []

    fig = plot_phi(master_dirs, case=case)
    f = os.path.join(out_d, "phi_vs_iter.png")
    fig.savefig(f, dpi=140, bbox_inches="tight")
    plt.close(fig)
    written.append(f)

    for meth, d in master_dirs.items():
        fig = plot_obs_vs_sim(d, template_d, case=case)
        f = os.path.join(out_d, f"obs_vs_sim_{meth}.png")
        fig.savefig(f, dpi=140, bbox_inches="tight")
        plt.close(fig)
        written.append(f)

        fig = plot_property_maps(d, template_d, truth_d, case=case)
        f = os.path.join(out_d, f"property_maps_{meth}.png")
        fig.savefig(f, dpi=140, bbox_inches="tight")
        plt.close(fig)
        written.append(f)

    print("wrote:")
    for f in written:
        print("   ", f)
    return written


def plot_model_map(model_ws, truth_d=None, case="synth2d", kper=-1, figsize=(12.0, 5.4)):
    """map of the model: boundaries, wells, observation sites, and head contours.

    left panel is the flow system - river along the west edge, specified inflow
    along the east edge, three pumping wells, and the simulated head field with
    contours.  right panel is the true hk field with the same features overlaid,
    so it is clear which parts of the domain the observations can actually see.
    """
    import flopy
    import synth2d_model as m2d

    sim = flopy.mf6.MFSimulation.load(sim_ws=model_ws, verbosity_level=0)
    gwf = sim.get_model()
    nrow, ncol = gwf.modelgrid.nrow, gwf.modelgrid.ncol
    extent = (0, ncol * m2d.DELR, nrow * m2d.DELC, 0)

    head = flopy.utils.HeadFile(os.path.join(model_ws, f"{case}.hds")).get_data(
        kstpkper=flopy.utils.HeadFile(
            os.path.join(model_ws, f"{case}.hds")).get_kstpkper()[kper])[0]
    head = np.where(head < -1e29, np.nan, head)

    def overlay(ax):
        # river on the west edge, specified inflow on the east edge
        ax.add_patch(plt.Rectangle((0, 0), m2d.DELR, nrow * m2d.DELC,
                                   fc=C_POST, ec="none", alpha=0.85, zorder=3))
        ax.add_patch(plt.Rectangle(((ncol - 1) * m2d.DELR, 0), m2d.DELR,
                                   nrow * m2d.DELC, fc=C_METHOD["ies"], ec="none",
                                   alpha=0.85, zorder=3))
        for (_, i, j) in m2d.well_cells(nrow, ncol):
            ax.plot((j + 0.5) * m2d.DELR, (i + 0.5) * m2d.DELC, "v", color="black",
                    ms=11, mfc="white", mew=1.8, zorder=5)
        for nme, (_, i, j) in m2d.head_obs_cells(nrow, ncol).items():
            ax.plot((j + 0.5) * m2d.DELR, (i + 0.5) * m2d.DELC, "o", color="black",
                    ms=8, mfc="white", mew=1.8, zorder=5)
            ax.annotate(nme, ((j + 0.5) * m2d.DELR, (i + 0.5) * m2d.DELC),
                        textcoords="offset points", xytext=(9, 5), fontsize=8,
                        zorder=6)
        ax.set_xlabel("x (m)")

    ncols = 2 if truth_d else 1
    fig, axes = plt.subplots(1, ncols, figsize=figsize, squeeze=False)
    ax = axes[0, 0]
    im = ax.imshow(head, cmap=CMAP_PROP, extent=extent, interpolation="bilinear")
    cs = ax.contour(head, levels=12, colors="white", linewidths=0.8, extent=extent,
                    origin="upper")
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")
    plt.colorbar(im, ax=ax, fraction=0.046, label="head (m)")
    overlay(ax)
    ax.set_ylabel("y (m)")
    ax.set_title("simulated head, with river (blue, west) and\n"
                 "specified inflow (red, east); flow is east to west", fontsize=10)

    if truth_d:
        ax = axes[0, 1]
        hk = np.loadtxt(os.path.join(truth_d, "truth_hk.dat"))
        im = ax.imshow(hk, cmap=CMAP_PROP, extent=extent, interpolation="nearest")
        plt.colorbar(im, ax=ax, fraction=0.046, label="hk (m/d)")
        overlay(ax)
        ax.set_title("true hk field\n"
                     "(triangles = pumping wells, circles = head observations)",
                     fontsize=10)
    plt.tight_layout()
    return fig


def _par_ensembles(master_d, pst, case="synth2d"):
    """(prior, posterior) parameter ensembles in the solve space (log10 for
    log-transformed parameters), plus the iteration numbers used"""
    its = sorted({int(f.split(".")[-3]) for f in os.listdir(master_d)
                  if f.startswith(case + ".")
                  and (f.endswith(".par.jcb") or f.endswith(".par.csv"))
                  and f.split(".")[-3].isdigit()})
    if len(its) < 2:
        raise FileNotFoundError(f"need two parameter ensembles in {master_d}, got {its}")

    def load(i):
        for ext in ("jcb", "csv"):
            f = os.path.join(master_d, f"{case}.{i}.par.{ext}")
            if os.path.exists(f):
                if ext == "jcb":
                    return pyemu.ParameterEnsemble.from_binary(pst=pst, filename=f)._df
                return pd.read_csv(f, index_col=0)
        raise FileNotFoundError(f"no parameter ensemble for iteration {i}")

    names = list(pst.adj_par_names)
    mask = (pst.parameter_data.loc[names, "partrans"] == "log").values

    def to_solve_space(df):
        v = df.loc[:, names].values.copy()
        v[:, mask] = np.log10(v[:, mask])
        return pd.DataFrame(v, index=df.index, columns=names)

    return to_solve_space(load(its[0])), to_solve_space(load(its[-1])), (its[0], its[-1])


def _variant_label(pst, template_d):
    """short description of which problem a figure came from"""
    grps = sorted(set(pst.parameter_data.loc[pst.adj_par_names, "pargp"].astype(str)))
    return (f"{os.path.basename(os.path.dirname(os.path.abspath(template_d)))}: "
            f"{pst.npar_adj} adjustable pars [{', '.join(grps)}], "
            f"{pst.nnz_obs} nz obs")


def plot_par_moments(master_dirs, template_d, case="synth2d", figsize=None):
    """prior vs posterior parameter moments, in the space the solve happens in.

    top row  - first moment: how far the mean of each parameter moved
    mid row  - second moment: prior sd against posterior sd.  points below the
               1:1 line are parameters whose uncertainty shrank; how far below is
               how much conditioning the ensemble absorbed
    bottom   - distribution of the variance-retention ratio by parameter group,
               which is the summary number for ensemble collapse
    """
    pst = pyemu.Pst(os.path.join(template_d, f"{case}.pst"))
    names = list(pst.adj_par_names)
    grp = pst.parameter_data.loc[names, "pargp"].values
    groups = sorted(set(grp))
    items = [(m, d) for m, d in master_dirs.items() if os.path.exists(d)]

    fig, axes = plt.subplots(3, len(items), figsize=figsize or (5.2 * len(items), 12.0),
                             squeeze=False)
    for c, (meth, d) in enumerate(items):
        pri, post, its = _par_ensembles(d, pst, case)
        col = C_METHOD.get(meth, C_POST)
        m0, m1 = pri.mean().values, post.mean().values
        s0, s1 = pri.std(ddof=1).values, post.std(ddof=1).values

        # first moment
        ax = axes[0, c]
        for g in groups:
            k = grp == g
            ax.scatter(m0[k], m1[k], s=14, alpha=0.7, label=g)
        lim = [min(m0.min(), m1.min()), max(m0.max(), m1.max())]
        ax.plot(lim, lim, "k--", lw=1, alpha=0.6)
        ax.set_xlabel("prior mean (log10)")
        ax.set_ylabel("posterior mean (log10)")
        ax.set_title(f"{LBL_METHOD.get(meth, meth)}: first moment\n"
                     f"(iter {its[0]} -> {its[1]})", fontsize=10)
        ax.grid(alpha=0.3)
        if c == 0:
            ax.legend(fontsize=7, frameon=False)

        # second moment
        ax = axes[1, c]
        for g in groups:
            k = grp == g
            ax.scatter(s0[k], s1[k], s=14, alpha=0.7)
        lim = [0, max(s0.max(), s1.max()) * 1.05]
        ax.plot(lim, lim, "k--", lw=1, alpha=0.6)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("prior sd (log10)")
        ax.set_ylabel("posterior sd (log10)")
        ratio = float(np.mean(s1 ** 2 / np.maximum(s0 ** 2, 1e-30)))
        ax.set_title(f"second moment - mean variance retained {ratio:.3f}\n"
                     "(below the 1:1 line = uncertainty reduced)", fontsize=10)
        ax.grid(alpha=0.3)

        # variance retention by group
        ax = axes[2, c]
        data = [(s1[grp == g] ** 2) / np.maximum(s0[grp == g] ** 2, 1e-30)
                for g in groups]
        bp = ax.boxplot(data, labels=groups, patch_artist=True, widths=0.6)
        for patch in bp["boxes"]:
            patch.set_facecolor(col); patch.set_alpha(0.45)
        for med in bp["medians"]:
            med.set_color("black")
        ax.axhline(1.0, color="black", ls="--", lw=1, alpha=0.6)
        ax.set_ylabel("posterior var / prior var")
        ax.set_title("variance retained by parameter group", fontsize=10)
        ax.grid(alpha=0.3, axis="y")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)

    fig.suptitle(_variant_label(pst, template_d), y=1.005, fontsize=11)
    plt.tight_layout()
    return fig
