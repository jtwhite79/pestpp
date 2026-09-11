"""pstfrom setup for the 2-D synthetic ies test problem.

parameterises hk and sy at BOTH grid scale and pilot points, which is the usual
highly-parameterised arrangement: the pilot points carry the broad structure and
the grid-scale multipliers carry the rest.  parameter count scales with the grid,
so nrow/ncol is the dial for problem dimension.

also writes the two things enif needs and ies does not: a geostatistical prior
ensemble AND the prior covariance matrix it was drawn from.
"""
import os
import shutil
import numpy as np
import pandas as pd
import pyemu
import flopy

import synth2d_model as m2d

MF6 = os.path.expanduser("~/bin/mf6")
PP_SPACE = 5          # pilot point every N cells
V_RANGE_FAC = 8.0     # variogram range as a multiple of cell size
NUM_REALS = 200       # prior ensemble size (one extra is drawn to be the truth)


def run_realization(template_d, parval, work_d, case="synth2d"):
    """run the forward model for one parameter realization and return the
    observation output plus the hk / sy fields it implies"""
    if os.path.exists(work_d):
        shutil.rmtree(work_d)
    shutil.copytree(template_d, work_d)
    pst = pyemu.Pst(os.path.join(work_d, f"{case}.pst"))
    pst.parameter_data.loc[parval.index, "parval1"] = parval.values
    pst.write_input_files(pst_path=work_d)
    pyemu.os_utils.run("python forward_run.py", cwd=work_d)
    head = pd.read_csv(os.path.join(work_d, "head_obs.csv"))
    riv = pd.read_csv(os.path.join(work_d, "riv_obs.csv"))
    hk = np.loadtxt(os.path.join(work_d, "hk.dat"))
    sy = np.loadtxt(os.path.join(work_d, "sy.dat"))
    return head, riv, hk, sy


def setup(new_d="synth2d_template", nrow=40, ncol=40, noise_frac=0.01, seed=99881,
          parameterization="full"):
    """build the model, parameterise it, draw the prior, and take the truth from
    that prior.  returns the pst.

    the truth is one realization OF the prior ensemble rather than an
    independently generated field.  that guarantees the truth is reachable
    within the parameter bounds - otherwise no method can fit the data and the
    comparison measures the setup instead of the algorithms.
    """
    org_d = new_d + "_org"
    base_d = org_d + "_base"
    m2d.build_model(base_d, nrow=nrow, ncol=ncol, run=True, exe_name=MF6)
    np.savetxt(os.path.join(base_d, "hk.dat"), np.full((nrow, ncol), m2d.K_BASE))
    np.savetxt(os.path.join(base_d, "sy.dat"), np.full((nrow, ncol), m2d.SY_BASE))

    sr = pyemu.helpers.SpatialReference(delr=np.full(ncol, m2d.DELR),
                                        delc=np.full(nrow, m2d.DELC))
    # pp_solve_num_threads=1 keeps the pilot point kriging single-process.  the
    # multiprocessing path needs a Manager, which is not available in every
    # environment, and the solve is small enough here that it costs nothing.
    pf = pyemu.utils.PstFrom(original_d=base_d, new_d=new_d, remove_existing=True,
                             longnames=True, spatial_reference=sr,
                             zero_based=False, start_datetime="1-1-2020",
                             pp_solve_num_threads=1)

    v = pyemu.geostats.ExpVario(contribution=1.0, a=V_RANGE_FAC * m2d.DELR)
    gs = pyemu.geostats.GeoStruct(variograms=v, transform="log")

    # which properties, and at which scales.  "full" is the highly-parameterised
    # arrangement; "hk_pp" is the deliberately small, well-posed variant with only
    # hk pilot points - there N can exceed the parameter count, so the H
    # regression is overdetermined instead of interpolating the ensemble.
    if parameterization == "full":
        specs = [("hk", 0.2, 5.0, "grid"), ("hk", 0.2, 5.0, "pilotpoints"),
                 ("sy", 0.5, 2.0, "grid"), ("sy", 0.5, 2.0, "pilotpoints")]
    elif parameterization == "hk_pp":
        specs = [("hk", 0.2, 5.0, "pilotpoints")]
    elif parameterization == "hk_gr":
        specs = [("hk", 0.2, 5.0, "grid")]
    elif parameterization == "hksy_gr":
        specs = [("hk", 0.2, 5.0, "grid"), ("sy", 0.5, 2.0, "grid")]
    else:
        raise ValueError(f"unknown parameterization '{parameterization}'")

    for tag, lb, ub, ptype in specs:
        kw = dict(filenames=f"{tag}.dat", par_type=ptype,
                  par_name_base=f"{tag}_{'pp' if ptype=='pilotpoints' else 'gr'}",
                  pargp=f"{tag}_{'pp' if ptype=='pilotpoints' else 'gr'}",
                  lower_bound=lb, upper_bound=ub, geostruct=gs, transform="log")
        if ptype == "pilotpoints":
            # try_use_ppu=False keeps kriging in pyemu rather than the pypestutils
            # shared library, which is not always in step with its python bindings
            kw.update(pp_space=PP_SPACE, pp_options={"try_use_ppu": False})
        pf.add_parameters(**kw)

    # mf6 writes observation column headers in upper case
    hcols = [c.upper() for c in m2d.head_obs_cells(nrow, ncol)]
    pf.add_observations("head_obs.csv", insfile="head_obs.csv.ins",
                        index_cols="time", use_cols=hcols,
                        prefix="gwlevel", obsgp="gwlevel")
    pf.add_observations("riv_obs.csv", insfile="riv_obs.csv.ins",
                        index_cols="time", use_cols=["RIVFLUX"],
                        prefix="rivflux", obsgp="rivflux")

    pf.mod_sys_cmds.append("mf6")
    pf.add_py_function("synth2d_setup.py", "apply_arrays()", is_pre_cmd=True)
    # apply pilot point factors with pyemu's own kriging rather than the
    # pypestutils shared library.  pyemu already falls back on ImportError, but a
    # stale ppu library raises AttributeError instead and escapes that guard.
    # forcing the python path also keeps the benchmark reproducible on machines
    # with different ppu builds.
    pf.pre_py_cmds.insert(
        0, "pyemu.geostats._try_import_ppu = lambda: (_ for _ in ()).throw("
           "ImportError('ppu disabled for this benchmark'))")
    pst = pf.build_pst(os.path.join(new_d, "synth2d.pst"), version=2)

    # prior covariance and ensemble.  draw one extra realization: the first
    # becomes the truth and is then removed, so the truth is a member of the
    # prior but not a member of the ensemble being conditioned.
    cov = pf.build_prior(fmt="coo", filename=os.path.join(new_d, "prior_cov.jcb"))
    pe = pf.draw(num_reals=NUM_REALS + 1, use_specsim=False)
    pe.enforce()
    print(f"  prior covariance {cov.shape}, prior ensemble {pe.shape}")

    truth_real = pe.index[0]
    os.makedirs(org_d, exist_ok=True)
    head, riv, hk, sy = run_realization(new_d, pe.loc[truth_real, :], org_d + "_run")
    for nme, arr in (("truth_hk.dat", hk), ("truth_sy.dat", sy)):
        np.savetxt(os.path.join(org_d, nme), arr)
    head.to_csv(os.path.join(org_d, "head_obs.csv"), index=False)
    riv.to_csv(os.path.join(org_d, "riv_obs.csv"), index=False)
    shutil.rmtree(org_d + "_run", ignore_errors=True)
    print(f"  truth is prior realization '{truth_real}': "
          f"hk {hk.min():.2f}-{hk.max():.2f}, sy {sy.min():.3f}-{sy.max():.3f}")

    _set_obs_from_truth(pst, org_d, noise_frac, seed)
    pe = pe.loc[pe.index[1:], :]
    pe.to_binary(os.path.join(new_d, "prior_pe.jcb"))
    print(f"  conditioning ensemble (truth removed): {pe.shape}")

    pst.control_data.noptmax = 0
    pst.write(os.path.join(new_d, "synth2d.pst"), version=2)
    shutil.copy2(MF6, os.path.join(new_d, "mf6"))
    print(f"\nsynth2d: {nrow}x{ncol} grid, {pst.npar_adj} adjustable parameters, "
          f"{pst.nnz_obs} non-zero-weight observations")
    return pst


def _set_obs_from_truth(pst, truth_d, noise_frac, seed):
    """observation values are the truth plus noise; weights are 1/sigma.

    matched through the oname/usecol/time columns pstfrom carries on
    observation_data rather than by rebuilding observation names, so this keeps
    working if pyemu's naming convention shifts.
    """
    rng = np.random.default_rng(seed)
    obs = pst.observation_data
    obs.loc[:, "weight"] = 0.0
    src = {"gwlevel": pd.read_csv(os.path.join(truth_d, "head_obs.csv")),
           "rivflux": pd.read_csv(os.path.join(truth_d, "riv_obs.csv"))}
    for d in src.values():
        d.columns = [c.lower() for c in d.columns]
        d.set_index("time", inplace=True)

    hit, truth_vals = 0, {}
    for nme, row in obs.iterrows():
        oname = str(row.get("oname", ""))
        if oname not in src:
            continue
        col, t = str(row["usecol"]).lower(), float(row["time"])
        d = src[oname]
        if col not in d.columns:
            continue
        # the truth run's time axis and the template's should agree exactly,
        # but match on nearest to survive any float formatting drift
        it = int(np.argmin(np.abs(d.index.values - t)))
        val = float(d[col].values[it])
        sigma = 0.10 if oname == "gwlevel" else max(abs(val) * noise_frac, 1.0)
        obs.loc[nme, "obsval"] = val + rng.normal(0.0, sigma)
        obs.loc[nme, "weight"] = 1.0 / sigma
        obs.loc[nme, "standard_deviation"] = sigma
        truth_vals[nme] = val
        hit += 1
    if hit == 0:
        raise RuntimeError("no observations matched the truth output - names changed?")
    pd.Series(truth_vals, name="truth").to_csv(
        os.path.join(os.path.dirname(pst.filename), "obs_truth.csv"))
    print(f"  set {hit} non-zero-weight observations from the truth run")


def apply_arrays():
    """runs on the agent before mf6: pstfrom has already written the multiplied
    hk.dat / sy.dat, so push them into the mf6 input files"""
    import flopy
    import numpy as np
    sim = flopy.mf6.MFSimulation.load(sim_ws=".", verbosity_level=0)
    gwf = sim.get_model()
    gwf.npf.k.set_data(np.loadtxt("hk.dat"))
    gwf.sto.sy.set_data(np.loadtxt("sy.dat"))
    sim.write_simulation()


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    setup(nrow=n, ncol=n)
