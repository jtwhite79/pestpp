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

def find_mf6():
    """mf6 from the path first - ci puts test_bin/<plat> there (mf6.exe on windows) -
    then the usual local install.  None when there is no mf6 anywhere, so callers can
    skip instead of running off into a path that was never going to exist: the old
    form handed back ~/bin/mf6 whether or not it was there, which is how a missing
    mf6 on ci turned into a confusing FileNotFoundError naming the runner's home dir"""
    exe = shutil.which("mf6")
    if exe is not None:
        return exe
    local = os.path.expanduser("~/bin/mf6")
    return local if os.path.exists(local) else None


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
          parameterization="full", truth_d=None, sy_max=0.25, num_reals=200,
          pp_space=5, pp_range_fac=60.0, gr_range_fac=3.0, pp_aniso=5.0, pp_bearing=45.0):
    """build the model, parameterise it, draw the prior, and take the truth from
    that prior.  returns the pst.

    the truth is one realization OF the prior ensemble rather than an
    independently generated field.  that guarantees the truth is reachable
    within the parameter bounds - otherwise no method can fit the data and the
    comparison measures the setup instead of the algorithms.

    truth_d points at another problem's truth directory (the "_org" one, holding
    head_obs.csv, riv_obs.csv, truth_hk.dat and truth_sy.dat).  when it is given,
    those become the truth here instead of a fresh draw, so several
    parameterisations can be fit to the SAME data.  the truth is then generally
    not reachable by a reduced parameterisation - that is the point of doing it,
    but it means phi cannot go to the noise floor for those problems.

    sy_max is the ultimate upper bound on the sy array mf6 reads, full_bc only.

    num_reals is the prior ensemble size; one extra is drawn to be the truth.

    pp_space puts a pilot point every that many cells.  the variogram ranges are
    multiples of cell size.  pyemu's ExpVario is exp(-d/a), so a is the
    e-folding length and the practical range is nearer 3a.  the two scales are
    meant to do different jobs: the pilot points carry the broad structure
    (60 cells = 6000 m, more than twice the domain) and the grid multipliers the
    short-scale roughness (3 cells = 300 m), so the pilot point range is an order
    of magnitude longer.  the pilot point trend is stretched along pp_bearing:
    the major axis is pp_range_fac cells long, the minor axis pp_aniso times
    shorter.  bearing is degrees counter-clockwise from east, so 45 runs from
    lower left to upper right.
    """
    mf6 = find_mf6()
    if mf6 is None:
        raise RuntimeError("no mf6 on the path or in ~/bin - cannot build the synth2d model")

    org_d = new_d + "_org"
    base_d = org_d + "_base"
    sim = m2d.build_model(base_d, nrow=nrow, ncol=ncol, run=True, exe_name=mf6)
    # base property values and geometry come from the model just built, so the
    # model keeps the only copy of them
    gwf = sim.get_model()
    np.savetxt(os.path.join(base_d, "hk.dat"), np.squeeze(gwf.npf.k.array))
    np.savetxt(os.path.join(base_d, "sy.dat"), np.squeeze(gwf.sto.sy.array))
    np.savetxt(os.path.join(base_d, "ss.dat"), np.squeeze(gwf.sto.ss.array))
    delr = gwf.dis.delr.array
    delc = gwf.dis.delc.array
    nper = sim.tdis.nper.get_data()

    sr = pyemu.helpers.SpatialReference(delr=delr, delc=delc)
    # pp_solve_num_threads=1 keeps the pilot point kriging single-process.  the
    # multiprocessing path needs a Manager, which is not available in every
    # environment, and the solve is small enough here that it costs nothing.
    pf = pyemu.utils.PstFrom(original_d=base_d, new_d=new_d, remove_existing=True,
                             longnames=True, spatial_reference=sr,
                             zero_based=False, start_datetime="1-1-2020",
                             pp_solve_num_threads=1)

    gs_pp = pyemu.geostats.GeoStruct(
        variograms=pyemu.geostats.ExpVario(contribution=1.0, a=pp_range_fac * delr[0],
                                           anisotropy=pp_aniso, bearing=pp_bearing),
        transform="log")
    gs_gr = pyemu.geostats.GeoStruct(
        variograms=pyemu.geostats.ExpVario(contribution=1.0, a=gr_range_fac * delr[0]),
        transform="log")

    # which properties, and at which scales.  "full" is the highly-parameterised
    # arrangement; "hk_pp" is the deliberately small, well-posed variant with only
    # hk pilot points - there N can exceed the parameter count, so the H
    # regression is overdetermined instead of interpolating the ensemble.
    if parameterization == "full":
        # the pilot points carry the broad trend, so they get a wider bound
        # range than the grid multipliers, which only add local roughness
        # sy is capped tighter than hk on purpose: base sy is 0.15 and the grid
        # multiplier can reach 2, so a pp bound past ~5 puts specific yield over
        # 1, which is not a real number for a granular aquifer
        specs = [("hk", 0.2, 5.0, "grid"), ("hk", 0.01, 100.0, "pilotpoints"),
                 ("sy", 0.5, 2.0, "grid"), ("sy", 0.2, 5.0, "pilotpoints")]
    elif parameterization == "full_bc":
        # the full arrangement plus domain-wide constants on hk and sy, and ss
        # at all three scales with the same geostructs as hk and sy.  the
        # boundary and pumping parameters are added below
        specs = [("hk", 0.2, 5.0, "grid"), ("hk", 0.01, 100.0, "pilotpoints"),
                 ("hk", 0.2, 5.0, "constant"),
                 # sy multipliers kept tight so their product on 0.15 seldom
                 # reaches the sy_max cap - where the cap bites, they do nothing
                 ("sy", 0.8, 1.25, "grid"), ("sy", 0.5, 1.5, "pilotpoints"),
                 ("sy", 0.8, 1.25, "constant"),
                 ("ss", 0.2, 5.0, "grid"), ("ss", 0.1, 10.0, "pilotpoints"),
                 ("ss", 0.2, 5.0, "constant")]
    elif parameterization == "hk_pp":
        specs = [("hk", 0.01, 100.0, "pilotpoints")]
    elif parameterization == "hk_gr":
        specs = [("hk", 0.2, 5.0, "grid")]
    elif parameterization == "hksy_gr":
        specs = [("hk", 0.2, 5.0, "grid"), ("sy", 0.5, 2.0, "grid")]
    else:
        raise ValueError(f"unknown parameterization '{parameterization}'")

    ptag = {"pilotpoints": "pp", "grid": "gr", "constant": "cn"}
    for tag, lb, ub, ptype in specs:
        kw = dict(filenames=f"{tag}.dat", par_type=ptype,
                  par_name_base=f"{tag}_{ptag[ptype]}", pargp=f"{tag}_{ptag[ptype]}",
                  lower_bound=lb, upper_bound=ub, transform="log",
                  geostruct={"pilotpoints": gs_pp, "grid": gs_gr}.get(ptype))
        if ptype == "pilotpoints":
            # try_use_ppu=False keeps kriging in pyemu rather than the pypestutils
            # shared library, which is not always in step with its python bindings
            kw.update(pp_space=pp_space, pp_options={"try_use_ppu": False})
        if parameterization == "full_bc" and tag == "sy":
            # cap the final sy array, not the multipliers: the product of three
            # multipliers on 0.15 otherwise runs well past a sensible specific yield
            kw.update(ult_ubound=sy_max)
        pf.add_parameters(**kw)

    if parameterization == "full_bc":
        # list files are "lay row col value(s) boundname", 1-based
        idx = [0, 1, 2]
        # left boundary: the river is written once and carried through every
        # period, so a constant here is constant in time too.  stage is an
        # addend in metres - it has to stay above rbot, which a multiplier on an
        # elevation does not respect in any meaningful way
        pf.add_parameters("riv.txt", par_type="constant", index_cols=idx, use_cols=[3],
                          par_style="add", par_name_base="rivstage", pargp="rivstage",
                          lower_bound=-0.5, upper_bound=0.5, transform="none")
        pf.add_parameters("riv.txt", par_type="constant", index_cols=idx, use_cols=[4],
                          par_name_base="rivcond", pargp="rivcond",
                          lower_bound=0.1, upper_bound=10.0, transform="log")
        # right boundary: one multiplier on the total specified inflow
        pf.add_parameters("inflow.txt", par_type="constant", index_cols=idx, use_cols=[3],
                          par_name_base="inflow", pargp="inflow",
                          lower_bound=0.5, upper_bound=2.0, transform="log")
        # pumping: a multiplier per well per stress period, independent in the prior
        for kper in range(1, nper):
            pf.add_parameters(m2d.pump_file(kper), par_type="grid", index_cols=idx,
                              use_cols=[3], par_name_base=f"pumpsp{kper:02d}",
                              pargp="pump", lower_bound=0.8, upper_bound=1.25,
                              transform="log")

        # water budget components as zero-weight observations
        bcols = write_budget(base_d)
        shutil.copy2(os.path.join(base_d, "budget.csv"), os.path.join(new_d, "budget.csv"))
        pf.add_observations("budget.csv", insfile="budget.csv.ins", index_cols="time",
                            use_cols=bcols, prefix="wbud",
                            obsgp="wbud")
        pf.add_py_function(os.path.abspath(__file__), "write_budget()", is_pre_cmd=False)

        # head snapshots in every active cell, zero weight.  one array file per
        # snapshot period, so each cell becomes its own observation
        for fname in write_heads(base_d):
            shutil.copy2(os.path.join(base_d, fname), os.path.join(new_d, fname))
            pf.add_observations(fname, insfile=fname + ".ins",
                                prefix=fname.split(".")[0], obsgp="hdsnap")
        pf.add_py_function(os.path.abspath(__file__), "write_heads()", is_pre_cmd=False)

    # mf6 writes observation column headers in upper case
    hcols = [c.upper() for c in m2d.head_obs_cells(nrow, ncol)]
    pf.add_observations("head_obs.csv", insfile="head_obs.csv.ins",
                        index_cols="time", use_cols=hcols,
                        prefix="gwlevel", obsgp="gwlevel")
    pf.add_observations("riv_obs.csv", insfile="riv_obs.csv.ins",
                        index_cols="time", use_cols=["RIVFLUX"],
                        prefix="rivflux", obsgp="rivflux")

    pf.mod_sys_cmds.append("mf6")
    # full path so this works when ci runs the tests from another folder
    pf.add_py_function(os.path.abspath(__file__), "apply_arrays()", is_pre_cmd=True)
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
    pe = pf.draw(num_reals=num_reals + 1, use_specsim=False)
    pe.enforce()
    print(f"  prior covariance {cov.shape}, prior ensemble {pe.shape}")

    truth_real = pe.index[0]
    os.makedirs(org_d, exist_ok=True)
    if truth_d is None:
        head, riv, hk, sy = run_realization(new_d, pe.loc[truth_real, :], org_d + "_run")
        for nme, arr in (("truth_hk.dat", hk), ("truth_sy.dat", sy)):
            np.savetxt(os.path.join(org_d, nme), arr)
        head.to_csv(os.path.join(org_d, "head_obs.csv"), index=False)
        riv.to_csv(os.path.join(org_d, "riv_obs.csv"), index=False)
        # the truth realization is dropped from the ensemble below, so keep its
        # parameter values here - otherwise the truth cannot be pulled apart
        # into its multipliers afterwards
        pe.loc[truth_real, :].to_csv(os.path.join(org_d, "truth_par.csv"))
        for nme in ("budget.csv", "ss.dat"):
            if os.path.exists(os.path.join(org_d + "_run", nme)):
                shutil.copy2(os.path.join(org_d + "_run", nme),
                             os.path.join(org_d, "truth_" + nme if nme == "ss.dat" else nme))
        shutil.rmtree(org_d + "_run", ignore_errors=True)
        print(f"  truth is prior realization '{truth_real}': "
              f"hk {hk.min():.2f}-{hk.max():.2f}, sy {sy.min():.3f}-{sy.max():.3f}")
    else:
        # shared truth: same fields, same model output, same data for every
        # parameterisation.  nothing is run here, the files are just carried over
        for nme in ("head_obs.csv", "riv_obs.csv", "truth_hk.dat", "truth_sy.dat"):
            shutil.copy2(os.path.join(truth_d, nme), os.path.join(org_d, nme))
        hk = np.loadtxt(os.path.join(org_d, "truth_hk.dat"))
        sy = np.loadtxt(os.path.join(org_d, "truth_sy.dat"))
        print(f"  shared truth from {truth_d}: "
              f"hk {hk.min():.2f}-{hk.max():.2f}, sy {sy.min():.3f}-{sy.max():.3f}")

    _set_obs_from_truth(pst, org_d, noise_frac, seed)
    pe = pe.loc[pe.index[1:], :]
    pe.to_binary(os.path.join(new_d, "prior_pe.jcb"))
    print(f"  conditioning ensemble (truth removed): {pe.shape}")

    pst.control_data.noptmax = 0
    pst.write(os.path.join(new_d, "synth2d.pst"), version=2)
    shutil.copy2(mf6, os.path.join(new_d, os.path.basename(mf6)))
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
    import os
    import flopy
    import numpy as np
    sim = flopy.mf6.MFSimulation.load(sim_ws=".", verbosity_level=0)
    gwf = sim.get_model()
    gwf.npf.k.set_data(np.loadtxt("hk.dat"))
    gwf.sto.sy.set_data(np.loadtxt("sy.dat"))
    # ss.dat is only parameterised in full_bc; the other variants leave ss alone
    if os.path.exists("ss.dat"):
        gwf.sto.ss.set_data(np.loadtxt("ss.dat"))
    # only rewrite the two packages that changed.  rewriting the whole
    # simulation would also rewrite the boundary list files pstfrom just wrote
    gwf.npf.write()
    gwf.sto.write()


def write_heads(ws="."):
    """runs on the agent after mf6: write the head array at the end of a few
    stress periods to hdssp<kper>.txt, one file per snapshot.  returns the file
    names.

    the snapshots are the steady state, the first pumping month, mid year and
    the last period.  the grid has no inactive cells, so every cell is written"""
    import os
    import numpy as np
    import flopy
    hds = flopy.utils.HeadFile(os.path.join(ws, "synth2d.hds"))
    names = []
    for kper in (0, 1, 6, 12):
        arr = hds.get_data(kstpkper=(0, kper))[0]
        fname = f"hdssp{kper:02d}.txt"
        np.savetxt(os.path.join(ws, fname), arr, fmt="%15.6E")
        names.append(fname)
    return names


def write_budget(ws="."):
    """runs on the agent after mf6: pull the rate budget out of the list file
    into budget.csv, one row per stress period.  returns the column names.

    the column map lives in here because pstfrom copies only this function
    into forward_run.py.  flopy numbers repeated package types in the order they
    are built, so WEL is the inflow package and WEL2 the pumping.  terms that are
    zero by construction (inflow out, pumping in) are left out, and no
    underscores in the column names - pstfrom's usecol parser splits on them"""
    import os
    import flopy
    cols = {"STO-SS_IN": "stossin", "STO-SS_OUT": "stossout",
            "STO-SY_IN": "stosyin", "STO-SY_OUT": "stosyout",
            "WEL_IN": "inflowin", "WEL2_OUT": "pumpout",
            "RIV_IN": "rivin", "RIV_OUT": "rivout",
            "PERCENT_DISCREPANCY": "pctdisc"}
    lb = flopy.utils.Mf6ListBudget(os.path.join(ws, "synth2d.lst"))
    inc, _ = lb.get_dataframes(start_datetime=None)
    df = inc.loc[:, list(cols.keys())].rename(columns=cols)
    df.index.name = "time"
    df.to_csv(os.path.join(ws, "budget.csv"))
    return list(cols.values())


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    setup(nrow=n, ncol=n)
