"""a small, adjustable 2-D groundwater model for high-dimensional ies testing.

one layer, nrow x ncol set by the caller so the problem can be scaled from
"runs in a second" to "genuinely high dimensional" without changing anything
else.  the flow system is deliberately simple and unambiguous:

    specified inflow along the RIGHT edge  ->  flows west  ->  river on the LEFT edge

with three pumping wells in the interior that only switch on for the transient
periods.  that gives a steady state to condition the conductivity field and a
transient response to condition storage.

observations are groundwater levels at a handful of cells and the total
groundwater flux to the river, which is the integrated response and the thing a
forecast would usually care about.
"""
import os
import numpy as np
import flopy


def well_cells(nrow, ncol):
    """three pumping wells spread through the interior, scaled with the grid"""
    return [(0, int(nrow * f), int(ncol * g))
            for f, g in ((0.30, 0.35), (0.55, 0.60), (0.75, 0.30))]


def pump_file(kper):
    """external list file holding the pumping well rates for one stress period"""
    return f"pump_sp{kper:02d}.txt"


def head_obs_cells(nrow, ncol):
    """a handful of monitoring locations, deliberately not on top of the wells"""
    # no underscores in observation names: pyemu's obs-name parser splits on
    # them and would truncate the usecol metadata column
    return {f"gw{i:02d}": (0, int(nrow * f), int(ncol * g))
            for i, (f, g) in enumerate(((0.20, 0.20), (0.45, 0.45),
                                        (0.65, 0.75), (0.85, 0.50)))}


def build_model(ws, nrow=40, ncol=40, run=False, exe_name="mf6", delr=100.0, delc=100.0,
                top=60.0, botm=0.0, strt=45.0, k=5.0, sy=0.15, ss=1.0e-5,
                riv_stage=44.0, riv_rbot=41.0, riv_cond=500.0, inflow_total=2500.0,
                wel_rate=-900.0, nper_trans=12, perlen_trans=30.4):
    """build (and optionally run) the mf6 model in directory ws.

    the defaults are the "reference" truth values.  top sits well above the
    simulated water table so every cell stays unconfined - otherwise cells go
    confined and sy drops out of the transient response, which would make the
    storage parameters unidentifiable.  inflow_total is the m^3/d entering along
    the right edge, wel_rate the m^3/d per pumping well in the transient periods.
    """
    if os.path.exists(ws):
        import shutil
        shutil.rmtree(ws)
    os.makedirs(ws)

    sim = flopy.mf6.MFSimulation(sim_name="synth2d", sim_ws=ws,
                                 exe_name=exe_name, version="mf6")

    # period 0 is steady state, the rest are transient
    perioddata = [(1.0, 1, 1.0)] + [(perlen_trans, 1, 1.0)] * nper_trans
    flopy.mf6.ModflowTdis(sim, nper=len(perioddata), perioddata=perioddata,
                          time_units="days")
    flopy.mf6.ModflowIms(sim, complexity="moderate",
                         outer_dvclose=1.0e-6, inner_dvclose=1.0e-7,
                         linear_acceleration="bicgstab")

    gwf = flopy.mf6.ModflowGwf(sim, modelname="synth2d", save_flows=True,
                               newtonoptions="NEWTON UNDER_RELAXATION")
    flopy.mf6.ModflowGwfdis(gwf, nlay=1, nrow=nrow, ncol=ncol,
                            delr=delr, delc=delc, top=top, botm=botm,
                            length_units="meters")
    flopy.mf6.ModflowGwfic(gwf, strt=strt)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=1, k=k, save_specific_discharge=True)
    flopy.mf6.ModflowGwfsto(gwf, iconvert=1, sy=sy, ss=ss,
                            steady_state={0: True},
                            transient={i: True for i in range(1, len(perioddata))})

    # the boundary inputs all go to external list files so pstfrom has a file to
    # put parameters on.  each is written once in the period it starts and mf6
    # carries it forward, except the pumping wells, which get their own file in
    # every transient period so each period can take its own rates.

    # river along the left edge; one boundname so the total flux is a single observation
    riv_spd = [[(0, i, 0), riv_stage, riv_cond, riv_rbot, "river"] for i in range(nrow)]
    riv = flopy.mf6.ModflowGwfriv(gwf, boundnames=True, save_flows=True, pname="riv",
                                  stress_period_data={0: {"filename": "riv.txt",
                                                          "data": riv_spd}})
    riv.obs.initialize(filename="synth2d.riv.obs",
                       continuous={"riv_obs.csv": [("rivflux", "riv", "river")]})

    # specified inflow along the right edge, its own package so it can be
    # parameterised apart from the pumping
    q_in = inflow_total / float(nrow)
    inflow = [[(0, i, ncol - 1), q_in, "inflow"] for i in range(nrow)]
    flopy.mf6.ModflowGwfwel(gwf, boundnames=True, save_flows=True, pname="inflow",
                            filename="synth2d.inflow.wel",
                            stress_period_data={0: {"filename": "inflow.txt",
                                                    "data": inflow}})

    # the three pumping wells: absent in the steady state period, on for the
    # transient ones, one file per period
    pumps = [[c, wel_rate, f"pump_{kw}"] for kw, c in enumerate(well_cells(nrow, ncol))]
    pump_spd = {kper: {"filename": pump_file(kper), "data": pumps}
                for kper in range(1, len(perioddata))}
    flopy.mf6.ModflowGwfwel(gwf, boundnames=True, save_flows=True, pname="pump",
                            filename="synth2d.pump.wel", stress_period_data=pump_spd)

    hobs = {"head_obs.csv": [(nm, "head", c) for nm, c in head_obs_cells(nrow, ncol).items()]}
    flopy.mf6.ModflowUtlobs(gwf, filename="synth2d.obs", continuous=hobs)

    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="synth2d.hds",
                           budget_filerecord="synth2d.cbc",
                           saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")])

    sim.write_simulation()
    if run:
        ok, buff = sim.run_simulation(silent=True)
        if not ok:
            raise RuntimeError("synth2d model failed to run:\n" + "\n".join(buff[-25:]))
    return sim


if __name__ == "__main__":
    import sys
    nrow = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    ncol = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    build_model("synth2d_check", nrow=nrow, ncol=ncol, run=True,
                exe_name=os.path.expanduser("~/bin/mf6"))
    import pandas as pd
    for f in ("head_obs.csv", "riv_obs.csv"):
        d = pd.read_csv(os.path.join("synth2d_check", f))
        print(f"\n{f}  shape={d.shape}")
        print(d.head(3).to_string(index=False))
        print("...")
        print(d.tail(2).to_string(index=False))
