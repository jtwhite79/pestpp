
import os, sys
import shutil
import numpy as np
import pandas as pd
import flopy
import matplotlib.pyplot as plt
import pyemu
def forward_run():

    N = 200
    nx = 100
    ny = 100
    nnodes = nx * ny
    ref_real_i = 10
    template_ws = os.getcwd()
    modelname = 'es_pmp'

    df_par = pd.read_csv("k_input.csv")
    mf = flopy.modflow.Modflow.load(f=modelname+'.nam', model_ws=template_ws, forgive=False)
    k_vals = df_par['parval'].values.reshape(nx,ny)
    mf.upw.hk = np.power(10.0,k_vals)
    mf.upw.write_file()

    pyemu.os_utils.run("mfnwt es_pmp.nam")

    # read output
    out_file = os.path.join("hob_out.csv")
    hds = flopy.utils.HeadFile(os.path.join(modelname + ".hds"))
    totims = hds.get_times()
    rr_ = np.linspace(10, 90, 10)
    ro, co = np.meshgrid(rr_, rr_)
    ro = ro.astype(int)
    co = co.astype(int)
    heads = []
    for tm in totims:
        # if np.mod(tm, 2) == 0:
        #     continue
        wt = hds.get_data(totim=tm)
        ho = wt[0][ro.flatten(), co.flatten()]
        ho = ho.tolist()

        heads = heads + ho

    obs_df = pd.DataFrame(columns=['sim', 'obsname'])
    obs_df['sim'] = heads
    obs_df['obsname'] = ["ho_{}".format(i) for i in range(len(heads))]
    obs_df.to_csv(out_file, index=False)

if __name__ == "__main__":
     forward_run()
    