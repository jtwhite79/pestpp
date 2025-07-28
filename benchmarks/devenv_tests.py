import os
import sys
import shutil
import platform
import numpy as np
import pandas as pd
import platform
import pyemu

bin_path = os.path.join("test_bin")
if "linux" in platform.platform().lower():
    bin_path = os.path.join(bin_path,"linux")
elif "darwin" in platform.platform().lower() or "macos" in platform.platform().lower() :
    bin_path = os.path.join(bin_path,"mac")
else:
    bin_path = os.path.join(bin_path,"win")

bin_path = os.path.abspath("test_bin")
os.environ["PATH"] += os.pathsep + bin_path


# bin_path = os.path.join("..","..","..","bin")
# exe = ""
# if "windows" in platform.platform().lower():
#     exe = ".exe"
# exe_path = os.path.join(bin_path, "pestpp-ies" + exe)

# case of either appveyor, travis or local
if os.path.exists(os.path.join("pestpp","bin")):
    bin_path = os.path.join("..","..","pestpp","bin")
else:
    bin_path = os.path.join("..","..","..","..","pestpp","bin")

        
if "windows" in platform.platform().lower():
    exe_path = os.path.join(bin_path, "win", "pestpp-ies.exe")
elif "darwin" in platform.platform().lower() or "macos" in platform.platform().lower() :
    exe_path = os.path.join(bin_path,  "mac", "pestpp-ies")
else:
    exe_path = os.path.join(bin_path, "linux", "pestpp-ies")

noptmax = 4
num_reals = 20
port = 4021



def mf6_v5_ies_test():
    model_d = "mf6_freyberg"

    t_d = os.path.join(model_d,"template")
    m_d = os.path.join(model_d,"master_ies_glm_loc")
    #if os.path.exists(m_d):
    #    shutil.rmtree(m_d)
    pst = pyemu.Pst(os.path.join(t_d,"freyberg6_run_ies.pst"))
    pst.control_data.noptmax = 0
    pst.write(os.path.join(t_d,"freyberg6_run_ies.pst"))
    pyemu.os_utils.run("{0} freyberg6_run_ies.pst".format(exe_path),cwd=t_d)

    pst.control_data.noptmax = 3
    par = pst.parameter_data

    eff_lb = (par.parlbnd + (np.abs(par.parlbnd.values)*.01)).to_dict()
    eff_ub = (par.parubnd - (np.abs(par.parlbnd.values)*.01)).to_dict()
    log_idx = par.partrans.apply(lambda x: x=="log").to_dict()
    for p,log in log_idx.items():
        if log:
            lb = np.log10(par.loc[p,"parlbnd"])
            eff_lb[p] = (lb + (np.abs(lb)*.01))
            ub = np.log10(par.loc[p,"parubnd"])
            eff_ub[p] = (ub - (np.abs(ub)*.01))

    pargp_map = par.groupby(par.pargp).groups
    print(pargp_map)

    


    m_d = os.path.join(model_d, "master_ies_glm_noloc_standard")
    if os.path.exists(m_d):
        shutil.rmtree(m_d)
    pst = pyemu.Pst(os.path.join(t_d, "freyberg6_run_ies.pst"))
    pst.pestpp_options.pop("ies_localizer",None)
    pst.pestpp_options.pop("ies_autoadaloc",None)
    pst.pestpp_options["ies_bad_phi_sigma"] = 2.5
    pst.pestpp_options["ies_num_reals"] = 30
    pst.pestpp_options["ensemble_output_precision"] = 40
    pst.control_data.noptmax = 3
    pst_name = "freyberg6_run_ies_glm_noloc_standard.pst"
    pst.write(os.path.join(t_d, pst_name))
    pyemu.os_utils.start_workers(t_d, exe_path, pst_name, num_workers=15,
                                 master_dir=m_d, worker_root=model_d, port=port)
    
    


    phidf = pd.read_csv(os.path.join(m_d,pst_name.replace(".pst",".phi.actual.csv")))
    assert phidf.shape[0] == pst.control_data.noptmax + 1
    for i in range(1,pst.control_data.noptmax+1):
        pcs = pd.read_csv(os.path.join(m_d,pst_name.replace(".pst",".{0}.pcs.csv".format(i))),index_col=0)
        #print(pcs)
        pe = pd.read_csv(os.path.join(m_d,pst_name.replace(".pst",".{0}.par.csv".format(i))),index_col=0)
        print(pe.shape)
        #print(pe)
        groups = pcs.index.values.copy()
        groups.sort()
        for group in groups:
            pnames = pargp_map[group].values
            lb_count,ub_count = 0,0
            for pname in pnames:
                lb,ub = eff_lb[pname],eff_ub[pname]
                v = pe.loc[:,pname].values.copy()
                if log_idx[pname]:
                    v = np.log10(v)
                low = np.zeros_like(v,dtype=int)
                low[v < lb] = 1
                high = np.zeros_like(v,dtype=int)
                high[v > ub] = 1
                lb_count += low.sum()
                ub_count += high.sum()
            print(i,group,len(pnames),lb_count,pcs.loc[group,"num_at_near_lbound"],ub_count,pcs.loc[group,"num_at_near_ubound"])
            assert lb_count == pcs.loc[group,"num_at_near_lbound"]
            assert ub_count == pcs.loc[group,"num_at_near_ubound"]
    
    


if __name__ == "__main__":
   
    mf6_v5_ies_test()
   