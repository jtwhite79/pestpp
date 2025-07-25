import os
import numpy as np
import pandas as pd
def zdt1(x):
    g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
    return (x[0], g * (1 - np.sqrt(x[0] / g))),[]

def helper(func=zdt1,pdf=None):
    if pdf is None:
        pdf = pd.read_csv("dv.dat",delim_whitespace=True,index_col=0, header=None, names=["parnme","parval1"]).values
    #obj1,obj2 = func(pdf.values)
    objs,constrs = func(pdf)
    
    with open("obj.dat",'w') as f:
        for i,obj in enumerate(objs):
            f.write("obj_{0} {1}\n".format(i+1,float(obj)))
        #f.write("obj_2 {0}\n".format(float(obj2)))
        for i,constr in enumerate(constrs):
            f.write("constr_{0} {1}\n".format(i+1,float(constr)))
    return objs



if __name__ == '__main__':
    helper()
