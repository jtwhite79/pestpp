import pyemu
import numpy as np

ib_str = """1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
      1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
      1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
      1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
      1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
      1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
      1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
      1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
      1  1  1  1  0  0  0  1  1  1  1  1  1  1  1  1  1  1  1  1
      1  1  1  1  0  0  0  0  1  1  1  1  1  1  1  1  1  1  1  1
      1  1  1  1  0  0  0  0  1  1  1  1  1  1  1  1  1  1  1  1
      1  1  1  1  0  0  0  0  1  1  1  1  1  1  1  1  1  1  1  1
      1  1  1  1  0  0  0  0  1  1  1  1  1  1  1  1  1  1  1  1
      1  1  1  1  0  0  0  0  1  1  1  1  1  1  1  1  1  1  1  1
      1  1  1  1  0  0  0  0  1  1  1  1  1  1  1  1  1  1  1  1
      1  1  1  1  0  0  0  0  1  1  1  1  1  1  1  1  1  1  1  1
      1  1  1  1  0  0  0  0  1  1  1  1  1  1  1  1  1  1  1  1
      1  1  1  1  0  0  0  0  1  1  1  1  1  1  1  1  1  1  1  1
      1  1  1  1  0  0  0  0  1  1  1  1  1  1  1  1  1  1  1  1
      1  1  1  1  0  0  0  0  1  1  1  1  1  1  1  1  1  1  1  1
      1  1  1  1  0  0  0  0  1  1  1  1  1  1  1  1  1  1  1  1
      1  1  1  1  0  0  1  1  1  1  1  1  1  1  1  1  1  1  1  1
      1  1  1  1  1  0  1  1  1  1  1  1  1  1  1  1  1  1  1  1
      1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
      1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
      1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
      1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
      1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
      1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
      1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
      1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
      1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
      0  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
      0  0  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
      0  0  0  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
      0  0  0  0  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  0
      0  0  0  0  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  0
      0  0  0  0  0  1  1  1  1  1  1  1  1  1  1  1  1  1  0  0
      0  0  0  0  0  1  1  1  1  1  1  1  1  1  1  1  1  0  0  0
      0  0  0  0  0  1  1  1  1  1  1  1  1  1  1  0  0  0  0  0"""

ib_arr = np.array(ib_str.split(),dtype=int).reshape(40,20)

ins_file = "freyberg6.lst.heads.ins"
out_file = "freyberg6.lst"
with open(ins_file,'w') as f:
	f.write("pif ~\n")
	for k in range(3):
		f.write("~HEAD IN LAYER   {0} AT END OF TIME STEP   1 IN STRESS PERIOD    1~\n".format(k+1))
		f.write("l5\n")
		irow, jcol = 0,0
		for i in range(80):
			f.write("l1 ")
			if i % 2 == 0:
				f.write(" !dum! ") # an appropriate name for the fawking row labels!
			for j in range(10):
				if ib_arr[irow,jcol] == 0:
					f.write(" !dum! ")
				else:
					f.write(" !head_{0:02d}_{1:03d}_{2:03d}! ".format(k,irow,jcol))
				jcol += 1
				if jcol == 20:
					irow += 1
					jcol = 0
			f.write("\n")

i = pyemu.pst_utils.InstructionFile(ins_file)
df = i.read_output_file(out_file)
df.to_csv("test.csv")