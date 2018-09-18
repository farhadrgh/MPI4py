# mpirun -np 4 pythonMPI.py
# comm.size : np
# comm.rand : p ID

import time
import numpy as np
from glob import glob
from mpi4py import MPI

comm = MPI.COMM_WORLD

def make_files(n):
    for _ in range(n):
        np.savetxt('file.{}'.format(_),
        np.random.normal(size=(100,5)))

        '''if comm.rank == 0:
            make_files(10000)
        comm.Barrier()
        exit()'''

t0 = time.time()

file_arr = np.array_split(glob('file.*'),comm.size) # split it to comm.size parts

for fname in file_arr[comm.rank]:
    arr = np.genfromtxt(fname)
    #print(comm.rank, fname, np.mean(arr[:,0]))
                
comm.Barrier()
if comm.rank == 0:
    print('Total time:{:.3}'.format(time.time()-t0))
