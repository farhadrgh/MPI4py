#! /usr/bin/env python
# Measure number of contacts between two
# selection groups given a cutoff

import time, itertools
import numpy as np
from glob import glob
from mpi4py import MPI
import MDAnalysis as md
from MDAnalysis.tests.datafiles import GRO,XTC
from scipy.spatial.distance import euclidean, pdist

comm = MPI.COMM_WORLD

u = md.Universe('confout.gro', 'traj_comp.xtc')
frames = len(u.trajectory)
sel2 = u.select_atoms('resname PEG')
sel1 = u.select_atoms('protein')
peg = sel2.atoms  # 450
bsa = sel1.atoms  # 1318
tot = len(peg) + len(bsa)
ind = list(itertools.combinations(range(tot), 2))

start= np.array([m * frames / comm.size for m in range(comm.size)])
t0 = time.time()

data = np.zeros(0, dtype='i')
lol = []
tmp = -1

comm.Barrier()
print("# trajectory loaded for process %d: [ %d\t%d ] " %\
 (comm.rank, start[comm.rank], start[comm.rank] + frames / comm.size - 1))

for ts in u.trajectory[start[comm.rank]: start[comm.rank] + frames / comm.size - 1]:
    if comm.rank == 0:
        print("timestep %d of %d" % (u.trajectory.frame * comm.size, frames * (comm.rank + 1)))
    temp = []
    for i in sel1.atoms:
        temp.append(list(i.position))
    Ipos = np.array(temp)
    temp = []
    for j in sel2.atoms:
        temp.append(list(j.position))
    Jpos = np.array(temp)
    X = np.concatenate((Ipos, Jpos), )  # bsa then peg coordinates
    dist_mat = pdist(X, 'euclidean')
    for k in range(len(ind)):
        if ind[k][0] < 1318 and ind[k][1] >= 1318:
            if dist_mat[k] < 7:
                bsa_i = ind[k][0]
                peg_i = ind[k][1]
                residue = bsa[bsa_i].resid
                if tmp != residue:
                    peg_chain = int((peg_i - 1318) / 9) + 1
                    row = [u.trajectory.frame, peg_chain, residue]
                    lol.append(row)  # (u.trajectory.frame, peg_chain, residue)
                tmp = residue

# count = np.zeros(583)
# for m in range(len(data)):
#     count[int(data[m])] += 1
# np.savetxt('contacthist.dat', zip(*count), delimiter='\t')

print("process %d finished" % comm.rank)

comm.Barrier()
data = comm.gather(lol, root=0)  # gather outputs list, here data is list len(8) of lols

if comm.rank == 0:
    f = open('timeline-test.dat', 'w')
    for i in range(len(data)):
        for j in range(len(data[i])):
            row = data[i][j]
            f.write("%d\t%d\t%d\n" % (row[0], row[1], row[2]))
    # np.savetxt('timeline-test.dat', data)
    print('Total time:{:.3}'.format(time.time() - t0))
