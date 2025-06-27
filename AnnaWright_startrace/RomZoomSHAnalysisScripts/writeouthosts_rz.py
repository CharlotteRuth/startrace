'''
Step 3 of stellar halo pipeline
For each snapshot, writes out a list of halos that formed stars 
between the last snapshot and this one and the number of stars formed; 
as in step 2, note that the IDs of these halos will be their index in 
the tangos database, not necessarily their amiga.grp ID. This is used 
to construct a unique ID for each star-forming halo in the next step.

Output: <sim>_halostarhosts.txt

Usage:   python writeouthosts_rz.py <sim>
Example: python writeouthosts_rz.py r634 

Note that this is currently set up for MMs, but should be easily adapted 
by e.g., changing the paths or adding a path CL argument. 
'''

import numpy as np
import h5py
import glob
import sys

def FindHosts(flist):
    tln = []
    ind = []
    for fname in flist: # for each of the files generated in step 2
        with h5py.File(fname,'r') as f:
            hostlist = f['particle_hosts'][:]
            tslist = f['timestep_location'][:]
        for t in np.unique(tslist): # for each snapshot
            tstr = str(int(t))
            tmask = np.where(tslist==t)[0]
            try:
                relh = hostlist[tmask]
            except:
                print(len(tslist), len(hostlist))
                print(tmask)
            for h in np.unique(relh): # for each halo that had a new star at that snapshot
                                      # record the tangos index and the number of stars
                                      # formed
                tstr = tstr + '\t'+str(int(h))+','+str(np.count_nonzero(relh==h, axis=0))
            tstr = tstr+'\n'
            tln.append(tstr)
            ind.append(int(t))
    return tln,ind
        
def main(cursim, odir='/home/christenc/Code/Datafiles/stellarhalo_trace_aw/'):
    hsfiles = glob.glob(odir+cursim+'_stardata_*.h5')
    ofile = odir+cursim+'_halostarhosts.txt'

    lnlist,ordind = FindHosts(sorted(hsfiles))

# write your data out with snapshots in chronological order
    lnlist = np.array(lnlist)
    ordind = np.array(ordind)
    argord = np.argsort(ordind)
    lnlist = lnlist[argord]
    with open(ofile, 'w') as f:
        for item in lnlist:
            f.write(item)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print ('Usage: python writeouthosts_rz.py <sim>')
        sys.exit()
    else:
        cursim = str(sys.argv[1])
    main(cursim)

    
'''
Created on Mar 4, 2024

@author: anna
'''
