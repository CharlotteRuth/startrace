'''
Optional: Step 6b of stellar halo pipeline
Updates the host_ID values stored in the allhalostardata hdf5 file based
on user input. This is designed as a follow-up to TrackDownStars and can 
be used in a couple of ways. If ffile=True, this script will look for 
numpy files with names <sim>_new_ID_?.npy and will assign the particles
with the iords in a given file to new_ID. If ffile=False, it will assign
all particles with a host_ID in the old_ID list to new_ID.

Output: <sim>_allhalostardata_upd.h5

Usage:   python FixHostIDs_rz.py <sim> [ffile]
         <sim>   : simulation name (e.g. r614)
         [ffile] : optional; "True" to reassign particles using .npy files saved by
                   TrackDownStars (default behaviour when omitted is "False").

Examples:
  # Reassign using .npy files produced by TrackDownStars:
  python FixHostIDs_rz.py r614 True

  # Reassign all particles matching old_ID (set in script) to new_ID (set in script):
  python FixHostIDs_rz.py r614
  python FixHostIDs_rz.py r614 False

When ffile=True the script globs for files matching <odir><sim>_????_*.npy.
The new host ID is parsed from the filename: e.g. r614_4096_1_2.npy → host "4096_1".
Copy the .npy files from the TrackDownStars output directory (odir in the notebook)
into the directory specified by odir in this script before running.

The script will print out all host_IDs for which the number of assigned particles 
changed and how many particles each gained/lost. If the output looks correct,
the user should manually rename <sim>_allhalostardata_upd.h5 to <sim>_allhalostardata_consolidated2.h5.
It's often necessary to go back and forth between this and TrackDownStars, in which 
case I usually move the *.npy files that have already been processed to a subfolder. 
'''

import tangos as db
import numpy as np
import h5py
from collections import defaultdict
import glob
import sys

if len(sys.argv) < 2 or len(sys.argv) > 3:
    print ('Usage: python FixHostIDs_rz.py <sim> [ffile]')
    print ('       ffile: "True" to use .npy files, "False" (default) to use old_ID/new_ID')
    sys.exit()
else:
    cursim = str(sys.argv[1])
    ffile_arg = sys.argv[2].lower() == 'true' if len(sys.argv) == 3 else False

# ffile = True
# If you're not using the autodetection from file name method,
# enter the host ID(s) you want to correct in old_ID and the IDs
# you want to replace them with in new_ID. If ffile=True, these
# will both be ignored
old_ID = ['0192_-4'] # Which star particle host_IDs do you want to change? This can be a single host_ID or a list
new_ID = '4096_1'
# odir = '/Users/Anna/Research/Outputs/M33Analogs/MM/'+cursim+'/' # Where does your allhalostardata hdf5 file live?
odir = '/home/christenc/Code/Datafiles/stellar_halos/'+cursim+'/' # Where does your allhalostardata hdf5 file live?

def main(odir, cursim, ffile=True):
    global new_ID, old_ID

    print(new_ID, old_ID)
    with h5py.File(odir+cursim+'_allhalostardata_consolidated2.h5','r') as f:
        hostids = f['host_IDs'].asstr()[:]
        partids = f['particle_IDs'][:]
        pct = f['particle_creation_times'][:]
        ph = f['particle_hosts'][:]
        pp = f['particle_positions'][:]
        ts = f['timestep_location'][:]
    uIDs = np.unique(hostids)
    print('Unique host_IDs before update: ', uIDs)

    # Make a dictionary of host_IDs
    orig = {}
    for i in uIDs:
        nparts = len(partids[hostids==i])
        orig[i] = nparts

    uphost = []
    newIDlist = []
    print(odir+cursim+'_????_*.npy')
    if ffile == True: # if we're using files 
        partfiles = glob.glob(odir+cursim+'_????_*.npy')
        uphost = np.copy(hostids)
        for pf in partfiles: # for each file
            nstr = pf.split('/')[-1].split('_') # figure out new host name
            new_ID = nstr[1]+'_'+nstr[2]
            if new_ID not in orig and new_ID not in newIDlist: 
                newIDlist.append(new_ID)
            curexparts = np.load(pf)
            uphost[np.isin(partids,curexparts)] = new_ID # update relevant particles
        uphost = uphost.tolist()
    else: # if we're going off of host names
        exparts = partids[np.isin(hostids,old_ID)] 
        print('Number of particles to reassign: ', len(exparts))
        for ctr in range(0,len(partids)):
            if partids[ctr] in exparts: # update relevant particles
                uphost.append(new_ID)
            else:
                uphost.append(hostids[ctr])
        if new_ID not in orig:
            newIDlist.append(new_ID)

    assert(len(hostids)==len(uphost)) # Make sure we didn't somehow lose some particles

    # write out new data
    with h5py.File(odir+cursim+'_allhalostardata_upd.h5','w') as f:
        f.create_dataset('particle_IDs', data=partids)
        f.create_dataset('particle_positions', data=pp)
        f.create_dataset('particle_creation_times', data=pct)
        f.create_dataset('timestep_location', data=ts)
        f.create_dataset('particle_hosts', data=ph)
        f.create_dataset('host_IDs', data=uphost, dtype="S10")

    # let the user know which hosts lost/gained particles
    for key,item in orig.items():
        upd_npart = len(partids[np.array(uphost)==key])
        d_npart = upd_npart-item
        if d_npart != 0:
            if d_npart>0:
                chword = 'gained'
            else:
                chword = 'lost'
            print (key+': '+str(np.abs(d_npart))+' particles '+chword)

    for i in newIDlist:
        upd_npart = len(partids[np.array(uphost)==i])
        print (i+': '+str(upd_npart)+' particles gained')

    print ('---------------------------------')
    print ('If this looks correct, run')
    print ('mv '+odir+cursim+'_allhalostardata_upd.h5 '+odir+cursim+'_allhalostardata_consolidated2.h5')

if __name__ == '__main__':
    main(odir, cursim, ffile=ffile_arg)

'''
Created on Mar 4, 2024

@author: anna
'''