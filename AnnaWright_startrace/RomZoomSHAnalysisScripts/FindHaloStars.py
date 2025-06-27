

'''
Spit off from LocAtCreation_pool_rz (Step 2) to better aid multiprocessing

'''

import numpy as np
import h5py
from astropy.table import Table
from multiprocessing import Pool
import pynbody
import tangos as db
from collections import defaultdict
import sys


import os
# NS: added this to make sure the path is correct
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for root, dirs, files in os.walk(base_path):
    if root not in sys.path:
        sys.path.append(root)

db_sim = 'snapshots_200crit_h329/'
odir = '/home/christenc/Code/python/AnnaWrite_startrace/'
simpath = '/data/Sims/'
cursim = simpath.split('/')[-2]
tst = []
tgyr = []
createtime = []
sim = []

def FindHaloStars(dsnames):
    halostarsfile = odir+simpath.split('/')[-2]+'_tf.npy'
    dat = np.load(halostarsfile) # load in data                                                                                    
    halostars = dat[0]
    createtime = dat[1]

    
    compidarr = []
    compposarr = []
    compctarr = []
    compsteparr = []
    comphostarr = []
    print ('MyFirstStep: ',dsnames[0].split('.')[-1])
    print(simpath)
    
    # initialize output hdf5 file
    filename = odir+cursim+'_stardata_'+dsnames[0].split('.')[-1]+'.h5'
    with h5py.File(filename,'w') as f:
        f.create_dataset('particle_IDs', (1000000,))
        f.create_dataset('particle_positions', (1000000,3))
        f.create_dataset('particle_creation_times', (1000000,))
        f.create_dataset('timestep_location', (1000000,))
        f.create_dataset('particle_hosts', (1000000,))


     
    # iterate through the snapshots this process has been assigned
    ctr = 0
    for step in dsnames:
        #s = pynbody.load(simpath+step+'/'+step) # load in snapshot
        print(simpath+'snapshots_200crit_h329/'+ step)
        s = pynbody.load(simpath+db_sim+step)
        assert(step==s.filename.split('/')[-1]) # and make sure it's the right one

        # identify the timespan we should be checking for new stars
        # i.e., the time between the previous snapshot and this one
        ind = np.where(np.array(tst)==s.filename.split('/')[-1])[0][0]
        if ind != 0:
            low_time = tgyr[ind-1]
        else:
            low_time = 0
        high_time = tgyr[ind]
        # Which stars formed during this span?
        starinds = np.where((createtime>=low_time) & (createtime<high_time))[0]
        print (str(len(starinds))+' relevant stars in '+str(step))
        
        # In addition to the iords and formation times, grab the formation positions
        # and formation hosts of each star particle. Also store the name of the snapshot
        # that this star particle is first found in for future convenience
        x = s.s['iord']
        y = halostars[starinds]
        index = np.argsort(x)
        sorted_x = x[index]
        sorted_index = np.searchsorted(sorted_x,y)
        yindex = np.take(index,sorted_index,mode="clip")
        mask = x[yindex] != y
        res = np.ma.array(yindex,mask=mask)
        posarr = s.s['pos'][np.ma.compressed(res)].in_units('Mpc')
        ctarr = s.s['tform'][np.ma.compressed(res)].in_units('Gyr')
        idarr = s.s['iord'][np.ma.compressed(res)]
        hostarr = s.s['amiga.grp'][np.ma.compressed(res)]
        starr = np.repeat(float(s.filename.split('.')[-1]),len(ctarr))

        # Creates a dictionary that relates the potential values of the amiga.grp array to the host index in the tangos database (fid)
        #     Convert the 0s that amiga uses for the hosts of particles that aren't bound to
        #     a halo to -1s and convert all other host IDs to their index in the tangos database
        fid = {}
        fid['0'] = -1
        for i in range(1,len(sim[int(ind)].halos[:])+1):
            # fid[str(sim[int(ind)][int(i)].finder_id)] = i  ### Original code by Anna. Causing a problem with my DB
            try:
                fid[str(sim[int(ind)][int(i)].halo_number)] = i  ### Altered version of above line
                # print(i, sim[int(ind)][int(i)].halo_number)
            except:
                print("Missing: ",str(sim[int(ind)]),str(i))

        dbhostarr = np.array([])
        print("Amiga Host IDs",np.unique(hostarr))
        # for each star being considered, take the host id (amiga.grp), use it to look up the fid and store it in an array called dbhostarr
        for x in hostarr:
            try:
                dbhostarr = np.append(dbhostarr, fid[str(x)])
            except:
                dbhostarr = np.append(dbhostarr, -1)  # CC: Is it possible that amiga is using -1 now for unassigned stars? These aren't being caught
        # CC: Anna's original code. Replaced with above for loop to catch the -1 cases, even though ugly
        # dbhostarr = np.array([fid[str(x)] for x in hostarr]) 
        assert(len(starinds)==len(idarr)) # make sure you got everything

        print('FID Host IDs', np.unique(dbhostarr))
        
        # periodically write data to output file
        compidarr.extend(idarr)
        compposarr.extend(posarr)
        compctarr.extend(ctarr)
        compsteparr.extend(starr)
        comphostarr.extend(dbhostarr)
        if ctr%4 == 0 or ctr==(len(dsnames)-1):
            with h5py.File(filename,'a') as f:
                del f['particle_IDs']
                del f['particle_creation_times']
                del f['timestep_location']
                del f['particle_positions']
                del f['particle_hosts']
                f.create_dataset('particle_IDs',data=compidarr)
                f.create_dataset('particle_creation_times',data=compctarr)
                f.create_dataset('particle_positions',data=compposarr)
                f.create_dataset('timestep_location',data=compsteparr)
                f.create_dataset('particle_hosts',data=comphostarr)
        ctr = ctr+1
    return
