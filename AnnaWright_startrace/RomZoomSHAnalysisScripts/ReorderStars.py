'''
Example of how to combine data from pynbody and data from the allhalostardata file.
This script will show the positions of the stars associated with the host <streamID> at 
z=0 using the central galaxy as a center. However, reorder_by_hdf5file should be broadly 
applicable (let me know if you have issues with it!)
'''

import numpy as np
import pynbody
import sys
import h5py
import matplotlib.pyplot as plt

hid = 1 # What's the amiga.grp ID of the halo you want to center on?
cursim = 'r442'
streamID = '3168_16'
opath = '/home/awright/dwarf_stellar_halos/'+cursim+'/' # Where should outputs be saved?
datapath = '/home/awright/dwarf_stellar_halos/'+cursim+'/' # Where does your allhalostardata hdf5 file live?
simdir = '/data/REPOSITORY/romulus_zooms/' # Where does your simulation live?

def reorder_by_hdf5file(starlist_pyn,starlist_h5):
    '''
    Takes a SimSnap of star particles (e.g., s.s - here starlist_pyn) and returns a SimSnap 
    that has been re-ordered (and clipped, if necessary) to match the order and contents of 
    starlist_h5 via a numpy mask. Note that starlist_pyn should be a SimSnap object and 
    starlist_h5 should be a list of particle IDs. This allows you to mix and match properties
    from pynbody (e.g., positions, velocities, angular momenta) with properties from 
    the allhalostardata file (e.g., hostID)
    '''

    # re-order to match hdf5 file if necessary
    if np.array_equal(starlist_h5,starlist_pyn['iord']):
        print ('Already fine.')
        return (starlist_pyn)
    else:
        allstarinds = starlist_pyn['iord']
        index = np.argsort(allstarinds)
        sorted_allstars = allstarinds[index]
        sorted_index = np.searchsorted(sorted_allstars,starlist_h5)
        pindex = np.take(index,sorted_index,mode="clip")
        mask = allstarinds[pindex] != starlist_h5
        res = np.ma.array(pindex,mask=mask)
        return starlist_pyn[np.ma.compressed(res)]


# grab data from allhalostars file
with h5py.File(opath+cursim+'_allhalostardata.h5','r') as f:
    partids = f['particle_IDs'][:]
    hostids = f['host_IDs'].asstr()[:]

# load in simulation, center on central halo, and rotate it to be side-on
s = pynbody.load(simdir+cursim+'.romulus25.3072g1HsbBH/'+cursim+'.romulus25.3072g1HsbBH.004096/'+cursim+'.romulus25.3072g1HsbBH.004096')
h = s.halos()
s.physical_units()
pynbody.analysis.angmom.sideon(h[hid])

# create SimSnap object that is in the same order as your allhalostardata file
relstars = reorder_by_hdf5file(s.s,partids)

# grab only the stars associated with the host <streamID>
mystars = np.where(hostids==streamID)
print (len(relstars[mystars]['iord']))

# plot the positions of these stars
plt.plot(relstars[mystars]['pos'][:,0],relstars[mystars]['pos'][:,2],linestyle='none',color='k',alpha=0.1,markersize=1,marker='o')
plt.xlim(-60,60)
plt.ylim(-60,60)
plt.show()