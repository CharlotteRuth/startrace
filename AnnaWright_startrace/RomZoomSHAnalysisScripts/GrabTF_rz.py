'''
Step 1 of stellar halo pipeline
Grabs formation time and iord for every star in specified simulation
Prints out number of stars found as a sanity check

Output: <sim>_tf.npy

Usage:   python GrabTF_rz.py <simpath> <opath>
Example: python GrabTF_rz.py r634

Note that this is currently set up for MMs, but has been adapted 
by e.g., changing the paths or adding a path CL argument. 
'''

import numpy as np
import pynbody
import sys
import os

def main(simpath, odir):

#ofile = '/Users/Anna/Research/Outputs/M33Analogs/'+cursim+'_tf.npy'
#simpath = '/Volumes/Audiobooks/RomZooms/'+cursim+'.romulus25.3072g1HsbBH/'

    ofile = os.path.join(odir, f"{simpath.split('/')[-1][:-7]}_tf.npy")

    # s = pynbody.load(simpath+simpath.split('/')[-2]+'.004096/'+simpath.split('/')[-2]+'.004096')
    s = pynbody.load(simpath)
    tf = s.s['tform'][s.s['tform']>0].in_units('Gyr')
    iord = s.s['iord'][s.s['tform']>0]

    print (str(len(tf))+' stars found!')

    outarr = np.vstack((iord,tf))

    print("Save to ",ofile)
    # NS: added 06272025, directory check
    if not os.path.exists(odir):
        print("Path doesn't exist. Creating directory: ", odir)
        os.makedirs(odir)

    np.save(ofile, outarr)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print ('Usage: python GrabTF_rz.py <sim>')
        sys.exit()
    else:
        simpath = str(sys.argv[1])
        odir = str(sys.argv[2])

    main(simpath, odir)



'''
Created on Mar 4, 2024

@author: anna

Modfied 6/11/2025 by Charlotte for near Mint JL
Modified 6/27/2025 by NS for directory check
'''
