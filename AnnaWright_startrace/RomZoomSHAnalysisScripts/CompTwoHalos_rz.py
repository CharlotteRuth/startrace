'''
Optional: Step 6c of stellar halo pipeline
Compares two halos to see how likely it is that one is
the main progenitor of the other based on how many particles
they have in common. This is particularly useful when you've
used a merger tree constructor that doesn't use phantoms or
some equivalent and may therefore fail to connect a halo at
snapshot1 to the same halo at snapshot3 if it lost track of it
at snapshot2. This information can then be used with FixHostIDs_rz
to merge two unique IDs and/or to create a new link in the 
relevant tangos db.

Usage: python CompTwoHalos_rz.py <sim> <halo1> <halo2>
Example: python CompTwoHalos_rz.py r718 0136_4 0192_3

Output: prints out the fraction of <halo1>'s DM particles
that are in <halo2> and vice-versa.

It takes three arguments: the simulation you're working with
and the tangos IDs of the two halos you want to compare, which
are formatted as <snapshot>_<IDatsnapshot>. Note that this ID
is assumed to be the tangos ID, not necessarily the amiga.grp
ID.
'''

import numpy as np
import tangos as db
from collections import defaultdict
import math
import pynbody
import sys

if len(sys.argv) != 4:
    print ('Usage: python CompTwoHalos_rz.py <sim> <halo1> <halo2>')
    sys.exit()
else:
    cursim = str(sys.argv[1])
    halo1 = str(sys.argv[2])
    halo2 = str(sys.argv[3])


sim = db.get_simulation(cursim+'%')
simdir = '/Volumes/Audiobooks/RomZooms/' # Where does your simulation live?

def snapshot_to_db_index(snap):
    '''
    Calculates the tangos database index of snapshot <snap>
    Returns the index if that snap is in the database. Otherwise,
    prints an error and exits
    '''
    dbind = -1
    for ctr,ts in enumerate(sim.timesteps):
        if str(snap).zfill(6) in str(ts):
            dbind = ctr
    if dbind>=0:
        return dbind
    else:
        print ('Snapshot '+str(snap)+' does not seem to exist in the current db.')
        exit()

def round_to_n(val,n):
    '''
    Round val to n significant figures
    '''
    if val == 0:
        return 0.0
    else:
        return round(val,int(n-math.ceil(math.log10(abs(val)))))
    

snnum = int(halo1.split('_')[0])
hnum = int(halo1.split('_')[1])
snind = int(snapshot_to_db_index(snnum))
fid = sim[snind][hnum].finder_id # grab amiga.grp ID
s = pynbody.load(simdir+cursim+'.romulus25.3072g1HsbBH/'+cursim+'.romulus25.3072g1HsbBH.'+str(snnum).zfill(6)+'/'+cursim+'.romulus25.3072g1HsbBH.'+str(snnum).zfill(6))
h = s.halos()
ptcls1 = h[fid].dm['iord'] # grab DM iords

snnum = int(halo2.split('_')[0])
hnum = int(halo2.split('_')[1])
snind = int(snapshot_to_db_index(snnum))
fid = sim[snind][hnum].finder_id # grab amiga.grp ID
s = pynbody.load(simdir+cursim+'.romulus25.3072g1HsbBH/'+cursim+'.romulus25.3072g1HsbBH.'+str(snnum).zfill(6)+'/'+cursim+'.romulus25.3072g1HsbBH.'+str(snnum).zfill(6))
h = s.halos()
ptcls2 = h[fid].dm['iord'] # grab DM iords

com1 = len(ptcls1[np.isin(ptcls1,ptcls2)])/len(ptcls1)
com2 = len(ptcls2[np.isin(ptcls2,ptcls1)])/len(ptcls2)

print (halo1+' contains '+str(100*round_to_n(com2,4))+'% of '+halo2+'\'s particles')
print (halo2+' contains '+str(100*round_to_n(com1,4))+'% of '+halo1+'\'s particles')


'''
Created on June 11, 2025

@author: anna
'''