'''
Script to identify halos that should be linked across snapshots
and link them in tangos, creating phantom halos between them if 
necessary. For each unique host ID listed in your allhalostardata 
file, it will check for potential descendants that a given halo is 
not already linked to in your tangos database and creates links if
the two share a sufficient fraction of particles. Also writes out a
log file recording what it did.
'''

import glob
import os
import numpy as np
import tangos as db
import yt
yt.set_log_level(1000)
from yt.frontends.rockstar.definitions import halo_dts
from collections import defaultdict
import math
import h5py
from tangos.core.halo import PhantomHalo

dbname = 'Maelstrom.9f11c.all.DD' # your tangos db
pathtofiles = '/home/user/Anna/Maelstrom/Maelstrom_rockstarbins/' # Where are your rockstar binary files?
halofile = '/home/user/Anna/Maelstrom/Maelstrom_RD0042_allhalostardata.h5' # Where is your allhalostardata file?
logfile = '/home/user/Anna/Maelstrom/mmlclog.txt' # Where should I write out the log file?
tlim = 30 # give up after failing to find a descendant for how many snapshots in a row?
maxfid = 10000 # maximum finder ID for a non-phantom halo

# Minimum fraction of particles halos must share to be linked. A potential descendant must meet both criteria for the code
# to actually consider it a match and make a link
# BELOW VALUES WERE SANE FOR FOGGIE+ROCKSTAR
# they may not be reasonable for ChaNGa+AHF
pthresh_f = 0.5 # what fraction of your halo's particles must a potential descendant have to be considered a match?  
pthresh_b = 0.4 # what fraction of a potential descendant's particles must have come from your halo to be considered a match?

# Parameters for creating finder IDs for phantoms
defphan = 10**19 # number above which any finder ID DEFINITELY belongs to a phantom
myphan = 3*10**18 # slightly lower number (but still definitely a phantom)

sim = db.get_simulation(dbname) 
tslist = [] # list of snapshots
ts = sim.timesteps
for d in ts:
    tslist.append(int(d.extension.split('/')[0].split('.')[-1]))
tslist = np.array(tslist)

lostcauses = [] # list of host IDs we've given up on

def round_to_n(val,n):
    '''
    Round val to n significant figures
    '''
    if val == 0:
        return 0.0
    else:
        return round(val,int(n-math.ceil(math.log10(abs(val)))))
        
def checkmatch_p(step,halo,hid,disp):
    proj = sim[int(step)][int(halo)].calculate_for_progenitors('finder_id()')[0]
    try:
        match = (proj[disp]==hid)
    except:
        match = False
    return match
    
def checkmatch_d(step,halo,hid,disp):
    proj = sim[int(step)][int(halo)].calculate_for_descendants('finder_id()')[0]
    try:
        match = (proj[disp]==hid)
    except:
        match = False
    return match

def trackforward(step,halo):
    # Can simplify considerably once earlier/later works with phantoms
    desc,fid = sim[int(step)][int(halo)].calculate_for_descendants('halo_number()','finder_id()')
    # Trim phantoms off the end of the list
    eol = fid[-1]
    while eol>maxfid:
        fid = fid[:-1]
        desc = desc[:-1]
        eol = fid[-1]
    nd = len(desc)-1
    match = checkmatch_p(step+nd,desc[nd],fid[0],nd)
    stat = int(match)
    last_t = 0
    if stat == 0:
        ma_desc = desc[fid<maxfid]
        refarr = np.cumsum(fid<maxfid)
        sf = len(ma_desc)
        s0 = 0
        while (s0 <= sf):
            ci = (s0+sf)//2
            ci_trans = np.argmax(refarr>ci)
            match = checkmatch_p(step+ci_trans,desc[ci_trans],fid[0],ci_trans)
            if match==True:
                s0 = ci+1
                last_t = ci_trans
            else:
                sf = ci-1
    else:
        last_t = nd
    return step+last_t,int(desc[last_t])

                      
def load_rockstar_particles_for_halo(snapnum,finderid,pathtofiles):
    '''
    Loads in list of particle IDs for a given galaxy - based heavily on 
    pynbody's method of reading in particle IDs from a rockstar catalog. 
    Pynbody is fast enough that this could probably just be 
    s = pynbody.load(simpath)
    h = s.halos()
    halo_ptcls = h[hid].dm['iord'] 
    for our galaxies, which has the benefit of being both concise and
    halo finder independent. This could also be modified to use pynbody's
    method of reading in AHF data.
    '''
    # head_type from pynbody
    head_type = np.dtype([('magic',np.uint64),('snap',np.int64),
                          ('chunk',np.int64),('scale','f'),
                          ('Om','f'),('Ol','f'),('h0','f'),
                          ('bounds','f',6),('num_halos',np.int64),
                          ('num_particles',np.int64),('box_size','f'),
                          ('particle_mass','f'),('particle_type',np.int64),
                          ('format_revision',np.int32),
                          ('rockstar_version',np.str_,12)])
                          
    binfiles = glob.glob(os.path.join(pathtofiles,'halos_'+str(snapnum)+'.*bin'))
    binfiles.sort() # DO NOT CHANGE! This deals with out-of-order rockstar nonsense
    
    cat = yt.frontends.rockstar.RockstarDataset(binfiles[0])
    cat.parameters['format_revision'] = 2
    cat_data = cat.all_data()
    idlist = cat_data['halos','particle_identifier']
    nump = cat_data['halos','num_p']
    
    halopos = np.where(idlist==finderid)[0][0]
    
    hcount = 0
    relfile = -1
    relmin = -1
    for bind,b in enumerate(binfiles):
        f = open(b,'rb')
        head = np.fromstring(f.read(head_type.itemsize),
                                   dtype=head_type)
        hrange = idlist[hcount:hcount+head['num_halos'][0]]
        if finderid in hrange:
            relfile = bind
            relmin = int(hrange[0].value)
        hcount += head['num_halos'][0]
        f.close()
    
    # based on pynbody's method
    f = open(binfiles[relfile],'rb')
    head = np.fromstring(f.read(head_type.itemsize),
                               dtype=head_type)
    offset_h = f.tell()
    n_halos = head['num_halos'][0]
    halo_offsets = np.empty(head['num_halos'][0],dtype=np.int64)
    halo_lens = np.empty(head['num_halos'][0],dtype=np.int64)
    halo_type = halo_dts[2]
    offset = offset_h+halo_type.itemsize*n_halos
    halo_min = int(np.fromfile(f, dtype=halo_type, count=1)['particle_identifier'])
    halo_max = int(halo_min+n_halos)
    f.close()
    
    this_id = relmin
    curind = int(halopos-(finderid-relmin))
    assert(relmin==idlist[curind].value)
    for h in range(n_halos):
        halo_offsets[this_id-relmin] = int(offset)
        num_ptcls = nump[curind].value
        halo_lens[this_id-relmin] = num_ptcls
        offset+=num_ptcls*np.dtype('int64').itemsize
        this_id+=1
        curind+=1
    f = open(binfiles[relfile],'rb')
    f.seek(halo_offsets[int(finderid-relmin)])
    halo_ptcls=np.fromfile(f,dtype=np.int64,count=halo_lens[int(finderid-relmin)])
    assert(len(halo_ptcls)==nump[halopos].value)
    f.close()
    return halo_ptcls

def find_orphans(ts,seekdir='b'):
    '''
    Locates halos that don't have a progenitor (seekdir='b') or a descendant (seekdir='f')
    in snapshot ts so that they can be compared to any halos that we're seeking descendants
    or progenitors for. 
    '''
    orphan_list = []
    
    if seekdir == 'b':
        for i in ts.halos:
            p,h = i.calculate_for_progenitors('halo_number()','finder_id()',nmax=1)
            if len(p[h<maxfid])<2:
                orphan_list.append(i.halo_number)
    elif seekdir =='f':
        for i in ts.halos:
            p,h = i.calculate_for_descendants('halo_number()','finder_id()',nmax=1)
            if len(p[h<maxfid])<2:
                orphan_list.append(i.halo_number)
    else:
        print ('Please select a valid direction for orphan finding')
        exit()
    
    return np.array(orphan_list,dtype=np.int64)

def comp_halo_parts(sim,ufc,h1,myveryowntimestep,potmatches):
    '''
    Attempt to find a match for a halo in snapshot myveryowntimestep
    sim is the db object
    ufc is the snapshot in which your halo lives
    h1 is the halo_number of your halo at that snapshot
    potmatches is a list of potential matches in myveryowntimestep - here a list of orphans
    Returns a list of the best potential matches and the fraction of common particles that each contains. 
    We'll call the original halo "halo 1" and any potential matches "halo 2(a/b)"
    '''

    fid = sim[int(ufc)][int(h1)].finder_id
    ptcls1 = load_rockstar_particles_for_halo(int(ufc),fid,pathtofiles) # grab list of halo 1's particles
    
    sim = db.get_simulation(dbname) # have to reload in case we've modified db
    
    bp = [] # fraction of halo 2's particles that are also in halo 1
    fp = [] # fraction of halo 1's particles that are also in halo 2
    for curm in potmatches: # for each of our potential matches...
        fid2 = sim[int(myveryowntimestep)][int(curm)].finder_id
        ptcls2 = load_rockstar_particles_for_halo(int(myveryowntimestep),fid2,pathtofiles) # grab list of halo 2's particles
        
        fp.append(len(ptcls1[np.isin(ptcls1,ptcls2)])/len(ptcls1))
        bp.append(len(ptcls2[np.isin(ptcls2,ptcls1)])/len(ptcls2))
    
    # Which of our potential matches has the most particles in common with halo 1? There are two ways to answer this (bp vs fp).
    # If the halo with max bp and max fp are the same and both are sufficiently high, we'll call this a match. If the fraction of 
    # common particles isn't high enough on either side, we'll deem this a failure. If the halos with max bp and max fp are different,
    # we need to check to see if adopting one descendant makes more sense than adopting the other. For instance, if halo 1 has fallen into
    # the central halo and is being stripped, the majority of its particles may be in the central halo (halo 2a) in this snapshot, so halo 2a
    # will have high fp. However, halo 2a will typically have low bp, since the majority of its particles did not come from halo 1. If there is
    # another halo (halo 2b) with bp above our threshold, we will check to see if it has enough of halo 1's particles to qualify as a match (e.g.,
    # if it is the stripped remnant of halo 1). I've picked 0.45 as a threshold for this secondary check, but this is somewhat arbitrary
    mbp = max(bp) 
    mfp =  max(fp)
    desc1 = np.array(potmatches)[np.array(fp)==mfp][0]
    desc2 = np.array(potmatches)[np.array(bp)==mbp][0]
    
    if mbp>pthresh_f and mfp>pthresh_b: # are there enough particles in common on both sides?
        if desc1!=desc2: # if our maximum particle holders aren't the same, does one have a reasonable bp AND fp?
            if np.array(fp)[np.array(potmatches)==desc2]>0.45: 
                mfp = np.array(fp)[np.array(potmatches)==desc2]
                descs = np.array([int(desc2),int(desc2)])
                with open(logfile,'a') as lf:
                    lf.write('Descendant found: '+str(desc2)+'\n')
                    lf.write(str(mbp)+', '+str(mfp)+'\n')
            elif np.array(bp)[np.array(potmatches)==desc1]>0.45:
                mbp = np.array(bp)[np.array(potmatches)==desc1]
                desc = np.array([int(desc1),int(desc1)])
                with open(logfile,'a') as lf:
                    lf.write('Descendant found: '+str(desc1)+'\n')
                    lf.write(str(mbp)+', '+str(mfp)+'\n')
            else: # if not, we'll call this a failure
                with open(logfile,'a') as lf:
                    lf.write('FAILURE: '+str(mbp)+', '+str(mfp)+'\n')
                descs = np.array([np.nan,np.nan])
        else: # if our maximum particle holders are the same, we've found a match!
            with open(logfile,'a') as lf:
                lf.write('Descendant found: '+str(desc1)+'\n')
                lf.write(str(mbp)+', '+str(mfp)+'\n')
            descs = np.array([int(desc1),int(desc1)])
    else: # if our maximum particle fractions aren't high enough, we'll call this a failure
        with open(logfile,'a') as lf:
            lf.write('FAILURE: '+str(mbp)+', '+str(mfp)+'\n')
        descs = np.array([np.nan,np.nan])
    fracc = np.array([mfp,mbp])
    return descs,fracc

def BringMeARing(hlist):
    for halo in hlist:
        print (halo)
        with open(logfile,'a') as lf:
            lf.write('----ID: '+halo+'----\n')
        # Check whether there's an updated Unique ID to make sure we don't repeat
        # any we've already taken care of
        sim = db.get_simulation(dbname) # have to reload in case we've updated db
        snnum = int(halo.split('_')[0])
        hnum = int(halo.split('_')[1])
        snind = np.where(snnum==tslist)[0][0] # index of this snapshot
        unis,unih = trackforward(snind,hnum)
        halo = str(sim[int(unis)]).split('/')[1][2:]+'_'+str(unih) 
        print ('Unique ID:',halo)
        with open(logfile,'a') as lf:
            lf.write('Unique ID: '+halo+'\n')
        # If we haven't already given up on this one, look for potential progenitors or descendants
        if halo not in lostcauses:
            hstr = halo.split('_')
            inist = int(hstr[0]) # starting point: snapshot
            inih = int(hstr[1]) # starting point: halo ID at that snapshot
            tlist = [] # list of snapshots we have checked
            hchain = [] # list of descendants. 0s indicate none in a given snapshot
            fconn = [] # fraction of particles descendant has in common with original halo 
            stind = np.where(inist==tslist)[0][0] # starting point: index of this snapshot
            ctr = stind+1
            fc = 0 # how many snapshots have we checked without finding a match?
            while ctr<=len(tslist) and fc<tlim: # start with snapshot after the one where original halo is last found
                print ('Searching step ',tslist[int(ctr)])
                sim = db.get_simulation(dbname)
                h1 = sim[int(stind)][int(inih)] # our original halo
                t2 = sim[int(ctr)] # the snapshot we're currently searching
                children = find_orphans(t2,seekdir='b') # check this snapshot for halos that have no progenitors
                desc = [] # likely descendants
                fcar = [] # fraction of particles carried over
                sim = db.get_simulation(dbname)
                d,f = comp_halo_parts(sim,int(stind),inih,int(ctr),children) # check how many particles each orphan has in common with original halo
                tlist.append(ctr)
                if np.isnan(d[0]): # if we didn't find any matches
                    hchain.append(0)
                    fconn.append(f)
                    ctr += 1
                    fc += 1 # increment number of sequential matchless snapshots
                elif d[0] == d[1]: # If we only found one potential descendant
                    fc = 0 # reset number of sequential matchless snapshots
                    hchain.append(d[0])
                    # does our new descendant have descendants of its own?
                    hid,fid,stp = sim[int(ctr)][int(d[0])].calculate_for_descendants('halo_number()','finder_id()','step_path()')
                    fconn.append(f)
                    if len(hid[fid<maxfid])>1: # if there's at least one non-phantom descendant, run down that chain and start from the
                                               # end of it
                        inist = int(stp[fid<maxfid][-1].split('/')[-1][2:]) # our new starting point: snapshot
                        inih = int(hid[fid<maxfid][-1]) # our new starting point: halo
                        ctr = inist+1
                    else: # If it has no descendants, treat the matched halo as our new starting point 
                        inist = ctr 
                        inih = d[0]
                        ctr += 1
                else: # if we have more than one potential match, make a note of it in the log file so you can come back to this and deal
                      # with it manually. Good chance this is a halo finder issue.
                    fconn.append(f)
                    hchain.append(-5)
                    with open(logfile,'a') as lf:
                        lf.write('PLEASE CHECK ON ME\n')
                    ctr += 1
                # Will need to check for double phantoms
            while len(hchain)>0 and hchain[-1] == 0: # Trim phantoms off the end of the list - should only be an issue for rockstar
                hchain.pop(-1)
                tlist.pop(-1)
            if len(tlist)>0: # if we actually found some non-phantom matches
                curt = int(halo.split('_')[0])
                curi = np.where(curt==tslist)[0][0]
                curh = int(halo.split('_')[-1])
                print ('MAKING LINKS FOR',halo)
            else: # if we found no matches before hitting our sequential matchless snapshot limit
                print ('GIVING UP ON',halo)
                lostcauses.append(halo)
            # Make the necessary links
            for h,t,f in zip(hchain,tlist,fconn): # for each match (includes phantoms between halo matches)
                if t-curt < 2: # check that steps are adjacent before you try to link them
                    with open(logfile,'a') as lf:
                        lf.write(str(t)+': '+str(curh)+'->'+str(h)+' ,'+str(f)+'\n')
                    if isinstance(curh,int): # check whether your current object is a halo or a phantom
                        myhalo = sim[int(curi)][int(curh)]
                    else: 
                        myhalo = sim[int(curi)].phantoms[int(curh[1:])-1]
                    if h != 0: # if this descendant is a halo, create link between it and our current object
                        link1 = db.core.HaloLink(myhalo, sim[int(t)][int(h)], db.core.get_or_create_dictionary_item(db.core.get_default_session(), 'manual_link'))
                        link2 = db.core.HaloLink(sim[int(t)][int(h)], myhalo, db.core.get_or_create_dictionary_item(db.core.get_default_session(), 'manual_link'))
                        db.core.get_default_session().add_all([link1, link2])
                        db.core.get_default_session().commit()
                        curh = int(h)
                    else:
                        futurehalo = myhalo.next
                        if futurehalo is None: # double check that our halo does not have a descendant in this snapshot
                            exph = sim[int(t)].phantoms.all() # grab a list of all phantoms in this snapshot
                            phnum = len(exph)+1 # how many phantoms are there?
                            with open(logfile,'a') as lf:
                                lf.write('Adding Phantom '+str(phnum)+' to step '+str(t)+'\n')
                             # Generate a finder ID for your phantom that's different from that of other phantoms
                            phids = np.array([p.finder_id for p in exph])
                            if len(phids[phids<defphan])>0:
                                phid = int(max(phids[phids<defphan])+1) # if there are other phantoms, just add 1 to max number
                            else:
                                phid = int(myphan) # if there are no phantoms, just do a big number
                            # create your phantom and link it to your halo object
                            phantom = PhantomHalo(sim[int(t)],phnum,phid)
                            db.core.get_default_session().add(phantom)
                            link1 = db.core.HaloLink(myhalo, phantom, db.core.get_or_create_dictionary_item(db.core.get_default_session(), 'manual_link'))
                            link2 = db.core.HaloLink(phantom, myhalo, db.core.get_or_create_dictionary_item(db.core.get_default_session(), 'manual_link'))
                            db.core.get_default_session().add_all([link1, link2])
                            db.core.get_default_session().commit()
                            curh = 'p'+str(phnum)
                        elif isinstance(futurehalo,PhantomHalo): # if your halo already has a phantom as its descendant here, do nothing
                            curh = 'p'+str(futurehalo.halo_number)
                        else: # if your halo already has a non-phantom descendant in this snapshot, do nothing
                            with open(logfile,'a') as lf:
                                lf.write('I already have a chain! I merge into halo '+str(futurehalo.halo_number)+'\n')
                curt = t
    return
    

# Load in your allhalostarsdata file and create list of unique hostIDs
with h5py.File(halofile) as f:
    hostids = f['host_IDs'].asstr()[:]
uIDs = np.unique(hostids)
hlist = []
for i in uIDs:
    if i!='': # Don't attempt to link hosts that don't have an associated halo
        hstr = i.split('_')
        if float(hstr[1])>0:
            hlist.append(i)

hlist = np.array(hlist) 

BringMeARing(hlist[:]) # Somewhere there's a parallelized version of this...
