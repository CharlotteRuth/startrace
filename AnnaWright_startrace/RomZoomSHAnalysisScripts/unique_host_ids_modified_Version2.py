'''
Step 4 of stellar halo pipeline
Creates a unique ID for each host that forms a star. The format of
this ID is <newest_snapshot-1>_<host_id_at_previous_snapshot>.
The process: find which halo at the newest snapshot this host merges into,
then go back one snapshot and use that halo's ID at that time.

Output: <sim>_uniquehalostarhosts.txt

Usage:   python IDUniqueHost_rz.py <sim>
Example: python IDUniqueHost_rz.py r634 
'''

import numpy as np
import tangos as db
from astropy.table import Table
import collections
import sys

def checkmatch_p(step,halo,hid,disp,sim):
    '''
    Check that <halo> (at snapshot number <step>) is the main progenitor of <hid> <disp> snapshots ago
    Returns True or False 
    '''
    proj = sim[int(step)][int(halo)].calculate_for_progenitors('finder_id()')[0]
    try:
        match = (proj[disp]==hid)
    except:
        match = False
    return match
    
def checkmatch_d(step,halo,hid,disp,sim):
    '''
    Check that <halo> (at snapshot number <step>) is the main descendant of <hid> <disp> snapshots in the future
    Returns True or False 
    '''
    proj = sim[int(step)][int(halo)].calculate_for_descendants('finder_id()')[0]
    try:
        match = (proj[disp]==hid)
    except:
        match = False
    return match

def find_merger_target_and_previous_id(step, halo, sim, newest_snapshot_index):
    '''
    Find which halo at the newest snapshot this halo merges into,
    then return that halo's ID at the previous snapshot (newest-1).
    Returns (previous_snapshot_number, halo_id_at_previous_snapshot)
    '''
    # First, track forward to find the final descendant
    final_step, final_halo = trackforward(step, halo, sim)
    
    # Find which halo at the newest snapshot this eventually merges into
    merger_target_at_newest = None
    
    if final_step == newest_snapshot_index:
        # This halo exists at the newest snapshot
        merger_target_at_newest = final_halo
    else:
        # Find which halo at the newest snapshot contains this as a progenitor
        try:
            newest_halos = sim[newest_snapshot_index].halos
            for candidate_halo in newest_halos:
                try:
                    # Get progenitors of this candidate halo
                    prog_steps = candidate_halo.calculate_for_progenitors('timestep.time_gyr')
                    prog_halos = candidate_halo.calculate_for_progenitors('halo_number()')
                    
                    # Check if our final halo is in the progenitor tree
                    final_time = sim[final_step].time_gyr
                    for i, prog_step in enumerate(prog_steps):
                        if abs(prog_step - final_time) < 1e-6:  # Same timestep
                            if prog_halos[i] == final_halo:
                                merger_target_at_newest = candidate_halo.halo_number
                                break
                    if merger_target_at_newest is not None:
                        break
                except:
                    continue
        except:
            pass
    
    if merger_target_at_newest is None:
        # Fallback: use the final halo we tracked to
        merger_target_at_newest = final_halo
    
    # Now find what this merger target halo was called at the previous snapshot
    previous_snapshot_index = newest_snapshot_index - 1
    if previous_snapshot_index < 0:
        # Edge case: only one snapshot exists
        return newest_snapshot_index, merger_target_at_newest
    
    try:
        # Get the halo at the newest snapshot
        target_halo_at_newest = sim[newest_snapshot_index][merger_target_at_newest]
        
        # Get its main progenitor at the previous snapshot
        progenitors = target_halo_at_newest.calculate_for_progenitors('halo_number()')
        if len(progenitors) > 1:  # Index 0 is self, index 1 is one step back
            previous_halo_id = progenitors[1]
        else:
            # No progenitor found, use the same ID
            previous_halo_id = merger_target_at_newest
            
    except Exception as e:
        print(f"Error finding previous ID for halo {merger_target_at_newest}: {e}")
        # Fallback: use the merger target ID
        previous_halo_id = merger_target_at_newest
    
    return previous_snapshot_index, previous_halo_id

def trackforward(step,halo,sim):
    '''
    Track <halo> (at snapshot number <step>) forward in time
    Returns latest snapshot number and halo number when <halo> was self-consistently identified
    '''
    # Make a list of the descendants of <halo> (at <step>)
    # Can simplify considerably once earlier/later works with phantoms
    desc,fid,phid = sim[int(step)][int(halo)].calculate_for_descendants('halo_number()','finder_id()','type()')

    # Trim phantoms off the end of the list
    eol = phid[-1]
    while eol>0:
        fid = fid[:-1]
        desc = desc[:-1]
        phid = phid[:-1]
        eol = phid[-1]

    # make sure that this list is self-consistent - i.e., the last halo in this list identifies
    # <halo> as its main progenitor
    nd = len(desc)-1
    match = checkmatch_p(step+nd,desc[nd],fid[0],nd,sim)
    stat = int(match)

    # If the final halo in the list doesn't identify <halo> as its main progenitor, identify
    # the last halo in the list that does. This halo provides the unique ID
    last_t = 0
    if stat == 0:
        ma_desc = desc[phid<1] # avoid phantoms
        refarr = np.cumsum(phid<1)
        sf = len(ma_desc)
        s0 = 0
        while (s0 <= sf):
            ci = (s0+sf)//2
            ci_trans = np.argmax(refarr>ci)
            match = checkmatch_p(step+ci_trans,desc[ci_trans],fid[0],ci_trans,sim)
            if match==True:
                s0 = ci+1
                last_t = ci_trans
            else:
                sf = ci-1
    else:
        last_t = nd
    return step+last_t,int(desc[last_t])

def main(sim, d, hsfile, ofile):
    # read in your list of hosts that had new stars at each snapshot
    # and initialize your list of unique IDs
    tslist = []
    hostlist = []
    nslist = []
    unilist = [] # list that will contain strings of unique IDs
    rf = open(hsfile,'r')
    rlist = rf.readline()
    badlist = []
    badlist_nums = []
    while rlist != '':
        rs = str(rlist).split()
        tslist.append(int(rs[0]))
        loch = []
        locn = []
        for p in rs[1:]:
            ps = p.split(',')
            loch.append(int(ps[0]))
            if len(ps)>1:
                locn.append(float(ps[1]))
        if int(rs[0]) in badlist:
            loch.append(-1)
            ind = np.where(np.array(badlist)==int(rs[0]))[0][0]
            print (ind)
            locn.append(badlist_nums[ind])
        hostlist.append(np.array(loch))
        if len(ps)>1:
            nslist.append(np.array(locn))
        unilist.append(np.array(['        ' for x in loch]))
        # currently assumes 8 characters is fine, so max timestep is 9999, max halo is 999
        # easy to change if you need to
        rlist = rf.readline()
    rf.close()

    # Find the newest snapshot and previous snapshot
    fulltslist = sim.timesteps
    fulltslist = np.array([int(str(tstr.extension).split('.')[-1]) for tstr in fulltslist])
    newest_snapshot_index = len(fulltslist) - 1
    newest_snapshot = fulltslist[newest_snapshot_index]
    
    if newest_snapshot_index > 0:
        previous_snapshot = fulltslist[newest_snapshot_index - 1]
        print(f"Using newest snapshot: {newest_snapshot}, previous snapshot: {previous_snapshot}")
    else:
        previous_snapshot = newest_snapshot
        print(f"Only one snapshot available: {newest_snapshot}")

    # for each host that has a new star in each snapshot, generate the unique ID
    # based on what the merger target halo was called at the previous snapshot
    kt = 0
    for i in range(0,len(fulltslist)):
        if fulltslist[i] in tslist: # for each snapshot
            print ('------',fulltslist[i])
            for ctr, hid in enumerate(hostlist[kt]): # for each host in that snapshot
                if hid<=0: # if this host is just "amiga.grp didn't find one"...
                    unilist[kt][ctr] = str(previous_snapshot).zfill(4)+'_0' # unique ID is <previous_snapshot>_0
                else: # Otherwise...
                    pstr = str(i).zfill(4)+','+str(hid)
                    print ('Current: '+str(fulltslist[i]).zfill(4)+'_'+str(hid))
                    if d[pstr] != []: # If we've already found the result for this host
                        keystr = d[pstr]
                        print ('Found key: ',keystr)
                    else: # Otherwise, find the merger target and its previous ID
                        try:
                            prev_step, prev_halo_id = find_merger_target_and_previous_id(i, hid, sim, newest_snapshot_index)
                            prev_snapshot_num = fulltslist[prev_step]
                            keystr = str(prev_snapshot_num).zfill(4)+'_'+str(prev_halo_id)
                            print ('Unique: ',keystr)
                            
                            # Store this result for future reference
                            d[pstr] = keystr
                            
                            # Also store for all progenitors in the chain
                            try:
                                proj,fid,phid = sim[i][hid].calculate_for_progenitors('halo_number()','finder_id()','type()')
                                for j in range(len(proj)):
                                    if phid[j] < 1:  # avoid phantoms
                                        prog_step = i - j
                                        if prog_step >= 0:
                                            pstr_prog = str(prog_step).zfill(4)+','+str(proj[j])
                                            d[pstr_prog] = keystr
                            except:
                                pass
                        except Exception as e:
                            print(f"Error processing halo {hid} at step {i}: {e}")
                            # Fallback to original method but use previous snapshot format
                            unis,unih = trackforward(i,hid,sim)
                            keystr = str(previous_snapshot).zfill(4)+'_'+str(unih)
                            d[pstr] = keystr
                    
                    unilist[kt][ctr] = keystr
            kt = kt+1

    # Write out your list of unique IDs
    wf = open(ofile,'w')
    for i in range(0,len(tslist)):
        tstr = str(tslist[i])
        for ctr in range(0,len(hostlist[i])):
            tstr = tstr+'\t'+str(unilist[i][ctr])+','+str(hostlist[i][ctr])
            if len(ps)>1:
                tstr = tstr+','+str(nslist[i][ctr])
        tstr = tstr+'\n'
        wf.write(tstr)
    wf.close()

if __name__ == '__main__':
    
    if len(sys.argv) != 3:
        print ('Usage: python writeouthosts_rz.py <sim> <odir>')
        sys.exit()
    else:
        cursim = str(sys.argv[1])
        odir = str(sys.argv[2])
        
    sim = db.get_simulation(cursim+'%')
    
    d = collections.defaultdict(list)

    hsfile = odir+cursim+'_halostarhosts.txt'
    ofile = odir+cursim+'_uniquehalostarhosts2.txt'

    main(sim, d, hsfile, ofile)

'''
Created on Aug 20, 2021

@author: anna
'''