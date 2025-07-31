

'''
Step 2 of stellar halo pipeline
Identifies the host of each star particle in <sim>_tf.npy at the 
time it was formed. Note that what is stored is NOT the amiga.grp 
ID, but the index of that halo in the tangos database. The amiga.grp
ID can be backed out via tangos with sim[stepnum][halonum].finder_id.

Output: <sim>_stardata_<snapshot>.h5
        where <snapshot> is the first snapshot that a given process
        analyzed. There will be <nproc> of these files generated
        and processes will not necessarily analyze adjacent snapshots

Usage:   python LocAtCreation_pool_rz.py <sim> optional:<nproc>
Example: python LocAtCreation_pool_rz.py r634 2

Includes an optional argument to specify number of processes to run
with; default is 4. Note that this will get reduced if you've specified
more processes than you have snapshots to process.

Note that this is currently set up for MMs, but should be easily adapted 
by e.g., changing the paths or adding a path CL argument. 
'''

import numpy as np
import h5py
from astropy.table import Table
from multiprocessing import Pool
import pynbody
import tangos as db
from collections import defaultdict
import sys
import tqdm.notebook as tqdm

import os
# NS: added this to make sure the path is correct
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print("Base path:", base_path)
# base_path = '/home/christenc/Code/python/AnnaWrite_startrace'

for root, dirs, files in os.walk(base_path):
    if root not in sys.path:
        sys.path.append(root)

import FindHaloStars as fhs

def main(simpath, cursim, db_sim, odir, n_processes=4, overwrite=True):
    pynbody.config['halo-class-priority'] = ['AmigaGrpCatalogue']

    # odir = '/Users/Anna/Research/Outputs/M33Analogs/MM/'+cursim+'/'
    # halostarsfile = '/Users/Anna/Research/Outputs/M33Analogs/'+cursim+'_tf.npy'
    # simpath = '/Volumes/Audiobooks/RomZooms/'+cursim+'.romulus25.3072g1HsbBH/'
    # cursim = simpath.split('/')[-2]

    fhs.odir = odir
    fhs.simpath = simpath
    fhs.cursim = cursim
    fhs.db_sim = db_sim
    # halostarsfile = os.path.join(odir, f"{simpath.split('/')[-2]}_tf.npy")
    halostarsfile = os.path.join(odir, f"{cursim}_tf.npy")

    dat = np.load(halostarsfile) # load in data
    fhs.dat = dat
    halostars = dat[0]
    createtime = dat[1]
    #fhs.createtime = createtime
    
    # Grab times for all available snapshots
    tst = [] # name of snapshot
    tgyr = [] # time of snapshot in Gyr
    #db_sim = 'snapshots_200crit_' + (simpath.split('/')[-2]).split('.')[0]
    sim = db.get_simulation(db_sim)
    ts = sim.timesteps
    for d in ts:
        tgyr.append(d.time_gyr)
        tst.append(d.extension.split('/')[0])
    tgyr = np.array(tgyr)
    tst = np.array(tst)
    sortord = np.argsort(tgyr)
    tgyr = list(tgyr[sortord])
    tst = list(tst[sortord])

    fhs.tst = tst
    fhs.tgyr = tgyr
    fhs.sim = sim
    
    # which snapshots actually contain new star particles?
    stardist = np.histogram(createtime,bins=[0]+tgyr)
    steplist = np.array(tst)[stardist[0]>0]
    print ('Stars from '+str(len(steplist))+' steps left to deal with')
    np.random.shuffle(steplist) # for load balancing

    nsteps = len(steplist)
    nprocesses = np.min([n_processes,nsteps]) # make sure we have at least as many steps as we have processes
    print ('Initializing ',nprocesses)

    #initialize the process pool and build the chunks to send to each process - adapted from powderday
    p = Pool(processes = nprocesses)
    nchunks = nprocesses
    chunk_start_indices = []
    chunk_start_indices.append(0)

    #this should just be int(nsteps/nchunks) but in case nsteps < nchunks, we need to ensure that this is at least  1
    delta_chunk_indices = np.max([int(nsteps / nchunks),1])

    for n in range(1,nchunks):
        chunk_start_indices.append(chunk_start_indices[n-1]+delta_chunk_indices)

    list_of_chunks = []
    for n in range(nchunks):
        steps_list_chunk = steplist[chunk_start_indices[n]:chunk_start_indices[n]+delta_chunk_indices]
        #if we're on the last chunk, we might not have the full list included, so need to make sure that we have that here
        if n == nchunks-1:
            steps_list_chunk = steplist[chunk_start_indices[n]::]
        list_of_chunks.append(steps_list_chunk)
    # for i in list_of_chunks:
    #     print('Chunk', i)
    # print('Chunks to process:',list_of_chunks)

    # Don't create a pool if running serially
    # shuffle
    list_of_chunks = np.random.permutation(list_of_chunks)
    print(f'Shuffled chunks:', list_of_chunks)
    pbar = tqdm.tqdm(total=len(list_of_chunks), desc=f'Processing', unit='chunks')
    for i, arg in enumerate(list_of_chunks):
        print(f"Processing chunk {i+1}/{len(list_of_chunks)}: {arg}")
        try:
            result = fhs.FindHaloStars(arg,overwrite=overwrite)
            print(f"  Completed: {result.split('.')[-2][-6:]}\n")
        except Exception as e:
            print(f"\tError processing chunk {i+1}: {e}")
        pbar.update(1)
    pbar.close()
    print("All chunks processed serially")

    # fhs.FindHaloStars(list_of_chunks[0])  # Useful for checking work on single snapshot
    #                                         #    before multiprocessing. Replaces next four lines of code

    # print('Starting multiprocessing with', nprocesses, 'processes')

    # idxs = list(range(len(list_of_chunks)))

    # # interweave indices with chunks
    # interleaved = list(zip(list_of_chunks, idxs))
    # for chunk in interleaved:
    #     print("Chunk:", chunk[0], "Index:", chunk[1])
    
    # try:
    #     # Use map and collect results
    #     results = p.map(multiprocessing_wrapper, interleaved)

    #     print("All processes completed successfully!")
    #     print("Output files created:")
                
    # except Exception as e:
    #     print(f"Error during multiprocessing: {e}")
    #     p.terminate()  # Force terminate if there's an error
        
    # finally:
    #     # Proper cleanup
    #     p.close()
    #     p.join()
    #     print('Pool properly closed and joined')
    

def multiprocessing_wrapper(interleaved):
    """
    Wrapper function for multiprocessing to call FindHaloStars.
    """
    chunks, j = interleaved
    list_of_chunks = np.random.permutation(chunks)
    print(f'Shuffled chunks {j}:', list_of_chunks)
    pbar = tqdm.tqdm(total=len(list_of_chunks), desc=f'Processing {j}', unit='chunks')
    for i, arg in enumerate(list_of_chunks):
        print(f"Processing chunk {i+1}/{len(list_of_chunks)}: {arg}")
        try:
            result = fhs.FindHaloStars([arg])
        except Exception as e:
            print(f"\tError processing chunk {i+1}: {e}")
        pbar.update(1)
        if result:
            print(f"  Completed: {result.split('.')[-2][-6:]}\n")
    pbar.close()
    print("All chunks processed serially")
    # for chunk in chunks:
    #     try:
    #         result = fhs.FindHaloStars(chunk)
    #         if result:
    #             print(f"Completed: {result.split('.')[-2][-6:]}")
    #     except Exception as e:
    #         print(f"Error processing chunk {chunk}: {e}")

    return 0

        
if __name__ == '__main__':
    n_processes = 4 # default number of processes 

    if len(sys.argv)<4 or len(sys.argv)>5:
        print ('Usage: python LocAtCreation_pool_rz.py <sim> <db_sim> <odir> opt:<nproc>')
        print ('       default number of processes is '+int(n_processes))
        sys.exit()
    elif len(sys.argv)==4:
        simpath = str(sys.argv[1])
        db_sim = str(sys.argv[2])
        odir = str(sys.argv[3])
    else:
        simpath = str(sys.argv[1])
        db_sim = str(sys.argv[2])
        odir = str(sys.argv[3])
        n_processes = int(sys.argv[4])

    main(simpath, db_sim, odir, n_processes) # NS: fix typo in var name
        
'''
Created on Mar 4, 2024

@author: anna
'''
