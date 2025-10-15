import os
import sys
import socket
hostname = socket.gethostname()
if 'emu' in hostname:
    os.environ['TANGOS_SIMULATION_FOLDER'] = '/home/ns1917/tangos_sims/'
    os.environ['TANGOS_DB_CONNECTION'] = '/home/ns1917/Databases/Marvel_BN_N10.db'
    # os.environ['TANGOS_DB_CONNECTION'] = '/home/ns1917/pynbody/Tangos/Marvel_BN_N10.db'
    os.chdir('/home/ns1917/pynbody/AnnaWright_startrace/')
else: # grinnell
    os.environ['TANGOS_SIMULATION_FOLDER'] = '/home/selvani/MAP/Sims/cptmarvel.cosmo25cmb/cptmarvel.cosmo25cmb.4096g5HbwK1BH/'
    # os.environ['TANGOS_DB_CONNECTION'] = '/home/selvani/MAP/Data/Marvel_BN_N10.db'
    os.environ['TANGOS_DB_CONNECTION'] = '/home/selvani/MAP/pynbody/Tangos/Marvel_BN_N10.db'
    os.chdir('/home/selvani/MAP/pynbody/AnnaWright_startrace/')

import pynbody
import numpy as np
import h5py
import math
import tangos as db
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import glob
from pynbody.array import SimArray
import pandas as pd
import tqdm.auto as tqdm

# Simulation name and path
if 'emu' in hostname:
    simpath = '/home/ns1917/tangos_sims/'
    outfile_dir = "/home/ns1917/pynbody/stellarhalo_trace_aw/"
    out2dir = "/home/ns1917/gdrive/ananke/"
else:
    simpath = '/home/selvani/MAP/Sims/cptmarvel.cosmo25cmb/cptmarvel.cosmo25cmb.4096g5HbwK1BH/'
    outfile_dir = "/home/selvani/MAP/pynbody/stellarhalo_trace_aw/"

simname = 'storm'
basename = f'{simname}.cosmo25cmb.4096g5HbwK1BH'
ss_dir = f'{simname}.4096g5HbwK1BH_bn'
sim_base = simpath + ss_dir + '/'
ss_z0 = sim_base + basename + '.004096'

# Read in data from Anna's pipeline
with h5py.File(outfile_dir+'/'+basename+'_allhalostardata_upd.h5','r') as f:
    hostids = f['host_IDs'].asstr()[:] # unique host IDs
    partids = f['particle_IDs'][:] # iords
    pct = f['particle_creation_times'][:] # formation times
    ph = f['particle_hosts'][:] # local host IDs (i.e., host at formation time)
    pp = f['particle_positions'][:] # position at formation time
    tsloc = f['timestep_location'][:] # snapshot where star particle first appears
uIDs = np.unique(hostids)

halo_particle_dict = {} # map iords to their unique host IDs
for i, part in enumerate(partids):
    halo_particle_dict[part] = hostids[i]

def main(idx):
    tqdm.tqdm.write(f"Starting processing for halo: {idx}")
    s = pynbody.load(ss_z0)
    h = s.halos(halo_numbers='v1')
    
    # Center the whole simulation on the halo of interest.
    pynbody.analysis.halo.center(h[int(idx)], vel=True)
    rvir = h[int(idx)].properties['Rvir']
    cen = [0, 0, 0]  # Center is at origin after centering
    sp = s[pynbody.filt.Sphere(SimArray([rvir], "kpc"), cen)].load_copy() # only show particles in the specified sphere
    sp.physical_units()
    tqdm.tqdm.write(f"Centered on halo: {idx}")

    # mask = s.s['amiga.grp'] == int(idx)
    mask = sp.s['tform'] > 0
    # mask = mask & mask2
    stars_to_consider = sp.s['iord'][mask] 
    unique_starids = np.unique([halo_particle_dict[star] for star in stars_to_consider])
    tqdm.tqdm.write(f"Number of unique star particles in the main halo: {len(stars_to_consider)}")
    tqdm.tqdm.write(f"Number of unique host halos these stars formed in: {(unique_starids)}")
    # pbar update 2?
    all_star_iords = np.sort(stars_to_consider)
    num_star_particles = len(all_star_iords)

    tstep = db.get_simulation(ss_dir).timesteps[-1]
    # s.physical_units()
    tqdm.tqdm.write(f"Loaded snapshot: {tstep.extension[-6:]}, ", end='')

    subs = sp.s[np.isin(sp.s['iord'], all_star_iords)]
    iords_in_subs = np.array(subs['iord'])

    star_uid = [halo_particle_dict[i] for i in iords_in_subs]
    star_pos = subs['pos']
    star_vel = subs['vel']
    star_mass = subs['mass']
    star_massform = subs['massform']
    star_age = subs['age']
    star_feh = subs['feh']
    star_oxh = subs['oxh']
    tqdm.tqdm.write(f"Processed {len(iords_in_subs)} star particles in snapshot {tstep.extension[-6:]}")
    # print(np.unique(star_uid, return_counts=True))
    output_filename = os.path.join(outfile_dir, 'ananke', simname, f"ananke_{basename}_{idx}_data.h5")

    with h5py.File(output_filename, 'w') as f:
        # Star data
        f.create_dataset('star_iords', data=iords_in_subs)
        f.create_dataset('star_uid', data=star_uid)#, dtype=h5py.string_dtype(encoding='utf-8'))
        f.create_dataset('star_pos', data=star_pos)
        f.create_dataset('star_vel', data=star_vel)
        f.create_dataset('star_mass', data=star_mass)
        f.create_dataset('star_massform', data=star_massform)
        f.create_dataset('star_age', data=star_age)
        f.create_dataset('star_feh', data=star_feh)
        f.create_dataset('star_oxh', data=star_oxh)
    tqdm.tqdm.write("Done.")
    return output_filename

if __name__ == "__main__":
    if len(sys.argv) == 2:
        tqdm.tqdm.write(f"Processing halo: {sys.argv[1]}")
        main(sys.argv[1])
    elif len(sys.argv) > 2:
        for arg in tqdm.tqdm(sys.argv[1:]):
            tqdm.tqdm.write(f"Processing halo: {arg}")
            main(arg)
    else:
        print("Usage: python savehaloinfoananke.py <halo_index1> <halo_index2> ...")
        sys.exit(1)