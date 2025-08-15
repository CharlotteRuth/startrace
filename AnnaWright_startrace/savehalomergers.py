import os
import sys
import socket
hostname = socket.gethostname()
if 'emu' in hostname:
    os.environ['TANGOS_SIMULATION_FOLDER'] = '/home/ns1917/tangos_sims/'
    # os.environ['TANGOS_DB_CONNECTION'] = '/home/ns1917/Databases/Marvel_BN_N10.db'
    os.environ['TANGOS_DB_CONNECTION'] = '/home/ns1917/pynbody/Tangos/Marvel_BN_N10.db'
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

def get_halo(snapshot, halo_number):
    ts = db.get_timestep(f"{ss_dir}/%{snapshot}")
    # print(f"Retrieved timestep: {ts}")
    return ts.halos.filter_by(halo_number=int(halo_number)).first()

# Simulation name and path
if 'emu' in hostname:
    simpath = '/home/ns1917/tangos_sims/'
    outfile_dir = "/home/ns1917/pynbody/stellarhalo_trace_aw/"
else:
    simpath = '/home/selvani/MAP/Sims/cptmarvel.cosmo25cmb/cptmarvel.cosmo25cmb.4096g5HbwK1BH/'
    outfile_dir = "/home/selvani/MAP/pynbody/stellarhalo_trace_aw/"

basename = 'storm.cosmo25cmb.4096g5HbwK1BH'
ss_dir = 'storm.4096g5HbwK1BH_bn'
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

s = pynbody.load(ss_z0)
h = s.halos(halo_numbers='v1')
main_halo = get_halo('004096', 4)
mask = s.s['amiga.grp'] == main_halo.halo_number

halo_numbers, dbids = main_halo.calculate_for_progenitors("halo_number()", "dbid()")
snapshots = [db.get_halo(dbid).timestep.extension[-6:] for dbid in dbids]
halo_snapshots_dict = {snapshot: halo_number for snapshot, halo_number in zip(snapshots, dbids)}
halo_snapshots_dict.keys()

stars_to_consider = s.s['iord'][mask]
unique_starids = np.unique([halo_particle_dict[star] for star in stars_to_consider])
print(f"Number of unique star particles in the main halo: {len(unique_starids)}")

def main(idx):
    # idx = '2304_14'
    snapshot, halo_num = idx.split('_')
    halo_merger = get_halo(snapshot, int(halo_num))

    halo_starmask = hostids == idx
    all_star_iords = partids[halo_starmask]
    all_star_tform = pct[halo_starmask]

    ndm, halonums, dbids2 = halo_merger.calculate_for_progenitors('NDM()', 'halo_number()', 'dbid()')
    halo_dm_max = db.get_halo(dbids2[np.argmax(ndm)])

    # get dark matter iords we want
    sim = pynbody.load(halo_dm_max.timestep.filename)
    mask = sim.dm['amiga.grp'] == int(halo_dm_max.halo_number)
    all_dm_iords = sim.dm['iord'][mask]

    timesteps_to_process = db.get_simulation(ss_dir).timesteps
    num_snaps = len(timesteps_to_process)
    num_star_particles = len(all_star_iords)
    num_dm_particles = len(all_dm_iords)

    # Use np.nan to fill arrays. This makes it clear if a particle was not present in a snapshot.
    star_iords = np.full((num_snaps, num_star_particles), np.nan)
    star_pos = np.full((num_snaps, num_star_particles, 3), np.nan)
    star_vel = np.full((num_snaps, num_star_particles, 3), np.nan)
    star_mass = np.full((num_snaps, num_star_particles), np.nan)
    star_age = np.full((num_snaps, num_star_particles), np.nan)
    star_feh = np.full((num_snaps, num_star_particles), np.nan) # Using Fe/H as a single metallicity value

    dm_pos = np.full((num_snaps, num_dm_particles, 3), np.nan)
    dm_vel = np.full((num_snaps, num_dm_particles, 3), np.nan)
    dm_mass = np.full((num_snaps, num_dm_particles), np.nan)

    zs = np.zeros(num_snaps)
    snaps = [ts.extension for ts in timesteps_to_process]

    all_star_iords = np.sort(all_star_iords)
    all_dm_iords = np.sort(all_dm_iords)

    star_iord_map = {iord: k for k, iord in enumerate(all_star_iords)}
    dm_iord_map = {iord: k for k, iord in enumerate(all_dm_iords)}

    prev_time = 0
    for i, tstep in enumerate(tqdm.tqdm(timesteps_to_process)):
        s = pynbody.load(tstep.filename)
        s.physical_units()
        h = s.halos(halo_numbers='v1')
        halo = db.get_halo(halo_snapshots_dict[tstep.extension[-6:]])
        print(f"Loaded snapshot: {tstep.extension[-6:]}, ", end='')
        zs[i] = s.properties['z']

        # Center the whole simulation on the halo of interest.
        pynbody.analysis.halo.center(h[halo.halo_number], vel=True)
        print(f"Centered on halo: {halo.halo_number}")

        stars_present_mask = pynbody.filt.HighPass('tform', prev_time)
        subs = s.s[stars_present_mask and np.isin(s.s['iord'], all_star_iords)]

        iords_in_subs = np.array(subs['iord'])
        k_indices_star = np.array([star_iord_map[iord] for iord in iords_in_subs])

        if len(k_indices_star) > 0:
            star_pos[i, k_indices_star, :] = subs['pos']
            star_vel[i, k_indices_star, :] = subs['vel']
            star_mass[i, k_indices_star] = subs['mass']
            star_age[i, k_indices_star] = subs['age']
            star_feh[i, k_indices_star] = subs['feh']
        print(f"Processed {len(k_indices_star)} star particles in snapshot {tstep.extension[-6:]}")

        subd = s.dm[np.isin(s.dm['iord'], all_dm_iords)]
        iords_in_subd = np.array(subd['iord'])
        k_indices_dm = np.array([dm_iord_map[iord] for iord in iords_in_subd])
        if len(k_indices_dm) > 0:
            dm_pos[i, k_indices_dm, :] = subd['pos']
            dm_vel[i, k_indices_dm, :] = subd['vel']
            dm_mass[i, k_indices_dm] = subd['mass']
        print(f"Processed {len(k_indices_dm)} DM particles in snapshot {tstep.extension[-6:]}")
        prev_time = tstep.time_gyr

    output_filename = os.path.join(outfile_dir, 'uw_boundfrac', f"{ss_dir}_{main_halo.halo_number}_{idx}_particle_data.h5")

    with h5py.File(output_filename, 'w') as f:
        f.create_dataset('snaps', data=np.bytes_(snaps))
        f.create_dataset('zs', data=zs)

        # Star data
        f.create_dataset('star_iords', data=all_star_iords)
        f.create_dataset('star_pos', data=star_pos)
        f.create_dataset('star_vel', data=star_vel)
        f.create_dataset('star_mass', data=star_mass)
        f.create_dataset('star_age', data=star_age)
        f.create_dataset('star_feh', data=star_feh)

        # DM data
        f.create_dataset('dm_iords', data=all_dm_iords)
        f.create_dataset('dm_pos', data=dm_pos)
        f.create_dataset('dm_vel', data=dm_vel)
        f.create_dataset('dm_mass', data=dm_mass)

    print("Done.")
    return output_filename

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(f"Processing halo: {sys.argv[1]}")
        main(sys.argv[1])