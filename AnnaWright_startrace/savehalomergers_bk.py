"""
savehalomergers.py

Track and save merger history particle data for stellar halos.

This script traces star and dark matter particles from merger progenitor halos
across all simulation snapshots. It identifies a specific merger event (by snapshot
and halo number) and extracts the full time evolution of particles that originated
in that merger progenitor, relative to a main host halo.

The script uses the Tangos database to track halo merger trees and pynbody to
extract particle data. Output is saved as HDF5 files containing positions, velocities,
masses, and other properties for both star and dark matter particles across all
available snapshots.

Key Features:
- Tracks particles from merger progenitors through cosmic time
- Centers all data on the main halo at each snapshot
- Preserves full particle histories (position, velocity, mass, age, metallicity)
- Handles particles that may not be present in all snapshots (using NaN)
- Identifies peak dark matter content to define merger progenitor particles

Author: Nithun Selva
Date: 2025

Usage:
    # Track merger 0384_47 into halo 7 in rogue simulation
    python savehalomergers.py --name rogue --halo 7 0384_47
    
    # Track multiple mergers into halo 10 in storm simulation
    python savehalomergers.py -n storm -H 10 0291_1 0384_47 0512_3
    
    # Show help
    python savehalomergers.py --help

Output:
    HDF5 files containing:
    - Star particle data: positions, velocities, masses, ages, metallicities
    - Dark matter particle data: positions, velocities, masses
    - Snapshot information: redshifts, timestep identifiers
"""

import os
import sys
import socket
import argparse
import tangos as db

#! Determine hostname to configure paths appropriately
hostname = socket.gethostname()
if 'emu' in hostname:
    # EMU system paths
    os.environ['TANGOS_SIMULATION_FOLDER'] = '/home/ns1917/tangos_sims/'
    os.environ['TANGOS_DB_CONNECTION'] = '/home/ns1917/Databases/Marvel_BN_N10.db'
    # os.environ['TANGOS_DB_CONNECTION'] = '/home/ns1917/pynbody/Tangos/Marvel_BN_N10.db'  # Alternative path
    # os.chdir('/home/ns1917/pynbody/AnnaWright_startrace/')
    os.chdir('/home/christenc/Code/python/NithunSelva_startrace/pynbody/AnnaWright_startrace/')

    simpath = '/data/REPOSITORY/romulus_zooms/'
    os.environ['TANGOS_SIMULATION_FOLDER'] = simpath
    db.core.init_db(simpath + '/rom25_dwarf_zooms.db')

else:  # Grinnell system paths
    os.environ['TANGOS_SIMULATION_FOLDER'] = '/home/selvani/MAP/Sims/cptmarvel.cosmo25cmb/cptmarvel.cosmo25cmb.4096g5HbwK1BH/'
    # os.environ['TANGOS_DB_CONNECTION'] = '/home/selvani/MAP/Data/Marvel_BN_N10.db'  # Alternative path
    os.environ['TANGOS_DB_CONNECTION'] = '/home/selvani/MAP/pynbody/Tangos/Marvel_BN_N10.db'
    os.chdir('/home/selvani/MAP/pynbody/AnnaWright_startrace/')

import pynbody
import numpy as np
import h5py
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import glob
from pynbody.array import SimArray
import pandas as pd
import tqdm.auto as tqdm


def get_halo(snapshot, halo_number, ss_dir):
    """Retrieve a specific halo from a snapshot using Tangos database.
    
    Args:
        snapshot (str): Snapshot identifier (e.g., '004096' or '000384')
        halo_number (int): Halo number within the snapshot
        ss_dir (str): Simulation directory name (e.g., 'rogue.4096g5HbwK1BH_bn')
    
    Returns:
        tangos.core.halo.Halo: The requested halo object from Tangos database
    """
    ts = db.get_timestep(f"{ss_dir}/%{snapshot}")
    # print(f"Retrieved timestep: {ts}")  # Uncomment for debugging
    return ts.halos.filter_by(halo_number=int(halo_number)).first()

def setup_paths_and_data(simname, main_halo_num):
    """Initialize simulation paths and load halo particle mapping data.
    
    This function sets up all necessary paths based on the hostname and simulation name,
    loads star particle data from Anna's pipeline, and initializes the main halo information.
    
    Args:
        simname (str): Name of simulation ('storm', 'elektra', 'rogue', or 'cptmarvel')
        main_halo_num (int): Main halo number to track mergers into
    
    Returns:
        tuple: (simpath, outfile_dir, basename, ss_dir, sim_base, ss_z0, 
                halo_particle_dict, main_halo, halo_snapshots_dict, hostids, partids, pct)
    """
    #! Configure paths based on hostname
    if 'emu' in hostname:
        simpath = '/home/ns1917/tangos_sims/'
        outfile_dir = "/home/ns1917/pynbody/stellarhalo_trace_aw/"

        simpath = '/data/REPOSITORY/romulus_zooms/'
        outfile_dir = "/home/christenc/Code/Datafiles/stellar_halos_bf/"
    else:
        simpath = '/home/selvani/MAP/Sims/cptmarvel.cosmo25cmb/cptmarvel.cosmo25cmb.4096g5HbwK1BH/'
        outfile_dir = "/home/selvani/MAP/pynbody/stellarhalo_trace_aw/"

    # Construct simulation-specific paths
    basename = f'{simname}.cosmo25cmb.4096g5HbwK1BH'
    ss_dir = f'{simname}.4096g5HbwK1BH_bn'
    sim_base = simpath + ss_dir + '/'
    ss_z0 = sim_base + basename + '.004096'  # z=0 snapshot

    # Read in data from Anna's pipeline - this contains star formation history info
    # Each star particle is mapped to its original host halo (where it formed)
    #! Update filepaths as needed
    with h5py.File(outfile_dir+'/'+basename+'_allhalostardata_upd.h5','r') as f:
        hostids = f['host_IDs'].asstr()[:]  # unique host IDs (strings like '0384_47')
        partids = f['particle_IDs'][:]  # iords (particle IDs)
        pct = f['particle_creation_times'][:]  # formation times
        ph = f['particle_hosts'][:]  # local host IDs (i.e., host at formation time)
        pp = f['particle_positions'][:]  # position at formation time
        tsloc = f['timestep_location'][:]  # snapshot where star particle first appears
    
    uIDs = np.unique(hostids)

    # Create mapping: particle ID -> original host halo ID
    halo_particle_dict = {}  # map iords to their unique host IDs
    for i, part in enumerate(partids):
        halo_particle_dict[part] = hostids[i]

    # Load z=0 snapshot to get main halo information
    s = pynbody.load(ss_z0)
    h = s.halos(halo_numbers='v1')
    
    # Get the main halo we're tracking mergers into
    main_halo = get_halo('004096', main_halo_num, ss_dir)
    
    # Filter stars: must be in main halo AND have positive formation time
    mask = s.s['amiga.grp'] == main_halo.halo_number
    mask2 = s.s['tform'] > 0  # Exclude wind particles
    mask = mask & mask2
    s.g = 0 # delete gas particles to save memory

    # Build merger tree: map each snapshot to the main halo's progenitor in that snapshot
    halo_numbers, dbids = main_halo.calculate_for_progenitors("halo_number()", "dbid()")
    snapshots = [db.get_halo(dbid).timestep.extension[-6:] for dbid in dbids]
    halo_snapshots_dict = {snapshot: halo_number for snapshot, halo_number in zip(snapshots, dbids)}
    
    # Diagnostic: show unique star particle hosts in main halo
    stars_to_consider = s.s['iord'][mask]
    unique_starids = np.unique([halo_particle_dict[star] for star in stars_to_consider])
    print(f"Number of unique star particles in the main halo: {len(unique_starids)}")
    
    return (simpath, outfile_dir, basename, ss_dir, sim_base, ss_z0, 
            halo_particle_dict, main_halo, halo_snapshots_dict, hostids, partids, pct)

def main(idx, simname, main_halo_num):
    """Process a specific merger event and save particle evolution data.
    
    This function tracks all star and dark matter particles from a merger progenitor
    halo (specified by idx) as it merges into the main halo. It extracts particle
    data across all snapshots and saves the time evolution to an HDF5 file.
    
    Args:
        idx (str): Merger identifier in format 'SSSS_HH' where SSSS is snapshot 
                   number and HH is halo number (e.g., '0384_47')
        simname (str): Name of simulation ('storm', 'elektra', 'rogue', or 'cptmarvel')
        main_halo_num (int): Main halo number to track mergers into
    
    Returns:
        str: Path to output HDF5 file
    """
    # Initialize paths and data for this simulation
    (simpath, outfile_dir, basename, ss_dir, sim_base, ss_z0, 
     halo_particle_dict, main_halo, halo_snapshots_dict, hostids, partids, pct) = setup_paths_and_data(simname, main_halo_num)
    
    # idx format: 'snapshot_halonumber', e.g., '0384_47'
    print(f'Processing merger: {idx}')
    print(f'Main halo: {main_halo.halo_number}')
    
    # Parse the merger identifier to get snapshot and halo number
    snapshot, halo_num = idx.split('_')
    halo_merger = get_halo(snapshot, int(halo_num), ss_dir)

    # Get all star particles that formed in this merger progenitor halo
    halo_starmask = hostids == idx
    all_star_iords = partids[halo_starmask]  # Particle IDs of stars from this merger
    all_star_tform = pct[halo_starmask]  # Formation times of these stars

    # Find the snapshot where this merger progenitor had maximum dark matter content
    # This represents the "peak" of the halo before it merged
    ndm, halonums, dbids2 = halo_merger.calculate_for_progenitors('NDM()', 'halo_number()', 'dbid()')
    halo_dm_max = db.get_halo(dbids2[np.argmax(ndm)])

    # Get dark matter particle IDs from the merger halo at its peak
    sim = pynbody.load(halo_dm_max.timestep.filename)
    mask = sim.dm['amiga.grp'] == int(halo_dm_max.halo_number)
    all_dm_iords = sim.dm['iord'][mask]

    # Get all simulation snapshots to process
    timesteps_to_process = db.get_simulation(ss_dir).timesteps
    # Optional: exclude specific snapshots if needed (uncomment to use)
    # if '000192' in timesteps_to_process, remove it
    # timesteps_to_process = [ts for ts in timesteps_to_process if ts.extension[-6:] != '000192']
    
    num_snaps = len(timesteps_to_process)
    num_star_particles = len(all_star_iords)
    num_dm_particles = len(all_dm_iords)
    
    print(f"Tracking {num_star_particles} star particles and {num_dm_particles} DM particles across {num_snaps} snapshots")

    # Initialize arrays to store particle data across all snapshots
    # Use np.nan to fill arrays - this makes it clear if a particle was not present in a snapshot
    # (e.g., stars before they formed, or particles that left the volume)
    
    # Star particle arrays - indexed by [snapshot, particle_index, (x/y/z)]
    star_iords = np.full((num_snaps, num_star_particles), np.nan)
    star_pos = np.full((num_snaps, num_star_particles, 3), np.nan)  # 3D positions
    star_vel = np.full((num_snaps, num_star_particles, 3), np.nan)  # 3D velocities
    star_mass = np.full((num_snaps, num_star_particles), np.nan)
    star_age = np.full((num_snaps, num_star_particles), np.nan)
    star_feh = np.full((num_snaps, num_star_particles), np.nan)  # Using Fe/H as a single metallicity value

    # Dark matter particle arrays
    dm_pos = np.full((num_snaps, num_dm_particles, 3), np.nan)
    dm_vel = np.full((num_snaps, num_dm_particles, 3), np.nan)
    dm_mass = np.full((num_snaps, num_dm_particles), np.nan)

    # Snapshot metadata
    zs = np.zeros(num_snaps)  # Redshifts
    snaps = [ts.extension for ts in timesteps_to_process]  # Snapshot identifiers

    # Sort particle IDs for consistent ordering
    all_star_iords = np.sort(all_star_iords)
    all_dm_iords = np.sort(all_dm_iords)

    # Create lookup dictionaries: particle ID -> array index
    # This allows fast mapping from particle IDs to their storage locations
    star_iord_map = {iord: k for k, iord in enumerate(all_star_iords)}
    dm_iord_map = {iord: k for k, iord in enumerate(all_dm_iords)}

    # Track previous timestep to filter for newly formed stars
    prev_time = 0
    
    # Loop through all snapshots and extract particle data
    for i, tstep in enumerate(tqdm.tqdm(timesteps_to_process, desc=f"Processing {simname} snapshots")):
        # Load the snapshot and convert to physical units
        s = pynbody.load(tstep.filename)
        s.physical_units()
        h = s.halos(halo_numbers='v1')  # Load halo catalog
        
        # Get the main halo's progenitor at this snapshot
        halo = db.get_halo(halo_snapshots_dict[tstep.extension[-6:]])
        print(f"Loaded snapshot: {tstep.extension[-6:]}, ", end='')
        
        # Store redshift for this snapshot
        zs[i] = s.properties['z']

        # Center the whole simulation on the main halo progenitor
        pynbody.analysis.halo.center(h[halo.halo_number], vel=True)
        print(f"Centered on halo: {halo.halo_number}")

        # --- Process Star Particles ---
        # Only include stars that have formed by this time (tform > prev_time)
        stars_present_mask = pynbody.filt.HighPass('tform', prev_time)
        # Filter for only the star particles we're tracking
        subs = s.s[stars_present_mask and np.isin(s.s['iord'], all_star_iords)]

        # Get particle IDs present in this snapshot
        iords_in_subs = np.array(subs['iord'])
        # Map particle IDs to their storage array indices
        k_indices_star = np.array([star_iord_map[iord] for iord in iords_in_subs])

        # Store star particle data for this snapshot
        if len(k_indices_star) > 0:
            star_pos[i, k_indices_star, :] = subs['pos']  # Positions (kpc)
            star_vel[i, k_indices_star, :] = subs['vel']  # Velocities (km/s)
            star_mass[i, k_indices_star] = subs['mass']  # Masses (solar masses)
            star_age[i, k_indices_star] = subs['age']  # Ages (Gyr)
            star_feh[i, k_indices_star] = subs['feh']  # Iron abundance [Fe/H]
        print(f"Processed {len(k_indices_star)} star particles in snapshot {tstep.extension[-6:]}")

        # --- Process Dark Matter Particles ---
        # Filter for only the DM particles we're tracking
        subd = s.dm[np.isin(s.dm['iord'], all_dm_iords)]
        iords_in_subd = np.array(subd['iord'])
        # Map particle IDs to their storage array indices
        k_indices_dm = np.array([dm_iord_map[iord] for iord in iords_in_subd])
        
        # Store DM particle data for this snapshot
        if len(k_indices_dm) > 0:
            dm_pos[i, k_indices_dm, :] = subd['pos']  # Positions (kpc)
            dm_vel[i, k_indices_dm, :] = subd['vel']  # Velocities (km/s)
            dm_mass[i, k_indices_dm] = subd['mass']  # Masses (solar masses)
        print(f"Processed {len(k_indices_dm)} DM particles in snapshot {tstep.extension[-6:]}")
        
        # Update time tracker for next iteration
        prev_time = tstep.time_gyr

    # --- Save Data to HDF5 File ---
    # Construct output path: organized by simulation name and main halo number
    output_filename = os.path.join(
        outfile_dir, 'uw_boundfrac', simname, str(main_halo.halo_number), 
        f"{ss_dir}_{main_halo.halo_number}_{idx}_particle_data.h5"
    )
    
    # Create output directory if it doesn't exist
    if not os.path.exists(os.path.dirname(output_filename)):
        os.makedirs(os.path.dirname(output_filename))
    print(f"Saving data to {output_filename}")

    # Write all data to HDF5 file
    with h5py.File(output_filename, 'w') as f:
        # Snapshot metadata
        f.create_dataset('snaps', data=np.bytes_(snaps))  # Snapshot identifiers
        f.create_dataset('zs', data=zs)  # Redshifts

        # Star particle data
        # Shape: (num_particles,) for IDs, (num_snaps, num_particles, 3) for positions/velocities
        f.create_dataset('star_iords', data=all_star_iords)  # Particle IDs
        f.create_dataset('star_pos', data=star_pos)  # Positions over time
        f.create_dataset('star_vel', data=star_vel)  # Velocities over time
        f.create_dataset('star_mass', data=star_mass)  # Masses over time
        f.create_dataset('star_age', data=star_age)  # Ages over time
        f.create_dataset('star_feh', data=star_feh)  # Metallicities over time

        # Dark matter particle data
        f.create_dataset('dm_iords', data=all_dm_iords)  # Particle IDs
        f.create_dataset('dm_pos', data=dm_pos)  # Positions over time
        f.create_dataset('dm_vel', data=dm_vel)  # Velocities over time
        f.create_dataset('dm_mass', data=dm_mass)  # Masses over time

    print("Done.")
    return output_filename

def _validate_merger_id(s):
    """Validate merger identifier strings like '0384_47' or '2304_14'.
    
    A valid merger ID has exactly four digits, an underscore, then one or more digits.
    This corresponds to snapshot number and halo number.
    
    Args:
        s (str): Merger identifier string to validate
    
    Returns:
        str: The validated string
        
    Raises:
        argparse.ArgumentTypeError: If format is invalid
    """
    import re
    if not re.match(r'^\d{4}_\d+$', s):
        raise argparse.ArgumentTypeError(
            f"merger_id must be format 'SSSS_HH' (e.g., '0384_47'), got: {s}"
        )
    elif int(s.split('_')[1]) <= 0:
        raise argparse.ArgumentTypeError(
            f"halo number in merger_id must be >=1, got: {s}"
        )
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Track and save merger history particle data for stellar halos.",
        epilog="""
Examples:
  # Track merger 0384_47 into halo 7 in rogue simulation
  python savehalomergers.py --name rogue --halo 7 0384_47
  
  # Track multiple mergers into halo 10 in storm simulation
  python savehalomergers.py -n storm -H 10 0291_1 0384_47 0512_3
  
  # Track merger in cptmarvel simulation into halo 4
  python savehalomergers.py --name cptmarvel --halo 4 1280_17

Merger ID Format:
  Merger IDs must be in format 'SSSS_HH' where:
    - SSSS is the 4-digit snapshot number (e.g., 0384, 2304)
    - HH is the halo number in that snapshot (e.g., 1, 47, 341)
  Examples: 0384_47, 2304_14, 0291_1

Output:
  HDF5 files saved to: {outfile_dir}/uw_boundfrac/{simname}/{halo_number}/
  Each file contains full particle evolution data:
    - Star particles: positions, velocities, masses, ages, metallicities
    - Dark matter particles: positions, velocities, masses
    - Metadata: snapshot IDs, redshifts
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '-n', '--name',
        choices=['storm', 'elektra', 'rogue', 'cptmarvel', '642'],
        default='rogue',
        help="Simulation name to process (default: rogue)"
    )
    parser.add_argument(
        '-H', '--halo',
        type=int,
        default=10,
        help="Main halo number to track mergers into (default: 10)"
    )
    parser.add_argument(
        'merger_ids',
        nargs='+',
        type=_validate_merger_id,
        metavar='MERGER_ID',
        help="One or more merger identifiers in format 'SSSS_HH' (e.g., 0384_47)"
    )
    
    args = parser.parse_args()
    
    simname = args.name
    main_halo_num = args.halo
    
    # Process single or multiple mergers
    if len(args.merger_ids) == 1:
        # Single merger: no progress bar needed
        merger_id = args.merger_ids[0]
        print(f"Processing merger {merger_id} into halo {main_halo_num} in {simname} simulation")
        main(merger_id, simname, main_halo_num)
    else:
        # Multiple mergers: use progress bar
        for merger_id in tqdm.tqdm(args.merger_ids, desc=f"Processing {simname} mergers"):
            tqdm.tqdm.write(f"Processing merger: {merger_id}")
            main(merger_id, simname, main_halo_num)