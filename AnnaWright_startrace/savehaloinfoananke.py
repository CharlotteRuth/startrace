"""
savehaloinfoananke.py

Save halo information from Marvel simulation for Ananke analysis.

This script extracts star particle data from cosmological simulation halos,
including positions, velocities, masses, ages, and metallicities. It centers
on a specified halo and saves stellar properties within the virial radius
to HDF5 files for analysis with Ananke.

Author: Nithun Selva
Date: 2025
Usage:
    python savehaloinfoananke.py --name storm <halo_index1> [<halo_index2> ...]
    python savehaloinfoananke.py -n rogue 4 7 12
    python savehaloinfoananke.py --help
"""

import os
import sys
import socket
import argparse


import pynbody
import numpy as np
import h5py
import math
import tangos as db
import glob
from pynbody.array import SimArray
import pandas as pd
import tqdm.auto as tqdm


def setup_paths(simname):
    """Configure simulation paths based on hostname and simulation name.
    
    Args:
        simname: Name of simulation ('storm', 'elektra', 'rogue', 'cptmarvel' or 'rXXX')
    
    Returns:
        tuple: (simpath, outfile_dir, basename, ss_dir, sim_base, ss_z0)
    """
    if 'emu' in hostname: #! change filepaths as needed
        if suite=='marvel_dwarfs':
            simpath = '/home/ns1917/tangos_sims/'
        else:
            simpath = '/data/REPOSITORY/romulus_zooms/'
        # outfile_dir = "/home/christenc/Code/python/NithunSelva_startrace/pynbody/stellarhalo_trace_aw/" #"/home/ns1917/pynbody/stellarhalo_trace_aw/"
        outfile_dir = "/home/christenc/Code/Datafiles/stellar_halos/"
    else:
        simpath = '/home/selvani/MAP/Sims/cptmarvel.cosmo25cmb/cptmarvel.cosmo25cmb.4096g5HbwK1BH/'
        outfile_dir = "/home/selvani/MAP/pynbody/stellarhalo_trace_aw/"
    
    if suite=='marvel_dwarfs':
        basename = f'{simname}.cosmo25cmb.4096g5HbwK1BH'
        ss_dir = f'{simname}.4096g5HbwK1BH_bn'
        sim_base = simpath + ss_dir + '/'
        ss_z0 = sim_base + basename + '.004096'
    else: 
        basename = f'{simname}'
        ss_dir = f'{simname}.romulus25.3072g1HsbBH'
        sim_base = simpath + ss_dir + '/'
        ss_z0 = sim_base + basename + '.romulus25.3072g1HsbBH.004096' + '/'+basename + '.romulus25.3072g1HsbBH.004096'


    
    return simpath, outfile_dir, basename, ss_dir, sim_base, ss_z0

def read_massform_file(massform_path, ngas, ndm, nstar, byteswap=False):
    """Read massform values directly from a *.massform auxiliary file.

    Bypasses pynbody's starlog reader (which causes a KeyError when 'iord' is
    already a family-level array) by reading the tipsy auxiliary file directly.

    The auxiliary file stores one value per particle for ALL particle types
    (gas, dm, star) in order.  Star values start at index ngas+ndm.

    Args:
        massform_path: Full path to the *.massform file
        ngas:     Number of gas particles in the full snapshot
        ndm:      Number of DM particles in the full snapshot
        nstar:    Number of star particles in the full snapshot
        byteswap: Whether the file is byte-swapped (big-endian)

    Returns:
        np.ndarray: massform values for star particles, in simulation mass units
    """
    import struct
    try:
        # --- Try ASCII format first ---
        with open(massform_path, 'r') as f:
            n_total = int(f.readline().strip())
            vals = np.fromiter(
                (float(line) for line in f if line.strip()),
                dtype=np.float32,
                count=n_total,
            )
    except (ValueError, UnicodeDecodeError):
        # --- Binary format: 4-byte int header then float32 values ---
        with open(massform_path, 'rb') as f:
            raw = f.read(4)
            n_total = struct.unpack(">i" if byteswap else "i", raw)[0]
            vals = np.frombuffer(f.read(n_total * 4), dtype=np.float32)
            if byteswap:
                vals = vals.byteswap()

    # Slice out only the star portion
    return vals[ngas + ndm: ngas + ndm + nstar]


def load_halo_data(outfile_dir, basename):
    """Load star particle data from Anna's pipeline.
    
    Args:
        outfile_dir: Directory containing the HDF5 file
        basename: Base name of the simulation
    
    Returns:
        dict: Dictionary mapping particle IDs to their unique host IDs
    """
    #! change/update filename as needed
    if suite=='marvel_dwarfs':
        starfile_name = outfile_dir+'/'+basename+'_allhalostardata_upd.h5'
    else:
        starfile_name = outfile_dir+'/'+basename+'/' + basename + '_allhalostardata_consolidated2.h5'
        
    with h5py.File(starfile_name,'r') as f:
        hostids = f['host_IDs'].asstr()[:]  # unique host IDs
        partids = f['particle_IDs'][:]  # iords
        pct = f['particle_creation_times'][:]  # formation times
        ph = f['particle_hosts'][:]  # local host IDs (i.e., host at formation time)
        pp = f['particle_positions'][:]  # position at formation time
        tsloc = f['timestep_location'][:]  # snapshot where star particle first appears
    
    uIDs = np.unique(hostids)
    
    halo_particle_dict = {}  # map iords to their unique host IDs
    for i, part in enumerate(partids):
        halo_particle_dict[part] = hostids[i]
    
    return halo_particle_dict

def main(idx, simname):
    """Extract and save star particle data for a given halo.
    
    Args:
        idx: Halo index to process
        simname: Name of simulation ('storm', 'elektra', 'rogue', or 'cptmarvel')
    
    Returns:
        str: Path to output HDF5 file
    """
    tqdm.tqdm.write(f"Starting processing for halo: {idx} in simulation: {simname}")
    
    # Setup paths for this simulation
    simpath, outfile_dir, basename, ss_dir, sim_base, ss_z0 = setup_paths(simname)
    
    # Load halo particle mapping
    halo_particle_dict = load_halo_data(outfile_dir, basename)
    
    s = pynbody.load(ss_z0)
    h = s.halos(halo_numbers='v1')

    # --- Build iord -> massform mapping directly from *.massform file ---
    # This avoids pynbody's starlog reader, which raises a KeyError when
    # 'iord' is already a family-level array.
    massform_path = ss_z0 + '.massform'
    if not os.path.exists(massform_path):
        candidates = glob.glob(ss_z0 + '*.massform')
        if candidates:
            massform_path = candidates[0]
        else:
            raise FileNotFoundError(
                f"No .massform file found for snapshot: {ss_z0}"
            )
    tqdm.tqdm.write(f"Reading massform from: {os.path.basename(massform_path)}")
    raw_massform = read_massform_file(
        massform_path,
        ngas=len(s.g),
        ndm=len(s.dm),
        nstar=len(s.s),
        byteswap=getattr(s, '_byteswap', False),
    )
    # Convert from simulation mass units to Msol
    mass_unit = s.s['mass'].units
    massform_physical = SimArray(raw_massform, mass_unit)
    massform_physical.convert_units('Msol')
    # Map star iord -> massform [Msol]
    iord_to_massform = dict(zip(s.s['iord'], massform_physical))
    
    # Center the whole simulation on the halo of interest.
    pynbody.analysis.halo.center(h[int(idx)], vel=True)
    #print(h[int(idx)].properties)
    if suite=='marvel_dwarfs':
        rvir = h[int(idx)].properties['Rvir']
    else:
        rvir = h[int(idx)].properties['Rhalo']
    cen = [0, 0, 0]  # Center is at origin after centering
    sp = s[pynbody.filt.Sphere(SimArray([rvir], "kpc"), cen)].load_copy() # only show particles in the specified sphere
    sp.physical_units()
    tqdm.tqdm.write(f"Centered on halo: {idx}")

    mask = sp.s['tform'] > 0 # filter by positive formation time
    stars_to_consider = sp.s['iord'][mask] 
    unique_starids = np.unique([halo_particle_dict[star] for star in stars_to_consider])
    tqdm.tqdm.write(f"Number of unique star particles in the main halo: {len(stars_to_consider)}")
    tqdm.tqdm.write(f"Number of unique host halos these stars formed in: {len(unique_starids)}")
    
    all_star_iords = np.sort(stars_to_consider)
    num_star_particles = len(all_star_iords)

    tstep = db.get_simulation(ss_dir).timesteps[-1] #! select different snapshot if needed
    tqdm.tqdm.write(f"Loaded snapshot: {tstep.extension[-6:]}, ", end='')
    
    # Create mask for filtering instead of creating subset
    mask = np.isin(sp.s['iord'], all_star_iords)
    
    star_uid = [halo_particle_dict[i] for i in sp.s['iord'][mask]]
    star_pos = sp.s['pos'][mask]
    star_vel = sp.s['vel'][mask]
    star_mass = sp.s['mass'][mask]
    star_massform = SimArray(
        [iord_to_massform[i] for i in sp.s['iord'][mask]], 'Msol'
    )
    star_age = sp.s['age'][mask]
    star_feh = sp.s['feh'][mask]
    star_oxh = sp.s['oxh'][mask]
    tqdm.tqdm.write(f"Processed {len(star_uid)} star particles in snapshot {tstep.extension[-6:]}")
    
    output_filename = os.path.join(outfile_dir, 'ananke', simname, f"ananke_{basename}_{idx}_data.h5")
    if not os.path.exists(os.path.dirname(output_filename)):
        tqdm.tqdm.write(f"Creating directory: {os.path.dirname(output_filename)}")
        os.makedirs(os.path.dirname(output_filename))

    with h5py.File(output_filename, 'w') as f:
        # Star data
        f.create_dataset('star_iords', data=sp.s['iord'][mask])
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
    parser = argparse.ArgumentParser(
        description="Extract star particle data from Marvel simulation halos for Ananke analysis.",
        epilog="""
Examples:
  # Process halo 4 from the storm simulation
  python savehaloinfoananke.py --name storm 4
  
  # Process multiple halos from the rogue simulation
  python savehaloinfoananke.py -n rogue 4 7 12
  
  # Process halos from cptmarvel simulation
  python savehaloinfoananke.py --name cptmarvel 1 2 3 4 5
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '-n', '--name', 
        #choices=['storm', 'elektra', 'rogue', 'cptmarvel'], 
        default='rogue',
        help="Simulation name to process (default: rogue)"
    )
    parser.add_argument(
        'halo_indices', 
        nargs='+', 
        type=int,
        help='One or more halo indices to process'
    )
    
    args = parser.parse_args()
    
    simname = args.name

    #! Configure paths based on hostname
    hostname = socket.gethostname()
    if simname=='rogue' or simname=='elektra' or simname=='storm':
        suite='marvel_dwarfs'
    else:
        suite='romulus_zooms'
    if 'emu' in hostname:
        if suite=='romulus_zooms':
            os.environ['TANGOS_SIMULATION_FOLDER'] = '/data/REPOSITORY/romulus_zooms/'
            os.environ['TANGOS_DB_CONNECTION'] = '/data/REPOSITORY/romulus_zooms/rom25_dwarf_zooms.db'
        else:
            os.environ['TANGOS_SIMULATION_FOLDER'] = '/home/ns1917/tangos_sims/'
            os.environ['TANGOS_DB_CONNECTION'] = '/home/ns1917/Databases/Marvel_BN_N10.db'

        os.chdir('/home/ns1917/pynbody/AnnaWright_startrace/')
    else:  # grinnell
        os.environ['TANGOS_SIMULATION_FOLDER'] = '/home/selvani/MAP/Sims/cptmarvel.cosmo25cmb/cptmarvel.cosmo25cmb.4096g5HbwK1BH/'
        os.environ['TANGOS_DB_CONNECTION'] = '/home/selvani/MAP/pynbody/Tangos/Marvel_BN_N10.db'
        os.chdir('/home/selvani/MAP/pynbody/AnnaWright_startrace/')

    # Process single or multiple halos
    if len(args.halo_indices) == 1:
        tqdm.tqdm.write(f"Processing halo: {args.halo_indices[0]}")
        main(args.halo_indices[0], simname)
    else:
        for halo_idx in tqdm.tqdm(args.halo_indices, desc=f"Processing {simname} halos"):
            tqdm.tqdm.write(f"Processing halo: {halo_idx}")
            main(halo_idx, simname)