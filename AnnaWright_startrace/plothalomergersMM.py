"""
plothalomergersMM.py

Plot stellar halo merger histories from Romulus zoom simulations.

This script generates time-series visualizations of stellar halo mergers by
tracking star particles from different progenitor halos across simulation
snapshots. It creates three 2D projection plots (XY, YZ, XZ) colored by
original host halo ID, showing how satellite galaxies merge into the main
halo over cosmic time.

The script uses the Tangos database to track halo properties and pynbody
for particle data analysis. It produces multiple zoom levels to capture
both large-scale and detailed structure.

Author: Nithun Selva
Date: 2025
Usage:
    # Plot merger history for simulation r442
    python plothalomergersMM.py --name r442
    
    # Plot with overwrite enabled for r329
    python plothalomergersMM.py -n r329 --overwrite
    
    # Show help
    python plothalomergersMM.py --help
"""

import os
import sys
import socket
import argparse
# Determine hostname to configure paths appropriately
hostname = socket.gethostname()
if 'emu' in hostname:
    os.environ['TANGOS_SIMULATION_FOLDER'] = '/home/ns1917/Sims/romulus_zooms/'
    os.environ['TANGOS_DB_CONNECTION'] = '/home/ns1917/Sims/romulus_zooms/rom25_dwarf_zooms.db'
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
import tangos as db
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import glob
from pynbody.array import SimArray
import pandas as pd
import tqdm.auto as tqdm
from createanimation import create_animation_with_padding, create_animation_simple

def plot_halo_mergers(sp,mask,haloids,color_map,timestep,rad=None,savepath=None):
    """Plots stellar positions from a simulation in three 2D projections.

    This function generates a 1x3 matplotlib figure showing XY, YZ, and XZ
    scatter plots of star particle positions. Particles are colored based on
    their unique host halo ID. The function includes an optional legend and
    can save the figure to a file.

    Args:
        sp (pynbody.snapshot.SimSnap): 
            The pynbody snapshot or sub-snapshot containing the particle data.
        mask (array): 
            A boolean mask or array of indices used to select the desired 
            star particles from `sp`.
        haloids (array): 
            A list of original host halo IDs, corresponding to the particles 
            selected by `mask`.
        color_map (dict): 
            A dictionary mapping each unique halo ID from `haloids` to a 
            matplotlib color.
        timestep (str or int): 
            The identifier for the current timestep, used in the main figure title.
        rad (float, optional): 
            The radius for the plot limits. If provided, the x and y axes for 
            each projection are set to `[-rad, rad]`. Defaults to None.
        savepath (str, optional): 
            The file path to save the generated plot. If None, the plot is 
            only displayed. Defaults to None.
            
    Returns:
        None
    """
    # Create an array of colors for each star particle
    particle_colors = [color_map[hid] for hid in haloids]
    # Create the scatter plot
    fig, axes = plt.subplots(1, 3, figsize=(54, 18))
    # Define projections
    projections = [
        {'ax': axes[0], 'dims': [0, 1], 'labels': ['x (kpc)', 'y (kpc)'], 'title': 'X-Y Projection'},
        {'ax': axes[1], 'dims': [1, 2], 'labels': ['y (kpc)', 'z (kpc)'], 'title': 'Y-Z Projection'},
        {'ax': axes[2], 'dims': [0, 2], 'labels': ['x (kpc)', 'z (kpc)'], 'title': 'X-Z Projection'}
    ]

    pos_data = sp.s['pos'][mask]

    for proj in projections:
        ax = proj['ax']
        dim1 = proj['dims'][0]
        dim2 = proj['dims'][1]

        ax.scatter(
            pos_data[:, dim1], 
            pos_data[:, dim2],
            c=particle_colors,
            s=25.0,  # Marker size
            alpha=0.25, # Use transparency to see overlapping structures
            edgecolors='none' # Remove marker edges for a cleaner look
        )

        ax.set_title(proj['title'], fontsize=14)
        ax.set_xlabel(proj['labels'][0], fontsize=12)
        ax.set_ylabel(proj['labels'][1], fontsize=12)

        if rad:
            ax.set_xlim(-rad, rad)
            ax.set_ylim(-rad, rad)

    # Display the legend
    legend_patches = [mpatches.Patch(color=color_map[hid], label=hid) for hid in np.unique(haloids)]
    if len(legend_patches) <= 40:
        axes[0].legend(handles=legend_patches, title="Original Halo ID", loc='upper right', fontsize=8)
    else:
        tqdm.tqdm.write(f"Warning: {len(legend_patches)} unique halos found. Legend will not be displayed to avoid clutter.")

    fig.suptitle(f"Snapshot {timestep}", fontsize=18, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    if savepath:
        tqdm.tqdm.write(f"Saving plot to {savepath}")
        plt.savefig(savepath, bbox_inches='tight')
    # plt.show()

def make_colormap(unique_haloids):
    """Creates a color map for the unique halo IDs.

    Args:
        unique_haloids (array): A list of unique halo IDs.

    Returns:
        dict: A dictionary mapping halo IDs to their corresponding colors.
    """
    if len(unique_haloids) <= 10:
        colors = plt.cm.get_cmap('Accent', len(unique_haloids))
    else:
        # For more than 10 halos, 'viridis' or 'plasma' can work, though colors may be less distinct
        colors = plt.cm.get_cmap('viridis', len(unique_haloids))

    color_map = {hid: colors(i) for i, hid in enumerate(unique_haloids)}
    return color_map

def get_halo(snapshot, halo_number, ss_dir):
    """Retrieve a specific halo from a snapshot using Tangos database.
    
    Args:
        snapshot (str): Snapshot identifier (e.g., '004096')
        halo_number (int): Halo number within the snapshot
        ss_dir (str): Simulation directory name
    
    Returns:
        tangos.core.halo.Halo: The requested halo object
    """
    ts = db.get_timestep(f"{ss_dir}/%{snapshot}")
    # print(f"Retrieved timestep: {ts}")
    return ts.halos.filter_by(halo_number=int(halo_number)).first()

def setup_paths(simname):
    """Configure simulation paths based on hostname and simulation name.
    
    Args:
        simname (str): Name of Romulus zoom simulation (e.g., 'r442', 'r329')
    
    Returns:
        tuple: (simpath, outfile_dir, annafile_dir, basename, ss_dir, sim_base, ss_z0)
    """
    #! Configure paths as needed
    if 'emu' in hostname:
        simpath = '/home/ns1917/Sims/romulus_zooms/'
        outfile_dir = "/home/ns1917/pynbody/stellarhalo_trace_aw/"
        annafile_dir = '/home/ns1917/pynbody/stellarhalo_trace_aw/updated_ids'
    else:
        simpath = '/home/selvani/MAP/Sims/cptmarvel.cosmo25cmb/cptmarvel.cosmo25cmb.4096g5HbwK1BH/'
        outfile_dir = "/home/selvani/MAP/pynbody/stellarhalo_trace_aw/"
        annafile_dir = '/home/selvani/MAP/pynbody/awright_mm'

    # Construct simulation-specific paths for Romulus zoom
    basename = f'{simname}.romulus25.3072g1HsbBH'
    ss_dir = basename
    sim_base = simpath + ss_dir + '/'
    ss_z0 = sim_base + basename + '.004096'
    
    return simpath, outfile_dir, annafile_dir, basename, ss_dir, sim_base, ss_z0

def load_halo_data(annafile_dir, simname):
    """Load star particle data from Anna's pipeline.
    
    Args:
        annafile_dir: Directory containing the HDF5 file
        simname: Base name of the simulation
    
    Returns:
        dict: Dictionary mapping particle IDs to their unique host IDs
        list: List of unique host IDs
        list: List of particle IDs
    """
    #! Configure paths 
    with h5py.File(annafile_dir+'/'+simname+'/'+simname+'_allhalostardata_consolidated2.h5','r') as f:
        hostids = f['host_IDs'].asstr()[:]  # unique host IDs
        partids = f['particle_IDs'][:]  # iords
        pct = f['particle_creation_times'][:]  # formation times
        ph = f['particle_hosts'][:]  # local host IDs (i.e., host at formation time)
        pp = f['particle_positions'][:]  # position at formation time
        tsloc = f['timestep_location'][:]  # snapshot where star particle first appears
        
    halo_particle_dict = {}  # map iords to their unique host IDs
    for i, part in enumerate(partids):
        halo_particle_dict[part] = hostids[i]
    
    return halo_particle_dict, hostids, partids

def initialize_simulation_data(simname):
    """Initialize simulation paths and load halo particle data.
    
    Args:
        simname (str): Name of Romulus zoom simulation (e.g., 'r442', 'r329')
    
    Returns:
        tuple: (simpath, outfile_dir, annafile_dir, basename, ss_dir, sim_base, ss_z0,
                halo_particle_dict, all_timesteps, halos_stars_dict, uIDs)
    """
    # Setup paths
    (simpath, outfile_dir, annafile_dir, basename, ss_dir, 
     sim_base, ss_z0) = setup_paths(simname)

    # Load z=0 snapshot to get halo information
    timestep = db.get_timestep(ss_dir+'/%'+str(4096))
    all_halos = timestep.halos.all()
    tqdm.tqdm.write("There are %d halos in the snapshot." % len(all_halos))

    # Filter for halos with stars and sufficient mass
    halos_with_stars = [h for h in all_halos if h.NStar > 0 and h['Mvir'] > 1e9]
    # for halo in halos_with_stars:
        # tqdm.tqdm.write("Halo ID: %s, Mass: %1.2eM☉, Stars: %d" % (halo.halo_number, halo['Mvir'], halo.NStar))

    # Read in data from Anna's pipeline - star formation history
    halo_particle_dict, hostids, partids = load_halo_data(annafile_dir, simname)

    # Build dictionary of halos with stars at each timestep
    all_timesteps = db.get_simulation(ss_dir).timesteps
    halos_stars_dict = {}
    for timestep in all_timesteps:
        # Get all halos in the current timestep with stars
        # Kinda overkill, since we only need ones at 4096 for now
        # tqdm.tqdm.write(f'Processing timestep: {timestep.extension[-6:]}')
        halos_stars_dict[timestep.extension[-6:]] = [h for h in timestep.halos.all() if h.NStar > 0]
    tqdm.tqdm.write('Created halos dictionary')

    # Test for existence of shrink_center property (needed for centering)
    all_timesteps = db.get_simulation(ss_dir).timesteps
    for timestep in all_timesteps:
        halo2 = timestep.halos.all()[1]
        try:
            halo2['shrink_center']
        except:
            tqdm.tqdm.write(timestep.extension[-6:])  # Log timesteps without shrink_center
    
    return (simpath, outfile_dir, annafile_dir, basename, ss_dir, sim_base, ss_z0,
            halo_particle_dict, all_timesteps, halos_stars_dict, hostids, partids)

def main(simname, overwrite=False, create_animation=False):
    """Generate merger history plots for a Romulus zoom simulation.
    
    Args:
        simname (str): Name of Romulus zoom simulation (e.g., 'r442', 'r329')
        overwrite (bool): Whether to overwrite existing plot files (default: False)
    """
    # Initialize simulation data
    (simpath, outfile_dir, annafile_dir, basename, ss_dir, sim_base, ss_z0,
     halo_particle_dict, all_timesteps, halos_stars_dict, hostids, partids) = initialize_simulation_data(simname)
    
    uIDs = np.unique(hostids)

    # Get the main halo (first halo with stars at z=0)
    halo = halos_stars_dict[all_timesteps[-1].extension[-6:]][0]
    tqdm.tqdm.write(f"Starting processing for simulation: {simname}, halo: {halo.halo_number} with {halo.NStar} star particles")

    # Build merger tree: map each snapshot to the main halo's progenitor in that snapshot
    halo_numbers, dbids = halo.calculate_for_progenitors("halo_number()", "dbid()")
    snapshots = [db.get_halo(dbid).timestep.extension[-6:] for dbid in dbids]
    halo_snapshots_dict = {snapshot: dbid for snapshot, dbid in zip(snapshots, dbids)}

    # Configure output directories for plots at different zoom levels
    save_bases = [os.path.join(outfile_dir, 'merge_plots', ss_dir+str(halo.halo_number)),
                  os.path.join(outfile_dir, 'merge_plots', ss_dir+str(halo.halo_number)+'zoom' ),
                  os.path.join(outfile_dir, 'merge_plots', ss_dir+str(halo.halo_number)+'morezoom')]
    tqdm.tqdm.write(f"Saving plots to {save_bases[0]}")
    
    # Create output directories if they don't exist
    for save_base in save_bases:
        if not os.path.exists(save_base):
            os.makedirs(save_base)
    filename_base = ss_dir + '_' + str(halo.halo_number) + '_'

    # Loop through all timesteps for this halo (in reverse chronological order)
    pbar = tqdm.tqdm(total=len(all_timesteps), desc=f"Processing {simname} snapshots")
    for i, timestep in enumerate(all_timesteps[::-1]):
        timestep = timestep.extension[-6:]

        # Get simulation and halo
        s = pynbody.load(sim_base + basename + '.' + timestep +'/'+ basename + '.' + timestep)
        s.physical_units()
        
        # Get the main halo's progenitor at this snapshot
        halo = db.get_halo(halo_snapshots_dict[timestep])
        tqdm.tqdm.write(f"Now on {timestep}")

        # Define output paths for the three zoom levels
        save_path = os.path.join(save_bases[0], filename_base + timestep + '.png')
        save_path2 = os.path.join(save_bases[1], filename_base + timestep + '.png')
        save_path3 = os.path.join(save_bases[2], filename_base + timestep + '.png')

        # First timestep
        if i == 0:
            # Reproduce unique color map seed for consistent colors across runs
            rng = np.random.default_rng(halo.halo_number*2) 
            uIDs_shuffled = rng.permutation(uIDs)
            tqdm.tqdm.write(f'Seed {halo.halo_number*2}: {str(uIDs_shuffled)}')
            colormap = make_colormap(uIDs_shuffled)

            # Set universal scaling radius (z=0 value) for consistent plot scale
            try:
                rad = halo['Rvir']  # Virial radius
            except:
                rad = halo['max_radius']  # Fallback to max radius

        # Skip if files exist and overwrite is False
        if not overwrite and (os.path.isfile(save_path) and os.path.isfile(save_path2) and os.path.isfile(save_path3)):
            tqdm.tqdm.write(f"File {save_path} already exists, skipping...")
            pbar.update(1)
            continue

        # Center on halo
        try:
            # Use existing value from tangos
            roughcen = halo['shrink_center']
        except:
            try:
                # Try calculating from pynbody if not available in database
                tqdm.tqdm.write(f'No shrink center on {timestep}')
                h = s.halos(halo_numbers='v1')
                roughcen = pynbody.analysis.halo.shrink_sphere_center(h[halo.halo_number])
                tqdm.tqdm.write(f'Calculated shrink center: {roughcen}')
            except:
                # Skip this snapshot if centering fails
                tqdm.tqdm.write('Failed again, skipping...')
                pbar.update(1)
                continue

        # Radius for adaptive zoom plots
        try:
            currad = halo['Rvir']
        except:
            currad = halo['max_radius']

        # Filter snapshot by only particles within the reference radius, then center on halo
        try:
            sp = s[pynbody.filt.Sphere(SimArray([rad], "kpc"), roughcen)].load_copy()
            sp.physical_units()
            sp['pos'] -= roughcen  # Shift coordinates so halo is at origin
            tqdm.tqdm.write(f"Num stars: {len(sp.s)}", end=', ')  # Total num of stars in sphere

            # Filter stars: exclude wind particles (negative formation times)
            # mask = np.where(sp.s['amiga.grp'] == halo.halo_number)[0]  # Alternative: filter by halo membership
            # mask = np.where(sp.s['amiga.grp'] == sp.s['amiga.grp'])[0]  # Alternative: dummy mask (all particles)
            mask = sp.s['tform'] > 0  # Exclude wind particles: FROM ANNA'S CODE
            
            # Map each star to its original host halo ID (where it formed)
            haloids = np.array([halo_particle_dict[part] for part in sp.s['iord'][mask]])
            tqdm.tqdm.write(f"Masked num stars: {len(sp.s[mask])}")

            # Generate plots at three zoom levels
            plot_halo_mergers(sp, mask, haloids, colormap, timestep, rad/np.sqrt(2), save_path)  # Standard: scaled to z=0 Rvir
            plot_halo_mergers(sp, mask, haloids, colormap, timestep, currad, save_path2)  # Zoom: current snapshot Rvir
            plot_halo_mergers(sp, mask, haloids, colormap, timestep, currad/(np.sqrt(2)*4), save_path3)  # More zoomed: 4x closer
            pbar.update(1)
        except Exception as e:
            tqdm.tqdm.write(f"\nERROR processing timestep {timestep}: {str(e)}")
            tqdm.tqdm.write(f"Terminating processing loop.")
            pbar.close()
            break

    if create_animation:
        zoom_suffixes = ['', '_zoom', '_morezoom']
        for folder_path, suffix in tqdm.tqdm(zip(save_bases, zoom_suffixes), total=3):
            base_name = os.path.basename(folder_path)
            # Extract just the first part (e.g., 'r442' from 'r442.romulus25.3072g1HsbBH1')
            base_id = base_name.split('.')[0]
            
            # output_gif_path = os.path.join(outfile_dir, 'merge_plots', f'{base_id}{suffix}_animation.gif')
            output_mp4_path = os.path.join(outfile_dir, 'merge_plots', f'{base_id}{suffix}_animation.mp4')
            # create_animation_simple(folder_path, output_gif_path, fps=12)
            create_animation_with_padding(folder_path, output_mp4_path, fps=6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate stellar halo merger history plots from Romulus zoom simulations.",
        epilog="""
Examples:
  # Plot merger history for r442 simulation
  python plothalomergersMM.py --name r442
  
  # Plot r329 simulation with overwrite enabled
  python plothalomergersMM.py -n r329 --overwrite
  
  # Plot r377 simulation (short form)
  python plothalomergersMM.py -n r377
  
  # Show help
  python plothalomergersMM.py --help

Supported Simulations:
  Romulus zoom simulations (e.g., r442, r329, r377, etc.)
  These are Milky Way-like galaxies from the Romulus simulation suite.

Output:
  Creates three versions of each plot with different zoom levels:
  - Standard: scaled to z=0 virial radius
  - Zoom: scaled to current snapshot virial radius  
  - More zoom: 4x closer view of current virial radius
  
  Plots are saved to: {outfile_dir}/merge_plots/{simname}.romulus25.3072g1HsbBH{halo_number}/

Note:
  This script automatically processes the main halo (halo 0) with stars.
  Unlike plothalomergers.py, it doesn't accept halo numbers as arguments.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'simnames',
        nargs='+',
        help="One or more Romulus zoom simulation names (e.g., r442, r329, r377)"
    )
    parser.add_argument(
        '-o', '--overwrite',
        action='store_true',
        help="Overwrite existing plot files (default: False, skip existing files)"
    )
    parser.add_argument(
        '-a', '--create-animation',
        action='store_true',
        help="Create animations from the generated plots (default: False)"
    )
    args = parser.parse_args()
    
    overwrite = args.overwrite
    create_animation = args.create_animation
    
    # Process single or multiple simulations
    if len(args.simnames) == 1:
        tqdm.tqdm.write(f"Processing simulation: {args.simnames[0]}")
        main(args.simnames[0], overwrite, create_animation)
    else:
        for simname in args.simnames:
            tqdm.tqdm.write(f"Processing simulation: {simname}")
            main(simname, overwrite, create_animation)