"""
Particle Tracking Analysis for Ram Pressure Stripping Studies

This module tracks gas particles in cosmological simulations to study ram pressure
stripping of satellite galaxies. It follows gas particles through time to analyze
their properties and environment (satellite, host, IGM, etc.).

Main components:
- Particle identification and tracking across snapshots
- Coordinate transformations relative to satellite and host halos
- Physical property calculations (density, temperature, velocities)
- Environment classification (satellite, host, IGM)
"""

from base import *
import os

# Global Hubble parameter for unit conversions
hubble = 0.6776942783267969  # h parameter from simulation

def handle_exception(exc_type, exc_value, exc_traceback):
    """
    Custom exception handler to log uncaught exceptions.
    
    This ensures that any unexpected errors during particle tracking
    are properly logged for debugging purposes.
    """
    global logger
    # Allow keyboard interrupts to exit normally
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    # Log all other exceptions with full traceback
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

# Set the custom exception handler
sys.excepthook = handle_exception


def get_iords(sim, z0haloid, filepaths, haloids):
    """
    Get particle indices (iords) for gas particles to track across snapshots.
    
    This function identifies all gas particles that have been part of the satellite
    halo at any point during the simulation. These particles are then tracked
    through all snapshots to study their evolution.
    
    Parameters:
    -----------
    sim : str
        Simulation name identifier
    z0haloid : int
        Halo ID of the satellite at z=0 (present day)
    filepaths : list
        List of paths to simulation snapshot files
    haloids : list
        List of halo IDs corresponding to the satellite across snapshots
    
    Returns:
    --------
    iords : numpy.array
        Array of particle indices to track
    """
    # Check if pre-computed iords file exists to save computation time
    path = f'../../Data/iords/{sim}_{z0haloid}.pickle'
    if os.path.exists(path):
        logger.debug(f'Found iords file at {path}, loading these')
        logger.warning(f'If you have recently changed something, these iords might not be correct')
        with open(path,'rb') as infile:
            iords = pickle.load(infile)
    
    else:
        logger.debug(f'Could not find iords file, computing iords to track')
        # Start with empty array and build union of all gas particle IDs
        iords = np.array([])
        
        # Loop through all snapshots to collect gas particles
        for f,haloid in zip(filepaths,haloids):
            # Load simulation snapshot and convert to physical units
            s = pynbody.load(f)
            s.physical_units()
            
            # Load halo catalog (AHF format)
            h = s.halos(halo_numbers='v1')
            halo = h[haloid]  # Get the satellite halo
            
            # Extract gas particle IDs from this snapshot
            iord = np.array(halo.gas['iord'], dtype=int)
            
            # Add to master list (union ensures no duplicates)
            iords = np.union1d(iords, iord)
        
        # Save computed iords for future use
        logger.debug(f'Saving iords file to {path}')
        with open(path,'wb') as outfile:
            pickle.dump(iords,outfile)

    return iords

def run_tracking(sim, z0haloid, filepaths, haloids, h1ids):
    """
    Main tracking function that follows particles through all snapshots.
    
    This function performs the core particle tracking analysis by:
    1. Getting the list of particles to track
    2. Loading each snapshot in sequence
    3. Finding tracked particles in each snapshot
    4. Running analysis on those particles
    5. Concatenating results across all snapshots
    
    Parameters:
    -----------
    sim : str
        Simulation identifier
    z0haloid : int
        Satellite halo ID at z=0
    filepaths : list
        Ordered list of snapshot file paths
    haloids : list
        Satellite halo IDs for each snapshot
    h1ids : list
        Host halo IDs for each snapshot
    
    Returns:
    --------
    output : pandas.DataFrame
        Complete tracking data for all particles and snapshots
    """
    # Get the master list of particle IDs to track
    iords = get_iords(sim, z0haloid, filepaths, haloids)
    
    # Flag to determine tracking method (iords vs bridge)
    use_iords = True
    output = pd.DataFrame()
    logger.debug('Starting tracking')
    
    # Loop through snapshots from high redshift to z=0
    for f, haloid, h1id in zip(filepaths, haloids, h1ids):
        # Load current snapshot
        s = pynbody.load(f)
        s.physical_units()
        
        # Load halo catalog and get satellite/host halos
        h = s.halos(halo_numbers='v1')
        halo = h[haloid]    # Satellite halo
        h1 = h[h1id]        # Host halo
        snapnum = f[-4:]    # Extract snapshot number from filename
        logger.debug(f'* Snapshot {snapnum}')

        if use_iords:
            # First snapshot: find particles by their IDs
            iord = np.array(s.gas['iord'], dtype=float)
            gas_particles = s.gas[np.isin(iord, iords)]
            use_iords = False
        else:
            # Subsequent snapshots: use bridge to track particles across time
            # This accounts for particles that may change type or position
            b = pynbody.bridge.OrderBridge(s_prev, s, allow_family_change=True)
            gas_particles = b(gas_particles_prev)

        # Analyze the tracked particles in this snapshot
        # This calculates all physical properties and environment classification
        snapshot_results = analysis(s, halo, h1, gas_particles, h, haloid, h1id)
        
        # Add results to master dataframe
        output = pd.concat([output, snapshot_results])

        # Store current snapshot data for next iteration's bridge
        gas_particles_prev = gas_particles
        snapnum_prev = snapnum
        s_prev = s
    
    return output

def analysis(s, halo, h1, gas_particles, h, haloid, h1id):
    """
    Comprehensive analysis of tracked gas particles in a single snapshot.
    
    This function calculates a wide range of physical properties and classifies
    the environment of each tracked particle. Properties are calculated in
    multiple coordinate systems (satellite-centered and host-centered).
    
    Parameters:
    -----------
    s : pynbody.snapshot
        Current simulation snapshot
    halo : pynbody.halo
        Satellite halo object
    h1 : pynbody.halo
        Host halo object
    gas_particles : pynbody.array
        Subset of gas particles being tracked
    h : pynbody.halo
        Complete halo catalog
    haloid : int
        Satellite halo ID
    h1id : int
        Host halo ID
    
    Returns:
    --------
    output : pandas.DataFrame
        Analysis results for all tracked particles in this snapshot
    """
    output = pd.DataFrame()
    
    # Scale factor for coordinate transformations
    a = float(s.properties['a'])

    # Verify all tracked particles are still gas particles
    if len(gas_particles) != len(gas_particles.g):
        raise Exception('Some particles are no longer gas particles...')

    # =====================================================================
    # CENTERING-INVARIANT PROPERTIES
    # These properties don't depend on coordinate system choice
    # =====================================================================
    
    # Time information
    output['time'] = np.array([float(s.properties['time'].in_units('Gyr'))] * len(gas_particles))
    
    # Particle identification
    output['pid'] = np.array(gas_particles['iord'], dtype=int)
    
    # Physical properties of gas particles
    # Density: convert from simulation units to physical units (amu/cm³)
    output['rho'] = np.array(gas_particles.g['rho'].in_units('Msol kpc**-3'), dtype=float) * 4.077603812e-8
    
    # Temperature in Kelvin
    output['temp'] = np.array(gas_particles.g['temp'].in_units('K'), dtype=float)
    
    # Particle mass in solar masses
    output['mass'] = np.array(gas_particles.g['mass'].in_units('Msol'), dtype=float)
    
    # Cooling time in Gyr
    output['coolontime'] = np.array(gas_particles.g['coolontime'].in_units('Gyr'), dtype=float)
    
    # =====================================================================
    # SATELLITE-CENTERED PROPERTIES
    # Calculate positions and velocities relative to satellite center
    # =====================================================================
    
    # Center coordinate system on satellite halo
    pynbody.analysis.halo.center(halo)
    
    # Get particle coordinates (now satellite-centered)
    x, y, z = gas_particles['x'], gas_particles['y'], gas_particles['z']
    
    # Satellite virial radius (corrected for cosmology)
    Rvir = halo.properties['Rvir'] * a / hubble
    
    # Distance from satellite center
    output['r'] = np.array(np.sqrt(x**2 + y**2 + z**2), dtype=float)
    output['r_per_Rvir'] = output.r / Rvir  # Normalized by virial radius
    
    # Cartesian coordinates relative to satellite
    output['x'] = x
    output['y'] = y
    output['z'] = z
    output['satRvir'] = np.array([Rvir] * len(x))
    output['a'] = np.array([a] * len(x))

    # Velocity components relative to satellite
    output['vx'] = np.array(gas_particles['vx'].in_units('km s**-1'), dtype=float)
    output['vy'] = np.array(gas_particles['vy'].in_units('km s**-1'), dtype=float)
    output['vz'] = np.array(gas_particles['vz'].in_units('km s**-1'), dtype=float)
    output['v'] = np.array(np.sqrt(output.vx**2 + output.vy**2 + output.vz**2))

    # =====================================================================
    # HOST-CENTERED PROPERTIES
    # Calculate positions and velocities relative to host galaxy center
    # =====================================================================
    
    # Re-center coordinate system on host halo
    pynbody.analysis.halo.center(h1)
    
    # Get particle coordinates (now host-centered)
    x, y, z = gas_particles['x'], gas_particles['y'], gas_particles['z']
    
    # Host virial radius
    Rvir = h1.properties['Rvir'] / hubble * a
    
    # Distance from host center
    output['r_rel_host'] = np.array(np.sqrt(x**2 + y**2 + z**2), dtype=float)
    output['r_rel_host_per_Rvir'] = output.r_rel_host / Rvir
    
    # Cartesian coordinates relative to host
    output['x_rel_host'] = x
    output['y_rel_host'] = y
    output['z_rel_host'] = z
    output['hostRvir'] = np.array([Rvir] * len(x))
    
    # Velocity components relative to host
    output['vx_rel_host'] = np.array(gas_particles['vx'].in_units('km s**-1'), dtype=float)
    output['vy_rel_host'] = np.array(gas_particles['vy'].in_units('km s**-1'), dtype=float)
    output['vz_rel_host'] = np.array(gas_particles['vz'].in_units('km s**-1'), dtype=float)
    output['v_rel_host'] = np.array(np.sqrt(output.vx_rel_host**2 + output.vy_rel_host**2 + output.vz_rel_host**2))

    # =====================================================================
    # HALO PROPERTIES
    # Store properties of satellite and host halos
    # =====================================================================
    
    # Satellite halo center positions (comoving coordinates)
    output['sat_Xc'] = halo.properties['Xc'] / hubble * a
    output['sat_Yc'] = halo.properties['Yc'] / hubble * a
    output['sat_Zc'] = halo.properties['Zc'] / hubble * a
    
    # Satellite halo velocities
    output['sat_vx'] = halo.properties['VXc']
    output['sat_vy'] = halo.properties['VYc']
    output['sat_vz'] = halo.properties['VZc']

    # Host halo center positions
    output['host_Xc'] = h1.properties['Xc'] / hubble * a
    output['host_Yc'] = h1.properties['Yc'] / hubble * a
    output['host_Zc'] = h1.properties['Zc'] / hubble * a

    # Host halo velocities
    output['host_vx'] = h1.properties['VXc']
    output['host_vy'] = h1.properties['VYc']
    output['host_vz'] = h1.properties['VZc']
    
    # Halo masses (stellar and gas components)
    output['sat_Mstar'] = halo.properties['M_star']
    output['sat_Mgas'] = halo.properties['M_gas']
    
    output['host_Mstar'] = h1.properties['M_star']
    output['host_Mgas'] = h1.properties['M_gas']
    

    # =====================================================================
    # SATELLITE CHARACTERISTIC RADII
    # Calculate half-mass radius and gas disk radius for satellite
    # =====================================================================
    
    try:
        # Align satellite to face-on orientation for profile calculation
        pynbody.analysis.angmom.faceon(halo)
        Rvir = halo.properties['Rvir'] / hubble * a 
        do_sat_radius = True
    except:
        # If alignment fails, set radii to NaN
        r_half = np.nan
        r_gas = np.nan
        do_sat_radius = False
        
    if do_sat_radius:
        # Calculate gas disk radius using Kennicutt star formation threshold
        try:
            bins = np.power(10, np.linspace(-1, np.log10(Rvir), 100))
            p_gas = pynbody.analysis.profile.Profile(halo.g, bins=bins)
            x, y = p_gas['rbins'], p_gas['density']
            sigma_th = 9e6  # Minimum surface density for SF (Kennicutt criterion)
            r_gas = np.average([np.max(x[y > sigma_th]), np.min(x[y < sigma_th])])
        except:
            r_gas = np.nan
        
        # Calculate stellar half-mass radius
        try:
            bins = np.power(10, np.linspace(-1, np.log10(0.2*Rvir), 500))
            p_stars = pynbody.analysis.profile.Profile(halo.s, bins=bins)
            x, y = p_stars['rbins'], p_stars['mass_enc']/np.sum(halo.s['mass'].in_units('Msol'))
            r_half = np.average([np.max(x[y < 0.5]), np.min(x[y > 0.5])])
        except:
            r_half = np.nan

    output['sat_r_half'] = r_half
    output['sat_r_gas'] = r_gas 

    # =====================================================================
    # HOST CHARACTERISTIC RADII
    # Calculate half-mass radius and gas disk radius for host
    # =====================================================================
    
    try:
        # Align host to face-on orientation
        pynbody.analysis.angmom.faceon(h1)
        Rvir = h1.properties['Rvir'] / hubble * a 
        do_host_radius = True
    except:
        r_half = np.nan
        r_gas = np.nan
        do_host_radius = False
        
    if do_host_radius:
        # Host gas disk radius
        try:
            bins = np.power(10, np.linspace(-1, np.log10(Rvir), 100))
            p_gas = pynbody.analysis.profile.Profile(h1.g, bins=bins)
            x, y = p_gas['rbins'], p_gas['density']
            sigma_th = 9e6  # Kennicutt threshold
            r_gas = np.average([np.max(x[y > sigma_th]), np.min(x[y < sigma_th])])
        except:
            r_gas = np.nan
        
        # Host stellar half-mass radius
        try:
            bins = np.power(10, np.linspace(-1, np.log10(0.2*Rvir), 500))
            p_stars = pynbody.analysis.profile.Profile(h1.s, bins=bins)
            x, y = p_stars['rbins'], p_stars['mass_enc']/np.sum(h1.s['mass'].in_units('Msol'))
            r_half = np.average([np.max(x[y < 0.5]), np.min(x[y > 0.5])])
        except:
            r_half = np.nan

    output['host_r_half'] = r_half
    output['host_r_gas'] = r_gas
    
    # =====================================================================
    # ENVIRONMENT CLASSIFICATION
    # Classify each particle's current environment
    # =====================================================================
    
    # Particle is in satellite if AHF identifies it as part of satellite halo
    in_sat = np.isin(output.pid, halo.g['iord'])
    
    # Particle is in host if AHF identifies it as part of host halo
    in_host = np.isin(output.pid, h1.g['iord'])
    
    # Collect particle IDs from all other satellite halos
    iords_other = np.array([])
    for i in range(len(h)):
        i += 1  # Halo numbering starts at 1
        # Skip the main satellite and host halos
        if (i != haloid) and (i != h1id):
            halo_other = h[i]
            # Only consider halos that are satellites (have a host)
            if halo_other.properties['hostHalo'] != -1:
                iords_other = np.append(iords_other, halo_other.g['iord'])
    
    # Classify particle environments
    in_other_sat = np.isin(output.pid, iords_other)  # In other satellite halos
    in_IGM = ~in_sat & ~in_host & ~in_other_sat      # In intergalactic medium
    
    # Store environment flags
    output['in_sat'] = in_sat
    output['in_host'] = in_host
    output['in_other_sat'] = in_other_sat
    output['in_IGM'] = in_IGM

    return output


if __name__ == '__main__':
    # =====================================================================
    # MAIN EXECUTION
    # Parse command line arguments and run the tracking analysis
    # =====================================================================
    
    # Parse command line arguments
    sim = str(sys.argv[1])          # Simulation identifier
    z0haloid = int(sys.argv[2])     # Satellite halo ID at z=0

    # Set up logging system
    if not os.path.exists('./logs/'):
        os.mkdir('./logs/')
    logging.basicConfig(filename=f'./logs/{sim}_{z0haloid}.log', 
                        format='%(asctime)s :: %(name)s :: %(levelname)-8s :: %(message)s', 
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.DEBUG)
    logger = logging.getLogger('PartTracker')

    logger.debug(f'--------------------------------------------------------------')
    logger.debug(f'Beginning particle tracking for {sim}-{z0haloid}')

    # Get pre-computed file paths and halo IDs across all snapshots
    logger.debug('Getting stored filepaths and haloids')
    filepaths, haloids, h1ids = get_stored_filepaths_haloids(sim, z0haloid)
    # Note: filepaths starts with z=0 and goes to high redshift

    # Determine which snapshot to start tracking from
    logger.debug('Getting starting snapshot (may take a while)')
    snap_start = get_snap_start(sim, z0haloid)
    logger.debug(f'Start on snapshot {snap_start}, {filepaths[snap_start][-4:]}')
    
    # Handle edge case where satellite doesn't exist in early snapshots
    if len(haloids) < snap_start:
        snap_start = len(haloids)
        raise Exception('Careful! You may have an error since the satellite doesnt have mergertree info out to the time where you want to start. This case is untested')
    
    # Trim arrays to only include snapshots from start_snap to z=0
    if len(haloids) > snap_start:
        filepaths = np.flip(filepaths[:snap_start+1])
        haloids = np.flip(haloids[:snap_start+1])
        h1ids = np.flip(h1ids[:snap_start+1])

    if len(haloids) == snap_start:
        filepaths = np.flip(filepaths[:snap_start])
        haloids = np.flip(haloids[:snap_start])
        h1ids = np.flip(h1ids[:snap_start])   
        
    # Now arrays go from high redshift (start_snap) to z=0
    assert len(filepaths) == len(haloids)
    assert len(haloids) == len(h1ids)

    # Run the complete particle tracking analysis
    output = run_tracking(sim, z0haloid, filepaths, haloids, h1ids)

    # Save results to HDF5 format (efficient for large datasets)
    savepath = '../../Data/tracked_particles.hdf5'
    logger.debug(f'Saving output to {savepath}')
    output.to_hdf(savepath, key=f'{sim}_{z0haloid}')

