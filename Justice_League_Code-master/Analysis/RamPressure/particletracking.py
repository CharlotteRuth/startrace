from base import *
import os

# function to log uncaught exceptions
def handle_exception(exc_type, exc_value, exc_traceback):
    global logger
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception

hubble =  0.6776942783267969

def get_iords(sim, z0haloid, filepaths, haloids):
    # '''Get the particle indices (iords) for all gas particles that have been in the halo since snap_start.''''
    path = f'../../Data/iords/{sim}_{z0haloid}.pickle'
    if os.path.exists(path):
        logger.debug(f'Found iords file at {path}, loading these')
        logger.warning(f'If you have recently changed something, these iords might not be correct')
        with open(path,'rb') as infile:
            iords = pickle.load(infile)
    
    else:
        logger.debug(f'Could not find iords file, computing iords to track')
        iords = np.array([])
        for f,haloid in zip(filepaths,haloids):
            s = pynbody.load(f)
            s.physical_units()
            h = s.halos()
            halo = h[haloid]
            iord = np.array(halo.gas['iord'], dtype=int)
            iords = np.union1d(iords, iord)
        
        logger.debug(f'Saving iords file to {path}')
        with open(path,'wb') as outfile:
            pickle.dump(iords,outfile)

    return iords

def run_tracking(sim, z0haloid, filepaths,haloids,h1ids):
    # now we need to start tracking, so we need to get the iords
    iords = get_iords(sim, z0haloid, filepaths, haloids)
    
    use_iords = True
    output = pd.DataFrame()
    logger.debug('Starting tracking')
    for f,haloid,h1id in zip(filepaths,haloids,h1ids):
        s = pynbody.load(f)
        s.physical_units()
        h = s.halos()
        halo = h[haloid]
        h1 = h[h1id]
        snapnum = f[-4:]
        logger.debug(f'* Snapshot {snapnum}')

        if use_iords:
            iord = np.array(s.gas['iord'],dtype=float)
            gas_particles = s.gas[np.isin(iord,iords)]
            use_iords = False
        else:
            b = pynbody.bridge.OrderBridge(s_prev,s,allow_family_change=True)
            gas_particles = b(gas_particles_prev)

        # run analysis on gas particles!
        # this calls the analysis function i've defined, and concatenates the output from this snapshot to the output overall
        output = pd.concat([output, analysis(s,halo,h1,gas_particles,h,haloid,h1id)])

        gas_particles_prev = gas_particles
        snapnum_prev = snapnum
        s_prev = s
    
    return output


def analysis(s,halo,h1,gas_particles,h,haloid,h1id):
    output = pd.DataFrame()
    a = float(s.properties['a'])

    if len(gas_particles) != len(gas_particles.g):
        raise Exception('Some particles are no longer gas particles...')

    # calculate properties that are invariant to centering
    output['time'] = np.array([float(s.properties['time'].in_units('Gyr'))]*len(gas_particles))
    output['pid'] = np.array(gas_particles['iord'],dtype=int)
    output['rho'] = np.array(gas_particles.g['rho'].in_units('Msol kpc**-3'), dtype=float) * 4.077603812e-8 # multiply to convert to amu/cm^3
    output['temp'] = np.array(gas_particles.g['temp'].in_units('K'), dtype=float)
    output['mass'] = np.array(gas_particles.g['mass'].in_units('Msol'), dtype=float)
    output['coolontime'] = np.array(gas_particles.g['coolontime'].in_units('Gyr'),dtype=float)
    
    # calculate properties centered on the satellite
    pynbody.analysis.halo.center(halo)
    x,y,z = gas_particles['x'],gas_particles['y'],gas_particles['z']
    Rvir = halo.properties['Rvir'] * a / hubble
    output['r'] = np.array(np.sqrt(x**2 + y**2 + z**2), dtype=float)
    output['r_per_Rvir'] = output.r / Rvir
    output['x'] = x
    output['y'] = y
    output['z'] = z
    output['satRvir'] = np.array([Rvir]*len(x))
    output['a'] = np.array([a]*len(x))

    output['vx'] = np.array(gas_particles['vx'].in_units('km s**-1'),dtype=float)
    output['vy'] = np.array(gas_particles['vy'].in_units('km s**-1'),dtype=float)
    output['vz'] = np.array(gas_particles['vz'].in_units('km s**-1'),dtype=float)
    output['v'] = np.array(np.sqrt(output.vx**2 + output.vy**2 + output.vz**2))

    # calculate properties centered on the host
    pynbody.analysis.halo.center(h1)
    x,y,z = gas_particles['x'],gas_particles['y'],gas_particles['z']
    Rvir = h1.properties['Rvir'] / hubble * a
    output['r_rel_host'] = np.array(np.sqrt(x**2 + y**2 + z**2), dtype=float)
    output['r_rel_host_per_Rvir'] = output.r_rel_host / Rvir
    output['x_rel_host'] = x
    output['y_rel_host'] = y
    output['z_rel_host'] = z
    output['hostRvir'] = np.array([Rvir]*len(x))
    
    output['vx_rel_host'] = np.array(gas_particles['vx'].in_units('km s**-1'),dtype=float)
    output['vy_rel_host'] = np.array(gas_particles['vy'].in_units('km s**-1'),dtype=float)
    output['vz_rel_host'] = np.array(gas_particles['vz'].in_units('km s**-1'),dtype=float)
    output['v_rel_host'] = np.array(np.sqrt(output.vx_rel_host**2 + output.vy_rel_host**2 + output.vz_rel_host**2))

    # positions and velocities of halos
    output['sat_Xc'] = halo.properties['Xc'] / hubble * a
    output['sat_Yc'] = halo.properties['Yc'] / hubble * a
    output['sat_Zc'] = halo.properties['Zc'] / hubble * a
    
    output['sat_vx'] = halo.properties['VXc']
    output['sat_vy'] = halo.properties['VYc']
    output['sat_vz'] = halo.properties['VZc']

    output['host_Xc'] = h1.properties['Xc'] / hubble * a
    output['host_Yc'] = h1.properties['Yc'] / hubble * a
    output['host_Zc'] = h1.properties['Zc'] / hubble * a

    output['host_vx'] = h1.properties['VXc']
    output['host_vy'] = h1.properties['VYc']
    output['host_vz'] = h1.properties['VZc']
    
    
    # masses of halos
    output['sat_Mstar'] = halo.properties['M_star']
    output['sat_Mgas'] = halo.properties['M_gas']
    
    output['host_Mstar'] = h1.properties['M_star']
    output['host_Mgas'] = h1.properties['M_gas']
    

    # defining r_g of satellite
    try:
        pynbody.analysis.angmom.faceon(halo)
        Rvir = halo.properties['Rvir'] / hubble * a 
        do_sat_radius = True
    except:
        r_half = np.nan
        r_gas = np.nan
        do_sat_radius = False
        
    if do_sat_radius:
        try:
            bins = np.power(10, np.linspace(-1, np.log10(Rvir), 100))
            p_gas = pynbody.analysis.profile.Profile(halo.g, bins=bins)
            x, y = p_gas['rbins'], p_gas['density']
            sigma_th = 9e6 # minimum surface density for SF according to Kennicutt
            r_gas = np.average([np.max(x[y > sigma_th]), np.min(x[y < sigma_th])])
        except:
            r_gas = np.nan
        
        try:
            bins = np.power(10, np.linspace(-1, np.log10(0.2*Rvir), 500))
            p_stars = pynbody.analysis.profile.Profile(halo.s, bins=bins)
            x, y = p_stars['rbins'], p_stars['mass_enc']/np.sum(halo.s['mass'].in_units('Msol'))
            r_half = np.average([np.max(x[y < 0.5]), np.min(x[y > 0.5])])
        except:
            r_half = np.nan

    output['sat_r_half'] = r_half
    output['sat_r_gas'] = r_gas 

    # defining r_g of host
    try:
        pynbody.analysis.angmom.faceon(h1)
        Rvir = h1.properties['Rvir'] / hubble * a 
        do_host_radius = True
    except:
        r_half = np.nan
        r_gas = np.nan
        do_host_radius = False
        
    if do_host_radius:
        try:
            bins = np.power(10, np.linspace(-1, np.log10(Rvir), 100))
            p_gas = pynbody.analysis.profile.Profile(h1.g, bins=bins)
            x, y = p_gas['rbins'], p_gas['density']
            sigma_th = 9e6 # minimum surface density for SF according to Kennicutt
            r_gas = np.average([np.max(x[y > sigma_th]), np.min(x[y < sigma_th])])
        except:
            r_gas = np.nan
        
        try:
            bins = np.power(10, np.linspace(-1, np.log10(0.2*Rvir), 500))
            p_stars = pynbody.analysis.profile.Profile(h1.s, bins=bins)
            x, y = p_stars['rbins'], p_stars['mass_enc']/np.sum(h1.s['mass'].in_units('Msol'))
            r_half = np.average([np.max(x[y < 0.5]), np.min(x[y > 0.5])])
        except:
            r_half = np.nan

    output['host_r_half'] = r_half
    output['host_r_gas'] = r_gas
    
    # we say the particle is in the satellite if its particle ID is one of those AHF identifies as part of the halo
    in_sat = np.isin(output.pid, halo.g['iord'])
    in_host = np.isin(output.pid, h1.g['iord'])
    
    iords_other = np.array([])
    for i in range(len(h)):
        i += 1
        if (i != haloid) and (i != h1id):
            halo_other = h[i]
            if halo_other.properties['hostHalo'] != -1:
                iords_other = np.append(iords_other, halo_other.g['iord'])
    
    in_other_sat = np.isin(output.pid, iords_other)
    in_IGM = ~in_sat & ~in_host & ~in_other_sat
    
    output['in_sat'] = in_sat
    output['in_host'] = in_host
    output['in_other_sat'] = in_other_sat
    output['in_IGM'] = in_IGM

    return output




if __name__ == '__main__':
    sim = str(sys.argv[1])
    z0haloid = int(sys.argv[2])

    if not os.path.exists('./logs/'):
        os.mkdir('./logs/')
    logging.basicConfig(filename=f'./logs/{sim}_{z0haloid}.log', 
                        format='%(asctime)s :: %(name)s :: %(levelname)-8s :: %(message)s', 
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.DEBUG)
    logger = logging.getLogger('PartTracker')

    logger.debug(f'--------------------------------------------------------------')
    logger.debug(f'Beginning particle tracking for {sim}-{z0haloid}')
    # in order: debug, info, warning, error

    logger.debug('Getting stored filepaths and haloids')
    filepaths, haloids, h1ids = get_stored_filepaths_haloids(sim,z0haloid)
    # filepaths starts with z=0 and goes to z=15 or so

    logger.debug('Getting starting snapshot (may take a while)')
    snap_start = get_snap_start(sim,z0haloid)
    logger.debug(f'Start on snapshot {snap_start}, {filepaths[snap_start][-4:]}')
    
    # fix the case where the satellite doesn't have merger info prior to 
    if len(haloids) < snap_start:
        snap_start = len(haloids)
        raise Exception('Careful! You may have an error since the satellite doesnt have mergertree info out to the time where you want to start. This case is untested')
    
    if len(haloids) > snap_start:
        filepaths = np.flip(filepaths[:snap_start+1])
        haloids = np.flip(haloids[:snap_start+1])
        h1ids = np.flip(h1ids[:snap_start+1])

    if len(haloids) == snap_start:
        filepaths = np.flip(filepaths[:snap_start])
        haloids = np.flip(haloids[:snap_start])
        h1ids = np.flip(h1ids[:snap_start])   
        
    # filepaths and haloids now go the "right" way, i.e. starts from start_snap and goes until z=0
    assert len(filepaths) == len(haloids)
    assert len(haloids) == len(h1ids)

    # we save the data as an .hdf5 file since this is meant for large datasets, so that should work pretty good
    output = run_tracking(sim, z0haloid, filepaths, haloids, h1ids)

    savepath = '../../Data/tracked_particles.hdf5'
    logger.debug(f'Saving output to {savepath}')
    output.to_hdf(savepath,key=f'{sim}_{z0haloid}')







    





                
