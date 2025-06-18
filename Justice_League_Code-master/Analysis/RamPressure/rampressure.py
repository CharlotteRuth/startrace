from analysis import *
from base import *
import sys
import os
import fsps

# used to align pynbody coordinate system to a specified vector
def vec_to_xform(vec):
    vec_in = np.asarray(vec)
    vec_in = vec_in / np.sum(vec_in ** 2).sum() ** 0.5
    vec_p1 = np.cross([1, 0, 0], vec_in)
    vec_p1 = vec_p1 / np.sum(vec_p1 ** 2).sum() ** 0.5
    vec_p2 = np.cross(vec_in, vec_p1)
    matr = np.concatenate((vec_p2, vec_in, vec_p1)).reshape((3, 3))
    return matr

# where the actual ram pressure calculations take place
def calc_ram_pressure(sim, z0haloid, filepaths, haloids, h1ids):
    output_tot = pd.DataFrame()
    
    
    for f,haloid,h1id in zip(filepaths,haloids,h1ids):
        # load simulation
        s = pynbody.load(f)
        s.physical_units()
        h = s.halos()
        sat = h[haloid]
        host = h[h1id]
        snapnum = f[-4:]
        logger.debug(f'* Snapshot {snapnum}')
        
        # save time t and scale factor a
        t = float(s.properties['time'].in_units('Gyr'))
        a = float(s.properties['a'])

        output = pd.DataFrame()
        output['t'] = [t]
        output['a'] = [a]
        
        # positions and velocities
        r_sat = np.array([sat.properties[k]/hubble*a for k in ['Xc','Yc','Zc']])
        r_host = np.array([host.properties[k]/hubble*a for k in ['Xc','Yc','Zc']])
        r_rel = r_sat - r_host
        h1dist = np.linalg.norm(r_rel)
        output['h1dist'] = [h1dist]
        logger.debug(f'Distance from host = {h1dist:.2f} kpc')
        
        v_sat = np.array([sat.properties[k] for k in ['VXc','VYc','VZc']])
        v_host = np.array([host.properties[k] for k in ['VXc','VYc','VZc']])
        v_rel = v_sat - v_host
        v_rel_mag = np.linalg.norm(v_rel)
        logger.debug(f'Relative velocity = {v_rel_mag:.2f} km/s')

        # nearest neighbor distance (topic of ongoing investigation)
        nDM = len(sat.dm)
        Xcs, Ycs, Zcs = np.array([]), np.array([]), np.array([])
        for halo in h:
            if len(halo.dm) > nDM*0.1:
                r = np.array([halo.properties[k]/hubble*a for k in ['Xc','Yc','Zc']])
                if not (r==r_sat).all():
                    Xcs = np.append(Xcs, r[0])
                    Ycs = np.append(Ycs, r[1])
                    Zcs = np.append(Zcs, r[2])
        
        
        r_others = np.array([Xcs, Ycs, Zcs]).T
        dists = np.linalg.norm(r_others - r_sat, axis=1)
        output['dNN'] = [np.min(dists)]
        
        # basic galaxy properties
        M_star = np.sum(sat.s['mass'].in_units('Msol'))
        M_gas = np.sum(sat.g['mass'].in_units('Msol'))
        rvir = sat.properties['Rvir']/hubble*a
        h1rvir = host.properties['Rvir']/hubble*a

        output['M_star'] = [M_star]
        output['M_gas'] = [M_gas]
        output['satRvir'] = [rvir]
        output['hostRvir'] = [h1rvir]
        logger.debug(f'Satellite M_gas = {M_gas:.1e} Msun')

        # simple ram pressure calculations: calculate rho_CGM from spherical density profile
        pynbody.analysis.halo.center(host)
        pg = pynbody.analysis.profile.Profile(s.g, min=0.01, max=2*h1dist, ndim=3)
        rbins = pg['rbins']
        density = pg['density']

        rho_CGM = density[np.argmin(np.abs(rbins-h1dist))]
        Pram = rho_CGM * v_rel_mag * v_rel_mag
        output['vel_CGM'] = [v_rel_mag]
        output['rho_CGM'] = [rho_CGM]
        output['Pram'] = [Pram]
        logger.debug(f'Simple v_rel = {v_rel_mag:.1f}')
        logger.debug(f'Simple rho_CGM = {rho_CGM:.1e}')
        logger.debug(f'Simple P_ram = {Pram:.1e}')


        # advanced ram pressure calculations: calculate rho, vel from cylinder in front of satellite
        
        # below code is adapted from pynbody.analysis.angmom.sideon() 
        # transform the snapshot so that the vector 'vel' points in the +y direction
        top = s
        logger.debug(f'Centering positions')
        cen = pynbody.analysis.halo.center(sat, retcen=True)
        tx = pynbody.transformation.inverse_translate(top, cen)
        logger.debug(f'Centering velocities')
        vcen = pynbody.analysis.halo.vel_center(sat, retcen=True) 
        tx = pynbody.transformation.inverse_v_translate(tx, vcen)
        
        # try to get vel from gas particles, but if there are no gas particles, use stars
        logger.debug(f'Getting velocity vector') 
        try:
            vel = np.average(sat.g['vel'], axis=0, weights=sat.g['mass'])
        except ZeroDivisionError:
            vel = np.average(sat.s['vel'], axis=0, weights=sat.s['mass'])
            
        vel_host = np.average(host.g['vel'], axis=0, weights=host.g['mass']) 
        vel -= vel_host

        logger.debug(f'Transforming snapshot')
        trans = vec_to_xform(vel)
        tx = pynbody.transformation.transform(tx, trans)
        
        # define cylinder size and filter out those particles
        radius = 0.5*rvir
        height = 0.75 * radius
        center = (0, rvir + height/2, 0)
        wind_filt = pynbody.filt.Disc(radius, height, cen=center)
        env = s[wind_filt].g
        logger.debug(f'Identified {len(env)} gas particles to calculate wind properties')
        output['n_CGM'] = [len(env)]

        # try to calculate CGM properties, but if you can't then set rho, vel to 0 (i.e. no gas particles)
        try:
            vel_CGM = np.linalg.norm(np.average(env['vel'],axis=0,weights=env['mass'])) # should be in units of Msun kpc**-3
            rho_CGM = np.average(env['rho'], weights=env['mass']) # should be in units of 
            std_rho_CGM = np.std(env['rho'])
            std_vel_CGM = np.std(np.linalg.norm(env['vel'], axis=1))
        except ZeroDivisionError:
            vel_CGM, rho_CGM = 0, 0
            std_vel_CGM, std_rho_CGM = 0, 0
            
        Pram = rho_CGM * vel_CGM * vel_CGM # overall units should be Msun kpc**-3 km**2 s**-2
        
        output['vel_CGM_adv'] = [vel_CGM]
        output['rho_CGM_adv'] = [rho_CGM]
        output['std_vel_CGM'] = [std_vel_CGM]
        output['std_rho_CGM'] = [std_rho_CGM]
        output['Pram_adv'] = [Pram]

        logger.debug(f'Advanced vel_CGM = {vel_CGM:.2f}')
        logger.debug(f'Advanced rho_CGM = {rho_CGM:.1e}')
        logger.debug(f'Advanced P_ram = {Pram:.1e}')

        # restoring pressure calculations
        # try to center the satellite. if you can't, then that means Prest = 0 (i.e. Mgas=0)
        try:
            pynbody.analysis.halo.center(sat)
            calc_rest = True
        except: 
            calc_rest = False
            Prest = 0.
            SigmaGas = 0.
            dphidz = 0.
        
        if calc_rest:
            p = pynbody.analysis.profile.Profile(s.g, min=0.01, max=rvir, ndim=3)
            percent_enc = p['mass_enc']/M_gas
            rhalf = np.min(p['rbins'][percent_enc > 0.5])
            SigmaGas = M_gas / (2*np.pi*rhalf**2)
            Rmax = sat.properties['Rmax']/hubble*a
            Vmax = sat.properties['Vmax']
            dphidz = Vmax**2 / Rmax
            Prest = dphidz * SigmaGas
        
        logger.debug(f'Prest = {Prest:.1e}')

        output['Prest'] = [Prest]
        output['SigmaGas'] = [SigmaGas]
        output['dphidz'] = [dphidz]
    
        
        # sfr calculations: use FSPS to calculate SFRs from formation masses of stars, not current masses
        star_masses = np.array(sat.s['mass'].in_units('Msol'),dtype=float)
        star_metals = np.array(sat.s['metals'], dtype=float)
        star_ages = np.array(sat.s['age'].in_units('Myr'),dtype=float)
        size = len(star_ages)
        
        # construct simple stellar population with fsps
        fsps_ssp = fsps.StellarPopulation(sfh=0,zcontinuous=1, imf_type=2, zred=0.0, add_dust_emission=False) 
        solar_Z = 0.0196
        
        star_masses = star_masses[star_ages <= 100]
        star_metals = star_metals[star_ages <= 100]
        star_ages = star_ages[star_ages <= 100]
        logger.debug(f'performing FSPS calculations on {len(star_masses)} star particles (subset of {size} stars)')
        
        if len(star_masses)==0:
            SFR = 0
        else:
            massform = np.array([])
            for age, metallicity, mass in zip(star_ages, star_metals, star_masses):
                fsps_ssp.params['logzsol'] = np.log10(metallicity/solar_Z)
                mass_remaining = fsps_ssp.stellar_mass
                massform = np.append(massform, mass / np.interp(np.log10(age*1e9), fsps_ssp.ssp_ages, mass_remaining)) 

            SFR = np.sum(massform)/100e6
            
        output['SFR'] = [SFR]
        output['sSFR'] = [SFR/M_star]
        logger.debug(f'sSFR = {SFR/M_star:.2e} yr**-1')
        
        output_tot = pd.concat([output_tot, output])

    return output_tot


if __name__ == '__main__':
    sim = str(sys.argv[1])
    z0haloid = int(sys.argv[2])

    if not os.path.exists('./logs/'):
        os.mkdir('./logs/')
    logging.basicConfig(filename=f'./logs/{sim}_{z0haloid}.log', 
                        format='%(asctime)s :: %(name)s :: %(levelname)-8s :: %(message)s', 
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.DEBUG)
    logger = logging.getLogger('RamPressure')

    logger.debug(f'--------------------------------------------------------------')
    logger.debug(f'Beginning ram pressure calculations for {sim}-{z0haloid}')
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

    output_tot = calc_ram_pressure(sim, z0haloid, filepaths, haloids, h1ids)
    savepath = '../../Data/ram_pressure.hdf5'
    logger.debug(f'Saving output to {savepath}')
    output_tot.to_hdf(savepath,key=f'{sim}_{z0haloid}')







    





                
