import pynbody
import pickle
import numpy as np
import pandas as pd
from base import *

### SELECT WHICH HALOS YOU WANT TO GET DATA FOR
data = read_timescales()
data = data[(~np.isnan(np.array(data.tinfall,dtype=float)))&(data.M_star > 2.45e5)]

print(f'Running for {len(data)} halos')

age = 13.800797497330507
hubble =  0.6776942783267969


with open('../../Data/QuenchingTimescales_InfallProperties.data','wb') as outfile:
    for sim in ['h329','h148','h242','h229']:
        filepaths, h1ids1, h1ids2 = get_stored_filepaths_haloids(sim, 1)        

        lbts = np.zeros(len(filepaths))
        for i, filepath in enumerate(filepaths):
            s = pynbody.load(filepath)
            lbt = age - float(s.properties['time'].in_units('Gyr'))
            lbts[i] = lbt

        print(f'Beginning sim {sim}') 

        for haloid in list(np.unique(data[data.sim==sim].haloid)):
            d = data[(data.sim==sim)&(data.haloid==haloid)]
            timesteps = read_timesteps(sim)
            timesteps = timesteps[timesteps.z0haloid==haloid]
            
            tinfall = d.tinfall.tolist()[0]
            tinfall_lower = d.tinfall_lower.tolist()[0]
            tinfall_upper = d.tinfall_upper.tolist()[0]

            tquench = d.tquench.tolist()[0]
            tquench_lower = d.tquench_lower.tolist()[0]
            tquench_upper = d.tquench_upper.tolist()[0]

            #n_star = d.n_star.tolist()[0]
            z0_M_star = d.M_star.tolist()[0]
            is_quenched = np.array(d.quenched,dtype=bool)[0]
            
            i = np.argmin(np.abs(lbts-tinfall)) # infall snapshot index
            f = filepaths[i] # infall snapshot filepath
            s = pynbody.load(f) # load in the snapshot at infall
            s.physical_units()
            h = s.halos()

            filepaths, haloids, h1ids = get_stored_filepaths_haloids(sim, haloid)

            host = h[h1ids[i]]
            sat = h[haloids[i]]

            print(f'Loaded halo {haloid}, tinfall = {tinfall:.2f} Gyr ago')
            print(haloids[i], f)

            sat_x, sat_y, sat_z = sat.properties['Xc']/hubble, sat.properties['Yc']/hubble, sat.properties['Zc']/hubble
            host_x, host_y, host_z = host.properties['Xc']/hubble, host.properties['Yc']/hubble, host.properties['Zc']/hubble
            r_sat = np.array([sat_x, sat_y, sat_z])
            r_host = np.array([host_x, host_y, host_z])

            v_sat = np.array([sat.properties['VXc'],sat.properties['VYc'],sat.properties['VZc']])
            v_host = np.array([host.properties['VXc'],host.properties['VYc'],host.properties['VZc']])

            v_rel = v_sat - v_host
            r_rel = r_sat - r_host
            v_rel_mag = np.sqrt(np.dot(v_rel,v_rel))
            h1dist = np.sqrt(np.dot(r_rel,r_rel))
            print(f'\t Relative velocity = {v_rel_mag:.2f} km/s')

            v_r = np.dot(v_rel, r_rel)/h1dist # magnitude of radial velocity vector in km/s
            theta = (180/np.pi)*np.arccos(np.abs(v_r)/np.sqrt(np.dot(v_rel,v_rel))) # angle of impact in degrees
            print(f'\t Impact angle = {theta:.2f} degrees')

            Vmax = sat.properties['Vmax']
            Rmax = sat.properties['Rmax']
            print(f'\t Max circular velocity = {Vmax:.2f} km/s')

            M_star = np.sum(sat.s['mass'].in_units('Msol'))
            M_gas = np.sum(sat.g['mass'].in_units('Msol'))
            M_halo = np.sum(sat.dm['mass'].in_units('Msol'))
            M_vir = M_star + M_gas + M_halo

            M_HI = np.sum(sat.gas['HI']*sat.gas['mass'].in_units('Msol'))
            print(f'\t HI mass = {M_HI:.2e} Msol')
            
            R_peri = np.min(np.array(timesteps.h1dist_kpc,dtype=float))
            print(f'\t Pericentric distance = {R_peri:.2f} kpc')

            pynbody.analysis.angmom.faceon(host)
            pg = pynbody.analysis.profile.Profile(s.g, min=0.01, max=2*h1dist, ndim=3)
            rbins = pg['rbins']
            density = pg['density']

            rho_cgm = density[np.argmin(np.abs(rbins-h1dist))]
            Pram = rho_cgm * np.sum(v_rel**2)
            print(f'\t Pram = {Pram:.1e}')
            
            try:
                pynbody.analysis.angmom.faceon(sat)
                rvir = sat.properties['Rvir']/hubble
                p = pynbody.analysis.profile.Profile(s.g, min=0.01, max=rvir)
                percent_enc = p['mass_enc']/M_gas
                rhalf = np.min(p['rbins'][percent_enc > 0.5])
                SigmaGas = M_gas / (2*np.pi*rhalf**2)
                dphidz = Vmax**2 / Rmax
                Prest = dphidz * SigmaGas
                print(f'\t Prest = {Prest:.1e}')
            except: 
                print('Failed to calculate Prest')
                Prest = None
            
            
            

            pickle.dump({
                'sim':sim,
                'snap':f,
                'haloid_snap':haloids[i],
                'haloid':haloid,
                'quenched': is_quenched,
                'tquench':tquench,
                'tquench_lower': tquench_lower,
                'tquench_upper': tquench_upper,
                'tinfall':tinfall,
                'tinfall_lower': tinfall_lower,
                'tinfall_upper': tinfall_upper,
                'z0_M_star': z0_M_star,
                #'n_star': n_star, 
                'M_star_at_infall':M_star, 
                'M_gas_at_infall':M_gas,
                'M_halo_at_infall':M_halo,
                'M_vir_at_infall':M_vir,
                'M_HI_at_infall':M_HI,
                'theta':theta,
                'v_r':v_r,
                'v_rel':v_rel_mag,
                'v_max':Vmax, 
                'r_peri':R_peri,
                'Pram':Pram,
                'rho_cgm':rho_cgm,
                'Prest':Prest
            }, outfile, protocol=2)

