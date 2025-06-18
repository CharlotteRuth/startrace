import matplotlib as mpl
mpl.use('Agg')
import pynbody
import matplotlib.pyplot as plt
import numpy as np
import pynbody.plot as pp
import pynbody.filt as filt
import pickle
import pandas as pd
import logging
from pynbody import array,filt,units,config,transformation
from pynbody.analysis import halo
import os

# set the config to prioritize the AHF catalog
pynbody.config['halo-class-priority'] =  [pynbody.halo.ahf.AHFCatalogue,
                                          pynbody.halo.GrpCatalogue,
                                          pynbody.halo.AmigaGrpCatalogue,
                                          pynbody.halo.legacy.RockstarIntermediateCatalogue,
                                          pynbody.halo.rockstar.RockstarCatalogue,
                                          pynbody.halo.subfind.SubfindCatalogue, pynbody.halo.hop.HOPCatalogue]
                                          

def bulk_processing(tstep, haloids, rvirs, snapshots, path, savepath):
    snapshot = snapshots[tstep]
    # load the relevant pynbody data
    s = pynbody.load(path+snapshot)
    s.physical_units()
    t = s.properties['time'].in_units('Gyr')
    print(f'Loaded snapshot {snapshot}, {13.800797497330507 - t} Gyr ago')
    h = s.halos()
    hd = s.halos(dummy=True)
    
    # get current halo ids
    current_haloids = np.array([])
    z0_haloids = np.array([])
    for key in list(haloids.keys())[1:]:
        if not haloids[key][tstep] == 0:
            z0_haloids = np.append(z0_haloids, key)
            current_haloids = np.append(current_haloids, haloids[key][tstep])

    # get current rvirs
    current_rvirs = np.array([])
    for key in list(rvirs.keys())[1:]:
        if not haloids[key][tstep] == 0:
            current_rvirs = np.append(current_rvirs, rvirs[key][tstep])
    
    print(f'Gathered {len(current_haloids)} haloids')
    
    h1id = haloids[1][tstep]
    # get h1 properties
    h1x = hd[h1id].properties['Xc']
    h1y = hd[h1id].properties['Yc']
    h1z = hd[h1id].properties['Zc']
    
    h1d = np.array([h1x, h1y, h1z]) # halo 1 position
    h1d = h1d / s.properties['h'] * s.properties['a']
    h1r = hd[h1id].properties['Rvir'] # halo 1 virial radius
    h1r = h1r / s.properties['h'] * s.properties['a'] # put halo 1 virial radius in physical units
    h1v = np.array([hd[h1id].properties['VXc'], hd[h1id].properties['VYc'], hd[h1id].properties['VZc']]) # halo 1 velocity, already in physical units
    
    pynbody.analysis.angmom.faceon(h[h1id])
    pg = pynbody.analysis.profile.Profile(s.g, min=0.01, max=10*h1r, ndim=3) # make gas density profile
    # pall = pynbody.analysis.profile.Profile(s, min=0.01, max=6*h1r, ndim=3, nbins=200) # make total density profile
    print('\t Made gas density profile for halo 1 (technically halo %s)' % h1id)
    rbins = pg['rbins']
    density = pg['density']
    # density_enc = pall['density_enc']
    # rbins_all = pall['rbins']
    # try:
    #     r200b = np.min(rbins_all[density_enc < 4416.922400090694])
    #     print(f'\t Halo 1 R_200,c = {h1r:.2f} kpc, R_200,b = {r200b:.2f} kpc')
    # except:
    #     r200b = None


    
    for i, rvir, z0haloid in zip(current_haloids, current_rvirs, z0_haloids):
        print('Major progenitor halod ID:', i)
        halo = h.load_copy(i)        
        properties = hd[i].properties

        x = (properties['Xc'] - h1x) / s.properties['h'] * s.properties['a'] 
        y = (properties['Yc'] - h1y) / s.properties['h'] * s.properties['a']
        z = (properties['Zc'] - h1z) / s.properties['h'] * s.properties['a']

        old_rvir = properties['Rvir'] / s.properties['h'] * s.properties['a'] # put rvir in physical units
        print(f'\t Adjusted virial radius {rvir:.2f} kpc, old virial radius {old_rvir:.2f} kpc.')

        # compute ram pressure on halo from halo 1
        # first compute distance to halo 1

        v_halo = np.array([properties['VXc'],properties['VYc'],properties['VZc']]) # halo i velocity
        v = v_halo - h1v

        d = np.array([properties['Xc'],properties['Yc'],properties['Zc']]) # halo i position
        d = d / s.properties['h'] * s.properties['a'] # halo i position in physical units
        d = np.sqrt(np.sum((d - h1d)**2)) # now distance from halo i to halo 1
                
        print('\t Distance from halo 1 (technically halo %s): %.2f kpc or %.2f Rvir' % (h1id,d,d/h1r))
                
        # now use distance and velocity to calculate ram pressure
        pcgm = density[np.argmin(abs(rbins-d))]
        Pram = pcgm * np.sum(v**2)
        print('\t Ram pressure %.2e Msol kpc^-3 km^2 s^-2' % Pram)
                
        try:
            pynbody.analysis.angmom.faceon(h[i])
            pynbody.analysis.angmom.faceon(halo)
            calc_rest = True
            calc_outflows = True
            if len(h[i].gas) < 30:
                raise Exception
        except:
            print('\t Not enough gas (%s), skipping restoring force and inflow/outflow calculations' % len(h[i].gas))
            calc_rest = False
            calc_outflows = False
                        
        # calculate restoring force pressure
        if not calc_rest:
            Prest = None
            ratio = None
            env_vel = None
            env_rho = None
        else:
            print('\t # of gas particles:',len(h[i].gas))
            try:
                p = pynbody.analysis.profile.Profile(h[i].g,min=.01,max=rvir)
                print('\t Made gas density profile for satellite halo %s' % i)
                Mgas = np.sum(h[i].g['mass'])
                percent_enc = p['mass_enc']/Mgas

                rhalf = np.min(p['rbins'][percent_enc > 0.5])
                SigmaGas = Mgas / (2*np.pi*rhalf**2)

                dphidz = properties['Vmax']**2 / properties['Rmax']
                Prest = dphidz * SigmaGas

                print('\t Restoring pressure %.2e Msol kpc^-3 km^2 s^-2' % Prest)
                ratio = Pram/Prest
                print(f'\t P_ram / P_rest {ratio:.2f}')
            except:
                print('\t ! Error in calculating restoring force...')
                Prest = None
                Pram = None

            # calculate nearby region density
            try:
                r_inner = rvir
                r_outer = 3*rvir
                inner_sphere = pynbody.filt.Sphere(str(r_inner)+' kpc', [0,0,0])
                outer_sphere = pynbody.filt.Sphere(str(r_outer)+' kpc', [0,0,0])
                env = s[outer_sphere & ~inner_sphere].gas

                env_volume = 4/3 * np.pi * (r_outer**3 - r_inner**3)
                env_mass = np.sum(env['mass'].in_units('Msol'))
                env_vel = np.mean(np.array(env['mass'].in_units('Msol'))[np.newaxis].T*np.array(env['vel'].in_units('kpc s**-1')), axis=0) / env_mass
                env_rho = env_mass / env_volume
                print(f'\t Environmental density {env_rho:.2f} Msol kpc^-3')
                print(f'\t Environmental wind velocity {env_vel} kpc s^-1')
            except:
                print('\t Could not calculate environmental density')
                env_rho, env_vel = None, None


        age = np.array(h[i].star['age'].in_units('Myr'),dtype=float)
        sfr = np.sum(np.array(h[i].star['mass'].in_units('Msol'))[age < 100]) / 100e6
        print(f'\t Star formation rate {sfr:.2e} Msol yr^-1')

        # calculate gas fraction

        mstar = np.sum(h[i].star['mass'].in_units('Msol'), dtype=float)
        mgas = np.sum(h[i].gas['mass'].in_units('Msol'), dtype=float)
        mass = np.sum(h[i]['mass'].in_units('Msol'),dtype=float)
        
        gas_density = np.array(h[i].g['rho'], dtype=float)
        gas_temp = np.array(h[i].g['temp'], dtype=float)
        gas_mass = np.array(h[i].g['mass'], dtype=float)
        gas_r = np.array(h[i].g['r'], dtype=float)
        hi = np.array(h[i].g['HI'], dtype=float)
        print(f'\t Number of particles in halo {len(gas_density):.2e}')
        
        gas_sphere = h[i][pynbody.filt.Sphere(str(rvir)+' kpc', [0,0,0])].gas
        gas_density_sphere = np.array(gas_sphere['rho'],dtype=float)
        gas_temp_sphere = np.array(gas_sphere['temp'],dtype=float)
        gas_mass_sphere = np.array(gas_sphere['mass'],dtype=float)
        gas_r_sphere = np.array(gas_sphere['r'], dtype=float)
        print(f'\t Number of particles in sphere {len(gas_density_sphere):.2e}')
            
        if mgas == 0 and mstar == 0:
            gasfrac = None
        else:
            gasfrac = mgas / (mstar + mgas)
            print('\t Gas fraction %.2f' % gasfrac)
        
        # calculate gas temperature
        gtemp = np.sum(h[i].gas['temp']*h[i].gas['mass'].in_units('Msol'))/mgas
        print('\t Gas temperature %.2f K' % gtemp)
        # atomic hydrogen gas fraction
        mHI = np.sum(h[i].gas['HI']*h[i].gas['mass'])
        mHII = np.sum(h[i].gas['HII']*h[i].gas['mass'])
        if mHI == 0 and mstar == 0:
            HIGasFrac = None
        else:
            HIGasFrac = mHI/(mstar+mHI)
            if mstar == 0:
                HIratio = None
            else:
                HIratio = mHI/mstar
            print('\t HI gas fraction %.2f' % HIGasFrac)
                
        # get gas coolontime and supernovae heated fraction
        if not mgas == 0:
            coolontime = np.array(h[i].gas['coolontime'].in_units('Gyr'), dtype=float)
            gM = np.array(h[i].gas['mass'].in_units('Msol'), dtype=float) 
            SNHfrac = np.sum(gM[coolontime > t]) / mgas
            print('\t Supernova heated gas fraction %.2f' % SNHfrac)	
        else:
            SNHfrac = None
                
                
        if not calc_outflows:
            GIN2,GOUT2,GIN_T_25,GOUT_T_25,GINL,GOUTL,GIN_T_L,GOUT_T_L = None,None,None,None,None,None,None,None
        else:
            # gas outflow rate
            dL = .1*rvir

            # select the particles in a shell 0.25*Rvir
            inner_sphere2 = pynbody.filt.Sphere(str(.2*rvir) + ' kpc', [0,0,0])
            outer_sphere2 = pynbody.filt.Sphere(str(.3*rvir) + ' kpc', [0,0,0])
            shell_part2 = halo[outer_sphere2 & ~inner_sphere2].gas

            print("\t Shell 0.2-0.3 Rvir")

            #Perform calculations
            velocity2 = shell_part2['vel'].in_units('kpc yr**-1')
            r2 = shell_part2['pos'].in_units('kpc')
            Mg2 = shell_part2['mass'].in_units('Msol')
            r_mag2 = shell_part2['r'].in_units('kpc')
            temp2 = shell_part2['temp']

            VR2 = np.sum((velocity2*r2), axis=1)

            #### first, the mass flux within the shell ####

            gin2 = []
            gout2 = []

            for y in range(len(VR2)):
                if VR2[y] < 0: 
                    gflowin2 = np.array(((VR2[y]/r_mag2[y])*Mg2[y])/dL)
                    gin2 = np.append(gin2, gflowin2)
                else: 
                    gflowout2 = np.array(((VR2[y]/r_mag2[y])*Mg2[y])/dL)
                    gout2 = np.append(gout2, gflowout2)
            GIN2 = np.sum(gin2)
            GOUT2 = np.sum(gout2)

            print("\t Flux in %.2f, Flux out %.2f" % (GIN2,GOUT2))

            ##### now, calculate temperatures of the mass fluxes ####

            tin2 = []
            min_2 = []
            tout2 = []
            mout_2 = []

            for y in range(len(VR2)):
                if VR2[y] < 0: 
                    intemp2 = np.array(temp2[y]*Mg2[y])
                    tin2 = np.append(tin2, intemp2)
                    min2 = np.array(Mg2[y])
                    min_2 = np.append(min_2, min2)
                else: 
                    outemp2 = np.array(temp2[y]*Mg2[y])
                    tout2 = np.append(tout2, outemp2)
                    mout2 = np.array(Mg2[y])
                    mout_2 = np.append(mout_2, mout2)

            in_2T = np.sum(tin2)/np.sum(min_2)
            out_2T = np.sum(tout2)/np.sum(mout_2)    
            GIN_T_25 = np.sum(in_2T)
            GOUT_T_25 = np.sum(out_2T)

            print("\t Flux in temp %.2f, Flux out temp %.2f" % (GIN_T_25,GOUT_T_25))

            print('\t Shell 0.9-1.0 Rvir')
            #select the particles in a shell
            inner_sphereL = pynbody.filt.Sphere(str(.9*rvir) + ' kpc', [0,0,0])
            outer_sphereL = pynbody.filt.Sphere(str(rvir) + ' kpc', [0,0,0])
            shell_partL = halo[outer_sphereL & ~inner_sphereL].gas

            #Perform calculations
            DD = .1*rvir
            velocityL = shell_partL['vel'].in_units('kpc yr**-1')
            rL = shell_partL['pos'].in_units('kpc')
            MgL = shell_partL['mass'].in_units('Msol')
            r_magL = shell_partL['r'].in_units('kpc')
            tempL = shell_partL['temp']

            VRL = np.sum((velocityL*rL), axis=1)

            #### First, the Mas Flux within a Shell ####

            ginL = []
            goutL = []

            for y in range(len(VRL)):
                if VRL[y] < 0: 
                    gflowinL = np.array(((VRL[y]/r_magL[y])*MgL[y])/DD)
                    ginL = np.append(ginL, gflowinL)
                else: 
                    gflowoutL = np.array(((VRL[y]/r_magL[y])*MgL[y])/DD)
                    goutL = np.append(goutL, gflowoutL)   
            GINL = np.sum(ginL)
            GOUTL = np.sum(goutL)

            print('\t Flux in %.2f, Flux out %.2f' % (GINL,GOUTL))

            ##### now calculate the temperature of the flux ####

            tinL = []
            min_L = []
            toutL = []
            mout_L = []

            for y in range(len(VRL)):
                if VRL[y] < 0: 
                    intempL = np.array(tempL[y]*MgL[y])
                    tinL = np.append(tinL, intempL)
                    minL = np.array(MgL[y])
                    min_L = np.append(min_L, minL)
                else: 
                    outempL = np.array(tempL[y]*MgL[y])
                    toutL = np.append(toutL, outempL)
                    moutL = np.array(MgL[y])
                    mout_L = np.append(mout_L, moutL)

            in_LT = np.sum(tinL)/np.sum(min_L)
            out_LT = np.sum(toutL)/np.sum(mout_L)    
            GIN_T_L = np.sum(in_LT)
            GOUT_T_L = np.sum(out_LT)

            print("\t Flux in temp %.2f, Flux out temp %.2f" % (GIN_T_L,GOUT_T_L))


        f = open(savepath, 'ab')
        pickle.dump({
                'time': t, 
                't':t,
                'redshift':s.properties['z'],
                'a':s.properties['a'],
                'haloid': i,
                'z0haloid':z0haloid,
                'mstar': mstar,
                'mgas': mgas,
                'mass':mass,
                'Rvir': rvir,
                #'r200b':r200b,
                'gas_rho': gas_density,
                'gas_temp':gas_temp,
                'gas_mass':gas_mass,
                'gas_r':gas_r,
                'gas_hi':hi,
                'gas_rho_sphere': gas_density_sphere,
                'gas_temp_sphere':gas_temp_sphere,
                'gas_mass_sphere':gas_mass_sphere,
                'gas_r_sphere':gas_r_sphere,
                'x':x,
                'y':y,
                'z':z,
                'sfr':sfr,
                'Pram': Pram, 
                'Prest': Prest, 
                'v_halo':v_halo,
                'v_halo1':h1v,
                'v_env':env_vel,
                'env_rho':env_rho,
                'ratio': ratio, 
                'h1dist': d/h1r, 
                'h1dist_kpc': d,
                'h1rvir':h1r,
                'gasfrac': gasfrac,
                'SNHfrac': SNHfrac,
                'mHI':mHI,
                'fHI': HIGasFrac, 
                'HIratio': HIratio,
                'gtemp': gtemp,
                'inflow_23':GIN2,
                'outflow_23':GOUT2,
                'inflow_temp_23':GIN_T_25,
                'outflow_temp_23':GOUT_T_25,
                'inflow_91':GINL,
                'outflow_91':GOUTL,
                'inflow_temp_91':GIN_T_L,
                'outflow_temp_91':GOUT_T_L
        },f,protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
