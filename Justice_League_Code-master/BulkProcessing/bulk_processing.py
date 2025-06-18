import matplotlib as mpl
mpl.use('Agg') # use a headless backend since this is meant to run outside of a notebook, on quirm
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

# this path needs to point to the directory where .data files are stored
z0data_prefix = '/home/akinshol/Data/Akins_Hollis_JL_Code/Data/z0_data/'




# set the config to prioritize the AHF catalog
# otherwise it prioritizes AmgiaGrpCatalogue and you lose a lot of important info
pynbody.config['halo-class-priority'] =  [pynbody.halo.ahf.AHFCatalogue,
                                          pynbody.halo.GrpCatalogue,
                                          pynbody.halo.AmigaGrpCatalogue,
                                          pynbody.halo.legacy.RockstarIntermediateCatalogue,
                                          pynbody.halo.rockstar.RockstarCatalogue,
                                          pynbody.halo.subfind.SubfindCatalogue, pynbody.halo.hop.HOPCatalogue]

####################################################################################################################
### codelock below is copied from the pynboady source code
### set pynbody to center a halo based on its stars, not its gas 
### ask Lucas Chamberland for more info as to why we did this
####################################################################################################################

logger = logging.getLogger('pynbody.analysis.angmom')

def ang_mom_vec(snap):
    angmom = (snap['mass'].reshape((len(snap), 1)) *
              np.cross(snap['pos'], snap['vel'])).sum(axis=0).view(np.ndarray)
    return angmom
def ang_mom_vec_units(snap):
    angmom = ang_mom_vec(snap)
    return array.SimArray(angmom, snap['mass'].units * snap['pos'].units * snap['vel'].units)
def calc_sideon_matrix(angmom_vec):
    vec_in = np.asarray(angmom_vec)
    vec_in = vec_in / np.sum(vec_in ** 2).sum() ** 0.5
    vec_p1 = np.cross([1, 0, 0], vec_in)
    vec_p1 = vec_p1 / np.sum(vec_p1 ** 2).sum() ** 0.5
    vec_p2 = np.cross(vec_in, vec_p1)

    matr = np.concatenate((vec_p2, vec_in, vec_p1)).reshape((3, 3))

    return matr
def calc_faceon_matrix(angmom_vec, up=[0.0, 1.0, 0.0]):
    vec_in = np.asarray(angmom_vec)
    vec_in = vec_in / np.sum(vec_in ** 2).sum() ** 0.5
    vec_p1 = np.cross(up, vec_in)
    vec_p1 = vec_p1 / np.sum(vec_p1 ** 2).sum() ** 0.5
    vec_p2 = np.cross(vec_in, vec_p1)

    matr = np.concatenate((vec_p1, vec_p2, vec_in)).reshape((3, 3))

    return matr
def sideon(h, vec_to_xform=calc_sideon_matrix, cen_size="1 kpc",
           disk_size="5 kpc", cen=None, vcen=None, move_all=True,
           **kwargs):
    global config

    if move_all:
        top = h.ancestor
    else:
        top = h
    if cen is None:
        logger.info("Finding halo center...")
        # or h['pos'][h['phi'].argmin()]
        cen = halo.center(h, retcen=True, **kwargs)
        logger.info("... cen=%s" % cen)

    tx = transformation.inverse_translate(top, cen)

    if vcen is None:
        vcen = halo.vel_center(h, retcen=True, cen_size=cen_size)

    tx = transformation.inverse_v_translate(tx, vcen)

    cen = h[filt.Sphere(disk_size)]

    logger.info("Calculating angular momentum vector...")
    trans = vec_to_xform(ang_mom_vec(cen))

    logger.info("Transforming simulation...")

    tx = transformation.transform(tx, trans)

    logger.info("...done!")

    return tx
def faceon(h, **kwargs):
    return sideon(h, calc_faceon_matrix, **kwargs)

####################################################################################################################
### end pynbody source code
####################################################################################################################

# primary function here is the bulk processing function, which takes in a filepath (to the raw simulation snapshot) and 
def bulk_processing(filepath,halo_nums, name):
        print('Running for simulation %s ' % filepath)
        # ZSOLAR = 0.0130215 
        # XSOLO = 0.84E-2 #What pynbody uses
        # XSOLH = 0.706

        # first, load in the simulation and halo catalog
        s = pynbody.load(filepath)
        s.physical_units()
        h = s.halos()

        # this is the #ID key which corresponds to the hostHalo property 
        id = []
        for i in halo_nums:
            id.append(h[i].properties['#ID'])

        print('Loaded simulation')
        # then we open a new .data file, into which we will save the "pickled" data
        with open(z0data_prefix+name+'.data','wb') as f:
                # we loop through all the halos, compute a value for that halo, and add it to the .data file
                X1 = h[1].properties['Xc']/s.properties['h']
                Y1 = h[1].properties['Yc']/s.properties['h']
                Z1 = h[1].properties['Zc']/s.properties['h'] # X,y,z coordinates in physical units

                for halo_num in halo_nums:
                        # we load the copy of the halo to minimize computational stress
                        halo = h.load_copy(halo_num)
                        halo.physical_units()

                        # hostHalo property in order to determine which halo is the host of a satellite
                        hostHalo = h[halo_num].properties['hostHalo']

                        # Masses and Numbers of Particles
                        npart = len(halo)
                        nstar = len(halo.star)
                        ngas  = len(halo.gas)
                        mstar = np.sum(halo.star['mass'])
                        mgas = np.sum(halo.gas['mass'])
                        totalmass = np.sum(halo['mass'])
                        ovdens = h[halo_num].properties['ovdens']

                        print('Halo %s, %s particles' % (halo_num,npart))

                        # Mean Temperature of Gas
                        gas_temp = np.sum(halo.gas['temp']*halo.gas['mass'])/np.sum(halo.gas['mass'])

                        # Morphology (work done by Lucas Chamberland)
                        try:
                                faceon(halo)
                                hpmorph = pynbody.analysis.profile.Profile(halo.s,rmin=.0001,rmax=300,nbins=10000)
                                hpmorph2 = pynbody.analysis.profile.VerticalProfile(halo.s, rmin=0.0001, rmax=300, zmax=300, ndim=2,nbins=10000)
                                z90 = np.min(hpmorph2['rbins'][hpmorph2['mass_enc'] >= 0.9*mstar])
                                r90 = np.min(hpmorph['rbins'][hpmorph['mass_enc'] >= 0.9*mstar])
                                c_a = z90/r90
                                V90 = np.min(hpmorph['v_circ'][hpmorph['rbins'] == r90])
                                Vdisp = np.sqrt(((np.std(halo.s['vx']))**2 + (np.std(halo.s['vy']))**2 + (np.std(halo.s['vz']))**2)/3)
                                V90_Vdisp = V90/Vdisp

                        except Exception as err:
                                print(err)
                                z90 = None
                                r90 = None
                                c_a = None
                                V90 = None
                                Vdisp = None
                                V90_Vdisp = None

                        print('\t c/a %s, V90/Vdisp %s' % (c_a,V90_Vdisp))

                        # Virial Radius and center coordinates
                        Rvir = h[halo_num].properties['Rvir']/s.properties['h']

                        if halo_num==1:
                                Xc = X1
                                Yc = Y1
                                Zc = Z1
                        else:
                                Xc = h[halo_num].properties['Xc']/s.properties['h']
                                Yc = h[halo_num].properties['Yc']/s.properties['h']
                                Zc = h[halo_num].properties['Zc']/s.properties['h']

                        print('\t Rvir %s' % Rvir)

                        # Virial radius of host halo and distance to host halos
                        id = np.array(id)
                        if hostHalo<np.min(id):
                            hostVirialR = None
                            hostDist = None
                        else:
                            try:
                                hostHaloid = np.array(np.array(halo_nums)[id==hostHalo])[0]
                                hostVirialR = h[hostHaloid].properties['Rvir']/s.properties['h']
                                hostDist = np.sqrt((h[hostHaloid].properties['Xc'] - h[halo_num].properties['Xc'])**2 + (h[hostHaloid].properties['Yc'] - h[halo_num].properties['Yc'])**2 + (h[hostHaloid].properties['Zc'] - h[halo_num].properties['Zc'])**2)/s.properties['h']
                            except IndexError as err:
                                print(err)
                                hostVirialR = None
                                hostDist = None

                        # Gas Outflows (work done by Anna Engelhardt)
                        # center on the halos
                        if len(halo.gas)==0: # If there are no gas particles then the value is equal to zero for all these properties
                            goutflow15 = 0
                            goutflow25 = 0
                            ginflow15 = 0
                            ginflow25  = 0
                            ginflow95 = 0
                            goutflow95 = 0
                            Gout_T = 0
                            Gin_T = 0
                        else:
                            try:
                                pynbody.analysis.halo.center(halo) # Centers the halo

                                #select the particles in a shell from 0.2 Rvir to 0.3 Rvir
                                inner_sphere25 = pynbody.filt.Sphere(str(.2*Rvir) + ' kpc', [0,0,0])
                                outer_sphere25 = pynbody.filt.Sphere(str(.3*Rvir) + ' kpc', [0,0,0])
                                shell_part25 = halo[outer_sphere25 & ~inner_sphere25].gas

                                #Perform calculations
                                dL = .1*Rvir
                                velocity25 = shell_part25['vel'].in_units('kpc yr**-1')
                                r25 = shell_part25['pos'].in_units('kpc')
                                Mg25 = shell_part25['mass'].in_units('Msol')
                                r_mag25 = shell_part25['r'].in_units('kpc')

                                G_in25 = []     #List of inflowing gas flux per particles at Rvir = .2-.3
                                G_out25 = []     #List of outflowing gas flux per particles at Rvir = .2-.3
                                vr25 = np.sum((velocity25*r25), axis=1)
                                for x in range(len(vr25)):
                                    if vr25[x] < 0:
                                        gin25 = np.array(((vr25[x]/r_mag25[x])*Mg25[x])/dL)
                                        G_in25 = np.append(G_in25, gin25)
                                    else:
                                        gout25 = np.sum(((vr25[x]/r_mag25[x])*Mg25[x])/dL)
                                        G_out25 = np.append(G_out25, gout25)
                                #Net flux for inflow and outflow at .25*Rvir
                                ginflow25 = np.sum(G_in25)
                                goutflow25 = np.sum(G_out25)


                                #Select particles in a shell from .9 Rvir to 1 Rvir
                                inner_sphere95 = pynbody.filt.Sphere(str(.9*Rvir) + ' kpc', [0,0,0])
                                outer_sphere95 = pynbody.filt.Sphere(str(Rvir) + ' kpc', [0,0,0])
                                shell_part95 = halo[outer_sphere95 & ~inner_sphere95].gas

                                #Perform calculations
                                velocity95 = shell_part95['vel'].in_units('kpc yr**-1')
                                r95 = shell_part95['pos'].in_units('kpc')
                                Mg95 = shell_part95['mass'].in_units('Msol')
                                r_mag95 = shell_part95['r'].in_units('kpc')

                                G_in95 = []     #List of inflowing gas flux per particles at Rvir = .9-1
                                G_out95 = []     #List of outflowing gas flux per particles at Rvir = .9-1
                                vr95 = np.sum((velocity95*r95), axis=1)

                                for x in range(len(vr95)):
                                    if vr95[x] < 0:
                                        gin95 = np.array(((vr95[x]/r_mag95[x])*Mg95[x])/dL)
                                        G_in95 = np.append(G_in95, gin95)
                                    else:
                                        gout95 = np.sum(((vr95[x]/r_mag95[x])*Mg95[x])/dL)
                                        G_out95 = np.append(G_out95, gout95)
                                #Net flux for inflow and outflow at .95*Rvir
                                ginflow95 = np.sum(G_in95)
                                goutflow95 = np.sum(G_out95)


                                #select the particles in a shell from 0.2 Rvir to 0.3 Rvir
                                inner_sphere15 = pynbody.filt.Sphere(str(.1*Rvir) + ' kpc', [0,0,0])
                                outer_sphere15 = pynbody.filt.Sphere(str(.2*Rvir) + ' kpc', [0,0,0])
                                shell_part15 = halo[outer_sphere15 & ~inner_sphere15].gas

                                #Perform calculations
                                velocity15 = shell_part15['vel'].in_units('kpc yr**-1')
                                r15 = shell_part15['pos'].in_units('kpc')
                                Mg15 = shell_part15['mass'].in_units('Msol')
                                r_mag15 = shell_part15['r'].in_units('kpc')

                                G_in15 = []     #List of inflowing gas flux per particles at Rvir = .1-.2
                                G_out15 = []    #List of outflowing gas flux per particles at Rvir = .1-.2
                                vr15 = np.sum((velocity15*r15), axis=1)

                                for x in range(len(vr15)):
                                    if vr15[x] < 0:
                                        gin15 = np.array(((vr15[x]/r_mag15[x])*Mg15[x])/dL)
                                        G_in15 = np.append(G_in15, gin15)
                                    else:
                                        gout15 = np.sum(((vr15[x]/r_mag15[x])*Mg15[x])/dL)
                                        G_out15 = np.append(G_out15, gout15)
                                #Net flux for inflow and outflow at .15*Rvir
                                ginflow15 = np.sum(G_in15)
                                goutflow15 = np.sum(G_out15)


                                #Finding the Temperature of inflowing and outlfowing gas
                                out_T = []     #Temperature of each ouflowing particle weighted by mass
                                pm_out = []     #The mass of each outflowing particle
                                in_T = []     #Temperature of each inflowing particle weighted by mass
                                pm_in = []     #The mass of each inflowing particle

                                VEL = halo.gas['vel'].in_units('kpc yr**-1')
                                R = halo.gas['pos'].in_units('kpc')
                                GM = halo.gas['mass'].in_units('Msol')
                                R_mag = halo.gas['r'].in_units('kpc')
                                T =halo.gas['temp']

                                VR = np.sum((VEL*R), axis =1)

                                for x in range(len(VR)):
                                    if VR[x]>0:
                                        gtmp = np.array(T[x]*GM[x])
                                        pmout = np.array(GM[x])
                                        out_T = np.append(out_T, gtmp)
                                        pm_out = np.append(pm_out, pmout)
                                    if VR[x]<0:
                                        gtm = np.array(T[x]*GM[x])
                                        pmin = np.array(GM[x])
                                        in_T = np.append(in_T, gtm)
                                        pm_in = np.append(pm_in, pmin)
                                #Weighted average of temperatures for outflowing and inflowing gas
                                Gout_T = np.sum(out_T)/np.sum(pm_out)
                                Gin_T = np.sum(in_T)/np.sum(pm_in)

                            except Exception as err: # If there are not enough particles to center the halo then we get Nan instead of an error
                                goutflow15 = None
                                goutflow25 = None
                                ginflow15 = None
                                ginflow25 = None
                                ginflow95 = None
                                goutflow95 = None
                                Gout_T = None
                                Gin_T = None
                                print(err)

                        # star formation rate and history
                        sfh, bins = pynbody.plot.stars.sfh(halo, filename=None, massform=False, clear=False, legend=False, subplot=False, trange=False, bins=128)
                        if bins[-1]>13.6:
                                sfr = sfh[-1]
                        else:
                                sfr = 0

                        sSFR = sfr/mstar

                        print('\t SFR %s, sSFR %s ' % (sfr,sSFR))
                        
                        # Stellar Metallicity [Fe/H]
                        try:
                                zstar = np.sum(halo.star['feh'] * halo.star['mass'])/mstar
                        except ValueError:
                                zstar = None

                        # try: # see note in pickle.dump about why these are commented out
                        #         stars_feh = np.array(halo.star['feh'],dtype=float)
                        #         stars_mass = np.array(halo.star['mass'], dtype=float)
                        #         stars_r = np.array(halo.star['r'],dtype=float)
                        #         stars_oxh = np.array(halo.star['oxh'],dtype=float)
                        #         stars_ofe = np.array(halo.star['ofe'],dtype=float)
                        # except:
                        #         stars_feh, stars_mass, stars_r, stars_oxh, stars_ofe = None, None, None, None, None

                        # Stellar Metallicity (simple)
                        zstar_simp = np.sum(halo.star['metals']*halo.star['mass'])/mstar

                        # gas metallicity
                        hi = h[halo_num].gas['HI']
                        gem = np.sum(hi*halo.gas['metals']*halo.gas['mass'])
                        zgas = gem/np.sum(halo.gas['mass']*hi)
                        print('\t Stellar Z %s, Gas Z %s' % (zstar,zgas))

                        # Magnitudes
                        V_mag = pynbody.analysis.luminosity.halo_mag(halo.star,band='v')
                        B_mag = pynbody.analysis.luminosity.halo_mag(halo.star,band='b')
                        U_mag = pynbody.analysis.luminosity.halo_mag(halo.star,band='u')
                        R_mag = pynbody.analysis.luminosity.halo_mag(halo.star,band='r')
                        I_mag = pynbody.analysis.luminosity.halo_mag(halo.star,band='i')
                        
                        for band in ['u','g','r','i','z']:
                            halo.star['sdss_'+band+'_mag'] = pynbody.analysis.luminosity.calc_mags(halo.star, 
                                                                                                  band='sdss_'+band,
                                                                                                  cmd_path='cmd.sdss_ugriz.npz')
                        
                        r_mag = pynbody.analysis.luminosity.halo_mag(halo.star,band='sdss_r')
                        print(f'\t M_v = {V_mag:.2f}, M_r = {r_mag:.2f}')

                        print('\t Vmag %s, Bmag %s' % (V_mag,B_mag))
                        # gas fractions
                        GasFrac = mgas/(mstar+mgas)
                        mHI = np.sum(hi*halo.gas['mass'])
                        HIGasFrac = mHI/mstar
                        print('\t Gas Frac %s' % GasFrac)
                        # age of youngest star
                        age_youngest = np.min(np.array(h[halo_num].star['age'].in_units('Myr'),dtype=float))

                        print('\t Age of youngest star %s' % age_youngest)
                        # color
                        BV = float(B_mag) - float(V_mag)
                        distance = np.sqrt((X1 - Xc)**2 + (Y1 - Yc)**2 + (Z1 - Zc)**2)

                        # other ID number
                        id2 = h[halo_num].properties['#ID']
                        try:
                            ### quenching time
                            sfhmstar = np.sum(sfh)
                            # create cumulative sfh
                            bincenters = 0.5*(bins[1:]+bins[:-1])
                            c_sfh = np.empty(shape=sfh.shape)
                            for i in range(len(bincenters)):
                                c_sfh[i] = np.sum(sfh[:i+1])/sfhmstar

                            time = np.min(bincenters[c_sfh > 0.9])

                            age = 13.800797497330507
                            tquench = age - time
                        except ValueError as err:
                            print(err)
                            tquench = None




                        ###################################

                        # now we add these to the .data file
                        pickle.dump({
                                'haloid': halo_num,  # always put the haloid here
                                'hostHalo':hostHalo,
                                'n_particles':npart,
                                'n_star': nstar,
                                'n_gas':ngas,
                                'M_star':mstar,
                                'M_gas':mgas,
                                'mass':totalmass,
                                'Rvir':Rvir,
                                'G_outflow_2.5':goutflow25,
                                'G_outflow_1.5':goutflow15,
                                'G_inflow_2.5':ginflow25,
                                'G_inflow_1.5':ginflow15,
                                'G_inflow_0': ginflow95,
                                'G_outflow_0': goutflow95,
                                'Gout_T':Gout_T,
                                'Gin_T': Gin_T,
                                'Xc':Xc,
                                'Yc':Yc,
                                'Zc':Zc,
                                'feh_avg': zstar,
                                'zstar':zstar_simp,
                                'zgas':zgas,
                				'g_temp': gas_temp,
                                'V_mag':V_mag,
                                'B_mag':B_mag,
                                'U_mag':U_mag,
                                'R_mag':R_mag,
                                'I_mag':I_mag,
                                'r_mag':r_mag,
                                'gasfrac':GasFrac,
                                'mHI':mHI,
                                'HIgasfrac':HIGasFrac,
                                'sfh':sfh,
                                'sfhbins':bins,
                                'SFR':sfr,
                                'sSFR':sSFR,
                                'tquench':tquench,
                                'age':age_youngest,
                                'B-V':BV,
                                'h1dist':distance,
                                'id2':id2,
                                'Rmax':h[halo_num].properties['Rmax'],
                                'ovdens':ovdens,
                                'fMhires':h[halo_num].properties['fMhires'],
                                'c_a':c_a,
                                'c':z90,
                                'a':r90,
                                # 'feh':stars_feh, # these metallicity calculations take up a lot of space...
                                # 'oxh':stars_oxh, # they also don't work with the data currently available for 200b simulations
                                # 'ofe':stars_ofe,
                                # 'stars_mass':stars_mass,
                                # 'stars_r':stars_r,
                                'V90_Vdisp':V90_Vdisp,
                                'hostVirialR':hostVirialR,
                                'hostDist':hostDist
            },f,protocol=2)   


if __name__ == '__main__':
        # for ease of use I have the filepaths for all of the simulations here
        # note that you must use the 200bkgdens simulaitons if you want these to match the timesteps data (which you do)
        h148 = '/home/akinshol/Data/Sims/h148_200bkgdens/h148.cosmo50PLK.3072g3HbwK1BH.004096'
        h229 = '/home/akinshol/Data/Sims/h229_200bkgdens/h229.cosmo50PLK.3072gst5HbwK1BH.004096'
        h242 = '/home/akinshol/Data/Sims/h242_200bkgdens/h242.cosmo50PLK.3072gst5HbwK1BH.004096'
        h329 = '/home/akinshol/Data/Sims/h329_200bkgdens/h329.cosmo50PLK.3072gst5HbwK1BH.004096'
        cptmarvel = '/home/akinshol/Data/Sims/cptmarvel.cosmo25cmb.4096g5HbwK1BH/cptmarvel.cosmo25cmb.4096g5HbwK1BH.004096.dir/cptmarvel.cosmo25cmb.4096g5HbwK1BH.004096'
        elektra = '/home/akinshol/Data/Sims/elektra.cosmo25cmb.4096g5HbwK1BH/elektra.cosmo25cmb.4096g5HbwK1BH.004096.dir/elektra.cosmo25cmb.4096g5HbwK1BH.004096'
        rogue = '/home/akinshol/Data/Sims/rogue.cosmo25cmb.4096g5HbwK1BH/rogue.cosmo25cmb.4096g5HbwK1BH.004096.dir/rogue.cosmo25cmb.4096g5HbwK1BH.004096'
        storm = '/home/akinshol/Data/Sims/storm.cosmo25cmb.4096g5HbwK1BH/storm.cosmo25cmb.4096g5HbwK1BH.004096/storm.cosmo25cmb.4096g5HbwK1BH.004096'

        # here is where we will put whatever halo numbers we decide are interesting and worth computing
        # i.e. all the halos with stars in them and have fMhires > 0.9 
        # these numbers have been updated with newest numbers and filepaths with AFH re-ran with 200 backround density (FOR JL ONLY)--- 09 August 2019
        
        nums_h148 = [1, 2, 3, 5, 6, 9, 10, 11, 13, 14, 21, 24, 27, 28, 30, 32, 36, 37, 41, 45, 47, 48, 58, 61, 65, 68, 80, 81, 96, 105, 119, 127, 128, 136, 163, 212, 265, 278, 283, 329, 372, 377, 384, 386, 442, 491, 620, 678, 699, 711, 759, 914, 1004, 1024, 1201, 1217, 1451, 2905, 5039]
        nums_h229 = [1, 2, 5, 7, 17, 20, 22, 23, 27, 29, 33, 52, 53, 55, 59, 61, 62, 73, 104, 113, 139, 212, 290, 549, 1047, 1374, 1483, 1558, 6045]
        nums_h242 = [1, 10, 12, 24, 30, 34, 40, 41, 44, 48, 49, 71, 78, 80, 86, 165, 223, 439, 480, 1872, 2885, 6054, 9380, 10426, 12297]
        nums_h329 = [1, 11, 31, 33, 40, 64, 103, 133, 137, 146, 185, 447, 729, 996, 1509]

        nums_cptmarvel = [1, 2, 4, 5, 6, 7, 10, 11, 13, 14, 27, 167, 455, 1328]
        nums_elektra = [1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 17, 18, 37, 75, 85, 245]
        nums_rogue = [1, 3, 7, 8, 10, 11, 12, 16, 17, 18, 30, 32, 34, 36, 61, 77, 123, 702, 848, 3626, 6092]
        nums_storm = [124, 125, 169, 192, 208, 218, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 14, 15, 16, 17, 23, 24, 28, 34, 35, 43, 47, 49, 50, 60, 109, 253, 292, 2423, 3127, 3738, 5779]


        satsims = [h242,h148,h229,h329] # the order here just determines which one runs first
        satnames = ['h242','h148','h229','h329'] # but must match these orders
        satnums = [nums_h242,nums_h148,nums_h229,nums_h329]
        fieldsims = [storm,cptmarvel,elektra,rogue]
        fieldnames = ['storm','cptmarvel','elektra','rogue']
        fieldnums = [nums_storm,nums_cptmarvel,nums_elektra,nums_rogue]
        sims = np.append(satsims,fieldsims)
        nums = np.append(satnums,fieldnums)
        names = np.append(satnames,fieldnames)

        for sim,num,name in zip(satsims,satnums,satnames): # for each *Justice League* simulation
            bulk_processing(sim,num,name) # run the bulk_processing function above

        # if you want to run bulk_processing on all eight simulations, comment out the lines above and uncomment the lines below
        #for sim,num,name in zip(sims,nums,names):
            #bulk_processing(sim,num,name)

