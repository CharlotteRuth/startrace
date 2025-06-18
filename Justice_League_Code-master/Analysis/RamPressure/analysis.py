 # The purpose of this file is to perform a series of data manipuation and processing commands to particle tracking data in bulk. 
# In particular, functions in this file import particle tracking and ram pressure data, join them as necessary, calculate kinetic 
# and potential energies of particles, classify particles as disk vs. halo, identify ejected or expulsed particles, and more. 
# The reason these functions are written here is so that we can ensure that we are using the same data processing procedures throughout 
# the analysis and not have to repeat this code for each analysis component. 
import pynbody
import pandas as pd
import numpy as np
import pickle
from base import *

def get_keys():
    keys = ['h148_10','h148_12','h148_2','h148_23','h148_249','h148_251','h148_27','h148_282','h148_34','h148_38','h148_4','h148_55','h148_6','h148_65','h148_7','h229_14','h229_18','h229_20','h229_22','h229_49','h242_10','h242_21','h242_30','h242_38','h242_401','h242_69','h242_8','h329_117','h329_29','h329_7']
    return keys





def read_tracked_particles(sim, haloid, verbose=False):
    
    if verbose: print(f'Loading tracked particles for {sim}-{haloid}...')
    
    key = f'{sim}_{str(int(haloid))}'

    # import the tracked particles dataset
    path = '../../Data/tracked_particles.hdf5'
    data = pd.read_hdf(path, key=key)
    
    time = np.unique(data.time)
    dt = time[1:]-time[:-1]
    dt = np.append(dt[0], dt)
    dt = dt[np.unique(data.time, return_inverse=True)[1]]
    data['dt'] = dt
    
    
    if verbose: print('Successfully loaded')
    
    r_gal = np.array([])
    for t in np.unique(data.time):
        d = data[data.time==t]
        r_gas = np.mean(d.sat_r_gas)
        r_half = np.mean(d.sat_r_half)
        rg = np.max([r_gas,r_half])

        if np.isnan(rg):
            rg = r_gal_prev

        if verbose: print(f't = {t:1f} Gyr, satellite R_gal = {rg:.2f} kpc')
        r_gal = np.append(r_gal,[rg]*len(d))

        r_gal_prev = rg

    data['r_gal'] = r_gal
    
    r_gal_prev = 0
    r_gal = np.array([])
    for t in np.unique(data.time):
        d = data[data.time==t]
        r_gas = np.mean(d.host_r_gas)
        r_half = np.mean(d.host_r_half)
        rg = np.max([r_gas,r_half])

        if np.isnan(rg):
            rg = r_gal_prev

        if verbose: print(f't = {t:1f} Gyr, host R_gal = {rg:.2f} kpc')
        r_gal = np.append(r_gal,[rg]*len(d))

        r_gal_prev = rg

    data['host_r_gal'] = r_gal
    
    thermo_disk = (np.array(data.temp) < 1.2e4) & (np.array(data.rho) > 0.1)
    
    in_sat = np.array(data.in_sat)
    other_sat = np.array(data.in_other_sat)
    in_host = np.array(data.in_host) & ~in_sat & ~other_sat
    
    sat_disk = in_sat & thermo_disk
    sat_halo = in_sat & ~thermo_disk
    
    host_disk = in_host & thermo_disk
    host_halo = in_host & ~thermo_disk
    
    IGM = np.array(data.in_IGM)
    
    
#    sat_disk = in_sat & (np.array(data.r) <= np.array(data.r_gal))
#     sat_halo = in_sat & (np.array(data.r) > np.array(data.r_gal))
#     sat_cool_disk = sat_disk & thermo_disk
#     sat_hot_disk = sat_disk & ~thermo_disk
#     sat_cool_halo = sat_halo & thermo_disk
#     sat_hot_halo = sat_halo & ~thermo_disk

#     in_host = np.array(data.in_host) & ~in_sat
#     host_disk = in_host & (np.array(data.r_rel_host) <= np.array(data.host_r_gal))
#     host_halo = in_host & (np.array(data.r_rel_host) > np.array(data.host_r_gal))

#     other_sat = np.array(data.in_other_sat)
#     IGM = np.array(data.in_IGM)
    
    
    # basic classifications
    data['sat_disk'] = sat_disk
    data['sat_halo'] = sat_halo
    data['host_disk'] = host_disk
    data['host_halo'] = host_halo
    data['other_sat'] = other_sat
    data['IGM'] = IGM
    
    # more advanced classifications
    #data['cool_disk'] = sat_cool_disk
    #data['hot_disk'] = sat_hot_disk
    #data['cool_halo'] = sat_cool_halo
    #data['hot_halo'] = sat_hot_halo

    return data

def calc_angles(d):
    # get gas particle velocity
    v = np.array([d.vx,d.vy,d.vz])

    # get velocity of CGM wind (host velocity relative to satellite)
    v_sat = np.array([d.sat_vx,d.sat_vy,d.sat_vz])
    v_host = np.array([d.host_vx,d.host_vy,d.host_vz])
    v_rel = v_host - v_sat # we want the velocity of the host in the satellite rest frame

    # take the dot product and get the angle, in degrees
    v_hat = v / np.linalg.norm(v)
    v_rel_hat = v_rel / np.linalg.norm(v_rel)
    angle = np.arccos(np.dot(v_hat,v_rel_hat)) * 180/np.pi

    d['angle'] = angle
        
    return d



def calc_angles_tidal(d):
    # get gas particle velocity
    v = np.array([d.vx,d.vy,d.vz])

    # instead of CGM velocity, get vector pointing from satellite to host (i.e. host position in the satellite rest frame) 
    r_sat =np.array([d.sat_Xc,d.sat_Yc,d.sat_Zc])
    r_host = np.array([d.host_Xc,d.host_Yc,d.host_Zc])
    r_rel = r_host - r_sat

    # take the dot product and get the angle, in degrees
    v_hat = v / np.linalg.norm(v)
    r_rel_hat = r_rel / np.linalg.norm(r_rel)
    angle = np.arccos(np.dot(v_hat,r_rel_hat)) * 180/np.pi

    d['angle_tidal'] = angle
        
    return d


def calc_ejected_expelled(sim, haloid, check_outflow_time=True, save=True, verbose=True):
    import tqdm
    data = read_tracked_particles(sim, haloid, verbose=verbose)

    if verbose: print(f'Now computing expelled particles for {sim}-{haloid}...')
    expelled = pd.DataFrame()
    accreted = pd.DataFrame()
    
    pids = np.unique(data.pid)
    for pid in tqdm.tqdm(pids):
        dat = data[data.pid==pid]

        sat_disk = np.array(dat.sat_disk, dtype=bool)
        sat_halo = np.array(dat.sat_halo, dtype=bool)
        in_sat = np.array(dat.in_sat, dtype=bool)
        outside_sat = ~in_sat

        host_halo = np.array(dat.host_halo, dtype=bool)
        host_disk = np.array(dat.host_disk, dtype=bool)
        IGM = np.array(dat.IGM, dtype=bool)
        other_sat = np.array(dat.other_sat, dtype=bool)
        
        time = np.array(dat.time,dtype=float)

        for i,t2 in enumerate(time[1:]):
                i += 1
                
                i_end = np.argmin(np.abs(t2+1-time)) # find the index at which the time is closest to 1 Gyr after i
                if check_outflow_time:
                    check = all(outside_sat[i:i_end])
                else:
                    check = outside_sat[i]
                    
                if in_sat[i-1] and check:
                    out = dat[time==t2].copy()
                    if sat_halo[i-1]:
                        out['state1'] = 'sat_halo'
                    elif sat_disk[i-1]:
                        out['state1'] = 'sat_disk'
                        
                    expelled = pd.concat([expelled, out])
                    
                if outside_sat[i-1] and in_sat[i]:
                    out = dat[time==t2].copy()
                    if sat_halo[i]:
                        out['state2'] = 'sat_halo'
                    elif sat_disk[i]:
                        out['state2'] = 'sat_disk'
                        
                    accreted = pd.concat([accreted, out])

    print('Calculating ejection angles')
    print('Calculating expulsion angles')
    expelled = expelled.apply(calc_angles, axis=1)
    
    if save:
        key = f'{sim}_{str(int(haloid))}'
        if check_outflow_time:
            filepath = '../../Data/expelled_particles.hdf5'
        else:
            filepath = '../../Data/expelled_particles_no1Gyr.hdf5'
        print(f'Saving {key} expelled particle dataset to {filepath}')
        expelled.to_hdf(filepath, key=key)
        if check_outflow_time:        
            filepath = '../../Data/accreted_particles.hdf5'
        else:
            filepath = '../../Data/accreted_particles_no1Gyr.hdf5'
        print(f'Saving {key} accreted particle dataset to {filepath}')
        accreted.to_hdf(filepath, key=key)
        
        
    print(f'Returning (expelled, accreted) datasets...')

    return expelled, accreted
        

def read_ejected_expelled(sim, haloid, suffix=''):
    key = f'{sim}_{str(int(haloid))}'
    #ejected = pd.read_hdf('../../Data/ejected_particles.hdf5', key=key)
    #cooled = pd.read_hdf('../../Data/cooled_particles.hdf5', key=key)
    expelled = pd.read_hdf('../../Data/expelled_particles.hdf5', key=key)
    accreted = pd.read_hdf('../../Data/accreted_particles.hdf5', key=key)
    print(f'Returning (expelled, accreted) for {sim}-{haloid}...')
    return expelled, accreted
        
    
def read_all_ejected_expelled():
#     ejected = pd.DataFrame()
#     cooled = pd.DataFrame()
    expelled = pd.DataFrame()
    accreted = pd.DataFrame()
    keys = get_keys()
    for key in keys:
        expelled1 = pd.read_hdf('../../Data/expelled_particles.hdf5', key=key)
        expelled1['key'] = key
        expelled = pd.concat([expelled, expelled1])
        accreted1 = pd.read_hdf('../../Data/accreted_particles.hdf5', key=key)
        accreted1['key'] = key
        accreted = pd.concat([accreted, accreted1])

    print(f'Returning (expelled, accreted) for all available satellites...')
    return expelled, accreted

def read_ram_pressure(sim, haloid, suffix=''):
    '''Function to read in the ram pressure dataset, merge it with particle and flow information, and return a dataset containing 
    rates of gas flow in additiont to ram pressure information.'''
    
    # Load in ram pressure data
    path = '../../Data/ram_pressure.hdf5'
    key = f'{sim}_{haloid}'
    data = pd.read_hdf(path, key=key)
    
    # convert data to numpy arrays (i.e. remove pynbody unit information) and calculate ratio
    data['Pram_adv'] = np.array(data.Pram_adv,dtype=float)
    data['Pram'] = np.array(data.Pram,dtype=float)
    data['Prest'] = np.array(data.Prest,dtype=float)
    data['ratio'] = data.Pram_adv / data.Prest
    
    ratio_prev = np.array(data.Pram_adv/data.Prest,dtype=float)
    ratio_prev = np.append(ratio_prev[0], ratio_prev[:-1])
    data['ratio_prev'] = ratio_prev
    
    dt = np.array(data.t)[1:] - np.array(data.t)[:-1]
    dt = np.append(dt[0],dt)
    data['dt'] = dt
    
    # Load timescales information to add quenching time and quenching timescale (tau)
    timescales = read_timescales()
    ts = timescales[(timescales.sim==sim)&(timescales.haloid==haloid)]
    data['tau'] = ts.tinfall.iloc[0] - ts.tquench.iloc[0]    
    data['tquench'] = age - ts.tquench.iloc[0]   
    data['tinfall'] = age - ts.tinfall.iloc[0]   

    # load ejected/expelled data
    expelled,accreted = read_ejected_expelled(sim, haloid)

    # Mgas_div is the gas mass we divide by when plotting rates. this is the gas mass 1 snapshot ago
    Mgas_div = np.array(data.M_gas,dtype=float)
    Mgas_div = np.append(Mgas_div[0], Mgas_div[:-1])
    data['Mgas_div'] = Mgas_div
    
    Mstar_div = np.array(data.M_star,dtype=float)
    Mstar_div = np.append(Mstar_div[0], Mstar_div[:-1])
    data['Mstar_div'] = Mstar_div
    
    # load in particle data
    particles = read_tracked_particles(sim,haloid)
    # m_disk = 0 if particle is not in the disk, = particle mass if it is. this allows us to compute total mass in the disk
    particles['m_disk'] = np.array(particles.mass,dtype=float)*np.array(particles.sat_disk,dtype=int)
    particles['m_SNeaff'] = np.array(particles.mass,dtype=float)*np.array(particles.coolontime > particles.time, dtype=int)
    
    # group the particles data by unique times and sum the mass of particles that are SNe affected, to get total mass
    data = pd.merge_asof(data, particles.groupby(['time']).m_SNeaff.sum().reset_index(), left_on='t', right_on='time', direction='nearest', tolerance=1)
    data = data.rename(columns={'m_SNeaff':'M_SNeaff'})
    
    # group the particles data by unique times and sum the mass of particles that are in the disk, to get total mass
    data = pd.merge_asof(data, particles.groupby(['time']).m_disk.sum().reset_index(), left_on='t', right_on='time', direction='nearest', tolerance=1)
    data = data.rename(columns={'m_disk':'M_disk'})
    
    # analagous to Mgas_div above
    Mdisk_div = np.array(data.M_disk,dtype=float)
    Mdisk_div = np.append(Mdisk_div[0], Mdisk_div[:-1])
    data['Mdisk_div'] = Mdisk_div
    
#     # get rates of heated (ejected) gas
#     data = pd.merge(data, ejected.groupby(['time']).mass.sum().reset_index(), how='left', left_on='t', right_on='time')
#     data = data.rename(columns={'mass':'M_ejected'}) # mass ejected in that snapshot
#     data['Mdot_ejected'] = data.M_ejected / data.dt # rate of mass ejection 
#     data['Mdot_ejected_by_Mgas'] = data.Mdot_ejected / Mgas_div # rate of ejection divided by M_gas
#     data['Mdot_ejected_by_Mdisk'] = data.Mdot_ejected / Mdisk_div # rate of ejection divided by M_disk

#     # next, cooled gas
#     data = pd.merge(data, cooled.groupby(['time']).mass.sum().reset_index(), how='left', left_on='t', right_on='time')
#     data = data.rename(columns={'mass':'M_cooled'})
#     data['Mdot_cooled'] = data.M_cooled / data.dt
#     data['Mdot_cooled_by_Mgas'] = data.Mdot_cooled / Mgas_div
#     data['Mdot_cooled_by_Mdisk'] = data.Mdot_cooled / Mdisk_div

    # next, expelled gas (including gas expelled directly from the disk and gas expelled within a certain exit angle)
    expelled_disk = expelled[expelled.state1 == 'sat_disk']
    expelled_halo = expelled[expelled.state1 == 'sat_halo']
    expelled_th30 = expelled[expelled.angle <= 30]
    expelled_th30_disk = expelled[(expelled.angle <= 30)&(expelled.state1=='sat_disk')]
    expelled_th30_halo = expelled[(expelled.angle <= 30)&(expelled.state1=='sat_halo')]
    
    data = pd.merge_asof(data, expelled.groupby(['time']).mass.sum().reset_index(), left_on='t', right_on='time', direction='nearest', tolerance=1)
    data = data.rename(columns={'mass':'M_expelled'})
    data['Mdot_expelled'] = data.M_expelled / data.dt
    data['Mdot_expelled_by_Mgas'] = data.Mdot_expelled / Mgas_div
    data['Mdot_expelled_by_Mstar'] = data.Mdot_expelled / Mstar_div

    data = pd.merge_asof(data, expelled.groupby(['time']).apply(lambda x: np.average(x.angle, weights=x.mass)).reset_index(), left_on='t', right_on='time', direction='nearest', tolerance=1)
    data = data.rename(columns={0:'theta_mean'})

    # gas expelled from the halo
    data = pd.merge_asof(data, expelled_halo.groupby(['time']).mass.sum().reset_index(), left_on='t', right_on='time', direction='nearest', tolerance=1)
    data = data.rename(columns={'mass':'M_expelled_halo'})
    data['Mdot_expelled_halo'] = data.M_expelled_halo / data.dt
    data['Mdot_expelled_halo_by_Mgas'] = data.Mdot_expelled_halo / Mgas_div
    data['Mdot_expelled_halo_by_Mhalo'] = data.Mdot_expelled_halo / (Mgas_div-Mdisk_div)
    
    data = pd.merge_asof(data, expelled_halo.groupby(['time']).apply(lambda x: np.average(x.angle, weights=x.mass)).reset_index(), left_on='t', right_on='time', direction='nearest', tolerance=1)
    data = data.rename(columns={0:'theta_mean_halo'})
    
    # gas expelled directly from the disk
    data = pd.merge_asof(data, expelled_disk.groupby(['time']).mass.sum().reset_index(), left_on='t', right_on='time', direction='nearest', tolerance=1)
    data = data.rename(columns={'mass':'M_expelled_disk'})
    data['Mdot_expelled_disk'] = data.M_expelled_disk / data.dt
    data['Mdot_expelled_disk_by_Mgas'] = data.Mdot_expelled_disk / Mgas_div
    data['Mdot_expelled_disk_by_Mdisk'] = data.Mdot_expelled_disk / Mdisk_div
    
    data = pd.merge_asof(data, expelled_disk.groupby(['time']).apply(lambda x: np.average(x.angle, weights=x.mass)).reset_index(), left_on='t', right_on='time', direction='nearest', tolerance=1)
    data = data.rename(columns={0:'theta_mean_disk'})

    # gas expelled within an exit angle of 30 degrees
    data = pd.merge_asof(data, expelled_th30.groupby(['time']).mass.sum().reset_index(), left_on='t', right_on='time', direction='nearest', tolerance=1)
    data = data.rename(columns={'mass':'M_expelled_th30'})
    data['Mdot_expelled_th30'] = data.M_expelled_th30 / data.dt
    data['Mdot_expelled_th30_by_Mgas'] = data.Mdot_expelled_th30 / Mgas_div
    
    # gas expelled from the disk within an exit angle of 30 degrees 
    data = pd.merge_asof(data, expelled_th30_disk.groupby(['time']).mass.sum().reset_index(), left_on='t', right_on='time', direction='nearest', tolerance=1)
    data = data.rename(columns={'mass':'M_expelled_th30_disk'})
    data['Mdot_expelled_th30_disk'] = data.M_expelled_th30_disk / data.dt
    data['Mdot_expelled_th30_disk_by_Mdisk'] = data.Mdot_expelled_th30_disk / Mdisk_div
    
    # gas expelled from the halo within an exit angle of 30 degrees 
    data = pd.merge_asof(data, expelled_th30_halo.groupby(['time']).mass.sum().reset_index(), left_on='t', right_on='time', direction='nearest', tolerance=1)
    data = data.rename(columns={'mass':'M_expelled_th30_halo'})
    data['Mdot_expelled_th30_halo'] = data.M_expelled_th30_halo / data.dt
    data['Mdot_expelled_th30_halo_by_Mhalo'] = data.Mdot_expelled_th30_halo / (Mgas_div-Mdisk_div)
    
    
    
    # finally, accreted gas
    accreted_disk = accreted[accreted.state2 == 'sat_disk']
    
    data = pd.merge_asof(data, accreted.groupby(['time']).mass.sum().reset_index(), left_on='t', right_on='time', direction='nearest', tolerance=1)
    data = data.rename(columns={'mass':'M_accreted'})
    data['Mdot_accreted'] = data.M_accreted / data.dt
    data['Mdot_accreted_by_Mgas'] = data.Mdot_accreted / Mgas_div
    data['Mdot_accreted_by_Mstar'] = data.Mdot_accreted / Mstar_div

    
    data = pd.merge_asof(data, accreted_disk.groupby(['time']).mass.sum().reset_index(), left_on='t', right_on='time', direction='nearest', tolerance=1)
    data = data.rename(columns={'mass':'M_accreted_disk'})
    data['Mdot_accreted_disk'] = data.M_accreted_disk / data.dt
    data['Mdot_accreted_disk_by_Mgas'] = data.Mdot_accreted_disk / Mgas_div
    data['Mdot_accreted_disk_by_Mdisk'] = data.Mdot_accreted_disk / Mdisk_div

    # overall rate of gas-loss
    dM_gas = np.array(data.M_gas,dtype=float)[1:] - np.array(data.M_gas,dtype=float)[:-1]
    dM_gas = np.append([np.nan],dM_gas)
    data['Mdot_gas'] = dM_gas / np.array(data.dt)
    
    # rate of gas-loss from the disk
    dM_disk = np.array(data.M_disk,dtype=float)[1:] - np.array(data.M_disk,dtype=float)[:-1]
    dM_disk = np.append([np.nan],dM_disk)
    data['Mdot_disk'] = dM_disk / np.array(data.dt)
    
    data['key'] = key
    
    # fraction of the inital gas mass still remaining in the satellite
    M_gas_init = np.array(data.M_gas)[np.argmin(data.t)]
    data['f_gas'] = np.array(data.M_gas)/M_gas_init
    
    timesteps = read_timesteps(sim)
    timesteps = timesteps[['t','z0haloid','x','y','z','mass']]
    timesteps_target = timesteps[timesteps.z0haloid==haloid].reset_index()
    timesteps_target = timesteps_target.sort_values('t')
    timesteps_target = pd.merge_asof(data, timesteps_target, on='t', tolerance=1, direction='nearest')

#     FG = np.array([])
#     for tt in timesteps_target.iterrows():
#         tt = tt[1]
#         timesteps_tt = timesteps[np.abs(timesteps.t-tt.t) < 1e-10]
#         dists_sq = (timesteps_tt.x-tt.x)**2 + (timesteps_tt.y-tt.y)**2 + (timesteps_tt.z-tt.z)**2
#         masses = np.array(timesteps_tt.mass)

#         mass_target = masses[np.argmin(dists_sq)]
#         masses = masses[dists_sq > 0]
#         dists_sq = dists_sq[dists_sq > 0]

#         FG = np.append(FG, np.array(np.sum((masses/mass_target)/dists_sq),dtype=float))

#     data['FG'] = FG
    
    ## calculate sigmadisk
    sat_particles = particles[particles.in_sat]
    disk_particles = particles[particles.sat_disk]

    df_r = disk_particles.groupby(['time']).r.apply(r_half).reset_index()
    df_m = disk_particles.groupby(['time']).mass.sum().reset_index()
    df1 = pd.merge_asof(df_r, df_m, on='time', direction='nearest', tolerance=1)
    df1['SigmaDisk'] = np.array(df1.mass / (2*np.pi*df1.r**2))

    df_r = sat_particles.groupby(['time']).r.apply(r_half).reset_index()
    df_m = sat_particles.groupby(['time']).mass.sum().reset_index()
    df2 = pd.merge_asof(df_r, df_m, on='time', direction='nearest', tolerance=1)
    df2['SigmaGas'] = np.array(df2.mass / (2*np.pi*df2.r**2))
    
    df = pd.merge_asof(df1,df2,on='time', direction='nearest', tolerance=1)
    df = df[['time','SigmaDisk','SigmaGas']]


    data = pd.merge_asof(data, df, left_on='t', right_on='time', direction='nearest', tolerance=1)
    
    p = particles.groupby(['time'])[['sat_Xc','sat_Yc','sat_Zc','host_Xc','host_Yc','host_Zc','hostRvir']].mean().reset_index()
    t = np.array(p.time)
    x = np.array(p.sat_Xc-p.host_Xc)
    y = np.array(p.sat_Yc-p.host_Yc)
    z = np.array(p.sat_Zc-p.host_Zc)
    r = np.array(p.hostRvir)
    from scipy.interpolate import UnivariateSpline
    sx = UnivariateSpline(t, x)
    sy = UnivariateSpline(t, y)
    sz = UnivariateSpline(t, z)
    sr = UnivariateSpline(t, r)
    tnew = np.linspace(np.min(t),np.max(t),300)
    xnew = sx(tnew)
    ynew = sy(tnew)
    znew = sz(tnew)
    rnew = sr(tnew)
    d = np.sqrt(x**2+y**2+z**2)/r
    dnew = np.sqrt(xnew**2+ynew**2+znew**2)/rnew
    from scipy.signal import argrelextrema
    t_1peri = np.min(tnew[argrelextrema(dnew, np.less)])
    data['t_1peri'] = t_1peri
    
    
    
    return data
 
def r_half(r):
    x = np.sort(r)
    y = np.arange(len(x))/len(x)
    return np.max(x[y <= 0.5])
    
    
def read_all_ram_pressure(suffix=''):
    data_all = pd.DataFrame()
    
    keys = get_keys()
    i = 1
    for key in keys:
        print(i, end=' ')
        i += 1
        sim = key[:4]
        haloid = int(key[5:])
        data = read_ram_pressure(sim, haloid, suffix=suffix)
        data_all = pd.concat([data_all,data])  
    
    return data_all
