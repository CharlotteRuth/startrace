import numpy as np
import pandas as pd
import pickle
import pynbody

# it is convenient to define these functions to read in our datasets for us

def read_timesteps(simname):
    '''Function to read in the timestep bulk-processing datafile (from /home/akinhol/Data/Timescales/DataFiles/{name}.data)'''
    data = []
    with open(f'../../Data/timesteps_data/{simname}.data','rb') as f:
        while True:
            try: 
                data.append(pickle.load(f))
            except EOFError:
                break
    
    data = pd.DataFrame(data)
    return data


names = ['h148','h242','h229','h329']

q_thresh = 1e-11 # sSFR threshold for queching, in yr^(-1)
i_thresh = 1.0 # distance threshold for infall, in host Rvir
age = 13.800797497330507


savepath = '../../Data/QuenchingTimescales.data'
with open(savepath,'wb') as f:
    for name in names:
        print(f'Simulation {name}')
        
        timesteps_all = read_timesteps(name)
        
        haloids = np.unique(np.array(timesteps_all.z0haloid, dtype=int)) # all the unique haloids we're interested in
        print('Found haloids (z=0): ', haloids)

        for haloid in haloids: # for each satellite
            timesteps = timesteps_all[np.array(timesteps_all.z0haloid,dtype=int)==haloid]
            
            # get the quenching time
            sfr, mstar, time = np.array(timesteps.sfr,dtype=float), np.array(timesteps.mstar,dtype=float), np.array(timesteps.time,dtype=float)
            nstar = np.array(timesteps.nstar, dtype=int)
            sfr = sfr[np.argsort(time)]
            mstar = mstar[np.argsort(time)]
            nstar = nstar[np.argsort(time)]
            time = np.sort(time)
            sSFR = sfr/mstar
            lbt = 13.8007 - time
            is_quenched = sSFR[-1] < q_thresh # boolean expression to determine whether the satellite is quenched
            if is_quenched:
                # then the halo is quenched, so we calculate the quenching time
                for i in range(0,len(lbt)):
                    t = np.flip(lbt)[i]
                    s = np.flip(sSFR)[i]
                    if s > 1e-11:
                        print(f'\t Halo {haloid}, Quenched (sSFR = {s}) {t} Gyr ago', end='  ')
                        tq = t
                        break
                    else:
                        continue
                        
                # lower limit on quenching time (same thing, but with 2e-11 as the threshold)
                for i in range(0,len(lbt)):
                    t = np.flip(lbt)[i]
                    s = np.flip(sSFR)[i]
                    if s > 2e-11:
                        tq_lower = t
                        break
                    else:
                        continue
                        
                # upper limit on quenching timem (same thing, but with 0 as the threshold)
                for i in range(0,len(lbt)):
                    t = np.flip(lbt)[i]
                    s = np.flip(sSFR)[i]
                    if s > 0:
                        tq_upper = t
                        break
                    else:
                        continue     
            else:
                print(f'\t Halo {haloid}, Unquenched at z = 0', end='  ')
                tq = None
                tq_lower = None
                tq_upper = None
                
            
            # get the infall time
            dist = np.array(timesteps.h1dist, dtype=float) # in Rvir
            time = age - np.array(timesteps.time, dtype=float) # in Gyr ago

            try:
                ti = np.max(time[dist <= i_thresh]) # maximum lookback time at which sat is < 1 Rvir from host (max LBT = earliest time)
                print(f'\t Halo {haloid}, infall {ti} Gyr ago')
            except ValueError:
                print(f'\t Halo {haloid}, never infell')
                ti = None
                
            try:
                ti_lower = np.max(time[dist <= 1.1])
            except ValueError:
                ti_lower = None
                
            try:
                ti_upper = np.max(time[dist <= 0.9])
            except ValueError:
                ti_upper = None
                
            z0mstar = mstar[-1] # for reference
            z0nstar = nstar[-1]

            pickle.dump({
                'haloid':haloid,
                'quenched':is_quenched,
                'tquench':tq,
                'tquench_lower': tq_lower,
                'tquench_upper': tq_upper,
                'tinfall':ti,
                'tinfall_lower': ti_lower,
                'tinfall_upper': ti_upper,
                'M_star': z0mstar,
                'n_star': z0nstar,
                'sim':name                
            }, f, protocol=2)
            

