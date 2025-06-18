from base import *
from particletracking import get_iords
import sys
import tqdm
import os
import fsps

hubble =  0.6776942783267969



def run_tracking(sim, z0haloid, filepaths,haloids,h1ids):
    # now we need to start tracking, so we need to get the iords
    iords = get_iords(sim, z0haloid, filepaths, haloids)

    output = pd.DataFrame()
    print('Starting tracking/analysis...')

    for f,haloid,h1id in tqdm.tqdm(zip(filepaths,haloids,h1ids),total=len(filepaths)):
        s = pynbody.load(f)
        s.physical_units()
        
        igasords = np.array(s.s['igasorder'],dtype=int)
        iordsStars = np.array(s.s['iord'],dtype=int)
        
        formedBool = np.isin(igasords,iords) # boolean array describing whether that star particle in the sim formed from one of our tracked gas particles        
        formedStars = s.s[formedBool] # formedStars is the star particles that formed from one of our gas particles 
        print(f'Identified {len(formedStars)} stars to track')
        
        # save formation times, masses, iords, and igasords of star particles that formed from gas particles we're tracking
        output = pd.concat([output, analysis(s, formedStars)])
    
    return output


def analysis(s, formedStars):
    output = pd.DataFrame()
    
    output['time'] = np.array([s.properties['time'].in_units('Gyr')]*len(formedStars.s), dtype=float)
    output['tform'] = np.array(formedStars.s['tform'].in_units('Gyr'), dtype=float)
    output['massform'] = np.array(formedStars.s['mass'].in_units('Msol'),dtype=float)
    output['pid'] = np.array(formedStars.s['iord'],dtype=int)
    output['igasorder'] = np.array(formedStars.s['igasorder'],dtype=int)
        
    return output


if __name__ == '__main__':
    sim = str(sys.argv[1])
    z0haloid = int(sys.argv[2])
    
    snap_start = get_snap_start(sim,z0haloid)
    filepaths, haloids, h1ids = get_stored_filepaths_haloids(sim,z0haloid)
    # filepaths starts with z=0 and goes to z=15 or so

    # fix the case where the satellite doesn't have merger info prior to 
    if len(haloids) < snap_start:
        snap_start = len(haloids)
        raise Exception('Careful! You may have an error since the satellite doesnt have mergertree info out to the time where you want to start. This case is untested')
    
    if len(haloids) >= snap_start:
        filepaths = np.flip(filepaths[:snap_start+1])
        haloids = np.flip(haloids[:snap_start+1])
        h1ids = np.flip(h1ids[:snap_start+1])
        
    # filepaths and haloids now go the "right" way, i.e. starts from start_snap and goes until z=0
    assert len(filepaths) == len(haloids)
    assert len(haloids) == len(h1ids)

    # we save the data as an .hdf5 file since this is meant for large datasets, so that should work pretty good
    output = run_tracking(sim, z0haloid, filepaths, haloids, h1ids)
    output.to_hdf('../../Data/tracked_stars.hdf5',key=f'{sim}_{z0haloid}')







    





                
