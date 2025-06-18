import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Tahoma','Lucida Grande','Verdana', 'DejaVu Sans']

import socket
if socket.gethostname() == "quirm.math.grinnell.edu":
    z0data_prefix = '/home/akinshol/Research/Justice_League_Code/Data/z0_data/'
else:
    z0data_prefix = '/Users/hollis/Google Drive/Grinnell/MAP/Justice_League_Code/Data/z0_data/' # filepath is different on my local machine


def read_file(simname):
    '''
    A function that returns the compiled .data file for the given simname (one of 'h148', 'h229', 'h242', or 'h329'). 
    '''
    data = []
    with open(z0data_prefix + simname + '.data', 'rb') as f:
        while True:
            try:
                data.append(pickle.load(f,encoding='latin1'))
            except EOFError:
                break

        data1 = pd.DataFrame(data)
        data1['sim'] = [simname]*len(data1)
        data1['g-r'] = 1.646*(data1['V_mag'] - data1['R_mag']) - 0.139
        return data1
    
def read_sat():
    '''Function that returns a combined pandas dataframe for all Justice League simulations.'''
    data = read_file('h148')
    data = data.append(read_file('h229'))
    data = data.append(read_file('h242'))
    data = data.append(read_file('h329'))
    return data

def read_field():
    '''Function that returns a combined pandas dataframe for all Marvel simulations.'''
    data = read_file('cptmarvel')
    data = data.append(read_file('elektra'))
    data = data.append(read_file('rogue'))
    data = data.append(read_file('storm'))
    return data

def read_all():
    '''Function that returns a combined pandas dataframe for the combined suite of Justice League and Marvel simulations.'''
    satsims = read_sat()
    fieldsims = read_field()
    return satsims.append(fieldsims)

def quenched(data):
    return np.array(data['sSFR']<1e-11)

# this function is used to determine which galaxies are satellites for QuenchedFractions
def distance_to_nearest_host(data):
    '''
    Finds the distance from each galaxy in the given dataset to its nearest "host " galaxy. 
    If the galaxy is in a Justice League simulation, this is simply the distance to the main host halo, i.e. halo 1.
    If the galaxy is in a Marvel simulation, this is the distance to the nearest massive DM halo (Mvir > 1e11.5 Msol). 
    '''
    distances = []
    hostrvirs = []
    for i in range(len(data)):
        s = data['sim'].tolist()[i]
        
        if s=='h148' or s=='h229' or s=='h242' or s=='h329': # if sat simulation, find distance to halo 1
            h1dist = data['h1dist'].tolist()[i]
            distances.append(h1dist)
        
            h1rvir = data['Rvir'][(data.sim==s) & (data.haloid==1)].tolist()[0]
            hostrvirs.append(h1rvir)
            
        else: # if field simulation, find distance to nearest massive DM halo (currently > 1e11.5 Msol)
            if s=='cptmarvel':
                path = '/home/akinshol/Data/Sims/cptmarvel.cosmo25cmb.4096g5HbwK1BH/cptmarvel.cosmo25cmb.4096g5HbwK1BH.004096.dir/cptmarvel.cosmo25cmb.4096g5HbwK1BH.004096'

            if s=='elektra':
                path = '/home/akinshol/Data/Sims/elektra.cosmo25cmb.4096g5HbwK1BH/elektra.cosmo25cmb.4096g5HbwK1BH.004096.dir/elektra.cosmo25cmb.4096g5HbwK1BH.004096'

            if s=='rogue':
                path = '/home/akinshol/Data/Sims/rogue.cosmo25cmb.4096g5HbwK1BH/rogue.cosmo25cmb.4096g5HbwK1BH.004096.dir/rogue.cosmo25cmb.4096g5HbwK1BH.004096'

            if s=='storm':
                path = '/home/akinshol/Data/Sims/storm.cosmo25cmb.4096g5HbwK1BH/storm.cosmo25cmb.4096g5HbwK1BH.004096/storm.cosmo25cmb.4096g5HbwK1BH.004096'
            
            coords = []
            with open(path+'.coords','rb') as f:
                while True:
                    try:
                        coords.append(pickle.load(f,encoding='latin1'))
                    except EOFError:
                        break
            coords = pd.DataFrame(coords)
            
            threshold = 10**(11.5) # this threshold can be adjusted, 
            # i tried to pick something similar to the virial masses of the host in the JL simulations
            
            coords = coords[coords.mass > threshold]
            
            halocoords = np.array([data['Xc'].tolist()[i],data['Yc'].tolist()[i],data['Zc'].tolist()[i]])

            x = np.array(coords['Xc'])/0.6776942783267969
            y = np.array(coords['Yc'])/0.6776942783267969
            z = np.array(coords['Zc'])/0.6776942783267969
            Rvir = np.array(coords['Rv'])/0.6776942783267969

            
            c = np.array([x,y,z])
            c = np.transpose(c)
            dist = np.sqrt(np.sum((halocoords-c)**2, axis=1))
            distances.append(np.min(dist))
            hostrvirs.append(Rvir[np.argmin(dist)])
            
    return np.array(distances),np.array(hostrvirs)




##############################################################################################################
### the following functions are obsolete in my code, but left here incase someone wants to mess with them
### that is, they probably will break if you try to use them right now
##############################################################################################################
def sat(data,whichhost=False):
    '''A function to determine which halos are thought to be satellites by AHF (based on the hostHalo property).'''
    output = []
    hostid = []
    for i in range(len(data)):
        s = data['sim'].tolist()[i]
        if s=='h148' or s=='h229' or s=='h242' or s=='h329' or s=='elektra':
            if data['hostHalo'].tolist()[i]==-1:
                output.append(False)
                hostid.append(-1)
            else:
                output.append(True)
                if whichhost:
                    hostid.append(data['haloid'][np.array(data['id2'])==np.array(data['hostHalo'].tolist()[i])].tolist()[0])
        elif s=='cptmarvel' or s=='rogue' or s=='storm':
            if data['hostHalo'].tolist()[i]==0:
                output.append(False)
                hostid.append(-1)
            else:
                output.append(True)
                if whichhost:
                    hostid.append(data['haloid'][np.array(data['id2'])==np.array(data['hostHalo'].tolist()[i])].tolist()[0])
        else: 
            raise Exception("simname must be either 'h148','h229','h242','h329','cptmarvel','elektra','rogue', or 'storm'")
    
    if whichhost:
        return np.array(hostid)
    else:
        return np.array(output)
    
def whichHost(data):
    '''Same as sat(), but returns the halo id for the galaxy's host galaxy if it is a satellite.'''
    return sat(data,whichhost=True)

def distance_to_nearest_halo(data):
    '''Finds the distance from each satellite to their nearest (stellar) neighbor.'''
    distances = []
    for i in range(len(data)):
        halocoords = np.array([data['Xc'].tolist()[i],data['Yc'].tolist()[i],data['Zc'].tolist()[i]])
        nstars = np.delete(data['n_star'].tolist(),i)
        x = np.delete(data['Xc'].tolist(),i)
        y = np.delete(data['Yc'].tolist(),i)
        z = np.delete(data['Zc'].tolist(),i)
        x = x[nstars >= 100].tolist()
        y = y[nstars >= 100].tolist()
        z = z[nstars >= 100].tolist()
        coords = np.array([x,y,z])
        coords = np.transpose(coords)
        dist = np.min(np.sqrt(np.sum((halocoords - coords)**2, axis=1)))
        distances.append(dist)
    return np.array(distances)/0.6776942783267969 # distances must be divided by H0 to correct for expansion

def distance_to_host(data,rvir=True):
    '''
    Returns an array of the distances to host galaxies for all halos in `data`. 
    Optional argument `rvir` changes whether or not to divide distance by host virial radius.
    '''
    distances = []
    for i in range(len(data)):
        host = whichHost(data)[i]
        if host == -1:
            distances.append(0)
        else:
            halocoords = np.array([data['Xc'].tolist()[i],data['Yc'].tolist()[i],data['Zc'].tolist()[i]])
            hostcoords = np.array([data['Xc'][data['haloid']==host].tolist()[0],data['Yc'][data['haloid']==host].tolist()[0],data['Zc'][data['haloid']==host].tolist()[0]])
            if rvir:
                if data['sim'].tolist()[i]=='h148':
                    r = data['Rvir'][(data['haloid']==host) & (data['sim']=='h148')].tolist()[0]
                elif data['sim'].tolist()[i]=='h229':
                    r = data['Rvir'][(data['haloid']==host) & (data['sim']=='h229')].tolist()[0]
                elif data['sim'].tolist()[i]=='h242':
                    r = data['Rvir'][(data['haloid']==host) & (data['sim']=='h242')].tolist()[0]
                elif data['sim'].tolist()[i]=='h329':
                    r = data['Rvir'][(data['haloid']==host) & (data['sim']=='h329')].tolist()[0]
                distances.append(np.sqrt(np.sum((hostcoords - halocoords)**2))/r)
            else:
                distances.append(np.sqrt(np.sum((hostcoords - halocoords)**2)))
    return np.array(distances)


