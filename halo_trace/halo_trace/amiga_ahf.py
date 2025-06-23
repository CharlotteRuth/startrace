#/usr/bin/env python

import numpy as np
import pandas as pd
import pynbody as pb
import matplotlib.pyplot as plt
import glob
from pathlib import Path


def MW_center(sim, cen, h=0.73, mw_mass=7.e11, pair_tolerance=1, ahf_dir=None):
    """Finds the mass and location of the center of the MW analog, in Mpc
    
    sim: path to simulation
    
    cen: a center (xc, yc, zc) [Mpc] from which distance to MW is calculated
    
    h (default 0.73, or 0.6777 for PLK): reduced hubble constant (to correct AHF values to physical)
    
    mw_mass (default 7.e11): minimum mass for a massive (MW-like) galaxy
    
    pair_tolerance (default 1 [Mpc]): minimum difference between distance of closest and
    second-closest massive galaxy
    """
    if 'PLK' in sim:
        h = 0.6777
    #finding and loading relevant info from AHF file
    pth = Path(sim)
    if ahf_dir is not None:
        AHF_halos_file = list(pth.parent.glob(f'{ahf_dir}/*AHF_halos'))
    else:
        AHF_halos_file = glob.glob(sim+'*AHF_halos')
    #assert len(AHF_halos_file) >= 1
    if len(AHF_halos_file) < 1:
        raise IOError
    if len(AHF_halos_file) > 1:
        print('Warning: more than one AHF_halos file found')
    AHF_halos_file = AHF_halos_file[0]
    a = np.genfromtxt(AHF_halos_file, usecols=[3, 5, 6, 7, 11])
    a[:,1:4] = a[:,1:4]/1.e3
    a = a/h
    
    #anything larger than a Milky Way?
    tempa = a[a[:,0]>mw_mass]
    if len(tempa)>0:
        a = tempa
        dists = np.sqrt(np.sum((np.array(cen) - a[:,1:4])**2, axis=1))
        a = a[np.argsort(dists)]
        if len(dists)>1 and (dists[1]-dists[0]<pair_tolerance):
            print("Error! More than one MW found!")
            print(np.sort(dists)[:5])
            print(a[:5, 0], '\n')
            return a[0]
        else:
            print('distance is {:.1f} Mpc'.format(np.min(dists)), 'Mass is {:.1e}\n'.format(a[0, 0]))
            return a[0]
    #nothing larger than a Milky Way, return largest halo within 5 Mpc
    else:
        dists = np.sqrt(np.sum((np.array(cen) - a[:,1:4])**2, axis=1))
        a = a[dists<5.0]
        assert len(a)>1
        a = a[np.argsort(a[:,0])[-1]]
        print("None quite as big as the MW\n", a)
        return a
        #dists = np.sqrt(np.sum((np.array(cen)-a[1:])**2))
        



def halo_centers(sim, grps, h=0.73, ahf_dir=None):
    """Returns positions of halos in AHF_halo file.
    
    By matching halo 1 from the amiga file to a halo in the AHF file, returns the positions of halos with
    amiga.grp "grps" in coordinates of the AHF file
    
    h (default 0.73, or 0.6777 for PLK): reduced hubble constant to correct units in AHF file
    """
    if 'PLK' in sim:
        h = 0.6777
    #finding and reading in relevant info f AHF and amiga files
    pth = Path(sim)
    if ahf_dir is not None:
        AHF_halos_file = list(pth.parent.glob(f'{ahf_dir}/*AHF_halos'))
    else:
        AHF_halos_file = glob.glob(sim+'*AHF_halos')
    #assert len(AHF_halos_file) >= 1
    if len(AHF_halos_file) < 1:
        raise IOError
    if len(AHF_halos_file) > 1:
        print('Warning, more than one AHF halos file found')
    AHF_halos_file = AHF_halos_file[0]
    if ahf_dir is not None:
        amiga_file = list(pth.parent.glob(f'{ahf_dir}/*amiga.stat'))[0]
    else:
        amiga_file = str(sim) + '.amiga.stat'

    ahf_halos = np.genfromtxt(AHF_halos_file, skip_header=1, usecols=(0, 3, 5, 6, 7, 11), 
                              dtype=[('ID', '<f8'), ('Mvir', '<f8'), ('Xc', '<f8'), 
                                     ('Yc', '<f8'), ('Zc', '<f8'), ('Rvir', '<f8')])
    amigastat = np.genfromtxt(amiga_file, skip_header=1, usecols=(0, 5, 6, 13, 14, 15), 
                              dtype=[('ID', '<i8'), ('Mvir', '<f8'), ('Rvir', '<f8'), 
                                     ('Xc', '<f8'), ('Yc', '<f8'), ('Zc', '<f8')])

    for key in ['Mvir', 'Rvir', 'Xc', 'Yc', 'Zc']:
        ahf_halos[key] = ahf_halos[key]/h

    #finding the likely index for AHF of amiga halo 1
    match_ind = np.argmin(abs(ahf_halos['Mvir'] - amigastat[0]['Mvir']))
    #print('Index of matching halo 1 is ', match_ind)
    
    #secondary check by comparing virial radii
    #assert (ahf_halos[match_ind]['Rvir'] - amigastat[0]['Rvir'])/amigastat[0]['Rvir'] < 0.0001
    print('Rvir is within 0.01%! Match confirmed.')

    #center of amiga halo 1 in coordinates of AHF
    cen1 = ahf_halos[match_ind]['Xc'], ahf_halos[match_ind]['Yc'], ahf_halos[match_ind]['Zc']
    xchange = cen1[0]/1.e3-amigastat[0]['Xc']
    ychange = cen1[1]/1.e3-amigastat[0]['Yc']
    zchange = cen1[2]/1.e3-amigastat[0]['Zc']
    
    centers = []
    for grp in grps:
        grp_ind = np.where(amigastat['ID']==grp)[0][0]
        x = amigastat[grp_ind]['Xc']+xchange
        y = amigastat[grp_ind]['Yc']+ychange
        z = amigastat[grp_ind]['Zc']+zchange
        centers.append([x, y, z, ahf_halos[match_ind]['Rvir']])
    
    return np.array(centers)


def read_amiga(sim, nearest=None, ahf_dir=None):
    """Reads in the amiga.stat file and populates a dataframe with information.
    
    Given the pynbody sim, reads the amiga.stat file. Checks for issues regarding the
    last two columns and makes an attempt to fix them. Additionally defines some extra infomation,
    like the location of halos in coordinates of the AHF file and the distance to the nearest massive galaxy
    """
    try:
        filename = sim.filename
        print(sim.filename, '\n')
    except AttributeError:
        filename = sim
    pth = Path(filename)
    if ahf_dir is not None:
        amiga_file = list(pth.parent.glob(f'{ahf_dir}/*amiga.stat'))[0]
    else:
        amiga_file = filename + '.amiga.stat'
    df = pd.read_csv(amiga_file,
            delim_whitespace=True, index_col='Grp')

    # below are attempts to realign incorrect columns in cases where the False
    # column is empty. A possible fix would be to use pandas read_fwf instead
    if 'False' in df.columns:  # rename column for consistency across amiga files
        df.rename(columns={'False' : 'False?'}, inplace=True)
    if df['False?'].dtype != 'object':
        df['False?'] = df['False?'].astype(str)
    mask = (df['False?'] != 'false') & (df['False?'] != 'false?')
    if 'N_BH' in df.columns:
        if df['N_BH'].dtype != 'int64':
            df.loc[mask, 'N_BH'] = df.loc[mask, 'False?'].astype(np.int32)
    df.loc[mask, 'False?'] = 'nan'
    
    df = df.reindex(columns=df.columns.tolist() + ['X_AHF', 'Y_AHF', 'Z_AHF', 'RVIR_AHF'])
    df[['X_AHF', 'Y_AHF', 'Z_AHF', 'RVIR_AHF']] = halo_centers(filename, df.index, ahf_dir=ahf_dir)

    
    if nearest is None:
        #info of the nearest MW-like halo [mass, x, y, z, rvir]
        mw_dat = MW_center(filename, df.loc[1, ['X_AHF', 'Y_AHF', 'Z_AHF']].astype(float).values, mw_mass=6.e11, ahf_dir=ahf_dir)
        df['Nearest'] = np.sqrt(np.sum((mw_dat[1:4]
                                                    - df[['Xc', 'Yc', 'Zc']])**2, axis=1)).values * 1000.0
        df['Nearest(Rvir)'] = df['Nearest'] / mw_dat[4]
    elif nearest is not False:
        df['Nearest'] = np.sqrt(np.sum((df.loc[nearest, ['Xc', 'Yc', 'Zc']] - df[['Xc', 'Yc', 'Zc']])**2, axis=1)).values * 1000.0
        df['Nearest(Rvir)'] = df['Nearest'] / df.loc[nearest, 'RVIR_AHF']
    
    return df

def clean_amiga(df, res=500):
    """Removes false and likely false halos from the amiga.stat dataframe"""
    if df['False?'].dtype != 'object':
        df['False?'] = df['False?'].astype(str)
    flag = df.loc[df['StarMass(M_sol)'] > 0.2*df['DarkMass(M_sol)']].copy() # looking for false halos
    flag = flag.query('Grp > 1') # clearly the main halo isn't false
    flag['False?'] = 'false'
    
    df.loc[flag.index, 'False?'] = flag['False?']
    df = df.loc[df.N_tot>res].copy()
    df = df.loc[(df['False?']!='false')].copy()
    
    return df

def read_ahf(sim, ahf_dir=None):
    try:
        filename = sim.filename
    except AttributeError:
        filename = sim
    pth = Path(filename)
    if ahf_dir is not None:
        AHF_halos_file = list(pth.parent.glob(f'{ahf_dir}/*AHF_halos'))
    else:
        AHF_halos_file = glob.glob(filename+'*AHF_halos')
    if len(AHF_halos_file) < 1:
        raise IOError
    if len(AHF_halos_file) > 1:
        print('Warning, more than one AHF halos file found')
    AHF_halos_file = AHF_halos_file[0]
    df = pd.read_csv(AHF_halos_file, delim_whitespace=True)
    
    return df


def return_step(sim_name, step):
    """Given a generic sim_name, return the path to a specific step"""
    if len(step) < 6:
        step = step.rjust(6, '0')
    newname = re.sub('______', step, sim_name)
    return newname

if __name__ == '__main__':
    print(0)
