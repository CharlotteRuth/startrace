import pickle
pickle.HIGHEST_PROTOCOL = 4 # For the file to be read by python 3.7, it needs a lower pickle protocol than the default 5

from . import paths
from . import amiga_ahf as amah

import pynbody as pb
import numpy as np
import os
import pandas as pd
import glob
import subprocess
from pathlib import Path

import tqdm.notebook as tqdm

def match_by_merit(bridge_1_to_2, sim1_grp_list, sim=False, ahf_dir=None, groups_1=None, groups_2=None):
    """Match most likely halo between snapshots
    
    Maximize the merit function, Mij = N^2 (i & j)/(Ni Nj)), over j
        Assumes Ni and Nj are the sum from within the transfer matrix, which
    may not be a great assumption

    if sim is not false, pass the pynbody simulation to check for too-high grp values
        This is important to prevent the matrices from getting too big
    """
    print('\tMatching halos by merit function')
    # max_ind is the maximum halo number considered in the match
    max_ind = int(max(sim1_grp_list) * 1.5)
    print(max_ind)
    if max_ind > 5000:
        if not sim:
            raise ValueError('You must provide a pynbody snapshot for argument "sim" when using such large grp values')
        # using wc in shell to find number of halos in AHF halos file
        pth = Path(sim.filename)
        if ahf_dir is not None:
            ahf_abspath = list(pth.parent.glob(f'{ahf_dir}/*AHF_halos'))
            print(ahf_abspath)
            stat_abspath = list(pth.parent.glob(f'{ahf_dir}/*stat'))
        else:
            ahf_abspath = glob.glob(sim.filename + '*z*AHF_halos')
            print(ahf_abspath)
            stat_abspath = glob.glob(sim.filename + '*amiga.stat')
        if len(ahf_abspath) > 0:
            nmax = int(subprocess.check_output(['wc', '-l', ahf_abspath[0]]).split()[0]) - 1
        elif len(stat_abspath) > 0:
            nmax = int(subprocess.check_output(['tail', '-1', stat_abspath[0]]).split()[0])
        else:
            nmax = max_ind # have to think more about this, this isn't ideal
        # should be as high as the longer timestep's max Grp, but not higher
        max_ind = max(max(sim1_grp_list), min(max_ind, nmax)) 
    if max_ind < 100:
        max_ind = 100
    mat = bridge_1_to_2.catalog_transfer_matrix(max_index=max_ind, use_family=pb.family.dm, groups_1=groups_1, groups_2=groups_2)
    # https://pynbody.readthedocs.io/latest/reference/_autosummary/pynbody.bridge.AbstractBridge.html#pynbody.bridge.AbstractBridge.count_particles_in_common
    # https://pynbody.readthedocs.io/v1-docs/reference/convenience.html#pynbody.bridge.Bridge.catalog_transfer_matrix
    # Switch to count_particles_in_common, deprecated
    # mat = bridge_1_to_2.count_particles_in_common(groups_1, groups_2, max_num_halos=max_ind, use_family=pb.family.dm)
    print(mat)
    Ni = np.sum(mat, axis=1) #number of particles in grp of sim 1
    print('\tNi=',Ni)
    Nj = np.sum(mat, axis=0) #number of particles in grp of sim 2
    print('\tNj=',Nj)

    # Check for zeros to prevent division by zero
    if np.any(Ni == 0) or np.any(Nj == 0):
        print('\tWarning: Some halos have zero particles')
        print(f'\tZeros in Ni: {np.sum(Ni == 0)}, Zeros in Nj: {np.sum(Nj == 0)}')
    
    # eps = 1e-10
    # Ni_safe = np.where(Ni == 0, eps, Ni)
    # Nj_safe = np.where(Nj == 0, eps, Nj)
    
    # merit_f = mat**2/np.outer(Ni_safe, Nj_safe)
    # # Set merit function to zero where either Ni or Nj was originally zero
    # zero_mask = np.outer(Ni == 0, np.ones_like(Nj, dtype=bool)) | np.outer(np.ones_like(Ni, dtype=bool), Nj == 0)
    # merit_f = np.where(zero_mask, 0.0, merit_f)
    
    #find merit function in both directions -- need to think if this is necessary. Each direction is transpose
    # of the other direction.
    merit_f = mat**2/np.outer(Ni, Nj)
    merit_f_back = merit_f.transpose() #mat.transpose()**2/np.outer(Nj, Ni)
    merit_f_mask = np.ma.array(merit_f, mask=np.isnan(merit_f))
    merit_f_back_mask = merit_f_mask.transpose() #np.ma.array(merit_f_back, mask=np.isnan(merit_f_back))
    guess = np.arange(1, len(Nj)+1, 1)[np.argmax(merit_f_mask, axis=1)]
    guess_back = np.arange(1, len(Nj)+1, 1)[np.argmax(merit_f_back_mask, axis=1)]
    
    flag_indices = np.arange(len(Nj))[guess_back[guess - 1] != np.arange(1, len(Nj)+1, 1)]
    guess[flag_indices] = -1
    sim2_grplist = [guess[grp - 1] if grp != -1 else -1 for grp in sim1_grp_list] #grp-1 because convert from grp to index
    
    return np.array(sim2_grplist)

def trace_halos(sim_base, trace_sats=False, grplist=None, steplist=None, maxstep=None, save_file=True,
                min_ntot=500, cross_check=True, ahf_dir=None, **kwargs):
    """Trace halos to between time steps
    
    Will find the lowest redshift available step, and will iterate through
    higher steps. For each higher redshift step, finds the most probable parent halos
    for the next lower redshift step
    
    sim_base - the base directory of the simulation you wish to match
    
    grplist(default None) - the desired halos to trace in the lowest redshift step
        if None, will trace all halos in amiga.stat file with more than min_ntot particles
    
    steplist(default None) - the steps to include in the trace. If None, uses all available
    
    maxstep(default None) - the earliest time step to include. If None, uses all available
    
    save_file(default True) - if True or string, save the resulting dataframe with default name
        or passed string. Will check for existence of file before overwrite
    
    min_ntot(default 500) - minimum number of particles in halos to track
    
    cross_check(default True) - if True, simulatenously link every step and every other step, and
        compare between the two to avoid losing track of halos

    ahf_dir(default None) - if not None, an extra (relative) path to append for the AHF catalogs

    
    ***********

    returns pandas DataFrame of matched halos, with -1 where matches were not found successfully
    
    """
    
    all_steplist = paths.list_steps(sim_base)
    #print(all_steplist)
    if steplist is None:
        steplist = all_steplist
    else:
        #for substep in steplist (e.g. [71, 96, 4096]), iterate through step in all_steplist to check if
        #substep is in step
        steplist = [step for step in all_steplist for substep in steplist if str(substep).rjust(6, '0') in step]
		
    if ahf_dir is not None:
        temp_steplist = []
        print(len(steplist))
        for step in steplist:
            # print('step',step)
            for subdir in Path(step).parent.iterdir():
                # print('\tsubdirr',subdir)
                # print('\t\tsubdir',str(subdir)[-len(ahf_dir):])
                if str(subdir)[-len(ahf_dir):] == ahf_dir:
                    # checking if the ahf_dir contains the AHF catalogs
                    if sum([str(subpath).endswith('AHF_halos') for subpath in subdir.iterdir()]) >= 1:
                        temp_steplist.append(step)

        print('Using only the following steps which contain ahf_dir and AHF catalogs:')
        print([step[-6:] for step in temp_steplist])
        steplist = temp_steplist
    
    if save_file:
        try:
            l = len(save_file)
        except TypeError:
            if trace_sats:
                save_file = steplist[0] + '.trace_back_sats.hdf5'
            else:
                save_file = steplist[0] + '.trace_back.hdf5'
        if os.path.isfile(save_file):
            print('File exists: ' + save_file)
            print('Aborting')
            return None
        else:
            print('File will be saved as ' + save_file)
    
    if maxstep is not None:
        steplist = [step for step in steplist if int(step.split('.')[-1]) >= int(maxstep)]
        print('Earliest step is ' + steplist[-1] + '\n')
        
    if grplist is None:
        if ahf_dir is not None:
            pth = Path(steplist[0])
            try:
                stat_abspath = str(list(pth.parent.glob(f'{ahf_dir}/*stat'))[0])
                print('grp list based on ' + stat_abspath + '\n')
            except IndexError:
                raise IOError('Given ahf_dir does not exist')
            all_grps =  np.genfromtxt(stat_abspath, skip_header=1, usecols=(0, 1),  dtype=[('ID', '<i4'), ('N_tot', '<f8')])
        else:
            all_grps = np.genfromtxt(steplist[0] + '.amiga.stat', skip_header=1, usecols=(0, 1), dtype=[('ID', '<i4'), ('N_tot', '<f8')])
        grplist = all_grps[all_grps['N_tot'] >= min_ntot]['ID']
    print('grplist=', grplist)

    if trace_sats:
        cross_check = False
        sats = pd.read_hdf(steplist[0] + '.sats.hdf5')
        grplist = sats.loc[steplist[0][-6:], 'Sats']

    uniqhaloid = [steplist[0][-6:] + str(haloid) for haloid in grplist.tolist()]
    df = pd.DataFrame(index=uniqhaloid, columns=[step[-6:] for step in steplist[1:]], )
    df.index.name = steplist[0][-6:]
    if len(steplist) > 2 and cross_check: #this will match every other as a cross-check
        df2 = pd.DataFrame(index=uniqhaloid, columns=[step[-6:] for step in steplist[1:]], )
        df2.index.name = steplist[0][-6:]
    elif cross_check:
        print('Must trace through at least 3 steps to use cross check')
        return
    
    print('simlow=',steplist[0])
    sim_low = pb.load(steplist[0])
    print('simlow=',sim_low)
    if ahf_dir is not None:
        pth = Path(sim_low.filename)
        print('pth=',pth)
        ahf_basename = str(list(pth.parent.glob(f'{ahf_dir}/*AHF_halos'))[0])[:-5]
        print('185ahfbasename=',ahf_basename)
        groups_1 = sim_low.halos(halo_numbers='v1', filename=ahf_basename)
    else:
        print('no ahf seen???')
        groups_1 = None

    pbar = tqdm.tqdm(total=len(steplist) - 1, desc='Tracing halos', unit='step')
    for i, step in enumerate(steplist[1:]):
        print('Starting step ' + step[-6:])
        
        if i == 0 and cross_check:
            grplist2 = grplist
            
        sim_high = pb.load(step)
        print('simhigh=',sim_high)
        b = pb.bridge.OrderBridge(sim_low, sim_high)
        #b = sim_low.bridge(sim_high)

        if i%2 == 0 and cross_check and i+2 < len(steplist):
            print('Advanced step ' + steplist[i+2][-6:])
            sim_high2 = pb.load(steplist[i+2])
            b2 = pb.bridge.OrderBridge(sim_low, sim_high2)
            #b2 = sim_low.bridge(sim_high2)
        try:
            if ahf_dir is not None:
                pth = Path(sim_high.filename)
                ahf_basename = str(list(pth.parent.glob(f'{ahf_dir}/*AHF_halos'))[0])[:-5]
                print('ahf_basename=',ahf_basename)
                groups_2 = sim_high.halos(halo_numbers='v1', filename=ahf_basename)
            else:
                groups_2 = None
            grplist = match_by_merit(b, grplist, sim_high, ahf_dir=ahf_dir, groups_1=groups_1, groups_2=groups_2)
            print('\tdone matching halos')
            df[step[-6:]] = grplist

            if trace_sats :
                if step[-6:] in sats.index:
                    newsat = np.in1d(np.array(sats.loc[step[-6:],'Sats']), np.array(grplist), invert = True)
                    grplist = np.append(grplist, (sats.loc[step[-6:],'Sats'])[newsat]) #append on the grp list any satellites at this timestep that were not already part of the calculation
                    print((sats.loc[step[-6:],'Sats'])[newsat])
                    uniqhaloid = [step[-6:] + str(haloid) for haloid in (sats.loc[step[-6:],'Sats'])[newsat]]
                    df_add = pd.DataFrame(index=uniqhaloid, columns=[step[-6:] for step in steplist[1:]], )
                    df_add[step[-6:]] = (sats.loc[step[-6:],'Sats'])[newsat]
                    df = df.append(df_add)
            
        except(ValueError, IndexError) as err:
            print(err)
            print('Problem encountered, skipping step ' + step[-6:] + '\n')

        if i%2 == 0 and cross_check and i+2 < len(steplist):
            try:
                if ahf_dir is not None:
                    pth = Path(sim_high2.filename)
                    ahf_basename = str(list(pth.parent.glob(f'{ahf_dir}/*AHF_halos'))[0])[:-5]
                    groups_2_2 = sim_high2.halos(filename=ahf_basename, halo_numbers='v1')
                else:
                    groups_2_2 = None
                grplist2 = match_by_merit(b2, grplist2, sim_high2, ahf_dir=ahf_dir, groups_1=groups_1, groups_2=groups_2_2)
                df2[steplist[i+2][-6:]] = grplist2
            except(ValueError, IndexError) as err:
                print(err)
                print('Problem encountered in cross-check, skipping ' + step[-6:] + '\n')
            try:
                del sim_high2
            except(NameError, UnboundLocalError):
                pass
        elif cross_check:# and step != steplist[-1]:
            print('Cross-checking step ' + step[-6:] +'\n')
            mask = (df[step[-6:]] != df2[step[-6:]]) & (df[step[-6:]] == -1)
            df.loc[mask, step[-6:]] = df2.loc[mask, step[-6:]]
            # check for duplicates introduced
            df.loc[(df[step[-6:]].duplicated(keep=False)) 
                   & (df[step[-6:]] != -1) 
                   & (df[steplist[i][-6:]] == -1), step[-6:]] = -1
            grplist = df[step[-6:]].values
        sim_low = sim_high
        groups_1 = groups_2
        del sim_high
        
        pbar.update(1)
        print('Finished\n')
    pbar.close()

    if save_file:
        df.to_hdf(save_file, key='ids')
    
    return df

def trace_quantity(sim_base, trace_df=None, quantity='Mvir(M_sol)', ahf_quantity=None, 
                   milky_way=1, ahf_dir=None, **kwargs):
    """Find the specified quantities halo has attained
    
    sim_base - the base directory of the simulation
    trace_df(default None) - if None, will trace through snapshots
        or, pandas DataFrame with matched data (from trace_halos())
        if string, will attempt to read_hdf on string
    quantity - a string or list of strings corresponding to amiga.stat file headers
    ahf_quantity - a string or list of strings corresponding to AHF_halos file headers
    ahf_dir - if not None, an extra (relative) path to append for the AHF catalogs
    
    **kwargs:
    grplist and steplist: specific halos at lowest-z step or specific
    steps to match, see trace_halos()
    """
    
    sim_base = paths.clean_simbase(sim_base)
    if trace_df is None:
            
        test_df = glob.glob(sim_base + '/*trace*hdf*')
        if len(test_df) == 0:
            test_df = glob.glob(sim_base + '*/*trace*hdf*')
        if len(test_df) > 0:
            trace_df = pd.read_hdf(test_df[-1])
            print('trace_df found, using ' + test_df[-1] + '\n')
        else:
            trace_df = trace_halos(sim_base, **kwargs)
    else:
        try:
            trace_df = pd.read_hdf(trace_df)
        except:
            pass

    if isinstance(quantity, str):
        quantity = np.array([quantity])
    else:
        quantity = np.array(quantity)
   
    if ahf_quantity is not None:
        if isinstance(ahf_quantity, str):
            ahf_quantity = np.array([ahf_quantity])
        else:
            ahf_quantity = np.array(ahf_quantity)

    quantity_dfs = []
    for i in range(len(quantity)):
        quantity_dfs.append(pd.DataFrame(index=trace_df.index, 
                               columns=np.concatenate(([trace_df.index.name], trace_df.columns.values))))
    if ahf_quantity is not None:
        for i in range(len(ahf_quantity)):
            quantity_dfs.append(pd.DataFrame(index=trace_df.index, 
                               columns=np.concatenate(([trace_df.index.name], trace_df.columns.values))))

    for i, step in enumerate(quantity_dfs[0].columns):
        skip = False
        filename = paths.step_path(sim_base, step)
        if filename is None:
            continue
        try:
            #amiga_df = pd.read_csv(filename + '.amiga.stat', delim_whitespace=True, index_col='Grp')
            if milky_way is None:
                nearest = None
            elif i == 0:
                nearest = milky_way
            else:
                nearest = trace_df.loc[milky_way, step]
                if np.isnan(nearest) or nearest == -1:
                    nearest = None
            amiga_df = amah.read_amiga(pb.load(filename), nearest=nearest, ahf_dir=ahf_dir)

            if ahf_quantity is not None:
                ahf_df = amah.read_ahf(pb.load(filename), ahf_dir=ahf_dir)
                # check a few random halos to make sure the files match up
                check_ids = amiga_df.sample(10).index
                for grp in check_ids:
                    assert amiga_df.loc[grp, 
                                        'N_tot'] == ahf_df.iloc[grp - 1, 
                                                                ahf_df.columns.get_loc('npart(5)')]
        except IOError:
            skip = True
            print('Missing amiga.stat or ahf file for ' + filename + '\n')
        if i == 0:
            for j, key in enumerate(quantity):
                quantity_dfs[j][step] = amiga_df.loc[trace_df.index, key].values
            if ahf_quantity is not None:
                for j, key in enumerate(ahf_quantity):
                    quantity_dfs[j + len(quantity)][step] = ahf_df.iloc[trace_df.index - 1, ahf_df.columns.get_loc(key)].values
        else:
            if not skip:
                #traced_halos = (trace_df[step] > -1).values
                #current_step_progenitors = trace_df.loc[traced_halos, step].values
                traced_halos = trace_df.loc[(trace_df[step] > -1), step]
                traced_halos = traced_halos.loc[pd.Index(traced_halos.values).isin(amiga_df.index)]
                #overlap_filt = pd.Index(current_step_progenitors).isin(amiga_df.index)
                #traced_halos = traced_halos & overlap_filt
                #current_step_progenitors = current_step_progenitors[overlap_filt]
                for j, key in enumerate(quantity):
                    # separated into 2 lines for debugging purposes
                    new_vals = amiga_df.loc[traced_halos.values, key].values
                    quantity_dfs[j].loc[traced_halos.index, step] = new_vals
                if ahf_quantity is not None:
                    for j, key in enumerate(ahf_quantity):
                        new_vals = ahf_df.iloc[traced_halos.values.astype(int) - 1, ahf_df.columns.get_loc(key)].values
                        quantity_dfs[j + len(quantity)].loc[traced_halos.index, step] = new_vals
                        #quantity_dfs[j + len(quantity)].loc[traced_halos.index, step] = ahf_df.iloc[traced_halos.values - 1, ahf_df.columns.get_loc(key)].values

    for i in range(len(quantity_dfs)):
        quantity_dfs[i].index = quantity_dfs[i].index.rename('Grp')
    return [df.astype(float, errors='ignore') for df in quantity_dfs]


# NS 06232025: added test functions for catalog info
def test_halo_catalogs(sim_base, ahf_dir=None, **kwargs):
    """Test all snapshots to see which ones have working halo catalogs
    
    Will attempt to load halo catalogs for each timestep and report which ones fail.
    Returns a list of working steps that can be used with trace_halos.
    
    sim_base - the base directory of the simulation you wish to test
    ahf_dir(default None) - if not None, an extra (relative) path to append for the AHF catalogs
    
    Returns:
    working_steps - list of steps that successfully loaded halo catalogs
    """
    
    all_steplist = paths.list_steps(sim_base)
    print(f"Testing {len(all_steplist)} snapshots for working halo catalogs...")
    
    # First filter by ahf_dir if specified (same logic as trace_halos)
    if ahf_dir is not None:
        temp_steplist = []
        for step in all_steplist:
            step_path = Path(step)
            ahf_path = step_path.parent / ahf_dir
            
            if ahf_path.exists() and ahf_path.is_dir():
                # Check if this directory contains AHF_halos files
                ahf_files = list(ahf_path.glob('*AHF_halos*'))
                if len(ahf_files) >= 1:
                    temp_steplist.append(step)
                else:
                    print(f"Skipping {step} - no valid AHF files found")
            else:
                print(f"Skipping {step} - no valid AHF directory found")

        print(f"Found {len(temp_steplist)} steps with ahf_dir containing AHF catalogs")
        steplist = temp_steplist
    else:
        steplist = all_steplist
    
    working_steps = []
    failed_steps = []
    
    for i, step in enumerate(steplist):
        step_name = step[-6:]  # Get the 6-digit step number
        print(f"Testing step {step_name} ({i+1}/{len(steplist)})... ", end="")
        
        try:
            # Load the simulation
            sim = pb.load(step)
            
            # Try to load the halo catalog
            if ahf_dir is not None:
                pth = Path(sim.filename)
                
                # Find AHF files
                ahf_files = list(pth.parent.glob(f'{ahf_dir}/*AHF_halos'))
                if len(ahf_files) == 0:
                    print(f"FAILED - No AHF_halos files found in {pth.parent / ahf_dir}")
                    failed_steps.append((step, "No AHF files found"))
                    continue
                
                # Try different ahf_basename constructions
                ahf_full_path = str(ahf_files[0])
                
                # Method 1: Remove .AHF_halos extension
                ahf_basename1 = ahf_full_path.replace('halos', '')
                
                # Method 2: Remove last 10 characters (.AHF_halos)
                ahf_basename2 = ahf_full_path[:-10]
                
                # Method 3: Just the filename without directory
                ahf_basename3 = ahf_full_path[:-9]

                # Method 4: Use the full path as basename
                ahf_basename4 = ahf_full_path

                # test opening file first
                try:
                    with open(ahf_full_path, 'rb') as f:
                        print(f"SUCCESS - AHF_halos file found: {ahf_full_path}")
                except Exception as e:
                    print(f"FAILED - Could not open AHF_halos file: {str(e)}")
                    failed_steps.append((step, f"File open error: {str(e)}"))
                    continue
                
                success = False
                for j, basename in enumerate([ahf_basename1, ahf_basename2, ahf_basename3, ahf_basename4]):
                    try:
                        groups = sim.halos(filename=basename, halo_numbers='v1')
                        print(f"SUCCESS (method {j+1}: {basename})")
                        working_steps.append(step)
                        success = True
                        break
                    except Exception as e:
                        continue
                
                if not success:
                    print(f"FAILED - Could not load halos with any basename method")
                    failed_steps.append((step, "Could not load halos"))
                        
            else:
                # Try default halo loading (no ahf_dir specified)
                try:
                    groups = sim.halos()
                    print("SUCCESS (default)")
                    working_steps.append(step)
                except Exception as e:
                    print(f"FAILED - {str(e)}")
                    failed_steps.append((step, str(e)))
            
            # Clean up
            del sim
            
        except Exception as e:
            print(f"FAILED - Could not load simulation: {str(e)}")
            failed_steps.append((step, f"Simulation load error: {str(e)}"))
    
    print(f"\n=== SUMMARY ===")
    print(f"Total snapshots tested: {len(steplist)}")
    print(f"Working snapshots: {len(working_steps)}")
    print(f"Failed snapshots: {len(failed_steps)}")
    
    if failed_steps:
        print(f"\nFailed snapshots:")
        for step, reason in failed_steps:
            print(f"  {step[-6:]}: {reason}")
    
    print(f"\nWorking snapshots can be used with:")
    print(f"steplist = {[step[-6:] for step in working_steps]}")
    
    return working_steps

def trace_halos_filtered(sim_base, test_first=True, **kwargs):
    """Wrapper around trace_halos that first tests all snapshots
    
    If test_first=True, will run test_halo_catalogs first and only use working snapshots.
    Otherwise, behaves exactly like trace_halos.
    """
    
    if test_first:
        print("Testing snapshots first...")
        working_steps = test_halo_catalogs(sim_base, **kwargs)
        
        if len(working_steps) == 0:
            print("No working snapshots found!")
            return None
        
        # Extract step numbers and pass to trace_halos
        working_step_numbers = [int(step.split('.')[-1]) for step in working_steps]
        kwargs['steplist'] = working_step_numbers
        print(f"\nProceeding with {len(working_steps)} working snapshots...")
    
    return trace_halos(sim_base, **kwargs)

def test_halo_catalogs_comprehensive(sim_base, ahf_dir=None, **kwargs):
    """Test all snapshots to see which ones have working simulation files AND halo catalogs
    
    This function tests both the simulation files and the halo catalogs to identify
    any corrupted or empty files that would cause the tracing to fail.
    
    sim_base - the base directory of the simulation you wish to test
    ahf_dir(default None) - if not None, an extra (relative) path to append for the AHF catalogs
    
    Returns:
    working_steps - list of steps that successfully loaded both simulation and halo catalogs
    """
    
    all_steplist = paths.list_steps(sim_base)
    print(f"Testing {len(all_steplist)} snapshots for working simulation files AND halo catalogs...")
    
    # First filter by ahf_dir if specified (same logic as trace_halos)
    if ahf_dir is not None:
        temp_steplist = []
        for step in all_steplist:
            for subdir in Path(step).parent.iterdir():
                if str(subdir)[-len(ahf_dir):] == ahf_dir:
                    if sum([str(subpath).endswith('AHF_halos') for subpath in subdir.iterdir()]) >= 1:
                        temp_steplist.append(step)
        steplist = temp_steplist
    else:
        steplist = all_steplist
    
    working_steps = []
    failed_steps = []
    
    for i, step in enumerate(steplist):
        print(f"\n--- Testing snapshot {i+1}/{len(steplist)}: {step[-6:]} ---")
        
        # Test 1: Can we load the simulation file?
        try:
            print("1. Loading simulation file...")
            sim = pb.load(step)
            print(f"   SUCCESS - Loaded {sim}")
            
            # Test 2: Can we access basic particle data?
            print("2. Testing particle data access...")
            try:
                n_particles = len(sim)
                print(f"   SUCCESS - {n_particles} particles")
                
                # Test 3: Can we access the order array (this is where the error occurs)?
                print("3. Testing particle order array...")
                try:
                    test_order = sim['iord'][:10]  # Just test first 10 particles
                    print(f"   SUCCESS - Order array accessible")
                    
                    # Test 4: Can we load halo catalogs?
                    print("4. Testing halo catalog loading...")
                    if ahf_dir is not None:
                        pth = Path(sim.filename)
                        ahf_files = list(pth.parent.glob(f'{ahf_dir}/*AHF_halos'))
                        
                        if len(ahf_files) == 0:
                            print("   FAILED - No AHF_halos files found")
                            failed_steps.append((step, "No AHF files found"))
                            continue
                        
                        # Try to load halos
                        ahf_basename = str(ahf_files[0])[:-5]  # Remove '.AHF_'
                        try:
                            groups = sim.halos(halo_numbers='v1', filename=ahf_basename)
                            print(f"   SUCCESS - Loaded {len(groups)} halo groups")
                            working_steps.append(step)
                            
                        except Exception as e:
                            print(f"   FAILED - Halo loading error: {str(e)}")
                            failed_steps.append((step, f"Halo loading error: {str(e)}"))
                    else:
                        # Try default halo loading
                        try:
                            groups = sim.halos()
                            print(f"   SUCCESS - Loaded {len(groups)} halo groups")
                            working_steps.append(step)
                        except Exception as e:
                            print(f"   FAILED - Default halo loading error: {str(e)}")
                            failed_steps.append((step, f"Default halo loading error: {str(e)}"))
                            
                except Exception as e:
                    print(f"   FAILED - Order array error: {str(e)}")
                    failed_steps.append((step, f"Order array error: {str(e)}"))
                    
            except Exception as e:
                print(f"   FAILED - Particle data error: {str(e)}")
                failed_steps.append((step, f"Particle data error: {str(e)}"))
                
        except Exception as e:
            print(f"   FAILED - Simulation loading error: {str(e)}")
            failed_steps.append((step, f"Simulation loading error: {str(e)}"))
    
    print(f"\n=== COMPREHENSIVE TEST SUMMARY ===")
    print(f"Total snapshots tested: {len(steplist)}")
    print(f"Working snapshots (both sim + halos): {len(working_steps)}")
    print(f"Failed snapshots: {len(failed_steps)}")
    
    if failed_steps:
        print(f"\nFailed snapshots and reasons:")
        for step, reason in failed_steps:
            print(f"  {step[-6:]}: {reason}")
    
    print(f"\nWorking snapshots can be used with:")
    print(f"steplist = {[step[-6:] for step in working_steps]}")
    
    return working_steps