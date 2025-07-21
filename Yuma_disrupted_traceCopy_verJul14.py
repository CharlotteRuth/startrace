#This is the code to trace the star particles of disrupted/surviving galaxies
import pynbody as pb
import numpy as np
import pandas as pd
import glob
import os
import h5py
import tables
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio
from natsort import natsorted
import importlib.util
import sys
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
import ast
import matplotlib.patches as patches
os.chdir('/home/takeichi/MAP/halo_tracing_code_Nithun')
import halo_trace as ht
from halo_trace import tracing
file_path = '/home/takeichi/MAP/Code_Yuma_2025/Code/star_trace_Yuma.py'
module_name = 'star_trace'

spec = importlib.util.spec_from_file_location(module_name, file_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

pb.config['halo-class-priority'] = ['HaloNumberCatalogue', 'AHFCatalogue',
  'AmigaGrpCatalogue',
  'VelociraptorCatalogue',
  'SubFindHDFHaloCatalogue',
  'RockstarCatalogue', 
  'SubfindCatalogue',
  'NewAdaptaHOPCatalogue',
  'NewAdaptaHOPCatalogueFullyLongInts',
  'AdaptaHOPCatalogue',
  'HOPCatalogue',
  'Gadget4SubfindHDFCatalogue',
  'ArepoSubfindHDFCatalogue',
  'TNGSubfindHDFCatalogue']




def sim_base(simulation_num):
    """
    Input:
        simulation_num (int): simulation number. note that you have to remove h. i.e. h329->329

    Output:
        simbase: where data of star particles at each snapshots are saved
        tracebase: where the traced halo id are saved

    Purpose:
        Make the path to the datafile
    """
    simbase = f'/data/Sims/h{simulation_num}.cosmo50PLK.3072g/h{simulation_num}.cosmo50PLK.3072gst5HbwK1BH/snapshots_200crit_h{simulation_num}/'
    tracebase = f'/data/Sims/h{simulation_num}.cosmo50PLK.3072g/h{simulation_num}.cosmo50PLK.3072gst5HbwK1BH/'
    if simulation_num == 148:
        simbase = f'/data/Sims/h{simulation_num}.cosmo50PLK.3072g/h{simulation_num}.cosmo50PLK.3072g3HbwK1BH/snapshots_200crit_h{simulation_num}/'
        tracebase = f'/data/Sims/h{simulation_num}.cosmo50PLK.3072g/h{simulation_num}.cosmo50PLK.3072g3HbwK1BH/'
    return simbase, tracebase

def trace_haloid(simulation_num, latestsnapshot, latesthaloid,
                steplist = ['004096', '004032', '003936', '003840', '003744', '003648', '003606', '003552', '003456', '003360', '003264', '003195', '003168', '003072', '002976', '002880', '002784', '002688', '002592', '002554', '002496', '002400', '002304', '002208', '002112', '002088', '002016', '001920', '001824', '001740', '001728', '001632', '001536', '001475', '001440', '001344', '001269', '001248', '001152', '001106', '001056', '000974', '000960', '000864', '000776', '000768', '000672', '000637', '000576', '000480', '000456', '000384', '000347', '000288', '000275', '000225', '000192', '000188', '000139', '000107', '000096', '000071']
                ):
    """
    Input:
        simulation_num (int): simulation number. note that you have to remove h. i.e. h329->329
        latestsnapshot (int): latest snapshot number before the halo is disrupted
        latesthaloid (int): latest halo id before the halo is disrupted
        steplist (list, optional): list of snapshot numbers. don't need to change

    Output:
        path to the halo id trace file
    Purpose:
        trace halo id
    """
    simbase, tracebase = sim_base(simulation_num)
    ahf_dir = 'ahf_200'
    halogrp=np.array([latesthaloid])
    target_str = f"{latestsnapshot:06d}"
    try:
        idx = steplist.index(target_str)
        result_list = steplist[idx:]
    except ValueError:
        result_list = []
        print(f"cannot find snapshot{latestsnapshot}")
    snapshotlist=result_list
    # use trace_halos function from tracingCopy1.py
    try:
        trace = tracing.trace_halos(sim_base=tracebase,
                               grplist=halogrp, steplist=result_list,
                               min_ntot=500)
    except Exception as e:
        print(f"Fail to trace the halo id. Detail: {e}")

    disruptedhalo_trace_files = f"/data/Sims/h{simulation_num}.cosmo50PLK.3072g/h{simulation_num}.cosmo50PLK.3072gst5HbwK1BH/h{simulation_num}.cosmo50PLK.3072gst5HbwK1BH.{target_str}/h{simulation_num}.cosmo50PLK.3072gst5HbwK1BH.{target_str}.trace_back3_{latesthaloid}.hdf5"
    disruptedhaloid = pd.read_hdf(disruptedhalo_trace_files, index_col=0)
    #print(disruptedhaloid)
    disruptedhaloid = disruptedhaloid.rename(index={f'{latestsnapshot:06d}{latesthaloid}': f'{latesthaloid}'})
    #print(disruptedhaloid)
    disruptedhaloid.to_hdf(disruptedhalo_trace_files, key='df', mode='w')
    a = disruptedhalo_trace_files
    return a


def mainhaloid_trace(simulation_num,
                    steplist = ['004096', '004032', '003936', '003840', '003744', '003648', '003606', '003552', '003456', '003360', '003264', '003195', '003168', '003072', '002976', '002880', '002784', '002688', '002592', '002554', '002496', '002400', '002304', '002208', '002112', '002088', '002016', '001920', '001824', '001740', '001728', '001632', '001536', '001475', '001440', '001344', '001269', '001248', '001152', '001106', '001056', '000974', '000960', '000864', '000776', '000768', '000672', '000637', '000576', '000480', '000456', '000384', '000347', '000288', '000275', '000225', '000192', '000188', '000139', '000107', '000096', '000071']
                    ):
    """
    Input:
        simulation_num (int): simulation number. note that you have to remove h. i.e. h329->329
        steplist (list, optional): list of snapshot numbers. don't need to change

    Output:
        path to the main halo id trace file
        
    Purpose:
        trace main halo id    
    """
    simbase, tracebase = sim_base(simulation_num)
    halogrp=np.array([1])
    #ahf_dir = 'ahf_200'
    # try:
    trace = tracing.trace_halos(sim_base=tracebase, ahf_dir="ahf_200",
                               grplist=halogrp, steplist=steplist,
                               min_ntot=500)
    # except Exception as e:
        # print(f"Fail to trace the halo id. Detail: {e}")
        
    mainhalo_trace_files = f"/data/Sims/h{simulation_num}.cosmo50PLK.3072g/h{simulation_num}.cosmo50PLK.3072gst5HbwK1BH/h{simulation_num}.cosmo50PLK.3072gst5HbwK1BH.004096/h{simulation_num}.cosmo50PLK.3072gst5HbwK1BH.004096.trace_back3_{halogrp[0]}.hdf5"
    if simulation_num == 148:
        mainhalo_trace_files = f"/data/Sims/h{simulation_num}.cosmo50PLK.3072g/h{simulation_num}.cosmo50PLK.3072g3HbwK1BH/h{simulation_num}.cosmo50PLK.3072g3HbwK1BH.004096/h{simulation_num}.cosmo50PLK.3072g3HbwK1BH.004096.trace_back3_{halogrp[0]}.hdf5"
    mainhaloid = pd.read_hdf(mainhalo_trace_files, index_col=0)
    #print(mainhaloid)
    return mainhalo_trace_files



def make_datafolder(simulation_num, latestsnapshot, latesthaloid):
    """
    Input:
        simulation_num (int): simulation number. note that you have to remove h. i.e. h329->329
        latestsnapshot (int): latest snapshot number before the halo is disrupted
        latesthaloid (int): latest halo id before the halo is disrupted

    Output/Purpose:
        Make the folder to store information
         Return the path to the datafolder. This will be /home/takeichi/MAP/Results/simulation_num/latestsnapshot(6d)/latesthaloid
        i.e. make_datafolder(329,864,8) make the datafolder /home/takeichi/MAP/Results/329/000864_8
    """
    datafolder0 = '/home/takeichi/MAP/Results'
    data_dir0 = os.path.join(datafolder0, f"{simulation_num}")
    os.makedirs(data_dir0, exist_ok=True)
    datafolder1 = f'/home/takeichi/MAP/Results/{simulation_num}'
    data_dir1 = os.path.join(datafolder1, f"{latestsnapshot:06d}_{latesthaloid}")
    os.makedirs(data_dir1, exist_ok=True)
    datafolder =  f'/home/takeichi/MAP/Results/{simulation_num}/{latestsnapshot:06d}_{latesthaloid}'
    return datafolder
    

def traceback(simulation_num, latestsnapshot, latesthaloid, dmainhalo,position_from_right):
    """
    Input:
        simulation_num (int): simulation number. note that you have to remove h. i.e. h329->329
        latestsnapshot (int): latest snapshot number before the halo is disrupted
        latesthaloid (int): latest halo id before the halo is disrupted
        dmainhalo: csv file which includes the coordinate of COM of the main halo at each snapshot
        ### Note ###
            Run the main halo id trace before running this code.
    Output:
        N/A (There is no output but this code made some files)

    Purpose:
        Trace the star particles when the galaxies are surviving. Use trace_copy1.py. Since this makes h5 file, we load the data and make csv files.
    """
    simbase, tracebase = sim_base(simulation_num)
    disruptedhalo_trace_files = trace_haloid(simulation_num, latestsnapshot, latesthaloid)
    print("--- finished tracing halo id of disrupted halo ---")
    datafolder = make_datafolder(simulation_num, latestsnapshot, latesthaloid)
    halos_grpNow = [latesthaloid]
    module.main(halos_grpNow=halos_grpNow, sim_base=simbase, 
            trace_file=disruptedhalo_trace_files, 
            data_folder=datafolder, starting_snapshot = position_from_right)
    #star_file=f"{datafolder}/star_trace/{halos_grpNow[0]:05d}/particle_data.h5"
    try:
        first_halo_file = f"{datafolder}/star_trace/{halos_grpNow[0]:05d}/particle_data.h5"
        with h5py.File(first_halo_file, 'r') as f:
            snapshot_list = sorted(f.keys(), key=int)
        print(f"Found {len(snapshot_list)} snapshots to process.")
    except Exception as e:
        print(f"[ERROR] Could not read snapshot list from {first_halo_file}. Exiting. Details: {e}")
        exit() # Exit if we can't get the snapshot list
    
    all_particles_data = []

    for halo_id in halos_grpNow:
        file_path = f"{datafolder}/star_trace/{halo_id:05d}/particle_data.h5"
        print(f"Processing Halo ID: {halo_id}")

        try:
            with h5py.File(file_path, "r") as f:
                for snap in snapshot_list:
                    # Get COM from the cache instead of recalculating
                    a=int(snap)
                    values_array = dmainhalo.loc[dmainhalo['snapshot'] == a, ['0', '1', '2']].values
                    
                    xmain, ymain, zmain = values_array[0][0],values_array[0][1],values_array[0][2]

                    if xmain is None:
                        continue # Skip if COM failed or snapshot not in this file

                    group = f[snap]
                    x_particles, y_particles, z_particles = group['x'][:], group['y'][:], group['z'][:]
                    mass_particles, tform_particles = group['mass'][:], group['tform'][:]
                    num_particles = len(x_particles)

                    if num_particles == 0:
                        continue

                    total_mass = np.sum(mass_particles)
                    x_com = np.sum(x_particles * mass_particles) / total_mass - xmain
                    y_com = np.sum(y_particles * mass_particles) / total_mass - ymain
                    z_com = np.sum(z_particles * mass_particles) / total_mass - zmain

                    for i in range(num_particles):
                        all_particles_data.append({
                            "snapshot": int(snap), "halo_id": halo_id, "particle_mass": mass_particles[i],
                            "particle_tform": tform_particles[i], "particle_x": x_particles[i],
                            "particle_y": y_particles[i], "particle_z": z_particles[i],
                            "halo_x_com": x_com, "halo_y_com": y_com, "halo_z_com": z_com,
                        })

        except Exception as e:
            print(f"--> [ERROR] Failed on halo {halo_id}. Details: {e}")
            continue


    if not all_particles_data:
        print("\nWarning: No data was processed. CSV file will not be created.")
    else:
        print("\nConverting results to DataFrame...")
        df = pd.DataFrame(all_particles_data)
        df.sort_values(by=["halo_id", "snapshot"], inplace=True)
        output_path = os.path.join(datafolder, f"halo_star_dist{latestsnapshot:06d}{latesthaloid}.csv")
        try:
            df.to_csv(output_path, index=False)
            print(f"　Successfully saved tidy data to: {output_path}")
        except Exception as e:
            print(f"--> [ERROR] Failed to save CSV file. Details: {e}")



def COM_main_halo(simulation_num, snapshot, mainhaloid):
    simbase, tracebase = sim_base(simulation_num)
    try:
        snapshot_num = int(snapshot)
        halodata=pb.load(f"{simbase}h{simulation_num}.cosmo50PLK.3072gst5HbwK1BH.{snapshot_num:06d}")
    except Exception as e:
        print(f"file cannot be loaded, {e}")
    boxcen = halodata.properties['boxsize'].in_units('kpc a')/2
    #mainhalo_trace_files = mainhaloid_trace(simulation_num)
    #mainhaloid = pd.read_hdf(mainhalo_trace_files, index_col=0)
    mainhaloid = mainhaloid
    
    if snapshot == 4096:
        h=halodata.halos(halo_numbers='v1')
        x_org=h[1].properties['Xc']
        y_org=h[1].properties['Yc']
        z_org=h[1].properties['Zc']        
        x,y,z=(x_org/halodata.properties['h']-boxcen)*halodata.properties['a'],(y_org/halodata.properties['h']-boxcen)*halodata.properties['a'],(z_org/halodata.properties['h']-boxcen)*halodata.properties['a']
    else:
        h=halodata.halos(halo_numbers='v1')
        x_org=h[mainhaloid[f"{snapshot:06d}"].iloc[0]].properties['Xc']
        y_org=h[mainhaloid[f"{snapshot:06d}"].iloc[0]].properties['Yc']
        z_org=h[mainhaloid[f"{snapshot:06d}"].iloc[0]].properties['Zc']  
        x,y,z=(x_org/halodata.properties['h']-boxcen)*halodata.properties['a'],(y_org/halodata.properties['h']-boxcen)*halodata.properties['a'],(z_org/halodata.properties['h']-boxcen)*halodata.properties['a']
    return x,y,z

def COM_halo(simulation_num, snapshot, haloid, latestsnapshot, latesthaloid, disruptedhaloid):
    simbase, tracebase = sim_base(simulation_num)
    try:
        haloid=int(haloid)
        snapshot_num = int(snapshot)
        halodata=pb.load(f"{simbase}h{simulation_num}.cosmo50PLK.3072gst5HbwK1BH.{snapshot_num:06d}")
    except Exception as e:
        print(f"file cannot be loaded, {e}")
    boxcen = halodata.properties['boxsize'].in_units('kpc a')/2

    #disruptedhalo_trace_files = trace_haloid(simulation_num, latestsnapshot, latesthaloid)
    #disruptedhaloid = pd.read_hdf(disruptedhalo_trace_files, index_col=0)
    disruptedhaloid = disruptedhaloid
    if snapshot==latestsnapshot:
        #halodata.physical_units()
        h=halodata.halos(halo_numbers='v1')
        x_org=h[latesthaloid].properties['Xc']
        y_org=h[latesthaloid].properties['Yc']
        z_org=h[latesthaloid].properties['Zc']
        x,y,z=x_org/halodata.properties['h']-boxcen,y_org/halodata.properties['h']-boxcen,z_org/halodata.properties['h']-boxcen
        x,y,z=(x_org/halodata.properties['h']-boxcen)*halodata.properties['a'],(y_org/halodata.properties['h']-boxcen)*halodata.properties['a'],(z_org/halodata.properties['h']-boxcen)*halodata.properties['a']
    else:
        #halodata.physical_units()
        h=halodata.halos(halo_numbers='v1')
        x_org=h[disruptedhaloid.loc[str(haloid), f"{snapshot:06d}"]].properties['Xc']
        y_org=h[disruptedhaloid.loc[str(haloid), f"{snapshot:06d}"]].properties['Yc']
        z_org=h[disruptedhaloid.loc[str(haloid), f"{snapshot:06d}"]].properties['Zc']
        x,y,z=x_org/halodata.properties['h']-boxcen,y_org/halodata.properties['h']-boxcen,z_org/halodata.properties['h']-boxcen
        x,y,z=(x_org/halodata.properties['h']-boxcen)*halodata.properties['a'],(y_org/halodata.properties['h']-boxcen)*halodata.properties['a'],(z_org/halodata.properties['h']-boxcen)*halodata.properties['a']
    return x,y,z


def Rvir_halo(simulation_num, snapshot, haloid, latestsnapshot, latesthaloid, disruptedhaloid):
    simbase, tracebase = sim_base(simulation_num)
    try:
        haloid=int(haloid)
        snapshot_num = int(snapshot)
        halodata=pb.load(f"{simbase}h{simulation_num}.cosmo50PLK.3072gst5HbwK1BH.{snapshot_num:06d}")
    except Exception as e:
        print(f"file cannot be loaded, {e}")
        
    #disruptedhalo_trace_files = trace_haloid(simulation_num, latestsnapshot, latesthaloid)
    #disruptedhaloid = pd.read_hdf(disruptedhalo_trace_files, index_col=0)
    disruptedhaloid = disruptedhaloid
    
    if snapshot==latestsnapshot:
        #halodata.physical_units()
        h=halodata.halos(halo_numbers='v1')
        Rvir=h[latesthaloid].properties['Rvir']/halodata.properties['h']*halodata.properties['a'] 
    else:
        #halodata.physical_units()
        h=halodata.halos(halo_numbers='v1')
        Rvir=h[disruptedhaloid.loc[str(haloid), f"{snapshot:06d}"]].properties['Rvir']/halodata.properties['h']*halodata.properties['a']       
    return Rvir
    

def track_star_particles(
    simulation_num,
    latestsnapshot: int,
    latesthaloid: int,
    snapshot_list: list = ['004096', '004032', '003936', '003840', '003744', '003648', '003606', '003552', '003456', '003360', '003264', '003195', '003168', '003072', '002976', '002880', '002784', '002688', '002592', '002554', '002496', '002400', '002304', '002208', '002112', '002088', '002016', '001920', '001824', '001740', '001728', '001632', '001536', '001475', '001440', '001344', '001269', '001248', '001152', '001106', '001056', '000974', '000960', '000864', '000776', '000768', '000672', '000637', '000576', '000480', '000456', '000384', '000347', '000288', '000275', '000225', '000192', '000188', '000139', '000107', '000096', '000071'],
    #data_path: str = '/data/Sims/h329.cosmo50PLK.3072g/h329.cosmo50PLK.3072gst5HbwK1BH/snapshots_200crit_h329/',
    #filename_template: str = 'h329.cosmo50PLK.3072gst5HbwK1BH.',
    output_csv_path: str = 'tracked_star_particles',
    #outputdir: str = '/home/takeichi/MAP/Datafiles/Disrupted_galaxies'
):
    """
    Args:
        latestsnapshot (int): the number of the snapshot which I start tracing
        latesthaloid (int): the target halo id at the first snapshot
                       note that this latesthaloid only works for the first snapshot
        snapshot_list (list): list of the snapshot number
                                         default is ['004096', '004032', '003936', '003840', '003744', '003648', '003606', '003552', '003456', '003360', '003264', '003195', '003168', '003072', '002976', '002880', '002784', '002688', '002592', '002554', '002496', '002400', '002304', '002208', '002112', '002088', '002016', '001920', '001824', '001740', '001728', '001632', '001536', '001475', '001440', '001344', '001269', '001248', '001152', '001106', '001056', '000974', '000960', '000864', '000776', '000768', '000672', '000637', '000576', '000480', '000456', '000384', '000347', '000288', '000275', '000225', '000192', '000188', '000139', '000107', '000096', '000071']
        data_path (str, optional): path to the snapshot data  
                                         default is '/data/Sims/h329.cosmo50PLK.3072g/h329.cosmo50PLK.3072gst5HbwK1BH/snapshots_200crit_h329/'
        filename_template (str, optional): snapshot name template
                                 ex: 'h329.cosmo50PLK.3072gst5HbwK1BH.{snapshot_num:06d}' -> 'h329.cosmo50PLK.3072gst5HbwK1BH.'
                                 
        output_csv_path (str, optional): path to the output csv
                                         default is 'tracked_star_particles'
        outputdir (str, optional): name of out put folder
                                         default is '/home/takeichi/MAP/Datafiles/Disrupted_galaxies'
    """
    data_path, tracebase = sim_base(simulation_num)
    filename_template = f'h{simulation_num}.cosmo50PLK.3072gst5HbwK1BH.'
    outputdir = make_datafolder(simulation_num, latestsnapshot, latesthaloid)
    
    print(f"--- getting star particles form halo {latesthaloid} in snapshot {latestsnapshot} ---")

    # 1. Load the star snapshot and identify the star iord which we are tracing
    try:
        # formatting filename
        start_filename = f'{filename_template}{latestsnapshot:06d}'
        start_filepath = os.path.join(data_path, start_filename)

        # load the snapshot
        s_initial = pb.load(start_filepath)
        s_initial.physical_units()

        # load the halo
        halos=s_initial.halos(halo_numbers='v1')
        
        # get the iord of the stars in designated halo
        initial_star_iords = halos[latesthaloid].s['iord']
        print(f"In halo {latesthaloid}, we find {len(initial_star_iords)} star particles.")

    except Exception as e:
        print(f"Error: fail to load the file or snapshot. Detail: {e}")
        return

    # List to preserve the data we get
    all_particle_data = []

    print("\n--- start tracing stars through snapshots ---")
    # modify snapshotlist
    start_str = f"{latestsnapshot:06d}"
    try:
        # find the index of the starting snapshot
        start_index = snapshot_list.index(start_str)
        # get the snapshot which we search for
        cut_list = snapshot_list[:start_index]
        # inverse
        snapshot_list = cut_list[::-1]
        print(f"We look for snapshots {snapshot_list}")
    
    except Exception as e:
        print(f"Error: We cannot find the starting snapshot in snapshot list. Detail: {e}")

    # 2. Loop the snapshots and trace stars
    for snap_num_str in snapshot_list:
        try:
            snap_num = int(snap_num_str)
            print(f"...processing: snapshot {snap_num}...")

            # make the file path and load snapshot
            filename = f"{filename_template}{snap_num:06d}"
            filepath = os.path.join(data_path, filename)
            s_next = pb.load(filepath)
            s_next.physical_units()

            # Use iord to find the star particles from the current snapshot
            stars=s_next.s
            #debug
            #print(f"  -> debug info: 'stars' type is {type(stars)}.")
            #print(f"  -> debug info: 'stars' available keys are: {stars.loadable_keys()}.")

            
            try:
                mask = np.isin(stars['iord'], initial_star_iords)
                tracked_stars = stars[mask]
            except TypeError as te:
                #debug
                #print(f"  -> error identify: While accessing 'stars['iord']', TypeError occurs.")
                #print(f"  -> Detail: {te}")
                continue 


            if len(tracked_stars) == 0:
                print(f"  -> in snapshot {snap_num_str}, we cannot find any star particles")
                continue
            
            # Get the data of the particles
            positions = tracked_stars['pos']
            velocities = tracked_stars['vel']
            iords = tracked_stars['iord']
            tforms = tracked_stars['tform']
            masses = tracked_stars['mass']
            fehs = tracked_stars['feh']

            
            for i in range(len(tracked_stars)):
                pos = positions[i]
                vel = velocities[i]
                iord = int(iords[i])
                tform = tforms[i]
                mass = masses[i]
                feh = fehs[i]
                
                all_particle_data.append({
                    'snapshot': snap_num,
                    'iord': iord,
                    'x': pos[0],
                    'y': pos[1],
                    'z': pos[2],
                    'vx': vel[0],
                    'vy': vel[1],
                    'vz': vel[2],
                    'tform': tform,
                    'mass': mass,
                    'feh': feh
                })


        except Exception as e:
            print(f"Error: While processing snapshot {snap_num_str}, we have error. Detail: {e}")
            continue

        print(f"--- Finished tracing in snapshot {snap_num_str} ---")

    if not all_particle_data:
        print("\n--- No tracing data. ---")
        return

    # 3. Output the result
    print(f"\n--- Finished Tracing ---")
    df = pd.DataFrame(all_particle_data)
    
    # sort by snapshot for clarity
    df.sort_values(by=['snapshot'], inplace=True)

    # make out put directory
    df.to_csv(os.path.join(outputdir, f"{output_csv_path}{latestsnapshot:06d}{latesthaloid}.csv"), index=False)
    print(f"Success: Data is saved")



def plot_particle_positions(
    simulation_num,
    latestsnapshot,
    latesthaloid,
    disruptedhaloid,
    dmainhalo,
    output_csv_path: str = 'tracked_star_particles'
    #csv_file_path: str = "/home/takeichi/MAP/Datafiles/Disrupted_galaxies/000864_8/tracked_star_particles.csv",
    #output_dir: str = "/home/takeichi/MAP/Datafiles/Disrupted_galaxies/000864_8"
):
    """
    Args:
        csv_file_path (str): Path to the input CSV file.
        plot_output_dir (str): Directory where the output plot images will be saved.
    """
    outputdir = make_datafolder(simulation_num, latestsnapshot, latesthaloid)
    csv_file_path = f"{outputdir}/{output_csv_path}{latestsnapshot:06d}{latesthaloid}.csv"
    output_dir=outputdir
    # --- 1. Load Data and Setup ---

    csv_file_path1 = f"{outputdir}/halo_star_dist{latestsnapshot:06d}{latesthaloid}.csv"
    
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_file_path)
        print(f"Successfully loaded data from '{csv_file_path}'.")
    except Exception as e:
        print(f"Error: File loading error for stars after disruption. Detail: {e}")
        

    try:
        # Read the CSV file into a pandas DataFrame
        df1 = pd.read_csv(csv_file_path1)
        print(f"Successfully loaded data from '{csv_file_path1}'.")
    except Exception as e:
        print(f"Error: File loading error for stars before disruption. Detail: {e}")
        



    # Define the final, fixed limits for the plots
    xlist = []
    ylist = []
    zlist = []
    
    # Get a list of unique snapshot numbers from the data
    
    # --- 2. Loop Through Each Snapshot and Plot ---
    try:
        unique_snapshots = sorted(df['snapshot'].unique())

        for snapshot in unique_snapshots:
        
            # Filter the DataFrame to get data for the current snapshot only
            snap_df = df[df['snapshot'] == snapshot]

            #fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,9))

            xstar=snap_df['x']
            ystar=snap_df['y']
            zstar=snap_df['z']
            a = int(snapshot)
            values_array = dmainhalo.loc[dmainhalo['snapshot'] == a, ['0', '1', '2']].values
                    
            x1, y1, z1 = values_array[0][0],values_array[0][1],values_array[0][2]

            if x1 is None:
                continue # Skip if COM failed or snapshot not in this file

            xlist.extend(xstar-x1)
            ylist.extend(ystar-y1)
            zlist.extend(zstar-z1)
    except Exception as e:
        print(f"Detail: {e}")
        


    for snapshot in df1["snapshot"].unique()[::-1]:
        sub = df1[df1["snapshot"] == snapshot]
        a= int(snapshot)
        values_array = dmainhalo.loc[dmainhalo['snapshot'] == a, ['0', '1', '2']].values
                    
        x1, y1, z1 = values_array[0][0],values_array[0][1],values_array[0][2]

        if x1 is None:
            continue # Skip if COM failed or snapshot not in this file
        for halo_id in sub["halo_id"].unique():
            sub_halo=sub[sub["halo_id"]==halo_id]

            x = sub_halo["particle_x"]
            x = x.to_numpy()
            y = sub_halo["particle_y"]
            y = y.to_numpy()
            z = sub_halo["particle_z"]
            z = z.to_numpy()
            mass = sub_halo["particle_mass"]
            mass = mass.to_numpy()

            x_com = sub_halo["halo_x_com"]
            x_com = x_com.to_numpy()
            y_com = sub_halo["halo_y_com"]
            y_com = y_com.to_numpy()
            z_com = sub_halo["halo_z_com"]
            z_com = z_com.to_numpy()
            x_com=x_com[0]
            y_com=y_com[0]
            z_com=z_com[0]

            
            Rvir = Rvir_halo(simulation_num, snapshot, halo_id, latestsnapshot, latesthaloid, disruptedhaloid)

            xstar_rel = x - x1
            ystar_rel = y - y1
            zstar_rel = z - z1
            xlist.extend(xstar_rel)
            ylist.extend(ystar_rel)
            zlist.extend(zstar_rel)

            xlist.append(x_com+Rvir)
            xlist.append(x_com-Rvir)
            ylist.append(y_com+Rvir)
            ylist.append(y_com-Rvir)
            zlist.append(z_com+Rvir)
            zlist.append(z_com-Rvir)

    if xlist:
        x_min, x_max = min(xlist), max(xlist)
        y_min, y_max = min(ylist), max(ylist)
        z_min, z_max = min(zlist), max(zlist)

        x_pad = (x_max - x_min) * 0.05
        y_pad = (y_max - y_min) * 0.05
        z_pad = (z_max - z_min) * 0.05

        xlim = (x_min - x_pad, x_max + x_pad)
        ylim = (y_min - y_pad, y_max + y_pad)
        zlim = (z_min - z_pad, z_max + z_pad)
        print(f"\nlimits determined:")
        print(f"xlim: {xlim}")
        print(f"ylim: {ylim}")
        print(f"zlim: {zlim}")
    else:
        print("\nNo data found to determine limits. Using default values.")
        xlim, ylim, zlim = (-100, 100), (-100, 100), (-100, 100)


    for snapshot in df1["snapshot"].unique()[::-1]:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 9))
        sub = df1[df1["snapshot"] == snapshot]
        a = int(snapshot)
        values_array = dmainhalo.loc[dmainhalo['snapshot'] == a, ['0', '1', '2']].values
        x1, y1, z1 = values_array[0][0],values_array[0][1],values_array[0][2]

        if x1 is None:
            continue # Skip if COM failed or snapshot not in this file


        for halo_id in sub["halo_id"].unique():
            sub_halo=sub[sub["halo_id"]==halo_id]

            x = sub_halo["particle_x"]
            x = x.to_numpy()
            y = sub_halo["particle_y"]
            y = y.to_numpy()
            z = sub_halo["particle_z"]
            z = z.to_numpy()
            mass = sub_halo["particle_mass"]
            mass = mass.to_numpy()

            x_com = sub_halo["halo_x_com"]
            x_com = x_com.to_numpy()
            y_com = sub_halo["halo_y_com"]
            y_com = y_com.to_numpy()
            z_com = sub_halo["halo_z_com"]
            z_com = z_com.to_numpy()
            x_com=x_com[0]
            y_com=y_com[0]
            z_com=z_com[0]

            
            Rvir = Rvir_halo(simulation_num, snapshot, halo_id, latestsnapshot, latesthaloid, disruptedhaloid)



            scatter=axes[0].scatter(x-x1, y-y1, s=5, label=f"Halo {halo_id}")
            plot_color = scatter.get_facecolor()[0]
            circle0 = patches.Circle(xy=(x_com,y_com), radius=Rvir, fill=False, ec=plot_color, lw=1, linestyle='--')
            axes[0].add_patch(circle0)
            axes[0].set_aspect('equal')

            scatter1=axes[1].scatter(x-x1, z-z1, s=5, label=f"Halo {halo_id}")
            plot_color1 = scatter1.get_facecolor()[0]
            circle1 = patches.Circle(xy=(x_com,z_com), radius=Rvir, fill=False, ec=plot_color1, lw=1, linestyle='--')
            axes[1].add_patch(circle1)
            axes[1].set_aspect('equal')
        
        axes[0].set_xlabel("x(kpc)")
        axes[0].set_ylabel("y(kpc)")
        axes[1].set_xlabel("x(kpc)")
        axes[1].set_ylabel("z(kpc)")

        axes[0].set_xlim(xlim)
        axes[0].set_ylim(ylim)
        
        axes[1].set_xlim(xlim)
        axes[1].set_ylim(zlim)
        axes[0].set_title(f"Star Distribution in xy plane at snapshot{snapshot}")
        axes[0].legend()
        axes[1].set_title(f"Star Distribution in xz plane at snapshot{snapshot}")
        axes[1].legend()
        filename=f"StarDist_{snapshot:06d}_halo{latesthaloid}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close(fig)
        print(f"save for {snapshot}")


    try:
        for snapshot in unique_snapshots:
            print(f"...Processing snapshot {snapshot}...")
        
            # Filter the DataFrame to get data for the current snapshot only
            snap_df = df[df['snapshot'] == snapshot]

            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18,9))

            xstar=snap_df['x']
            ystar=snap_df['y']
            zstar=snap_df['z']
            a = int(snapshot)
            values_array = dmainhalo.loc[dmainhalo['snapshot'] == a, ['0', '1', '2']].values
            x1, y1, z1 = values_array[0][0],values_array[0][1],values_array[0][2]

            if x1 is None:
                continue # Skip if COM failed or snapshot not in this file

            axes[0].scatter(xstar-x1, ystar-y1, s=5, label=f"disrupted stars from halo{latesthaloid}")
            axes[1].scatter(xstar-x1, zstar-z1, s=5, label=f"disrupted stars from halo{latesthaloid}")        
            axes[0].set_xlim(xlim)
            axes[0].set_ylim(ylim)
        
            axes[1].set_xlim(xlim)
            axes[1].set_ylim(zlim)
            axes[0].set_xlabel("x(kpc)")
            axes[0].set_ylabel("y(kpc)")
            axes[1].set_xlabel("x(kpc)")
            axes[1].set_ylabel("z(kpc)")
            axes[0].set_aspect('equal')
            axes[1].set_aspect('equal')
            axes[0].set_title(f"Star Distribution in xy plane at snapshot{snapshot}")
            axes[0].legend()
            axes[1].set_title(f"Star Distribution in xz plane at snapshot{snapshot}")
            axes[1].legend()

            # Save the figure
            filename=f"StarDist_{snapshot:06d}_halo{latesthaloid}.png"
            plt.savefig(os.path.join(output_dir, filename))
            plt.close(fig)
            print(f"save for {snapshot}")

        print("\n--- Plotting finished successfully! ---")
    except Exception as e:
        print(f"Detail: {e}")


def create_movie_with_imageio(
    simulation_num,
    latestsnapshot,
    latesthaloid,
    #input_dir: str = "/home/takeichi/MAP/Datafiles/Disrupted_galaxies/000864_8",
    #output_filename: str = "star_distribution_movie.mp4",
    fps: int = 10
):
    """
    Creates an MP4 video from a series of PNG images using the imageio library.
    The files are sorted alphanumerically before being added to the video.

    Args:
        input_dir (str): Path to the directory where the PNG images are saved.
        output_filename (str): Filename for the output MP4 video.
        fps (int): Frame rate of the video (frames per second).
    """
    
    input_dir=make_datafolder(simulation_num, latestsnapshot, latesthaloid)
    output_filename = f"star_distribution_movie{latestsnapshot:06d}{latesthaloid}.mp4"
    # --- 1. Get and sort the list of image files ---
    search_pattern = os.path.join(input_dir, "StarDist_*.png")
    image_files = sorted(glob.glob(search_pattern))

    if not image_files:
        print(f"Error: No PNG files matching the pattern '{os.path.basename(search_pattern)}' were found in '{input_dir}'.")
        return

    print(f"Found {len(image_files)} image files. Starting video creation.")
    
    # --- 2. Create the video ---
    output_path = os.path.join(input_dir, output_filename)
    
    # Use imageio.get_writer to create a writer object.
    # The 'with' statement ensures the writer is properly closed.
    with imageio.get_writer(output_path, fps=fps) as writer:
        for i, filename in enumerate(image_files):
            print(f"\rProcessing: Frame {i + 1}/{len(image_files)}", end="")
            # Read each image file
            image = imageio.imread(filename)
            # Append the image to the video
            writer.append_data(image)

    print(f"\n\n--- Video creation complete ---")
    print(f"Video saved to '{output_path}'.")





    
    
    
    

def run_star_trace_analysis(simulation_num, latestsnapshot, latesthaloidlist):
    output_folder = "/home/takeichi/MAP/Code_Yuma_2025/Code/Datafiles"
    com_output_path = os.path.join(output_folder, f"main_halo_com_{simulation_num}.csv")
    try:
        dmainhalo = pd.read_csv(com_output_path)
    except exception as e:
        print(f"cannot read the csv: {e}")

    for i in latesthaloidlist:
        print(f"Working on {i}")
        latesthaloid = i
        print(f"Starting full analysis for simulation {simulation_num}, halo {latesthaloid} from snapshot {latestsnapshot}...")
        # Step 0: Trace the halo id of the target star
        disruptedhalo_trace_files = trace_haloid(simulation_num, latestsnapshot, latesthaloid)
        try:
            disruptedhaloid = pd.read_hdf(disruptedhalo_trace_files, index_col=0)
        except Exception as e:
            print(f"We cannot load the data, {e}")
            disruptedhaloid = None
        
        print(disruptedhaloid)
        print("-----------------")
        # Check if the data is a DataFrame or a Series and ensure it can be treated as a Series.
        if isinstance(disruptedhaloid, pd.DataFrame):
            print("Input is a DataFrame, so converting the first row to a Series.")
            # If it's a DataFrame, extract the first row (iloc[0]) to process it as a Series.
            series_to_check = disruptedhaloid.iloc[0]
        else:
            # If it's already a Series, use it as is.
            series_to_check = disruptedhaloid

        # Filter the Series to get only the elements with positive values.
        positive_series = series_to_check[series_to_check > 0]

        # Get the index from the filtered Series.
        positive_indices = positive_series.index

        # Check if any positive indices were found.
        if not positive_indices.empty:
            # Get "the first positive index when viewed from the right".
            last_positive_index = positive_indices[-1]

            # From here, it is for debugging
            # Get all indices from the original Series as a list.
            all_indices_list = series_to_check.index.tolist()
    
            # Find the position of the target index (0-indexed from the left).
            position_from_left = all_indices_list.index(last_positive_index)
    
            # Calculate the position from the right (0-indexed) by subtracting the left position from the total count.
            # This value is used for main function later in the star tracing for surviving galaxy.
            position_from_right = len(all_indices_list) - 1 - position_from_left
    
            print(f"--- Finished Analysing the halo ids ---")
            print(f"The first positive index from the right is: {last_positive_index}")
            print(f"That index is at position {position_from_right}")

            if position_from_right <3:
                position_from_right = 3
                print("We will trace from snapshot 139")
        # if not found, print that there is no positive values
        else:
            print(f"\n[Final Result]")
            print("No positive values were found in the Series.")
            print("Inserting the initial halo state as the first column.")
            position_from_right = len(series_to_check)
            print(disruptedhaloid)
            print(f"Only the lastsnapshot is the snapshot the galaxy exists. At that point, index is {position_from_right}")

        print("-----------------")
        
        # Step 1: Trace the halo's history and its stars. This creates the primary data folder and the halo_star_dist.csv.
        print("Step 1")
        traceback(simulation_num, latestsnapshot, latesthaloid,dmainhalo,position_from_right)
        print("Step 1 finished")
        print("-----------------")
        
        # Step 2: Track the individual star particles from the initial halo (the stellar stream). This creates the tracked_star_particles.csv.
        print("Step 2")
        track_star_particles(simulation_num, latestsnapshot, latesthaloid)
        print("Step 2 finished")
        print("-----------------")
    
        # Step 3: Generate plots for each snapshot using the data from the previous steps.
        print("Step 3")
        plot_particle_positions(simulation_num, latestsnapshot, latesthaloid, disruptedhaloid, dmainhalo)
        print("Step 3 finished")
        print("-----------------")
        # Step 4: Create a movie from the generated plots.
        print("Step 4")
        create_movie_with_imageio(simulation_num, latestsnapshot, latesthaloid)
        print("Step 4 finished")
        print("-----------------")
        print(f"\n Analysis complete for simulation {simulation_num}, halo {latesthaloid}.")
        datafolder = make_datafolder(simulation_num, latestsnapshot, latesthaloid)
        print(f"Find your results in: {datafolder}")



    
    