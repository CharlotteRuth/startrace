#Bound fraction to Marvelous Dwarfs
import h5py
import numpy as np
import pandas as pd
from bound_fraction import compute_boundness_recursive_BFE
from morphology import local_velocity_dispersion
import astropy.units as u

def bound_fraction_task(results_df, dm_pos, dm_vel, dm_masses, star_pos, star_vel, star_masses, h5file, index_match=None):
    if index_match is None:
        index_match = np.array(range(len(star_masses)))
    dfi = results_df.index.to_numpy()[-1]

    results_df.loc[dfi, 'n_part_dm'] = np.sum(~np.isnan(dm_masses))
    results_df.loc[dfi, 'n_part'] = np.sum(~np.isnan(star_masses))
    results_df.loc[dfi, 'stellar_mass'] = np.sum(star_masses[~np.isnan(star_masses)])

    results = []
    methods = [['star', True], ['star', False], ['both', True], ['both', False]]

    for i in range(len(methods)+1):
        if i != len(methods):
            print(f'Running method {i}')
            result = compute_boundness_recursive_BFE(
                dm_pos,     dm_vel,     dm_masses,
                star_pos,   star_vel,   star_masses, 
                center_on=methods[i][0], center_vel_with_KDE=methods[i][1]
            )
            results.append(result)
            stri = f'_{i}'
            h5index = f'_method{i}'
        else:
            m_i = np.argmax([np.mean(ii[0][1]==1) for ii in results])
            result = results[m_i]
            stri = ''
            h5index = ''
        
        
        h5file[f'dm_bound{h5index}'][dfi] = result[0][0]
        h5file[f'star_bound{h5index}'][dfi][index_match] = result[0][1]
        
        results_df.loc[dfi, f'f_bound_part_dm{stri}'] = np.mean(result[0][0]==1)
        results_df.loc[dfi, f'f_bound_mass_dm{stri}'] = np.sum(dm_masses[result[0][0]==1])/np.sum(dm_masses)
        results_df.loc[dfi, f'f_bound_part{stri}'] = np.mean(result[0][1]==1)
        results_df.loc[dfi, f'f_bound_mass{stri}'] = np.sum(star_masses[result[0][1]==1])/np.sum(star_masses)
        results_df.loc[dfi, f'center_x{stri}'] = result[1][0]
        results_df.loc[dfi, f'center_y{stri}'] = result[1][1]
        results_df.loc[dfi, f'center_z{stri}'] = result[1][2]
        results_df.loc[dfi, f'center_v_x{stri}'] = result[2][0]
        results_df.loc[dfi, f'center_v_y{stri}'] = result[2][1]
        results_df.loc[dfi, f'center_v_z{stri}'] = result[2][2]

            
def calc_fbound_veldisp(ids, dirpath = '', output_path='f_bound.csv'):
    '''
    Calculates the bound fraction using energy. Output saved as csv file. 
    There are four different centering methods and the center with the maximum bound fraction is the default bound fraction.
    '''
    #We should write this to be the same file name structure across different halos
    idpath = ids
    #The h5 file should have the following keys
            # snaps, zs, 
            # star_pos, star_vel, star_mass, star_age,
            # dm_pos, dm_vel, dm_mass
            # Keys that change with time (pos, vel) should be formatted as
            # shape = (num_snaps, num_tot_particles, data_length)
    
    #There's gotta be a better way to write this.
    results_df = pd.DataFrame(
        columns=[
            'snap', 'vel_dispersion_16', 'vel_dispersion_50', 'vel_dispersion_84', 
            'n_part_dm', 'n_part', 'stellar_mass',
            'f_bound_part_dm', 'f_bound_mass_dm', 'f_bound_part', 'f_bound_mass', 
            'center_x', 'center_y', 'center_z', 'center_v_x', 'center_v_y', 'center_v_z',
            'f_bound_part_dm_0', 'f_bound_mass_dm_0', 'f_bound_part_0', 'f_bound_mass_0', 
            'center_x_0', 'center_y_0', 'center_z_0', 'center_v_x_0', 'center_v_y_0', 'center_v_z_0',
            'f_bound_part_dm_1', 'f_bound_mass_dm_1', 'f_bound_part_1', 'f_bound_mass_1', 
            'center_x_1', 'center_y_1', 'center_z_1', 'center_v_x_1', 'center_v_y_1', 'center_v_z_1',
            'f_bound_part_dm_2', 'f_bound_mass_dm_2', 'f_bound_part_2', 'f_bound_mass_2', 
            'center_x_2', 'center_y_2', 'center_z_2', 'center_v_x_2', 'center_v_y_2', 'center_v_z_2',
            'f_bound_part_dm_3', 'f_bound_mass_dm_3', 'f_bound_part_3', 'f_bound_mass_3', 
            'center_x_3', 'center_y_3', 'center_z_3', 'center_v_x_3', 'center_v_y_3', 'center_v_z_3'
        ]
    )

    with h5py.File(f'{dirpath}{idpath}.h5', 'r+') as data:
        snaps = data['snaps']
        star_masses = np.asarray(data['star_mass'])
        dm_masses = np.asarray(data['dm_mass'])
        try:
            del data["star_bound"], data["dm_bound"], data["star_vel_dsp"]
            for i in range(4):
                del data[f"star_bound_method{i}"], data[f"dm_bound_method{i}"]
        except:
            pass

        data.create_dataset("star_bound", star_masses.shape, dtype=int)
        data.create_dataset("star_vel_dsp", star_masses.shape, dtype=int)
        data.create_dataset("dm_bound", dm_masses.shape, dtype=int)
        for i in range(4):
            data.create_dataset(f"star_bound_method{i}", star_masses.shape, dtype=int)
            data.create_dataset(f"dm_bound_method{i}", dm_masses.shape, dtype=int)
        
    
        for i in range(len(snaps)):
            print(snaps[i])
            results_df.loc[i] = np.zeros(shape=57)*np.nan
            results_df.loc[i, 'snap'] = snaps[i]
            star_mass = star_masses[i]
            if np.any(~np.isnan(star_mass)):
                bound_fraction_task(
                    results_df, 
                    np.asarray(data['dm_pos'])[i], np.asarray(data['dm_vel'])[i], dm_masses[i], 
                    np.asarray(data['star_pos'])[i][~np.isnan(star_mass)], 
                    np.asarray(data['star_vel'])[i][~np.isnan(star_mass)], 
                    star_mass[~np.isnan(star_mass)],
                    data,
                    index_match = ~np.isnan(star_mass)
                )
            else:
                pass
            output = local_velocity_dispersion(
                np.asarray(data['star_pos'])[i][~np.isnan(star_mass)]*u.kpc, 
                np.asarray(data['star_vel'])[i][~np.isnan(star_mass)]*u.km/u.s, 
                return_disps=True
            )
            data['star_vel_dsp'][i][~np.isnan(star_mass)] = output[1]
            results_df.loc[i, 'vel_dispersion_16'] = output[0][0]
            results_df.loc[i, 'vel_dispersion_50'] = output[0][1]
            results_df.loc[i, 'vel_dispersion_84'] = output[0][2]
    
    results_df.to_csv(output_path)
        
calc_fbound_veldisp('storm.4096g5HbwK1BH_bn_4_2304_14_particle_data', dirpath='/home/graythoron/projects/marv_dwarf/marvel_dwarfs/')