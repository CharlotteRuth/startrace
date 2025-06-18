# Ram Pressure Stripping

In this directory lies code I've used for my Spring 2021 MAP. 
The goal of this project is to use particle tracking code to analyze ram pressure stripping and quenching in dwarf galaxies. 
Here I describe the basic structure of the code I've written.

## Base Code 

I've created a few files to store useful functions for data manipulation, calculations, etc. These files are imported into almost every other bit of code I run. 

> `base.py`

This file stores basic data manipulation functions: reading in timesteps data, timescales data, infall properties, getting the filepaths and haloids of the galaxies of interest, and more. It also defines some useful constants and sets up my `matplotlib` plotting preferences. 

> `analysis.py`

This file stores functions for reading in particle tracking data and doing some key calculations, such as identifying expelled/accreted particles. It's built on `base.py` and is imported into pretty much every notebook I have. 


## Scripts

> `particletracking.py`
- This script runs particle tracking on a particular satellite in a particular simulation, which are specified at the command line (e.g. `python particletracking.py h329 11`). 
- The code tracks gas particles, starting at the snapshot where the satellite first crosses 2 Rvir from its host and ending at redshift 0. Gas particles that are tracked are *those that are in the satellite for at least one snapshot in this range*.
- Uses as input data: simulation snapshots, `../../Data/filepaths_haloids.pickle`.
- Produces as output data: `../../Data/iords/sim_haloid.pickle`, `../../Data/tracked_particles.hdf5`.
- Note: the bash script `runall.sh` can be used to run particle tracking on multiple satellites at once, to speed things along. 

> `rampressure.py`

- This script calculates the ram pressure a satellite experiences, its gravitational restoring pressure per unit area, and a few other galaxy properties over time for each satellite. 
- Ram pressure calculations are somewhat computationally intensive, which is why this is done separate from any other analysis code. 
- Like `particletracking.py`, the simulation and satellite are specified at the command line, i.e. `python rampressure.py h148 68`. 
- Uses as input data: simulation snapshots, `../../Data/filepaths_haloids.pickle`.
- Produces as output data: `../../Data/ram_pressure.hdf5`.
- Note: the bash script `runall_rp.sh` can be used to calculate ram pressures for multiple satellites at once, to speed things along. 

> `particletracking_stars.py`
- This script tracks the stars that formed from gas particles tracked by `particletracking.py`.
- Uses as input data: simulation snapshots, `../../Data/filepaths_haloids.pickle`.
- Produces as output data: `../../Data/tracked_stars.hdf5`.
- Note: the bash script `runall_stars.sh` can be used to run particle tracking on multiple satellites at once, to speed things along. 

## Notebooks

> `ParticleTrackingPlots.ipynb`

Notebook where I produce basic plots of particle tracking data, including pathline plots and fractions plots. 

> `RamPressure.ipynb`

Notebook where I merge together particle tracking data and ram pressure data to do analysis on gas expulsion rates, etc. This notebook is where the bulk of my plots come from. 

> `ExitAngle.ipynb`

Notebook where I utilize the spatial information in the particle tracking data to study the exit angle of expelled gas particles. 

> `Consumption.ipynb` 

Notebook where I try to distinguish between gas-loss due to removal and gas-loss due to ongoing consumption by star formation. 

## Data

The `particletracking.py` script draws data directly from the simulation snapshots. To speed up the process of analyzing these satellites over time, I have stored the simulation snapshot filepaths and main progenitor haloids for each redshift 0 satellite at `/Data/filepaths_haloids.pickle`. The scripts in this directory utilize this pickle file to get satellite haloid information. 

Most of the datasets I've created as part of this project are in the form of `.hdf5` files. 
The HDF5 file format stores the output data efficiently and allows for data from all the galaixes to be stored in one file. The HDF5 file type can be read into python easily as a `pandas` DataFrame using `pandas.read_hdf(filepath, key=key)`, where the `key` option specifies the table you want. Each galaxy is saved with a key identifying its host and redshift 0 haloid, i.e. `key='h148_13'`. 

The `tracked_particles.hdf5` file is not stored on Github (as it is too large) but can be found on `quirm` at 
> `/home/akinshol/Research/Justice_League_Code/Data/tracked_particles.hdf5`


