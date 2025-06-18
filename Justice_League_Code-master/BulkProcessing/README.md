# Bulk Processing

When analyzing simulations, one rapidly encounters a fundamental problem: the analysis techniques that are useful can take a lot of to process on such large simulations. For example, to calculate the star-formation histories of every satellite in the Justice League simulations is not a quick task, even for a powerful computer. 

So, to workaround this, we write code in `.py` scripts that reads in the *raw* simulation files and returns a compiled dataset of relevant satellite properties. That is, we calculate these properties *in bulk*. These scripts can take as much as 12-18 hours to run, depending on what they are doing (and how efficiently the code is written). 

These bulk processing scripts produce the `.data` files that much of the analysis code uses. The `.data` extension is a convention used whenever we "pickle" data---that is, whenever we use the Python `pickle` module to store data. The `pickle` module is able to save Python objects to a file and reopen them without losing the same data format---you don't have to worry about whether the obeject was a list, array, SimArray, etc. For more information on how data is stored, see [Data](../Data/).

The bulk processing scripts in this directory generate a few different sets of data:

## Redshift Zero

The first of these scripts, `bulk_processing.py`, analyzes the redshift zero snapshots of the eight simulations (4 Justice League, 4 Marvel). It calculates basic properties for all satellites (at least, all satellites with stars) including metallicity, magnitude, color, gas inflow/outflow rates, etc. 

In order for this script to know what satellites to make these calculations for, i.e. what satellites are interesting, you have to provide it a list of halo IDs. Presently, the script includes a list of haloids for each of the 8 simulations; these are the haloids for satellites that contain at least 1 star particle. 

This Python script can be run directly, as long as you are on a computer that can be left running for several hours. You do not need to be on `quirm` to run this script, as the redshift 0 simulations are stored in my MathLAN directory. 

## Timestep Bulk Processing

What if you want to know more than just present-day satellite properties? This set of scripts calculates properties for satellites at each snapshot in the simulation, and stores these data. The scripts include `timescales_bulk.py` and `h###bulk_200b.py`. 

In order to calculate properties of satellites consistently over time, these scripts utilize a simulation's *merger trees*, a set of information about which satellites merge with which between each snapshot. For more information see [Merger Trees](../MergerTrees/)

For the purposes of bulk processing, the merger trees take the form of a Python `dictionary` object, defining the haloids for a given redshift 0 halo in a given simulation. For example, the `h329bulk_200b.py` file defines a dictionary called `haloids`. Within this dictionary are several lists, each assigned an ID---the redshift zero halo ID. So, if we call `haloids[11]`, then we get the list of haloids assigned to halo 7 in our redshift 0 snapshot. This list corresponds to the haloid of the satellite's progenitor galaxy at each snapshot, i.e. the main progenitor branch of the merger tree. If we call `h329_haloids[7][0]` we get the ID at z=0 and if we call `h329_haloids[7][-1]` we get the ID at the earliest snapshot available. This is a bit confusing, so the screenshot from `h329bulk_200b.py` below may make it more clear: 

![explanatory graphic](graphic.png)

Importantly, these scripts include a similar dictionary defining virial radii for each halo at each snapshot. The values in this dictionary overwrite the virial radii given by AHF for the purposes of most calculations in the script, as we noticed that AHF tend to underestimate the virial radius immediately after infall. 

The heart of these scripts is `timescales_bulk.py`. This file defines the bulk_processing function for timestep data, however, it will do nothing if run on its own. Instead, timestep bulk processing is run for the individual simulations in order to maximize efficienty (multiple can be run at once). For example, if I wanted to run the timestep bulk processing on `h242`, I could run the script `h242bulk_200b.py`, which imports the `bulk_processing` function defined in `timescales_bulk.py`. 





