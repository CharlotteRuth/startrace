# Luminosity Function

*Notebook: `LumFunction_F19.ipynb`.*

The code in this notebook is used to generate the satellite luminosity functions (Akins et al 2020, fig 2) of the Justice League satellite systems. 

*Notebook: `Lum_Mass_Occupation.ipynb`.*

The code in this notebook is used to generate satellite LFs, mass functions, and occupation fractions (the fraction of DM halos that are occupied by stellar disks). 

## Data used

Luminosity functions are generated from the redshift 0 dataset generated on 2019/11/29 (`/Data/z0_data/*`). Mass functions and occupation fractions are generated from the raw simulation data, since the compiled datasets ignore dark halos and instead only compute properties for "occupied" halos. 

Data for observational comparisons are written out as lists in the Jupyter notebooks, with source papers cited (for full citaiton, see Akins et al. 2020). 
A compiled dataset of SAGA hosts and satellites is stored in `SAGA_data_cleaned.csv`, for LF comparisons. 

## Output files

These notebooks produce a few figures: 

1. Luminosity function figure (Akins et al. 2020, Fig. 2) | `LF_F19.png`
2. Same luminosity function figure but with SAGA comparisons | `LF_F19_SAGA.png`
3. Luminosity function, mass function, and occupation fraction | `lum_mass_occupation.png`

and a few data products 

4. Data from luminosity function figure | `LuminosityFunction.csv`
5. Data from occupation fraction figure | `Occupation_Fraction_1dex.csv`
5. Masses and stellar content for all DM halos | `Occupation_Fraction_Data.csv` 



