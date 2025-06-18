import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import pickle
import pandas as pd
import matplotlib.cm as cm
import matplotlib as mpl
import pynbody
import math
import numpy as np
import socket
from bulk import *
import copy

# we set the global matplotlib parameters so the fonts are all consistent and serif 
mpl.rc('font',**{'family':'serif','monospace':['Palatino']})
mpl.rc('text', usetex=True)
mpl.rcParams.update({'font.size': 9})

def read_file(simname):
    ### leave this section as is
    data = []
    with open('/home/akinshol/Data/DataFiles_Archive_20191129/'+ simname + '.data', 'rb') as f:
        while True:
            try:
                data.append(pickle.load(f,encoding='latin1'))
            except EOFError:
                break

        data1 = pd.DataFrame(data)
        data1['sim'] = [simname]*len(data1)
        data1['g-r'] = 1.646*(data1['V_mag'] - data1['R_mag']) - 0.139
        return data1
    


    
fig = plt.figure(dpi=300, figsize=(7.5, 5.6))

gs = mpl.gridspec.GridSpec(3,3, height_ratios=[1,1,0.05], figure=fig)
gs.update(hspace=0.035, wspace=0.05, right=0.99, top=0.99, left=0.01, bottom=0.08)

# top row, Sonia, bottom row, Sandra

min_nstar =  50 #Minimum number of stars for
min_vmass = 1e8
min_noncontamFrac = 0.9 #Mass fraction in low mass DM particles

dm_vmin, dm_vmax = 1e2, 9e7
gas_vmin, gas_vmax = 1e2, 9e7
star_vmin, star_vmax = 1e2, 1e9

image_width = 1000 # kpc

ax_d_sonia = plt.subplot(gs[0,0])
ax_g_sonia = plt.subplot(gs[0,1])
ax_s_sonia = plt.subplot(gs[0,2])
ax_d_sandra = plt.subplot(gs[1,0])
ax_g_sandra = plt.subplot(gs[1,1])
ax_s_sandra = plt.subplot(gs[1,2])
cbax_d = plt.subplot(gs[2,0])
cbax_g = plt.subplot(gs[2,1])
cbax_s = plt.subplot(gs[2,2])


# TOP ROW: SONIA
sim = '/home/christenc/Data/Sims/h242.cosmo50PLK.3072g/h242.cosmo50PLK.3072gst5HbwK1BH/snapshots_200bkgdens/h242.cosmo50PLK.3072gst5HbwK1BH.004096'
name = 'h242'
s = pynbody.load(sim)
s.physical_units()
h = s.halos()
h_dummy = s.halos(dummy = True)

print(s.properties['boxsize'])
s.properties['boxsize'] = pynbody.units.Unit("100 Mpc")
print(s.properties['boxsize'])

print('Loaded simulation')

labels = []

i = 0

h1x = h_dummy[1].properties['Xc']
h1y = h_dummy[1].properties['Yc']
h1z = h_dummy[1].properties['Zc']


smin, smax = -image_width//2, image_width//2
print('Aligning halo 1')
pynbody.analysis.angmom.faceon(h[1])
s.physical_units()

print('Making DM image...')
im = pynbody.plot.sph.image(s.dm[pynbody.filt.Sphere('%s kpc' % str((smax-smin)))], width = '%s kpc' % str(smax-smin), cmap='Greys_r', 
                            av_z = 'rho', ret_im = True, show_cbar = False, noplot=True)

im = ax_d_sonia.imshow(im, norm=colors.LogNorm(), cmap='Greys_r', extent = [smin,smax,smin,smax], vmin=dm_vmin, vmax=dm_vmax,origin='lower')

cbar = fig.colorbar(im, cax=cbax_d, orientation="horizontal")
cbar.set_label(r'Dark Matter Density [$\mathrm{M}_{\odot}\ \mathrm{kpc}^{-3}$]')
cbar.ax.minorticks_on()

print('Making gas image...')
im = pynbody.plot.sph.image(s.gas[pynbody.filt.Sphere('%s kpc' % str((smax-smin)))], width = '%s kpc' % str(smax-smin), cmap='viridis', 
                            av_z = 'rho', ret_im = True, show_cbar = False, noplot=True)

im = ax_g_sonia.imshow(im, norm=colors.LogNorm(), cmap='viridis', extent = [smin,smax,smin,smax], vmin=gas_vmin, vmax=gas_vmax,origin='lower')

cbar = fig.colorbar(im, cax=cbax_g, orientation="horizontal")
cbar.set_label(r'Gas Density [$\mathrm{M}_{\odot}\ \mathrm{kpc}^{-3}$]')
cbar.ax.minorticks_on()

print('Making star image...')
im = pynbody.plot.sph.image(s.s[pynbody.filt.Sphere('%s kpc' % str((smax-smin)))], width = '%s kpc' % str(smax-smin), cmap='cubehelix', 
                            av_z = 'rho', ret_im = True, show_cbar = False, noplot=True)

my_cmap = copy.copy(mpl.cm.get_cmap('cubehelix')) # copy the default cmap
my_cmap.set_bad(my_cmap(0))

im = ax_s_sonia.imshow(im, norm=colors.LogNorm(), cmap=my_cmap, extent= [smin,smax,smin,smax], vmin=star_vmin, vmax=star_vmax,origin='lower')

cbar = fig.colorbar(im, cax=cbax_s, orientation="horizontal")
cbar.set_label(r'Stellar Density [$\mathrm{M}_{\odot}\ \mathrm{kpc}^{-3}$]')
cbar.ax.minorticks_on()

data = read_file(name) 

h1rvir = 0

print('Adding circles...')
for ihalo in range(1,len(h)): #len(h)-1):
    nstar = h_dummy[ihalo].properties['n_star']
    fMhires = h_dummy[ihalo].properties['fMhires']
    mass = h_dummy[ihalo].properties['mass']
    valid_halo = (nstar >= min_nstar) and (fMhires >= min_noncontamFrac) and (mass >= min_vmass)

    if valid_halo:
        labels.append(str(ihalo))
        print(h_dummy[ihalo].properties['halo_id'])

        #x = (h_dummy[ihalo].properties['Xc']-h1x)/0.6776942783267969
        #y = (h_dummy[ihalo].properties['Yc']-h1y)/0.6776942783267969
        #z = (h_dummy[ihalo].properties['Zc']-h1z)/0.6776942783267969

        dm_particles = h[ihalo].dm
        x_coords = dm_particles['x']
        y_coords = dm_particles['y']
        z_coords = dm_particles['z']
        masses = dm_particles['mass']
        x = np.average(x_coords, weights=masses)
        y = np.average(y_coords, weights=masses)
        z = np.average(z_coords, weights=masses)

        rvir = h_dummy[ihalo].properties['Rvir']/0.6776942783267969

        if float(data[data.haloid==ihalo].sSFR.tolist()[0]) < 1e-11:
            colorVal = '#ff837a'
        else:
            colorVal = '#79aefc'

        if ihalo==1:
            h1rvir = rvir
            style = '-'
            colorVal = 'w'
            width=1.5
        else:
            width=1.
            dist = math.sqrt(x*x + y*y + z*z)
            if dist < h1rvir:
                style = '--'
            else:
                style = '-'

        circlexy1 = plt.Circle((x,y),rvir,color = colorVal, linestyle=style, fill=False, linewidth=width)
        circlexy2 = plt.Circle((x,y),rvir,color = colorVal, linestyle=style, fill=False, linewidth=width)
        circlexy3 = plt.Circle((x,y),rvir,color = colorVal, linestyle=style, fill=False, linewidth=width)

        ax_d_sonia.add_artist(circlexy1)

        if ihalo==1:
            ax_g_sonia.add_artist(circlexy2)
            ax_s_sonia.add_artist(circlexy3)
            
        
# BOTTOM ROW: SANDRA
sim = '/home/christenc/Data/Sims/h148.cosmo50PLK.3072g/h148.cosmo50PLK.3072g3HbwK1BH/snapshots_200bkgdens/h148.cosmo50PLK.3072g3HbwK1BH.004096'
name = 'h148'
s = pynbody.load(sim)
s.physical_units()
h = s.halos()
h_dummy = s.halos(dummy = True)
print('Loaded simulation')

print(s.properties['boxsize'])
s.properties['boxsize'] = pynbody.units.Unit("100 Mpc")
print(s.properties['boxsize'])

labels = []

i = 0

h1x = h_dummy[1].properties['Xc']
h1y = h_dummy[1].properties['Yc']
h1z = h_dummy[1].properties['Zc']


smin, smax = -image_width//2, image_width//2
print('Aligning halo 1')
pynbody.analysis.angmom.faceon(h[1])
s.physical_units()

print('Making DM image...')
im = pynbody.plot.sph.image(s.dm[pynbody.filt.Sphere('%s kpc' % str((smax-smin)))], width = '%s kpc' % str(smax-smin), cmap='Greys_r', 
                            av_z = 'rho', ret_im = True, show_cbar = False, noplot=True)

im = ax_d_sandra.imshow(im, norm=colors.LogNorm(), cmap='Greys_r', extent = [smin,smax,smin,smax], vmin=dm_vmin, vmax=dm_vmax,origin='lower')

print('Making gas image...')
im = pynbody.plot.sph.image(s.gas[pynbody.filt.Sphere('%s kpc' % str((smax-smin)))], width = '%s kpc' % str(smax-smin), cmap='viridis', 
                            av_z = 'rho', ret_im = True, show_cbar = False, noplot=True)

im = ax_g_sandra.imshow(im, norm=colors.LogNorm(), cmap='viridis', extent = [smin,smax,smin,smax], vmin=gas_vmin, vmax=gas_vmax,origin='lower')

print('Making star image...')
im = pynbody.plot.sph.image(s.s[pynbody.filt.Sphere('%s kpc' % str((smax-smin)))], width = '%s kpc' % str(smax-smin), cmap='cubehelix', 
                            av_z = 'rho', ret_im = True, show_cbar = False, noplot=True)

my_cmap = copy.copy(mpl.cm.get_cmap('cubehelix')) # copy the default cmap
my_cmap.set_bad(my_cmap(0))

im = ax_s_sandra.imshow(im, norm=colors.LogNorm(), cmap=my_cmap, extent= [smin,smax,smin,smax], vmin=star_vmin, vmax=star_vmax,origin='lower')

data = read_file(name) 

h1rvir = 0

print('Adding circles...')
for ihalo in range(1,len(h)): #len(h)-1):
    nstar = h_dummy[ihalo].properties['n_star']
    fMhires = h_dummy[ihalo].properties['fMhires']
    mass = h_dummy[ihalo].properties['mass']
    valid_halo = (nstar >= min_nstar) and (fMhires >= min_noncontamFrac) and (mass >= min_vmass)

    if valid_halo:
        labels.append(str(ihalo))
        print(h_dummy[ihalo].properties['halo_id'])

        #x = (h_dummy[ihalo].properties['Xc']-h1x)/0.6776942783267969
        #y = (h_dummy[ihalo].properties['Yc']-h1y)/0.6776942783267969
        #z = (h_dummy[ihalo].properties['Zc']-h1z)/0.6776942783267969i
        
        dm_particles = h[ihalo].dm
        x_coords = dm_particles['x']
        y_coords = dm_particles['y']
        z_coords = dm_particles['z']
        masses = dm_particles['mass']
        x = np.average(x_coords, weights=masses)
        y = np.average(y_coords, weights=masses)
        z = np.average(z_coords, weights=masses)

        rvir = h_dummy[ihalo].properties['Rvir']/0.6776942783267969

        if float(data[data.haloid==ihalo].sSFR.tolist()[0]) < 1e-11:
            colorVal = '#ff837a'
        else:
            colorVal = '#79aefc'

        if ihalo==1:
            h1rvir = rvir
            style = '-'
            colorVal = 'w'
            width=1.5
        else:
            width=1.
            dist = math.sqrt(x*x + y*y + z*z)
            if dist < h1rvir:
                style = '--'
            else:
                style = '-'

        circlexy1 = plt.Circle((x,y),rvir,color = colorVal, linestyle=style, fill=False, linewidth=width)
        circlexy2 = plt.Circle((x,y),rvir,color = colorVal, linestyle=style, fill=False, linewidth=width)
        circlexy3 = plt.Circle((x,y),rvir,color = colorVal, linestyle=style, fill=False, linewidth=width)

        ax_d_sandra.add_artist(circlexy1)

        if ihalo==1:
            ax_g_sandra.add_artist(circlexy2)
            ax_s_sandra.add_artist(circlexy3)
            
    
for ax in [ax_d_sonia, ax_g_sonia, ax_s_sonia, ax_d_sandra, ax_g_sandra, ax_s_sandra]:
    ax.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
    ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)

ax_g_sonia.annotate('Sonia', (0.03, 0.97), xycoords='axes fraction', va='top', ha='left', color='white')
ax_g_sandra.annotate('Sandra', (0.03, 0.97), xycoords='axes fraction', va='top', ha='left', color='white')

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
bar = AnchoredSizeBar(ax_s_sonia.transData, 500, '500 kpc', loc='upper right', pad=0.5, sep=5, frameon=False, color='white')
ax_s_sonia.add_artist(bar)


fig.savefig('images_sonia_sandra.pdf', dpi=500)
fig.show()

