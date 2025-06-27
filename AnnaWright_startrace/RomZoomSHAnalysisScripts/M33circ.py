'''
Step 7 of stellar halo pipeline
Calculate and save the orbital circularity of each star particle

Usage:   python M33circ.py <sim>
Example: python M33circ.py r634 

Outputs: plots showing orbital circularity of stars as a function of 
         radius and shaded by stellar ages
         plots showing face-on and edge-on images of gas density and 
         stars for the chosen AM method
         <sim>_allhalostardata_circ.h5, which is a version of 
         the allhalostardata hdf5 file with the circularity of
         each star particle saved

There are several combinations of options you can use for this and 
what works best may depend on the galaxy in question. To calculate
the angular momentum vector, you can either use the AM of the stars 
with ages < <ysage> Myr or the gas with T< <gastemp> K within a radius 
of <disklim>. To calculate circularity, you can either use the method 
from Stinson+10 ('Stinson' or 'Stinson_W24') or the method from Abadi+03 
('Abadi'). 'Stinson' will give you pynbody's implementation of Stinson+10, 
while 'Stinson_W24' will give you the simpler version I used in Wright+24.
Ideally, you want a disk to be very clear as a distribution of young 
stars at small radii with a circularity somewhere around 1 so that it's 
easy to later make clear cuts for stellar halo stars (e.g., for well-behaved 
FOGGIE galaxies, I used r<~30 kpc and Stinson circ=0.65-1.35 for the disk, and 
everything else was either bulge (r>5 kpc) or stellar halo (r=5-350 kpc) -
see Fig 1 of Wright+24). However, not all galaxies have nice thin co-planar disks. 
Hopefully the combination of images produced by this script will help you 
figure out whether any of these parameters makes sense for your galaxy. 
Once you're happy with the AM method you've chosen, you can set 
<savecirc> to True to save the circularity of each star particle.
'''

import numpy as np
import matplotlib.pyplot as plt
import pynbody
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import datashader as dsh
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
import seaborn as sns
import h5py
import sys
from datashader.mpl_ext import dsshow
import tangos as db
from pynbody.analysis import Profile

fsize = 16

plt.rcParams['xtick.labelsize'] = fsize
plt.rcParams['ytick.labelsize'] = fsize

if len(sys.argv) != 2:
    print ('Usage: python M33circ.py <sim>')
    sys.exit()
else:
    cursim = str(sys.argv[1])

am_method = 'young_stars' # 'young_stars': use stars with ages < ysage Myr to ID AM vector
                          # 'gas': use gas with T<gastemp K to ID AM vector
halolim = 50 # how far from the center of the galaxy should your plot go?
hid = 1 # What is the amiga.grp ID of the halo we're centering on? Almost always 1 for MMs
disklim = 10 # How far out should we check for young stars or cool gas for AM calc (in kpc)?
ysage = 250 # We will use stars with age<ysage Myr to calculate AM if am_method='young_stars'
gastemp = 1e3 # temperature below which to use gas for AM calc if am_method='gas'
savecirc = False # Are you ready to save the circularity values to your hdf5 file?
circ_method = 'Stinson_W24' # options are 'Abadi', 'Stinson', and 'Stinson_W24'

opath = '/Users/Anna/Research/Outputs/M33analogs/MM/'+cursim+'/' # Where should outputs be saved?
datapath = '/Users/Anna/Research/Outputs/M33analogs/MM/'+cursim+'/' # Where does your allhalostardata hdf5 file live?
simdir = '/Volumes/Audiobooks/RomZooms/' # Where does your simulation live?

age_color_map = sns.blend_palette(("black", "#16263B", "#386094", "#4575b4", "#4daf4a","#FFD24D", "darkorange"), as_cmap=True)

keyadd = ''
if ysage != 25 and am_method=='young_stars':
    keyadd += '_ys'+str(ysage)
if disklim != 10:
    keyadd += '_dl'+str(disklim)
if gastemp != 1e3 and am_method=='gas':
    keyadd += '_gt'+str(gastemp)
    

s = pynbody.load(simdir+cursim+'.romulus25.3072g1HsbBH/'+cursim+'.romulus25.3072g1HsbBH.004096/'+cursim+'.romulus25.3072g1HsbBH.004096')
h = s.halos()
s.physical_units()

# get AM vector
pynbody.analysis.halo.center(h[hid]) # this centers both spatially and in terms of velocity; we are moving the entire snapshot
if am_method == 'gas':
    amsp = h[hid].g[pynbody.filt.Sphere(disklim)][pynbody.filt.LowPass('temp',str(gastemp)+' K')]
else:
    amsp = h[hid].s[pynbody.filt.Sphere(disklim)][pynbody.filt.LowPass('age',str(ysage)+' Myr')]
L = pynbody.analysis.angmom.ang_mom_vec(amsp)
norm_L = L/np.sqrt((L**2).sum())
print ('AM vector:',norm_L)

disk_rot_arr = pynbody.analysis.angmom.calc_faceon_matrix(norm_L)
pynbody.transformation.Rotation.rotate(s,disk_rot_arr) # rotate so that our disk is face-on (again moving everything)

allstars = s.s[s.s['tform']>0] # no black holes

# Calculate circularity
if circ_method == 'Stinson':
    pynbody.analysis.morphology.estimate_jcirc_from_rotation_curve(allstars,particles_per_bin=1000)
    circ = allstars['j'][:,2]/allstars['j_circ']
elif circ_method == 'Stinson_W24':
    # Calculate z-component of angular momentum
    jz = allstars['pos'][:,0]*allstars['vel'][:,1] - allstars['pos'][:,1]*allstars['vel'][:,0]
    # Calculate angular momentum of particle in perfect circular orbit
    stars_r = np.linalg.norm(allstars['pos'],axis=1)
    radii = np.logspace(np.log10(min(stars_r)),np.log10(halolim+30),2000)
    p = pynbody.analysis.profile.Profile(h[hid],bins=radii)
    totmass_enc = p['mass_enc']
    Menc_profile = IUS(np.concatenate(([0],radii[1:])), np.concatenate(([0],totmass_enc)))
    Menc = Menc_profile(stars_r)
    grav_pot = 6.67408*10**-8*Menc*(1.988*10**33)/(stars_r*3.086*10**21)
    vc = np.array([np.sqrt(gr)*10**-5 for gr in grav_pot])
    jc = vc*stars_r
    # Calculate circularity
    circ = jz/jc
elif circ_method == 'Abadi': 
    pynbody.analysis.morphology.estimate_jcirc_from_energy(allstars,particles_per_bin=1000)
    circ = allstars['j'][:,2]/allstars['j_circ']
else:
    print ('Unknown method requested')
    exit()


# Make radius-circularity-age figure
sim = db.get_simulation('%'+cursim+'%')
stars_r = np.linalg.norm(allstars['pos'],axis=1)
df = pd.DataFrame({'radius':stars_r, 'circularity':circ, 'age':(sim[-1].time_gyr-allstars['tform'].in_units('Gyr'))/sim[-1].time_gyr})
fig = plt.figure(figsize=(8,8),dpi=120)

ax1 = fig.add_axes([0.1, 0.1, 0.85, 0.85])

artist = dsshow(df,dsh.Point('radius','circularity'),dsh.mean('age'),norm='linear',cmap=age_color_map,x_range=(0,halolim),y_range=(-2,2), vmin=0,vmax=1,aspect='auto',ax=ax1)
ax1.set_xlim(0,50)
if circ_method == 'Stinson' or circ_method=='Stinson_W24':
    ax1.set_ylim(-2,2)
elif circ_method == 'Abadi':
    ax1.set_ylim(-1.1,1.1)
ax1.set_xlabel('Galactocentric Distance (kpc)',fontsize=20)
ax1.set_ylabel('Circularity',fontsize=20)

ax2 = fig.add_axes([0.605, 0.89, 0.33, 0.04])
norm = mpl.colors.Normalize(vmin=0, vmax=1)
cb1 = mpl.colorbar.ColorbarBase(ax2,cmap=age_color_map,norm=norm,ticks=np.arange(0,1.1,0.2),orientation='horizontal',label='Age/Current Time')
cb1.ax.xaxis.get_label().set_fontsize(20)

ax2.spines["top"].set_color('white')
ax2.spines["bottom"].set_color('white')
ax2.spines["left"].set_color('white')
ax2.spines["right"].set_color('white')
ax2.set_yticklabels([])
ax2.set_yticks([])
plt.savefig(opath+cursim+'_circularity_'+am_method+'_'+circ_method+keyadd+'.png',bbox_inches='tight')
plt.close()
# plt.show()

# Make images of gas and stars so you can sanity check your choice of AM
ml = -5 # log min gas density in g/cm^2
mh = -1 # log max gas density in g/cm^2
w = 20 # image width in kpc
f3 = plt.figure(figsize=(9,10))
pynbody.plot.stars.render(h[hid].s,dynamic_range=3,width=2*w)
# pynbody.plot.stars.render(h[hid].s,starsize=res,mag_range=(19,27),clear=False,width=40) 
plt.xlabel('kpc',fontsize=20)
plt.ylabel('kpc',fontsize=20)
plt.savefig(opath+cursim+'_stars_faceon_'+am_method+keyadd+'.png',bbox_inches='tight')
plt.close()

f3 = plt.figure(figsize=(9,10))
cmap = plt.cm.cubehelix
norm = mpl.colors.Normalize(vmin=ml, vmax=mh)
pynbody.plot.sph.image(h[hid].g,units="g cm**-2",width = 2*w,cmap='cubehelix',vmin=10**ml,vmax=10**mh,show_cbar=False)
plt.xlabel('kpc',fontsize=20)
plt.ylabel('kpc',fontsize=20)
plt.savefig(opath+cursim+'_gas_faceon_'+am_method+keyadd+'.png',bbox_inches='tight')
plt.close()

h[hid].rotate_x(90)
f3 = plt.figure(figsize=(9,10))
pynbody.plot.stars.render(h[hid].s,dynamic_range=3,width=2*w)
plt.xlabel('kpc',fontsize=20)
plt.ylabel('kpc',fontsize=20)
plt.savefig(opath+cursim+'_stars_sideon_'+am_method+keyadd+'.png',bbox_inches='tight')
plt.close()

f3 = plt.figure(figsize=(9,10))
cmap = plt.cm.cubehelix
norm = mpl.colors.Normalize(vmin=ml, vmax=mh)
pynbody.plot.sph.image(h[hid].g,units="g cm**-2",width = 2*w,cmap='cubehelix',vmin=10**ml,vmax=10**mh,show_cbar=False)
plt.xlabel('kpc',fontsize=20)
plt.ylabel('kpc',fontsize=20)
plt.savefig(opath+cursim+'_gas_sideon_'+am_method+keyadd+'.png',bbox_inches='tight')
plt.close()

# If you're happy with the AM vector you've calculated, 
# write out your data (but make sure to read in the old data first)
if savecirc == True:
    with h5py.File(datapath+cursim+'/'+cursim+'_allhalostardata.h5','r') as f:
        hostids = f['host_IDs'][:]
        partids = f['particle_IDs'][:]
        pct = f['particle_creation_times'][:]
        ph = f['particle_hosts'][:]
        pp = f['particle_positions'][:]
        ts = f['timestep_location'][:]
    
    assert(np.array_equal(partids,s.s['iords'][s.s['tform']>0])) # make absolutely certain these are the same.
                                                                 # If they're not, then we have to do everything
                                                                 # manually, rather than using pynbody's methods 
                                                                 # (plus we lost stars somewhere between GrabTF and
                                                                 # here)

    ofile = datapath+cursim+'/'+cursim+'_allhalostardata_circ.h5'
    with h5py.File(ofile,'w') as f:
        f.create_dataset('particle_IDs', data=partids)
        f.create_dataset('particle_positions', data=pp)
        f.create_dataset('particle_creation_times', data=pct)
        f.create_dataset('timestep_location', data=ts)
        f.create_dataset('particle_hosts', data=ph)
        f.create_dataset('host_IDs', data=hostids, dtype="S10")
        f.create_dataset(circ_method+'_circ', data=circ)

'''
Created on Mar 4, 2024

@author: anna
'''