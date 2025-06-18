### This code generates a figure showing the motion of a galaxy through the CGM alongside its ram pressure ratio over time
### This will be updated in the future, since I don't really like the way this plot fits into the overall narrative


from analysis import *
import sys

sim = str(sys.argv[1])
haloid = int(sys.argv[2])

filepaths, haloids, h1ids = get_stored_filepaths_haloids(sim,haloid)
path = filepaths[0]
print('Loading',path)

s = pynbody.load(path)
s.physical_units()
h = s.halos()

print('Centering halo 1')
pynbody.analysis.halo.center(h[1])


fig = plt.figure(dpi=300, figsize=(7.5,3), constrained_layout=True)
gs = mpl.gridspec.GridSpec(nrows=1, ncols=2, width_ratios = [1,1.2], figure=fig)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax = [ax0,ax1]

a = float(s.properties['a'])
host_Rvir = h[1].properties['Rvir'] / hubble * a
width = round(2.4*host_Rvir, 1)

filt = pynbody.filt.Cuboid(x1='-500 kpc', y1='-500 kpc', z1='-20 kpc', x2='500 kpc', y2='500 kpc', z2='20 kpc')

print('Making gas image')    
im = pynbody.plot.sph.velocity_image(s[filt].g, 
                                     width=f'{width:.1f} kpc', # set to 2.8 Rvir
                                     cmap='cubehelix',
                                     vector_color = 'cyan', 
                                     vector_resolution = 30,
                                     scale = 4000*pynbody.units.Unit('km s**-1'),
                                     av_z = 'rho', # slicing manually
                                     ret_im=True, denoise=False, approximate_fast=False, subplot=ax[0], show_cbar=False, quiverkey=False)

#plt.colorbar(

print('Plotting circle')
circ = plt.Circle((0,0), host_Rvir, color = 'w', linestyle=':', fill=False, linewidth=1)
ax[0].add_artist(circ)

print('Plotting satellite orbit')
data = read_tracked_particles(sim, haloid)
X = np.array([])
Y = np.array([])
Z = np.array([])
time = np.unique(data.time)
for t in time:
    d = data[data.time==t]
    x = np.mean(d.sat_Xc) - np.mean(d.host_Xc)
    y = np.mean(d.sat_Yc) - np.mean(d.host_Yc)
    z = np.mean(d.sat_Zc) - np.mean(d.host_Zc)
    X = np.append(X,x) #/hubble*a)
    Y = np.append(Y,y) #/hubble*a)
    Z = np.append(Z,z) #/hubble*a)

print('Performing spline fits...')
tnew = np.linspace(np.min(time), np.max(time), 1000)
from scipy.interpolate import UnivariateSpline
sX, sY, sZ = UnivariateSpline(time, X), UnivariateSpline(time, Y), UnivariateSpline(time, Z)
Xnew, Ynew, Znew = sX(tnew), sY(tnew), sZ(tnew)
    
from matplotlib.collections import LineCollection
points = np.array([Xnew, Ynew]).T.reshape(-1, 1, 2)
lwidths = -(Znew - max(Znew))/100 + 1
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = LineCollection(segments, linewidths=lwidths,color='w')
ax[0].add_collection(lc)

ax[0].arrow(Xnew[-2], Ynew[-2], Xnew[-1]-Xnew[-2], Ynew[-1]-Ynew[-2], width=lwidths[-1]+3, color='w', length_includes_head=False, head_length=15)
    

ax[0].set_xlabel(r'$x$ [kpc]')
ax[0].set_ylabel(r'$y$ [kpc]')

print('Plotting ram pressure')
data = pd.read_hdf('../../Data/ram_pressure.hdf5', key=f'{sim}_{str(haloid)}')
x = np.array(data.t,dtype=float)
y = np.array(data.Pram,dtype=float)/np.array(data.Prest,dtype=float)
ax[1].plot(x,y, label='Spherically-averaged CGM', color='k', linestyle='--')

x = np.array(data.t,dtype=float)
y = np.array(data.Pram_adv,dtype=float)/np.array(data.Prest,dtype=float)
ax[1].plot(x,y, label='True CGM', color='k', linestyle='-')
ax[1].legend(loc='upper left', frameon=False)

ax[1].semilogy()
ax[1].set_xlabel('Time [Gyr]')
ax[1].set_ylabel(r'$\mathcal{P} \equiv P_{\rm ram}/P_{\rm rest}$')

plt.savefig(f'ram_pressure_{sim}_{str(haloid)}.pdf')
plt.close()
    
