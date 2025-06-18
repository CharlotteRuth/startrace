import numpy as np
import matplotlib.pyplot as plt
import pynbody
import pickle

path = '/home/christenc/Data/Sims/h229.cosmo50PLK.3072g/h229.cosmo50PLK.3072gst5HbwK1BH/snapshots_200bkgdens/h229.cosmo50PLK.3072gst5HbwK1BH.004096'
#path = '/home/christenc/Data/Sims/h148.cosmo50PLK.3072g/h148.cosmo50PLK.3072g3HbwK1BH/snapshots_200bkgdens/h148.cosmo50PLK.3072g3HbwK1BH.004096'

s = pynbody.load(path)
s.physical_units()
h = s.halos()

print('centering...')
pynbody.analysis.halo.center(h[1])
print('done')
hubble = 0.6776942783267969
a = float(s.properties['a'])
Rvir = h[1].properties['Rvir'] / hubble * a
radius = round(4*Rvir,2)
sfilt = pynbody.filt.Sphere(f'{radius} kpc')
width = f'{round(2.8*Rvir,2)} kpc'

print('Making image...')
im = pynbody.plot.sph.image(s.dm[sfilt], width = width, cmap='Greys', av_z = 'rho', ret_im = True, show_cbar = False, noplot=True)

print('Saving image...')
with open('../../Data/h229_z0_dm_image.pickle','wb') as outfile:
    pickle.dump(im, outfile)