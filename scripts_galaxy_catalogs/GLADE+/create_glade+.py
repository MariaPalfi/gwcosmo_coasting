import gwcosmo
import numpy as np
import astropy
import astropy.constants as constants
from astropy.table import Table
from astropy import units as u
import pickle

import h5py
import healpy as hp


GLADE = Table.read("GLADE+_reduced.txt",format='ascii')
print("GLADE has a total of "+str(len(GLADE))+" objects.")

# remove galaxies with no reported redshift
no_zs = GLADE['col9'] == 'null'
GLADE.remove_rows(no_zs)

del no_zs

print("Removed null redshift: GLADE has "+str(len(GLADE))+" galaxies with a measured redshift.")

# remove QSOs and galaxy clusters.
no_clusters = GLADE['col3'] == 'C'
GLADE.remove_rows(no_clusters)
del no_clusters

no_QSOs = GLADE['col3'] == 'Q'
GLADE.remove_rows(no_QSOs)
del no_QSOs

print("Removed Quasars and clusters: GLADE has "+str(len(GLADE))+" galaxies.")

# Make sure redshifts are positive
mask = GLADE['col9'].astype('float64') < 0
GLADE.remove_rows(mask)
print("Removed negative redshift: GLADE has "+str(len(GLADE))+" galaxies.")
del mask

#  Remove galaxies without peculiar velocity corrections unless the galaxy is above redshift 0.05
mask = (GLADE['col10'] == 0) & (GLADE['col9'].astype('float64') <= 0.05)
GLADE.remove_rows(mask)
print("Removed Galaxies without peculiar correction: GLADE has "+str(len(GLADE))+" galaxies.")
del mask

#  Remove galaxies without measured redshift and distance (0) and with redshift from distance measure (2)
mask = (GLADE['col12'] == 0) | (GLADE['col12'] == 2)
GLADE.remove_rows(mask)
print("Removed galaxies without redshift and with redshift from dl: GLADE has "+str(len(GLADE))+" galaxies.")
del mask

GLADE.add_column(np.zeros(len(GLADE)),name='zerror')

z_error_pec = GLADE['col11'].data
mask = z_error_pec == 'null'
z_error_pec[mask] = '0'
z_error_pec=z_error_pec.astype(np.float)
del mask

# Z error for galaxies with spectroscopic errors, this is an absolute error
zerror_sp = 1.5e-4
zs_sp = GLADE['col12'] == 3
print('ciao2',zerror_sp)
GLADE['zerror'][zs_sp] = np.sqrt(zerror_sp**2.+z_error_pec[zs_sp]**2.)

del zs_sp

# Z error for galaxies with WISE, this is a relative error of 4%(From Maciej discussion)
zerror_wise = 0.04
zs_wise = GLADE['col2'] != 'null'
z_wise = GLADE['col9'][zs_wise]
z_wise = z_wise.astype('float64')
GLADE['zerror'][zs_wise] = np.sqrt((zerror_wise*(1+z_wise))**2.+z_error_pec[zs_wise]**2.)

del zs_wise
del z_wise


# Z error for galaxies with photoz this is an absolute error
zerror_ph = 1.5e-2
zs_ph = GLADE['col12'] == 1
GLADE['zerror'][zs_ph] = np.sqrt(zerror_ph**2.+z_error_pec[zs_ph]**2.)

del zs_ph

# Z error for galaxies from Hyperleda, this is a relative error. Since these are small redshift the relative error is multiplied only to z
zerror_hyperleda = 0.36
zs_hyperleda = (GLADE['col1'] != 'null')
z_hyperleda = GLADE['col9'][zs_hyperleda]
z_hyperleda = z_hyperleda.astype('float64')
GLADE['zerror'][zs_hyperleda] = np.sqrt((zerror_hyperleda*z_hyperleda)**2.+z_error_pec[zs_hyperleda]**2.)

del zs_hyperleda
del z_hyperleda

# Select columns of interest
ra = GLADE['col4'].astype('float64')*np.pi/180.
dec = GLADE['col5'].astype('float64')*np.pi/180.
z = GLADE['col9'].astype('float64')
B = GLADE['col6']
K = GLADE['col7']
W1 = GLADE['col8']
sigmaz = GLADE['zerror']

del GLADE

ra = ra.astype('float64')
dec = dec.astype('float64')
z = z.astype('float64')
B = B.data
K = K.data
W1= W1.data

mask = B == 'null'
B[mask] = '0'
del mask
mask = K == 'null'
K[mask] = '0'
del mask
mask = W1 == 'null'
W1[mask] = '0'
del mask

B = B.astype('float64')
K = K.astype('float64')
W1 = W1.astype('float64')

B[B==0] = np.nan
K[K==0] = np.nan
W1[W1==0] = np.nan

import h5py
import healpy as hp

# Save GLADE catalog
bands=['B', 'K','W1']
nGal = len(z)
m = np.column_stack((B,K,W1))

del B
del K
del W1

nside = 1024
ind = hp.pixelfunc.ang2pix(nside,dec+np.pi/2,ra)

ra_dec_lim = 0
ra_min = 0.0
ra_max = np.pi*2.0
dec_min = -np.pi/2.0
dec_max = np.pi/2.0 

with h5py.File("glade+.hdf5", "w") as f:
    f.create_dataset("ra", data=ra)
    f.create_dataset("dec", data=dec)
    f.create_dataset("z", data=z)
    f.create_dataset("sigmaz", data=sigmaz)
    f.create_dataset("skymap_indices", data=ind)
    f.create_dataset("radec_lim", data=np.array([ra_dec_lim,ra_min,ra_max,dec_min,dec_max]))
    for j in range(len(bands)):
        f.create_dataset("m_{0}".format(bands[j]), data=m[:,j])