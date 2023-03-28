#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import astropy
import astropy.constants as constants
from astropy.table import Table
from astropy import units as u
import pickle

#Global Imports
import matplotlib 
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family']= 'Times New Roman'
matplotlib.rcParams['font.sans-serif']= ['Bitstream Vera Sans']
matplotlib.rcParams['text.usetex']= True
matplotlib.rcParams['mathtext.fontset']= 'stixsans'

import seaborn as sns
sns.set_context('talk')
sns.set_style('ticks')
sns.set_palette('colorblind')
sns.set_palette('colorblind')
cs=sns.color_palette('colorblind')

c=constants.c.to('km/s').value

def sph2vec(ra,dec):
    return np.array([np.sin(np.pi/2.-dec)*np.cos(ra),np.sin(np.pi/2.-dec)*np.sin(ra),np.cos(np.pi/2.-dec)])

def zhelio_to_zcmb(ra,dec,z_helio):
    ra_cmb = 167.99*np.pi/180
    dec_cmb = -7.22*np.pi/180
    v_cmb = 369.
    z_gal_cmb = v_cmb*np.dot(sph2vec(ra_cmb,dec_cmb),sph2vec(ra,dec))/c
    return (1.+z_helio)*(1.+z_gal_cmb)-1.


# In[ ]:


GLADE = Table.read("GLADE_2.4.txt",format='ascii')

print("GLADE has "+str(len(GLADE))+" objects.")


# In[ ]:


# remove galaxies with no reported redshift
no_zs = GLADE['col11'] == 'null'
GLADE.remove_rows(no_zs)

print("GLADE has "+str(len(GLADE))+" galaxies with a measured redshift.")


# In[4]:


# remove QSOs and galaxy clusters.
no_clusters = GLADE['col6'] == 'C'
GLADE.remove_rows(no_clusters)

no_QSOs = GLADE['col6'] == 'Q'
GLADE.remove_rows(no_QSOs)

print("GLADE has "+str(len(GLADE))+" galaxies.")


# In[5]:


# Apply CMB frame correction
ra = GLADE['col7']*np.pi/180.
dec = GLADE['col8']*np.pi/180.
z = GLADE['col11']

ra = ra.astype('float64')
dec = dec.astype('float64')
z = z.astype('float64')

zcmb = zhelio_to_zcmb(ra,dec,z)

GLADE.add_column(zcmb,name='zcmb')

# Make sure redshifts are positive
mask = GLADE['zcmb'] < 0
GLADE.remove_rows(mask)
print("GLADE has "+str(len(GLADE))+" galaxies.")


# In[6]:


# Assign redshift errors to spec, photo, and hyperleda galaxies
zerror_sp = 1.5e-4
zerror_ph = 1.5e-2
zerror_hyperleda = 0.36

zs_sp = GLADE['col21'] == 3
zs_ph = GLADE['col21'] == 1
zs_hyperleda = (GLADE['col21'] == 2) & (GLADE['col3'] != 'null')

zerrors = np.zeros(len(GLADE))
GLADE.add_column(zerrors,name='zerror')

z_sp = GLADE['zcmb'][zs_sp]
z_ph = GLADE['zcmb'][zs_ph]
z_hyperleda = GLADE['zcmb'][zs_hyperleda]

z_sp = z_sp.astype('float64')
z_ph = z_ph.astype('float64')
z_hyperleda = z_hyperleda.astype('float64')

GLADE['zerror'][zs_sp] = zerror_sp*np.ones_like(z_sp)
GLADE['zerror'][zs_ph] = zerror_ph*np.ones_like(z_ph)
GLADE['zerror'][zs_hyperleda] = zerror_hyperleda*z_hyperleda


# In[7]:


# remove GWCC galaxies (distances to redshift gals)
no_errors = GLADE['zerror'] == 0
GLADE.remove_rows(no_errors)

print("GLADE has "+str(len(GLADE))+" galaxies.")


# In[8]:


# Select columns of interest
ra = GLADE['col7']*np.pi/180.
dec = GLADE['col8']*np.pi/180.
z = GLADE['zcmb']
B = GLADE['col12']
K = GLADE['col19']
sigmaz = GLADE['zerror']

ra = ra.astype('float64')
dec = dec.astype('float64')
z = z.astype('float64')
B = B.data
K = K.data

mask = B == 'null'
B[mask] = '0'
mask = K == 'null'
K[mask] = '0'

B = B.astype('float64')
K = K.astype('float64')

B[B==0] = np.nan
K[K==0] = np.nan


# In[9]:


import h5py
import healpy as hp

# Save GLADE catalog
bands=['B', 'K']
nGal = len(z)
m = np.column_stack((B,K))

nside = 1024
ind = hp.pixelfunc.ang2pix(nside,dec+np.pi/2,ra)

ra_dec_lim = 0
ra_min = 0.0
ra_max = np.pi*2.0
dec_min = -np.pi/2.0
dec_max = np.pi/2.0 

with h5py.File("glade.hdf5", "w") as f:
    f.create_dataset("ra", data=ra)
    f.create_dataset("dec", data=dec)
    f.create_dataset("z", data=z)
    f.create_dataset("sigmaz", data=sigmaz)
    f.create_dataset("skymap_indices", data=ind)
    f.create_dataset("radec_lim", data=np.array([ra_dec_lim,ra_min,ra_max,dec_min,dec_max]))
    for j in range(len(bands)):
        f.create_dataset("m_{0}".format(bands[j]), data=m[:,j])


# In[18]:


len(np.where(~np.isnan(B)==True)[0])


# In[19]:


len(np.where(~np.isnan(K)==True)[0])


# In[ ]:




