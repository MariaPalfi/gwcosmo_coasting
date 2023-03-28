import os
import glob
import pickle
import math
import array
import numpy as np
from optparse import make_option

import h5py
import pkg_resources
import healpy as hp

from ..likelihood.skymap import ipix_from_ra_dec
from ..utilities.cache import get_cachedir
from ..utilities.calc_kcor import calc_kcor

DEG2RAD = math.pi/180.0

color_names = {'B': None, 'K': None, 'u': 'u - r', 'g': 'g - r', 'r': 'g - r',
               'i': 'g - i', 'z': 'r - z', 'W1': None,'bJ': None}

# Taken from limits of K corrections from fig 4 of
# https://ui.adsabs.harvard.edu/abs/2010MNRAS.405.1409C/abstract
color_limits = {'u - r': [-0.1, 2.9], 'g - r': [-0.1, 1.9], 'g - i': [0, 3],
                'r - z': [0, 1.5], None : [-np.inf, np.inf]}

# istarmap.py for Python 3.8+
import multiprocessing.pool as mpp

def get_catalogfile(filename):
    if 'GWCOSMO_CATALOG_PATH' in os.environ.keys():
        file = os.path.join(os.environ['GWCOSMO_CATALOG_PATH'],filename)
    else:
        file = filename
    if os.path.exists(file):
        return file
    else:
        raise FileNotFoundError(f"Unable to locate {filename}. Make sure your $GWCOSMO_CATALOG_PATH is set")

def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    """
    try:
        self._check_running()
    except AttributeError:
        if self._state != mpp.RUN:
            raise ValueError("Pool not running")
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    try:
        result = mpp.IMapIterator(self)
    except TypeError:
        result = mpp.IMapIterator(self._cache)

    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)


mpp.Pool.istarmap = istarmap


"""
These are the command line options to control the catalog
"""

catalog_options = [
    make_option("--catalog", default=None, metavar="NAME",
               help="""Specify a galaxy catalog by name. Known catalogs are: DESI, DES, GLADE, GLADE+"""),
    make_option("--catalog_band", default='B', type=str,
            help = "Observation band of galaxy catalog (B,K,W1,bJ,u,g,r,i,z) (must be compatible with the catalogue provided)"),
]

def load_catalog(name, band):
    # Load raw catalog
    if name == 'GLADE':
        cat = OldStyleGLADE(band=band)
    elif name == 'DES':
        cat = OldStyleDES(band=band)
    elif name == 'DESI':
        cat = OldStyleDESI(band=band)
    elif name == 'GLADE+':
        cat = OldStyleGLADEPlus(band=band)
    else:
        raise ValueError(f"Unable catalog {name}")
    if band is not None:
        cat = cat.remove_missing_magnitudes(band)
    return cat

def load_catalog_from_opts(opts):
    name = opts.catalog
    band = opts.catalog_band
    return load_catalog(name, band)

from abc import ABC, abstractmethod

class GalaxyCatalog:
    """
    Interface for a generic galaxy catalog
    """
    colnames = {'z','sigmaz','ra','dec'}

    def __init__(self, data = None, name = 'Unknown Catalog',
                 supported_bands=None, cachedir=None):
        self.data = data
        self.name = name
        self.supported_bands = supported_bands
        if supported_bands is not None:
            self.colnames.union([f'm_{band}' for band in supported_bands])
        # Cache for pixel index to array index lookup
        self.pixmap = {}

    def __getitem__(self, *args, **kwargs):
        return self.data.__getitem__(*args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        return self.data.__setitem__(*args, **kwargs)

    def __len__(self):
        if self.data is None:
            return 0
        else:
            return len(self.data)

    def build_pixel_index_file(self, nside, cachedir=None, nested=True):
        if cachedir is None:
            cachedir = get_cachedir()

        filepath = self._cachefile(nside, nested=nested, cachedir=cachedir)
        pixlists = pixelate(self, nside, nested=nested)
        with open(filepath,'wb') as idxfile:
            pickle.dump(pixlists, idxfile)

    def _cachefile(self, nside, nested=True, cachedir=None):
        if cachedir is None:
            cachedir = get_cachedir()

        return os.path.join(cachedir,
                           '_'.join([self.name, str(nside), str(nested), 'pixidx.pkl'] )
                           )

    def read_pixel_index_cache(self, nside, cachedir=None):
        cachefile = self._cachefile(nside, cachedir=cachedir)
        if os.path.exists(cachefile):
            try:
                self.pixmap[nside] = pickle.load(open(cachefile,'rb'))
                return True
            except:
                print(f"Warning, unable to open pixel cache file {cachefile}, possible corrupted file")
                return False
        else:
            return False

    def clean_cache(self, mtime, cachedir=None):
        """
        Remove any index cache files older than mtime
        """
        if cachedir is None:
            cachedir = get_cachedir()
        for f in glob.glob(cachedir+'/'+self.name+'_*'):
            try:
                if os.path.getmtime(f) < mtime:
                    os.remove(f)
            except FileNotFoundError:
                # Here in case a parallel job has removed the file
                pass

    def magnitude_thresh(self, band, ra=None, dec=None):
        """
        Return the magnitude threshold for a specific
        sky position if specified. If fewer than 10 galaxies
        exist then return inf
        TODO: Currently this doesn't select based on ra,dec
        """
        assert (ra is None and dec is None) or (ra is not None and dec is not None)

        print(f'Computing magnitude threshold for {ra},{dec}')
        print(f'Ngal = {len(self)}')
        if len(self) < 10:
            mth = np.inf
        else:
            m = self.get_magnitudes(band)
            mth = np.median(m[np.isfinite(m)])
        return mth

    def select_pixel(self, nside, pixel_index, nested=True):
        """
        Keep only galaxies in the desired healpix pixel indices
        """
        # Try to load an index file, and create one if not
        if not self.read_pixel_index_cache(nside):
            # Try to make the index file
            self.build_pixel_index_file(nside, nested=nested)
            # Check if it can be read
            if not self.read_pixel_index_cache(nside):
                # If not, build it here
                print('No cache file found, generating pixel index (may take some time)')
                self.pixmap[nside] = pixelate(self, nside, nested=nested)

        pixmap = self.pixmap[nside]

        idx = pixmap[pixel_index]
        new = GalaxyCatalog(data = self.data[idx], name = self.name+f'_nside{nside}_pixel{pixel_index}',
                            supported_bands = self.supported_bands)
        return new

    def idx2pixdict(self, nside, idx, nested=True):
        ra, dec = self['ra'][idx], self['dec'][idx]
        ipix = ipix_from_ra_dec(nside, ra, dec, nest=nested)
        return (ipix, idx)

    def get_magnitudes(self, band):
        """
        Get magnitudes for a particular band
        """
        assert band in self.supported_bands, f"error: catalog doesn't support {band}-band magnitudes"
        return self[f'm_{band}']

    def get_color(self, band):
        """
        Return color index for K corrections
        """
        Kcorr_bands = {'B': 'B', 'K': 'K','bJ':'bJ','u': 'r', 'g': 'r', 'r': 'g', 'i': 'g', 'z': 'r'}
        Kcorr_signs = {'B': 1, 'K': 1,'bJ':1 ,'u': 1, 'g': 1, 'r': -1, 'i': -1, 'z': -1}
        if not band in Kcorr_bands.keys():
            print('Band {band} not supported for color-based K corrections')
            color = np.zeros(len(self))
        else:
            m = self.get_magnitudes(band)
            m_K = self.get_magnitudes(Kcorr_bands[band])
            color = Kcorr_signs[band]*(m - m_K)
        return color

    def apply_redshift_cut(self, zcut):
        idx = np.where(((self['z']-3*self['sigmaz']) <= zcut))
        return GalaxyCatalog(data = self.data[idx], name = self.name+f'_zcut{zcut}',
                             supported_bands = self.supported_bands)

    def remove_missing_magnitudes(self, band):
        m = self.get_magnitudes(band)
        idx = np.where(np.isfinite(m))
        return GalaxyCatalog(data = self.data[idx], name = self.name+f'_valid_{band}_mags',
                             supported_bands = self.supported_bands)

    def apply_color_limit(self, band, cmin, cmax):
        """
        Apply the given color limits to the given band.
        Will filter out any galaxies whose magnitude is missing in given band.
        """
        if band == 'W1':
            print('Not applying color limits for W1 band, as we use the z-dependent k-correction')
            return self
        idx = np.where((cmin <= self.get_color(band)) & (cmax >= self.get_color(band)))
        return GalaxyCatalog(data = self.data[idx], name = self.name+f'_climit_{band}_{cmin}_{cmax}',
                             supported_bands = self.supported_bands)

    def apply_magnitude_limit(self, band, magmax):
        idx = np.where(self.get_magnitudes(band) <= magmax)
        return GalaxyCatalog(data = self.data[idx], name = self.name+f'_mlimit_{band}_{magmax}',
                             supported_bands = self.supported_bands)

    def get_k_correction(self, band, z, color_name, color_value):
        "Apply K-correction"
        #https://arxiv.org/pdf/1709.08316.pdf
        if band == 'W1':
            k_corr = -1*(4.44e-2+2.67*z+1.33*(z**2.)-1.59*(z**3.)) #From Maciej email
            return k_corr
        elif band == 'K':
             # https://iopscience.iop.org/article/10.1086/322488/pdf 4th page lhs
            to_ret=-6.0*np.log10(1+z)
            to_ret[z>0.3]=-6.0*np.log10(1+0.3)
            return -6.0*np.log10(1+z)
        elif band == 'bJ':
            # Fig 5 caption from https://arxiv.org/pdf/astro-ph/0111011.pdf
            # Note that these corrections also includes evolution corrections
            return (z+6*np.power(z,2.))/(1+15*np.power(z,3.))
        else:
            try:
                kcor = calc_kcor(band, z, color_name, color_value)
                return kcor
            except:
                print(f'Cannot calculate k-corrections for band {band} color name {color_name}')
                print('Will return 0 k corrections')
                return np.zeros(len(z))

def newarr():
    return array.array('Q')

def pixelate(cat, nside, allowed_pixels=None, nested=True):
    from multiprocessing import Pool
    from collections import defaultdict
    from tqdm import tqdm
    Nprocs = 2 # higher values use more RAM and disk bandwidth.

    # Arrays of unsigned 64-bit integers for the galaxy indices
    pixlists = defaultdict(newarr)

    with Pool(processes=Nprocs) as pool:

        for (pixidx, galidx) in \
            pool.istarmap(cat.idx2pixdict,
                          ( [nside, idx] for idx in range(len(cat)) ),
                          chunksize=1000000
                         ):
                pixlists[pixidx].append(galidx)
    for k,v in pixlists.items():
        pixlists[k]=np.sort(v)
    return pixlists


class OldStyleCatalog(GalaxyCatalog):
    """
    Catalog in the old GWCosmo format. Must have been
    preprocessed into HDF5 files.
    """
    filename = None
    def __init__(self,
                     catalog_file=None,
                     name = None):

        self.filename = get_catalogfile(catalog_file)

        super().__init__(supported_bands = self.supported_bands, name = name)
        self.populate()
        self.clean_cache(os.path.getmtime(self.filename))

    def populate(self):
        """
        This is a separate step to load the data
        """
        f = h5py.File(self.filename, 'r')
        names = []
        for n in self.colnames:
            if n in f:
                names.append(n)
            else:
                print(f'Unable to find column for {n}-band')
        self.data = np.rec.fromarrays([f[n] for n in names], names = names)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['data']
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.populate()


class OldStyleDESI(OldStyleCatalog):
    supported_bands = {'W1'}
    def __init__(self, catalog_file = 'DESI.hdf5', band='W1'):
        print('WARNING: DESI catalog is not fully supported yet')
        self.colnames = set(self.colnames).union([f'm_{b}' for b in self.supported_bands])
        super().__init__(catalog_file = catalog_file, name = 'DESI')

class OldStyleGLADEPlus(OldStyleCatalog):
    supported_bands = {'B', 'K', 'W1','bJ'}
    def __init__(self, catalog_file = 'glade+.hdf5', band='bJ'):
        self.colnames = set(self.colnames).union([f'm_{b}' for b in self.supported_bands])
        super().__init__(catalog_file = catalog_file, name = 'GladePlus')

class OldStyleGLADE(OldStyleCatalog):
    supported_bands = {'B','K'}
    def __init__(self, catalog_file = 'glade.hdf5', band='B'):
        self.colnames = set(self.colnames).union([f'm_{b}' for b in self.supported_bands])
        super().__init__(catalog_file = catalog_file, name = 'Glade')

class OldStyleDES(OldStyleCatalog):
    supported_bands = {'g','i','r','z'}
    def __init__(self, catalog_file = 'des.hdf5', band='G'):
        self.colnames = set(self.colnames).union([f'm_{b.lower()}' for b in self.supported_bands])
        super().__init__(catalog_file = catalog_file, name = 'DES')

