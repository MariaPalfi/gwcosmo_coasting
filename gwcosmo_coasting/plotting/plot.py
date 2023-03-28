"""
Modified by Mária Pálfi and Péter Raffai (2023)
"""

import gwcosmo_coasting
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from scipy.integrate import quad
from scipy.stats import norm
from scipy.interpolate import interp1d

from gwcosmo_coasting.utilities import coasting_cosmology # use coasting_cosmology, changed by Mária Pálfi
from gwcosmo_coasting.utilities import schechter_params
from gwcosmo_coasting.utilities import schechter_function, calc_kcor
from gwcosmo_coasting.prior.catalog import color_names

from ligo.skymap.io import fits
from ligo.skymap import plot
from ligo.skymap import postprocess

from tqdm import tqdm


def mth2hpx(ra, dec, m, nside):
    """
    Map an array of values ``m'' defined for the sky points ra and dec in a
    HEALPY array.

    Parameters
    ----------
    ra, dec : (ndarray, ndarray)
        Coordinates of the sources in radians.

    nside : int
        HEALPix nside of the target map

    Return
    ------
    hpx_map : ndarray
        HEALPix map of the catalogue map.

    """

    # The number of pixels based on the chosen value of nside
    npix = hp.nside2npix(nside)

    # conver to theta, phi
    theta = np.pi/2.0 - dec
    phi = ra

    # convert to HEALPix indices (each galaxy is assigned to a single healpy pixel)
    indices = hp.ang2pix(nside, theta, phi)

    # sort the indices into ascending order
    idx_sort = np.argsort(indices)
    sorted_indices = indices[idx_sort]

    # idx: the healpy index of each pixel containing a galaxy (arranged in ascending order)
    # idx_start: the index of 'sorted_indices' corresponding to each new pixel
    # count: the number of galaxies in each pixel
    idx, idx_start,count = np.unique(sorted_indices,return_counts=True,return_index=True)

    # splits indices into arrays - 1 per pixel
    res = np.split(idx_sort, idx_start[1:])

    mths = np.zeros(npix)

    marray = np.array(m)
    count=0
    for i in range(npix):
        if i in idx:
            # select all the apparent magnitudes of galaxies in the ith pixel
            ms = marray[res[count]]
            count += 1
            # find the median apparent magnitude within this pixel (mth stays as zero if no galaxies are present)
            mths[i] = np.median(ms)

    # fill the fullsky map
    hpx_map = np.zeros(npix)
    hpx_map[range(npix)] = mths

    return hpx_map

def overdensity_given_skypos(galaxy_cat,band,Schparam,ra,dec,zproxy,k,nside=64,h0=67.7, 
			    lweight=True,redunc=False):
    """
    This function returns the redshift prior from the galaxy catalog, from the empty-catalog case and the
    index corresponsing to galaxies in the pixel corresponding to a given sky location.

    Parameters
    ----------
    galaxy_cat: galaxy catalog from gwcosmo_coasting
    Schparam: str
        Schechter class for the parameters
    ra: float
        right ascension in radiants where to compute the priors
    dec: float
        declination in radiants where to compute the priors
    zproxy: numpy array
        Redshift at which to calculate the overdensity
    nside: integer
        Nside parameter for the sky division in pixels by HEALPY, must be power of 2
    h0: float
        Hubble constant
    k: integer
    	Curvature parameter of the coasting cosmology (1, 0, or -1) # added by Mária Pálfi
    lweight: bool
        If to use luminosity weightening or not
    redunc: bool
        False if you dont want redshift uncertainties. Note that using redshift uncertainties will only generate a proxy figure (true distribution)
        convoluted with error function.
    """
    
    cosmo = coasting_cosmology.fast_cosmology(k, zmax=10.0) # added by Mária Pálfi
    to_return = np.zeros_like(zproxy)
    # The number of pixels based on the chosen value of nside
    npix = hp.nside2npix(nside)
    theta = np.pi/2.0 - galaxy_cat.data['dec']
    phi = galaxy_cat.data['ra']
    indices = hp.ang2pix(nside, theta, phi)
    index_point = hp.ang2pix(nside, np.pi/2.0 - dec, ra)
    gal_index = np.where(indices==index_point)[0]

    if not gal_index.size:
        print('No galaxy overlapping with ra={:.2f} and dec={:.2f}'.format(ra,dec))
        return to_return

    allm=galaxy_cat.data['m_'+band][gal_index]
    allra=galaxy_cat.data['ra'][gal_index]
    alldec=galaxy_cat.data['dec'][gal_index]
    allz=galaxy_cat.data['z'][gal_index]
    allsigmaz=galaxy_cat.data['sigmaz'][gal_index]

    mth = np.median(allm)
    selected_galaxies = np.where(allm<mth)[0]

    allm=allm[selected_galaxies]
    allra=allra[selected_galaxies]
    alldec=alldec[selected_galaxies]
    allz=allz[selected_galaxies]
    allsigmaz=allsigmaz[selected_galaxies]
    gal_index=gal_index[selected_galaxies]

    Kcorr = galaxy_cat.get_k_correction(band,allz,color_names[band],color_value=allm)
    lweights = coasting_cosmology.L_mdl(allm, cosmo.dl_zH0(allz, h0, k),Kcorr=Kcorr) # changed by Mária Pálfi

    alpha,Mstar_obs,Mmin_source,Mmax_source = Schparam.alpha,Schparam.Mstar,Schparam.Mmin,Schparam.Mmax

    Lmax=coasting_cosmology.L_M(Mmin_source)
    Lmin=coasting_cosmology.L_M(Mmax_source)

    index_lum = np.where((lweights<Lmax) & (lweights>Lmin))[0]
    allm=allm[index_lum]
    allra=allra[index_lum]
    alldec=alldec[index_lum]
    allz=allz[index_lum]
    allsigmaz=allsigmaz[index_lum]
    lweights=lweights[index_lum]
    gal_index=gal_index[index_lum]
    zprior = coasting_cosmology.redshift_prior(k=k, zmax=cosmo.zmax) # changed by Mária Pálfi

    if redunc:
        for i in range(len(allz)):
            zcut = np.linspace(0,allz[i]+9*allsigmaz[i],5000)
            priordens=zprior.p_z(zcut)
            zl = (1./(1+zcut))*priordens*norm.pdf(zcut,allz[i],allsigmaz[i])
            zl /=np.trapz(zl,zcut)
            interpo = interp1d(zcut,zl,bounds_error=False,fill_value=0.)

            if lweight:
                to_return+=interpo(zproxy)*lweights[i]*skymap_gwcosmo_coasting.skyprob(allra[i],alldec[i])
            else:
                to_return+=interpo(zproxy)*skymap_gwcosmo_coasting.skyprob(allra[i],alldec[i])
        galdens= to_return/np.trapz(to_return,zproxy)
    else:
        edges=zproxy-0.5*(zproxy[1]-zproxy[0])
        edges=np.append(edges,zproxy[-1]+0.5*(zproxy[1]-zproxy[0]))
        
        if lweight:
            to_return, edged = np.histogram(allz,bins=edges,weights=lweights/(1+allz))
        else:
            to_return, edged = np.histogram(allz,bins=edges,weights=1/(1+allz))
        galdens= to_return/np.sum(to_return*(zproxy[1]-zproxy[0]))    

    priordens=zprior.p_z(zproxy)
    priordens/=np.trapz(priordens,zproxy)
    return galdens,priordens,gal_index


def overdensity_given_GWskyarea(galaxy_cat,band,Schparam,fits_file,CL,zproxy,k,h0=67.7,nside=None,
                                lweight=True,redunc=False):
    """
    This function returns the redshift prior from the galaxy catalog, from the empty-catalog case and the
    index corresponsing to galaxies in the pixel corresponding to a given sky location.

    Parameters
    ----------
    galaxy_cat: galaxy catalog from gwcosmo_coasting
    Schparam: str
        Schechter class for the parameters
    fits_file: string
        path to the GWskymap
    CL: float
        What confidence level interval you want,e.g. 95
    zproxy: numpy array
        Redshift at which to calculate the overdensity
    h0: float
        Hubble constant
    k: integer
    	Curvature parameter of the coasting cosmology (1, 0, or -1) # added by Mária Pálfi
    lweight: bool
        If to use luminosity weightening or not
    redunc: bool
        False if you dont want redshift uncertainties. Note that using redshift uncertainties will only generate a proxy figure (true distribution)
        convoluted with error function.
    """
    cosmo = coasting_cosmology.fast_cosmology(k=k, zmax=10.0) # added by Mária Pálfi
    skymap_gwcosmo_coasting = gwcosmo_coasting.likelihood.skymap.skymap(fits_file)
    gal_index = np.where(skymap_gwcosmo_coasting.samples_within_region(galaxy_cat.data['ra'],galaxy_cat.data['dec'],CL,nside=nside))[0]
    to_return = np.zeros_like(zproxy)

    if not gal_index.size:
        print('No galaxy overlapping')
        return to_return

    allm=galaxy_cat.data['m_'+band][gal_index]
    allra=galaxy_cat.data['ra'][gal_index]
    alldec=galaxy_cat.data['dec'][gal_index]
    allz=galaxy_cat.data['z'][gal_index]
    allsigmaz=galaxy_cat.data['sigmaz'][gal_index]

    mth = np.median(allm)
    selected_galaxies = np.where(allm<mth)[0]

    allm=allm[selected_galaxies]
    allra=allra[selected_galaxies]
    alldec=alldec[selected_galaxies]
    allz=allz[selected_galaxies]
    allsigmaz=allsigmaz[selected_galaxies]
    gal_index=gal_index[selected_galaxies]


    Kcorr = galaxy_cat.get_k_correction(band,allz,color_names[band],color_value=allm)
    lweights = coasting_cosmology.L_mdl(allm, cosmo.dl_zH0(allz, h0, k),Kcorr=Kcorr) # changed by Mária Pálfi

    alpha,Mstar_obs,Mmin_source,Mmax_source = Schparam.alpha,Schparam.Mstar,Schparam.Mmin,Schparam.Mmax

    Lmax=coasting_cosmology.L_M(Mmin_source)
    Lmin=coasting_cosmology.L_M(Mmax_source)

    index_lum = np.where((lweights<Lmax) & (lweights>Lmin))[0]
    allm=allm[index_lum]
    allra=allra[index_lum]
    alldec=alldec[index_lum]
    allz=allz[index_lum]
    allsigmaz=allsigmaz[index_lum]
    lweights=lweights[index_lum]
    gal_index=gal_index[index_lum]
    zprior = coasting_cosmology.redshift_prior(k=k, zmax=cosmo.zmax) # changed by Mária Pálfi
    
    if redunc:
        for i in range(len(allz)):
            zcut = np.linspace(0,allz[i]+9*allsigmaz[i],5000)
            priordens=zprior.p_z(zcut)
            zl = (1./(1+zcut))*priordens*norm.pdf(zcut,allz[i],allsigmaz[i])
            zl /=np.trapz(zl,zcut)
            interpo = interp1d(zcut,zl,bounds_error=False,fill_value=0.)

            if lweight:
                to_return+=interpo(zproxy)*lweights[i]*skymap_gwcosmo_coasting.skyprob(allra[i],alldec[i])
            else:
                to_return+=interpo(zproxy)*skymap_gwcosmo_coasting.skyprob(allra[i],alldec[i])
        galdens= to_return/np.trapz(to_return,zproxy)
    else:
        edges=zproxy-0.5*(zproxy[1]-zproxy[0])
        edges=np.append(edges,zproxy[-1]+0.5*(zproxy[1]-zproxy[0]))
        
        if lweight:
            to_return, edged = np.histogram(allz,bins=edges,weights=lweights/(1+allz))
        else:
            to_return, edged = np.histogram(allz,bins=edges,weights=1/(1+allz))
        galdens= to_return/np.sum(to_return*(zproxy[1]-zproxy[0]))    

    priordens=zprior.p_z(zproxy)
    priordens/=np.trapz(priordens,zproxy)
    return galdens,priordens,gal_index


def Completeness(H0,z_array,mth,Schparam,weighted=True):
    """
    Returns p(G|H0,z)
    The probability that the host galaxy is in the catalogue given detection and H0.

    Parameters
    ----------
    H0 : float
        Hubble constant value in kms-1Mpc-1
    z_array : float, array_like
        An array of redshifts
    mth : float
        The apparent magnitude threshold of the galaxy catalog under consideration
    Schparam: str
        Schechter class for the parameters
    pdet_path: str
        path to pdet
    weighted : bool, optional
        Is the GW host luminosity weighted? (default=False)

    Returns
    -------
    float or array_like
        p(G|H0,z)
    """

    alpha,Mstar_obs,Mmin_source,Mmax_source = Schparam.alpha,Schparam.Mstar,Schparam.Mmin,Schparam.Mmax

    zprior = coasting_cosmology.redshift_prior(k = k) # changed by Mária Pálfi
    cosmo = coasting_cosmology.fast_cosmology(k = k ) # changed by Mária Pálfi
    num = np.zeros(len(z_array))
    den = np.zeros(len(z_array))

    for i in range(len(z_array)):

        def I(M):
            temp = schechter_function.SchechterMagFunction(alpha=alpha,Mstar_obs=Mstar_obs)(M,H0)
            if weighted:
                return temp*coasting_cosmology.L_M(M)
            else:
                return temp

        Mmin = schechter_function.M_Mobs(H0,Mmin_source)
        Mmax = schechter_function.M_Mobs(H0,Mmax_source)
        lim = coasting_cosmology.M_mdl(mth,cosmo.dl_zH0(z_array[i],H0, k)) # changed by Mária Pálfi
        if lim > Mmax:
            num[i]=1.0
            den[i]=1.0
        else:
            num[i] = quad(I,Mmin,lim,epsabs=1.49e-13,epsrel=1.49e-13)[0]
            den[i] = quad(I,Mmin,Mmax,epsabs=1.49e-13,epsrel=1.49e-13)[0]

    pGD = num/den
    return pGD


def plot_skymap(map_plot,mask=None,ax=None,**kwargs):
    """
    Plot a skymap using contourf of an HEALPY array:

    Parameters
    ----------
    map_plot: healpy array
        An healpy array with the values of the map you want to plot inside
    mask: array
        Array of the same size of the numpy array with False and True.
    ax: matplolib lib axes
        Axes where to plot.
    **kwargs:
        Keyword argumets to pass to contourf

    Returns
    -------
    axes handling, image handling for skymaps
    """

    if ax is None:
        ax = plt.axes(projection='astro hours mollweide')

    if mask is not None:
        indx0= np.where(mask)[0]

    im=ax.contourf_hpx(map_plot,**kwargs)
    return ax,im

def plot_loc_contour(fits_file,ax=None,**kwargs):
    """
    Plot the CL counters of a skymap

    Parameters
    ----------
    fits_file: fits file for the GW skymap
    ax: matplolib lib axes
        Axes where to plot.
    **kwargs:
        Keyword argumets to pass to contour. If you want to plot levels of CL, provide
        levels=[68,90] etc.
    Returns
    -------
    axes handling, image handling for skymaps
    """

    if ax is None:
        ax = plt.axes(projection='astro hours mollweide')
    skymap, metadata = fits.read_sky_map(fits_file, nest=None)
    cls = 100 * postprocess.find_greedy_credible_levels(skymap)
    cs = ax.contour_hpx((cls, 'ICRS'), nested=metadata['nest'],**kwargs)
    return ax, cs
