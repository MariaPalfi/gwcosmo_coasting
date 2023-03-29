"""
Module for coasting cosmologies with k = -1, 0, 1.

Constants
---------
c : speed of light in km/s
H0 : Hubble parameter
k: curvature parameter of the coasting cosmology (-1, 0, or 1)

Mária Pálfi and Péter Raffai (2023) based on standard_cosmology.py (Archisman Ghosh, 2013-Nov)
"""

import numpy as np
from scipy import integrate
from scipy.interpolate import splrep, splev
import lal

c = lal.C_SI/1000.  # 2.99792458e+05 # in km/s

def h(z):
    """
    Returns dimensionless redshift-dependent hubble parameter.

    Parameters
    ----------
    z : redshift

    Returns
    -------
    dimensionless h(z) = 1+z
    """
    return 1+z


def dcH0overc(z, k):
    """
    Returns dimensionless combination dc*H0/c given redshift.

    Parameters
    ----------
    z : redshift

    Returns
    -------
    dimensionless combination dc*H0/c = ln(1+z) for k = 0
    dimensionless combination dc*H0/c = sin(ln(1+z)) for k = 1
    dimensionless combination dc*H0/c = (z^2+2z)/(2*(z+1)) for k = -1
    """
    #print('k=', k)
    if k == 0:
        val = np.log(1+z)
    if k == 1:
        val = np.sin(np.log(1+z))
    if k == -1:
        val = (z**2+2*z)/(2*z+2)
    return val



def dLH0overc(z, k):
    """
    Returns dimensionless combination dL*H0/c redshift.

    Parameters
    ----------
    z : redshift

    Returns
    -------
    dimensionless combination dL*H0/c = (1+z) * dc*H0/c
    """
    #print('k=', k)
    return (1+z)*dcH0overc(z, k)


def volume_z(z, k):
    """
    Returns the cosmological volume at the given redshift.

    Parameters
    ----------
    z : redshift

    Returns
    -------
    volume element (\int_0^z dz'/h(z'))^2 / h(z): dimensionless
    """
    return dcH0overc(z, k)**2/h(z)


def volume_time_z(z, k):
    """
    Returns the cosmological volume time element at a given redshift.

    Parameters
    ----------
    z : redshift

    Returns
    -------
    volume time element (\int_0^z dz'/h(z'))^2 / (1+z)h(z)
    """
    return volume_z(z, k)/(1.0+z)


def prefactor_volume_dHoc(dHoc, k, tolerance_z=1e-06, z=None):
    """
    Returns the prefactor modifying dL^2*ddL
    for the cosmological volume element.

    Parameters
    ----------
    dLH0overc : dimensionless combination dL*H0/c
    tolerance_z : (optional) tolerated error in redshift. default = 1e-06
    z : (optional) redshift, if it has been calculated already

    Returns
    -------
    prefactor, (1+z)^(-3) * (1 - 1 / (1 + (1+z)^2/(dHoc*h(z))))
    """
    if z is None:
        z = redshift(dHoc, k, tolerance_z)
    return (1+z)**(-3.) * (1 - 1. / (1 + (1+z)**2/(dHoc*h(z))))


def volume_dHoc(dHoc, k, tolerance_z=1e-06, z=None):
    """
    Returns cosmological volume at the given dL*H0/c.

    Parameters
    ----------
    dLH0overc : dimensionless combination dL*H0/c
    tolerance_z : (optional) tolerated error in redshift. default = 1e-06
    z : (optional) redshift, if it has been calculated already

    Returns
    -------
    volume, dHoc^2 * (1+z)^(-3) * (1 - 1 / (1 + (1+z)^2/(dHoc*h(z))))
    """
    return dHoc**2*prefactor_volume_dHoc(dHoc, k, tolerance_z, z=z)


def redshift(dHoc, k, tolerance_z=1e-06):
    """
    Returns redshift given dimensionless combination dL*H0/c.

    Parameters
    ----------
    dLH0overc : dimensionless combination dL*H0/c
    tolerance_z : (optional) tolerated error in redshift. default = 1e-06.

    Returns
    -------
    redshift, z
    """
    min_z = 0.
    max_z = 1.
    error_z = max_z-min_z
    while error_z > tolerance_z:
        if dLH0overc(max_z, k) < dHoc:
            min_z = max_z
            max_z *= 2
        elif dLH0overc((max_z+min_z)/2., k) < dHoc:
            min_z = (max_z+min_z)/2.
        else:
            max_z = (max_z+min_z)/2.
        error_z = max_z-min_z
    return (max_z+min_z)/2.


# Distance modulus given luminosity distance
def DistanceModulus(dL):
    """
    Returns distance modulus given luminosity distance

    Parameters
    ----------
    dL : luminosity distance in Mpc

    Returns
    -------
    distance modulus = 5*np.log10(dL)+25
    """
    return 5*np.log10(dL)+25  # dL has to be in Mpc


def dl_mM(m, M, Kcorr=0.):
    """
    returns luminosity distance in Mpc given
    apparent magnitude and absolute magnitude

    Parameters
    ----------
    m : apparent magnitude
    M : absolute magnitude in the source frame
    Kcorr : (optional) K correction, to convert absolute magnitude from the
        observed band to the source frame band (default=0).  If fluxes are
        bolometric, this should be left as 0. If not, a K correction of 0 is
        only valid at low redshifts.

    Returns
    -------
    Luminosity distance in Mpc
    """
    return 10**(0.2*(m-M-Kcorr-25))


def L_M(M):
    """
    Returns luminosity when given an absolute magnitude.
    The constant used here corresponds to the conversion between bolometric mangitude and luminosity.
    It does not matter for the H0 inference, so please use with care when using with band specific magnitudes.

    Parameters
    ----------
    M : absolute magnitude in the source frame

    Returns
    -------
    Luminosity in Watts
    """
    # TODO: double check use of L0=3.0128e28
    return 3.0128e28*10**(-0.4*M)


def M_mdl(m, dl, Kcorr=0.):
    """
    Returns a source's absolute magnitude given
    apparent magnitude and luminosity distance
    If a K correction is supplied it will be applied

    Parameters
    ----------
    m : apparent magnitude
    dl : luminosity distance in Mpc
    Kcorr : (optional) K correction, to convert absolute magnitude from the
        observed band to the source frame band (default=0).  If fluxes are
        bolometric, this should be left as 0. If not, a K correction of 0 is
        only valid at low redshifts.

    Returns
    -------
    Absolute magnitude in the source frame
    """
    return m - DistanceModulus(dl) - Kcorr


def L_mdl(m, dl, Kcorr=0.):
    """
    Returns luminosity when given apparent magnitude and luminosity distance
    If a K correction is supplied it will be applied

    Parameters
    ----------
    m : apparent magnitude
    dl : luminosity distance in Mpc
    Kcorr : (optional) K correction, to convert absolute magnitude from the
        observed band to the source frame band (default=0).  If fluxes are
        bolometric, this should be left as 0. If not, a K correction of 0 is
        only valid at low redshifts.

    Returns
    -------
    Luminosity in the source frame
    """
    return L_M(M_mdl(m, dl, Kcorr=Kcorr))


# Rachel: I've put dl_zH0 and z_dlH0 in as place holders.
def dl_zH0(z, k, H0=70.):
    """
    Returns luminosity distance given distance and cosmological parameters

    Parameters
    ----------
    z : redshift
    H0 : Hubble parameter in km/s/Mpc (default=70.)

    Returns
    -------
    luminosity distance, dl (in Mpc)
    """
    return dLH0overc(z, k)*c/H0


def z_dlH0(dl, k, H0=70.):
    """
    Returns redshift given luminosity distance and cosmological parameters

    Parameters
    ----------
    dl : luminosity distance in Mpc
    H0 : Hubble parameter in km/s/Mpc (default=70.)

    Returns
    -------
    redshift, z
    """
    return redshift(dl*H0/c, k)


class redshift_prior(object):
    """
    p(z): Uniform in comoving volume distribution of galaxies

    Parameters
    ----------
    zmax : upper limit for redshift (default=10.0)
    """
    def __init__(self, k, zmax=10.0):
        self.zmax = zmax
        self.k = k
        z_array = np.logspace(-5, np.log10(self.zmax), 12000)
        lookup = np.log10(np.array([volume_z(z, self.k)
                          for z in z_array]))
        self.interp = splrep(np.log10(z_array), lookup)

    def p_z(self, z):
        return 10.**splev(np.log10(z), self.interp, ext=3)

    def __call__(self, z):
        return self.p_z(z)
         
class fast_cosmology(object):
    """

    Parameters
    ----------
    zmax : upper limit for redshift (default=10.0)
    """
    def __init__(self, k, zmax=10.0):
        self.zmax = zmax
        self.k = k

    def dl_zH0(self, z, k, H0):
        """
        Returns luminosity distance given distance and cosmological parameters

        Parameters
        ----------
        z : redshift
        H0 : Hubble parameter in km/s/Mpc

        Returns
        -------
        luminosity distance, dl (in Mpc)
        """
        return dLH0overc(z, k)*c/H0
   

    def E(self,z):
        """
        Returns the E(z) factor

        Parameters
        ----------
        z : redshift
        """
        return 1+z

    def dL_by_z_H0(self, z, H0, k):
        """
        Returns the derivative of the luminosity distance w.r.t. redshift

        Parameters
        ----------
        z : redshift
        H0 : Hubble constant in km Mpc-1 s-1
	"""
        if k == 0:
            return c/H0*(np.log(1+z)+1)
        if k == 1:
            return c/H0*(np.sin(np.log(1+z))+np.cos(np.log(1+z)))
        if k == -1:
            return c/H0*(1+z)
