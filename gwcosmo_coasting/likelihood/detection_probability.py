"""
Detection probability
Rachel Gray, John Veitch, Ignacio Magana, Dominika Zieba
Modified by Mária Pálfi (Eötvös University, Budapest, Hungary, e-mail: marika97@student.elte.hu)
            and Péter Raffai (Eötvös University, Budapest, Hungary) (2023)
"""
import lal
import lalsimulation as lalsim
import numpy as np
from scipy.interpolate import interp1d, splev, splrep, interp2d
from scipy.integrate import quad
from scipy.stats import ncx2
from scipy.special import logit, expit
from gwcosmo_coasting.prior.priors import mass_prior
from gwcosmo_coasting.utilities.coasting_cosmology import fast_cosmology,z_dlH0,dl_zH0 # use coasting_cosmology, changed by Mária Pálfi
import progressbar
import pkg_resources
import pickle
import os

class DetectionProbability(object):
    """
    Class to compute the detection probability p(D |z, H0) as a function of z and H0
    by marginalising over masses, inclination, polarisation, and sky location for a
    set of detectors at a given sensitivity.

    Parameters
    ----------
    mass_distribution : str
        Choose between BNS or NSBH/BBH-powerlaw, NSBH/BBH-powerlaw-gaussian, 
        NSBH/BBH-broken-powerlaw mass distributions for default Pdet calculations."
    psd : str
        Select between None for marginalization over the sensitivity or 'O1', 'O2', 
        'O3', 'O4low', 'O4high', 'O5' or the 'MDC' PSDs.
    detectors : list of str, optional
        list of detector names (default=['H1','L1', 'V1'])
        Select from 'L1', 'H1', 'V1', 'K1'.
    detected_masses : bool, optional 
        Set to True if you want to keep track of the detected masses (default=False)
    det_combination : bool, optional
        Set to True if you want to marginalize over detector combinations (default=True)
    Nsamps : int, optional
        Number of samples for monte carlo integration (default=5000)
    H0 : float or array, optional
        Value(s) of H0 at which to compute Pdet. If constant_H0 is True (default=70)
    k : int
        Curvature parameter of the coasting cosmology.
    network_snr_theshold : float, optional
        snr threshold for an individual detector (default=12)
    basic : bool, optional
        if True, don't redshift masses (for use with the MDC) (default=False)
    alpha : float, optional
        slope of the power law p(m) = m^-\alpha where alpha > 0 (default=1.6)
    Mmin : float, optional
        specify minimum source frame mass for BBH-powerlaw distribution (default=5)
    Mmax : float, optional
        specify maximum source frame mass for BBH-powerlaw distribution  (default=50)
    beta : float, optional
        Set powerlaw slope for the second black hole. (default=0)
    alpha_2 : float, optional
        Set second powerlaw slope for BBH with broken powerlaw mass distribution
    mu_g : float, optional
        Set the mu of the gaussian peak in case of BBH-powerlaw-gaussian mass distribution
    sigma_g : float, optional
        Set the sigma of the gaussian peak in case of BBH-powerlaw-gaussian mass distribution
    lambda_peak : float, optional
        Set the lambda of the gaussian peak in case of BBH-powerlaw-gaussian mass distribution.
    delta_m : float, optional
        Set the smoothing parameter in case of BBH-powerlaw-gaussian or BBH-broken-powerlaw mass distributions
    b : float, optional
        Set the fraction at which the powerlaw breaks in case of BBH-broken-powerlaw mass distribution
    M1, M2 : float, optional
        specify source masses in solar masses if using BBH-constant mass distribution (default=50,50)
    constant_H0 : bool, optional
        if True, set Hubble constant to 70 kms-1Mpc-1 for all calculations (default=False)
    full_waveform: bool, optional
        if True, use LALsimulation simulated inspiral waveform, otherwise use just the inspiral (default=True)
    seed : int, optional
        Set the random seed (default=1000)
    path: str, optional
        The output path of the file(used for the checkpointing)
    

    """
    def __init__(self, mass_distribution, asd=None, detectors=['H1', 'L1', 'V1'], detected_masses=False, det_combination=True,
                 Nsamps=5000, H0=70, k=None, network_snr_threshold=12, basic=False,
                 alpha=1.6, Mmin=5., Mmax=50., beta=0., alpha_2=0., mu_g=35., sigma_g=5., lambda_peak=0.2,
                 delta_m=0., b=0.5, M1=50., M2=50., constant_H0=False, full_waveform=True, seed=1000,path='./'): # k=None added by Mária Pálfi

        self.data_path = pkg_resources.resource_filename('gwcosmo_coasting', 'data/')
        self.mass_distribution = mass_distribution
        self.asd = asd
        self.detectors = detectors
        self.Nsamps = Nsamps
        self.H0vec = H0
        self.snr_threshold = network_snr_threshold
        self.alpha = alpha
        self.Mmin = Mmin
        self.Mmax = Mmax
        self.alpha_2 = alpha_2
        self.beta = beta
        self.mu_g = mu_g
        self.sigma_g = sigma_g
        self.lambda_peak = lambda_peak
        self.delta_m = delta_m
        self.b = b
        self.M1 = M1
        self.M2 = M2
        self.full_waveform = full_waveform
        self.constant_H0 = constant_H0
        self.seed = seed
        self.detected_masses = detected_masses
        self.det_combination = det_combination
        self.k = k # curvature parameter, added by Mária Pálfi
        self.cosmo = fast_cosmology(k = self.k) # added by Mária Pálfi
        self.path = str(path)+'_checkpoint.p'


        if self.full_waveform is True:
            self.z_array = np.logspace(-4.0, 1., 500)
        else:
            # TODO: For higher values of z (z=10) this goes
            # outside the range of the psds and gives an error
            self.z_array = np.logspace(-4.0, 0.5, 500)

        checkpoint_z = 0
        prob_checkpoint = np.zeros(len(self.z_array))
        detect = np.ones(self.Nsamps)
        self.detected = 0
        if self.detected_masses==True:
            self.detected = np.zeros((len(self.z_array),self.Nsamps),dtype=np.float32)

        if os.path.isfile(self.path):
            pdet_checkpoint = pickle.load(open(self.path, 'rb'))
            self.z_array = pdet_checkpoint['z_array']
            checkpoint_z = pdet_checkpoint['checkpoint_z']
            prob_checkpoint = pdet_checkpoint['prob_checkpoint']
            self.detected = pdet_checkpoint['detected']
            self.seed = pdet_checkpoint['seed']
            detect = pdet_checkpoint['detect']
        np.random.seed(self.seed)
        # set up the samples for monte carlo integral
        N = self.Nsamps
        self.RAs = np.random.rand(N)*2.0*np.pi
        r = np.random.rand(N)
        self.Decs = np.arcsin(2.0*r - 1.0)
        q = np.random.rand(N)
        self.incs = np.arccos(2.0*q - 1.0)
        self.psis = np.random.rand(N)*2.0*np.pi
        self.phis = np.random.rand(N)*2.0*np.pi

        self.hyper_params_dict = {'alpha':self.alpha,'alpha_2':self.alpha_2,'mmin':self.Mmin,'mmax':self.Mmax,
                                  'beta':self.beta,'sigma_g':self.sigma_g,'lambda_peak':self.lambda_peak,
                                  'mu_g':self.mu_g,'delta_m':self.delta_m,'b':self.b}

        sampling = mass_prior(name=self.mass_distribution, hyper_params_dict=self.hyper_params_dict)
        m1, m2 = sampling.sample(N)

        if self.mass_distribution == 'BNS' or self.mass_distribution.startswith('NSBH'):
            self.dl_array = np.linspace(1.0e-100, 1000.0, 500)
        else:
            self.dl_array = np.linspace(1.0e-100, 15000.0, 500)

        self.m1 = m1*1.9884754153381438e30
        self.m2 = m2*1.9884754153381438e30

        self.M_min = np.min(self.m1)+np.min(self.m2)

        self.df = 1          #set sampling frequency interval to 1 Hz
        self.f_min = 10      #10 Hz minimum frequency
        self.f_max = 4999    #5000 Hz maximum frequency

        if (self.asd == 'MDC' or self.asd == 'O1') and ('V1' in self.detectors):
            self.detectors.remove('V1')

        duty_factors = {'O3':{'H1':0.75,'L1':0.75,'V1':0.75},'O2':{'H1':0.60,'L1':0.60,'V1':-1},'O1':{'H1':0.60,'L1':0.50,'V1':-1},'MDC':{'H1':1.,'L1':1.,'V1':-1.}}
        days_of_run = {'O3':361,'O2':268,'O1':129 ,'MDC':100} # data taken from https://www.gw-openscience.org/timeline for O1,O2

        if self.asd == None:
            self.duty_factor = {'O3':{},'O2':{},'O1':{}}
        else:
            self.duty_factor={asd:{}}

        self.days_of_runs ={}
        for psd in self.duty_factor:
            self.days_of_runs[psd] = days_of_run[psd]

        if self.det_combination==True:
            for p in self.duty_factor:
                for d in duty_factors[p]:
                    if d in self.detectors:
                        self.duty_factor[p][d] = duty_factors[p][d]
                    else:
                        self.duty_factor[p][d] = -1.0

        else:
            for p in self.duty_factor:
                for d in duty_factors[p]:
                    if d in self.detectors:
                        self.duty_factor[p][d] = 1.0
                    else:
                        self.duty_factor[p][d] = -1.0
                    if p=='O1' and d=='V1':
                        self.duty_factor[p][d] = -1.0

        total_days = 0
        for key in self.days_of_runs:
            total_days+=self.days_of_runs[key]
        self.prob_of_run = {}
        for key in self.days_of_runs:
            self.prob_of_run[key] = self.days_of_runs[key]/total_days
        self.psds = []
        self.dets = []
        p = np.random.rand(N)

        if self.asd == None:
            self.asds = {'O3':{},'O2':{},'O1':{}}    #this is now a dictionary of functions
            ASD_data = {'O3':{},'O2':{},'O1':{}}
            self.__interpolnum = {'O3':{},'O2':{},'O1':{}}
            for i in range(N):
                if 0<=p[i]<=self.prob_of_run['O1']:
                    psd = 'O1'
                elif self.prob_of_run['O1']<p[i]<=self.prob_of_run['O1']+self.prob_of_run['O2']:
                    psd = 'O2'
                else:
                    psd = 'O3'
                self.psds.append(psd)
        else:
            self.asds = {self.asd:{}}
            ASD_data = {self.asd:{}}
            self.__interpolnum = {self.asd:{}}
            for i in range(N):
                self.psds.append(self.asd)

        for i in range(N):
            d = []
            while len(d)==0:
                h = np.random.rand()
                l = np.random.rand()
                v = np.random.rand()
                if (h<=self.duty_factor[self.psds[i]]['H1']) and ('H1' in self.detectors):
                    d.append('H1')
                if (l<=self.duty_factor[self.psds[i]]['L1']) and ('L1' in self.detectors):
                    d.append('L1')
                if (v<=self.duty_factor[self.psds[i]]['V1']) and ('V1' in self.detectors):
                    d.append('V1')
            self.dets.append(d)

        for run in self.asds:
            for det in self.duty_factor[run]:
                if self.duty_factor[run][det]>0:
                    if run == 'MDC':
                        ASD_data[run][det] = np.genfromtxt(self.data_path + 'PSD_L1_H1_mid.txt')
                    else:
                        ASD_data[run][det] = np.genfromtxt(self.data_path +str(det)+ '_'+ str(run) + '_strain.txt')
                    self.asds[run][det] = interp1d(ASD_data[run][det][:, 0], ASD_data[run][det][:, 1])
                    if basic==True or full_waveform==False:
                        self.__interpolnum[run][det] = self.__numfmax_fmax(self.M_min, det, run)

        if self.asd != None:
            print("Calculating pdet with " + self.asd + " sensitivity and " +
                    self.mass_distribution + " mass distribution.")
        else:
            print("Calculating pdet with marginalizing over sensitivity and with " +
                    self.mass_distribution + " mass distribution.")
        if basic is True:
            self.interp_average_basic = self.__pD_dl_basic()

        elif constant_H0 is True:
            self.prob = self.__pD_zH0(H0,prob_checkpoint,detect,checkpoint_z)  
            logit_prob=logit(self.prob)
            logit_prob=np.where(logit_prob==float('+inf'), 100, logit_prob)
            logit_prob=np.where(logit_prob==float('-inf'), -33, logit_prob)
            self.interp_average = interp1d(self.z_array, logit_prob, kind='cubic')

        else:
            self.prob = self.__pD_zH0_array(self.H0vec,prob_checkpoint,detect,checkpoint_z)

            #interpolation of prob is done in logit(prob)=prob/(1-prob)
            #this prevents values from going above 1 and below 0
            #if prob=1, logit(prob)=inf.
            #to solve this for interpolation purposes, set logit(prob=1)=100, so then expit(100)=logit^-1(100)=1
            #instead of 100 anything from 35 to sys.float_info.max can be set as in this range expit is effectively 1
            #yet: higher values make interpolation less effective

            logit_prob=logit(self.prob)
            for i in range (len(logit_prob)):
                logit_prob[i]=np.where(logit_prob[i]==float('+inf'), 100, logit_prob[i])
            self.interp_average = interp2d(self.z_array, self.H0vec, logit_prob, kind='cubic')

    def mchirp(self, m1, m2):
        """
        Calculates the source chirp mass

        Parameters
        ----------
        m1,m2 : float
            source rest frame masses in kg

        Returns
        -------
        Source chirp mass in kg
        """
        return np.power(m1*m2, 3.0/5.0)/np.power(m1+m2, 1.0/5.0)

    def mchirp_obs(self, m1, m2, z=0):
        """
        Calculates the redshifted chirp mass from source masses and a redshift

        Parameters
        ----------
        m1,m2 : float
            source rest frame masses in kg
        z : float
            redshift (default=0)

        Returns
        -------
        float
            Redshifted chirp mass in kg
        """
        return (1+z)*self.mchirp(m1, m2)

    def __mtot(self, m1, m2):
        """
        Calculates the total source mass of the system

        Parameters
        ----------
        m1,m2 : float
            source rest frame masses in kg

        Returns
        -------
        float
            Source total mass in kg
        """
        return m1+m2

    def __mtot_obs(self, m1, m2, z=0):
        """
        Calculates the total observed mass of the system

        Parameters
        ----------
        m1,m2 : float
            source rest frame masses in kg
        z : float
            redshift (default=0)

        Returns
        -------
        float
            Observed total mass in kg
        """
        return (m1+m2)*(1+z)


    def __Fplus(self, detector, RA, Dec, psi, gmst):
        """
        Computes the 'plus' antenna pattern

        Parameters
        ----------
        detector : str
            name of detector in network (eg 'H1', 'L1')
        RA,Dec : float
            sky location of the event in radians
        psi : float
            source polarisation in radians
        gmst : float
            Greenwich Mean Sidereal Time in seconds

        Returns
        -------
        float
            F_+ antenna response
        """
        detector = lalsim.DetectorPrefixToLALDetector(detector)
        return lal.ComputeDetAMResponse(detector.response, RA,
                                        Dec, psi, gmst)[0]

    def __Fcross(self, detector, RA, Dec, psi, gmst):
        """
        Computes the 'plus' antenna pattern

        Parameters
        ----------
        detector : str
            name of detector in network (eg 'H1', 'L1')
        RA,Dec : float
            sky location of the event in radians
        psi : float
            source polarisation in radians
        gmst : float
            Greenwich Mean Sidereal Time in seconds

        Returns
        -------
        float
            F_x antenna response
        """
        detector = lalsim.DetectorPrefixToLALDetector(detector)
        return lal.ComputeDetAMResponse(detector.response, RA,
                                        Dec, psi, gmst)[1]

    def simulate_waveform(self, m1, m2, dl, inc, phi,
                          S1x=0., S1y=0., S1z=0., S2x=0., S2y=0., S2z=0., lAN=0., e=0., Ano=0.):
        """
        Simulates frequency domain inspiral waveform

        Parameters
        ----------
        m1, m2 : float
            observed source masses in kg
        dl : float
            luminosity distance in Mpc
        inc: float
            source inclination in radians
        phi : float
            source reference phase in radians
        S1x, S1y, S1z : float, optional
            x,y,z-components of dimensionless spins of body 1 (default=0.)
        S2x, S2y, S2z : float, optional
            x,y,z-components of dimensionless spin of body 2 (default=0.)
        lAN: float, optional
            longitude of ascending nodes (default=0.)
        e: float, optional
            eccentricity at reference epoch (default=0.)
        Ano: float, optional
            mean anomaly at reference epoch (default=0.)

        Returns
        -------
        lists
            hp and hc
        """

        hp, hc = lalsim.SimInspiralChooseFDWaveform(
                    m1, m2,
                    S1x, S1y, S1z, S2x, S2y, S2z,
                    dl*1e6*lal.PC_SI, inc, phi, lAN, e, Ano,
                    self.df, self.f_min, self.f_max, 20,
                    lal.CreateDict(),
                    lalsim.IMRPhenomD)
        hp = hp.data.data
        hc = hc.data.data

        return hp,hc

    def simulate_waveform_response(self, hp, hc, RA, Dec, psi, gmst, detector):
        """
        Applies antenna response to frequency domain inspiral waveform

        Parameters
        ----------
        hp, hc : lists
            plus and cross components of the frequency domain inspiral waveform
        RA, Dec : float
            sky location of the event in radians
        psi : float
            source polarisation in radians
        phi : float
            source reference phase in radians
        gmst : float
            Greenwich Mean Sidereal Time in seconds
        detector : str
            name of detector in network (eg 'H1', 'L1')

        Returns
        -------
        complex array
            complex frequency series - detected h(f)

        real array
            array of frequencies corresponding to h(f)
        """

        #apply antenna response
        hf = hp*self.__Fplus(detector, RA, Dec, psi, gmst) + hc*self.__Fcross(detector, RA, Dec, psi, gmst)

        #recreate frequency array
        f_array=self.df*np.arange(len(hf))
        start=np.where(f_array == self.f_min)[0][0]
        end=np.where(f_array == self.f_max)[0][0]

        return hf[start: end + 1], f_array[start: end + 1]


    def snr_squared_waveform(self, hp, hc, RA, Dec, psi, gmst, detector, psd):
        """
        Calculates SNR squared of the simulated inspiral waveform for single detector

        Parameters
        ----------
        hp, hc : lists
            plus and cross components of the frequency domain inspiral waveform
        RA, Dec : float
            sky location of the event in radians
        psi : float
            source polarisation in radians
        gmst : float
            Greenwich Mean Sidereal Time in seconds
        detector : str
            name of detector in network (eg 'H1', 'L1')

        Returns
        -------
        float
            SNR squared

        """

        hf, f_array = self.simulate_waveform_response(hp, hc, RA, Dec, psi, gmst, detector)
        df = f_array[1]-f_array[0]
        SNR_squared=4*np.sum((np.abs(hf)**2/self.asds[psd][detector](f_array)**2)*df)
        return SNR_squared


    def __reduced_amplitude(self, RA, Dec, inc, psi, detector, gmst):
        """
        Component of the Fourier amplitude, with redshift-dependent
        parts removed computes:
        [F+^2*(1+cos(i)^2)^2 + Fx^2*4*cos(i)^2]^1/2 * [5*pi/96]^1/2 * pi^-7/6

        Parameters
        ----------
        RA,Dec : float
            sky location of the event in radians
        inc : float
            source inclination in radians
        psi : float
            source polarisation in radians
        detector : str
            name of detector in network as a string (eg 'H1', 'L1')
        gmst : float
            Greenwich Mean Sidereal Time in seconds

        Returns
        -------
        float
            Component of the Fourier amplitude, with
            redshift-dependent parts removed
        """
        Fplus = self.__Fplus(detector, RA, Dec, psi, gmst)
        Fcross = self.__Fcross(detector, RA, Dec, psi, gmst)
        return np.sqrt(Fplus**2*(1.0+np.cos(inc)**2)**2 + Fcross**2*4.0*np.cos(inc)**2)*np.sqrt(5.0*np.pi/96.0)*np.power(np.pi, -7.0/6.0)

    def __fmax(self, M):
        """
        Maximum frequency for integration, set by the frequency of
        the innermost stable orbit (ISO)
        fmax(M) 2*f_ISO = (6^(3/2)*pi*M)^-1

        Parameters
        ----------
        M : float
            total mass of the system in kg

        Returns
        -------
        float
            Maximum frequency in Hz
        """
        return 1/(np.power(6.0, 3.0/2.0)*np.pi*M) * lal.C_SI**3/lal.G_SI

    def __numfmax_fmax(self, M_min, detector, psd):
        """
        lookup table for snr as a function of max frequency
        Calculates \int_fmin^fmax f'^(-7/3)/S_h(f')
        df over a range of values for fmax
        fmin: 10 Hz
        fmax(M): (6^(3/2)*pi*M)^-1
        and fmax varies from fmin to fmax(M_min)

        Parameters
        ----------
        M_min : float
            total minimum mass of the distribution in kg

        Returns
        -------
        Interpolated 1D array of \int_fmin^fmax f'^(-7/3)/S_h(f')
        for different fmax's
        """
        ASD = self.asds[psd][detector]
        fmax = lambda m: self.__fmax(m)
        I = lambda f: np.power(f, -7.0/3.0)/(ASD(f)**2)
        f_min = self.f_min  # Hz, changed this from 20 to 10 to resolve NaN error
        f_max = fmax(M_min)

        arr_fmax = np.linspace(f_min, f_max, self.Nsamps)
        num_fmax = np.zeros(self.Nsamps)
        bar = progressbar.ProgressBar()
        print("Calculating lookup table for snr as a function of max frequency.")
        for i in bar(range(self.Nsamps)):
            num_fmax[i] = quad(I, f_min, arr_fmax[i],epsabs=0, epsrel=1.49e-4)[0]

        return interp1d(arr_fmax, num_fmax)


    def __snr_squared(self, RA, Dec, m1, m2, inc, psi, detector, gmst, z, H0, psd):
        """
        the optimal snr squared for one detector, used for marginalising
        over sky location, inclination, polarisation, mass

        Parameters
        ----------
        RA,Dec : float
            sky location of the event in radians
        m1,m2 : float
            source masses in kg
        inc : float
            source inclination in radians
        psi : float
            source polarisation in radians
        detector : str
            name of detector in network as a string (eg 'H1', 'L1')
        gmst : float
            Greenwich Mean Sidereal Time in seconds
        z : float
            redshift
        H0 : float
            value of Hubble constant in kms-1Mpc-1

        Returns
        -------
        float
            snr squared for given parameters at a single detector
        """
        mtot = self.__mtot_obs(m1, m2, z)
        mc = self.mchirp_obs(m1, m2, z)
        A = self.__reduced_amplitude(RA, Dec, inc, psi, detector, gmst) * np.power(mc, 5.0/6.0) / (self.cosmo.dl_zH0(z, H0, k)*lal.PC_SI*1e6)

        fmax = self.__fmax(mtot)
        num = self.__interpolnum[psd][detector](fmax)

        return 4.0*A**2*num*np.power(lal.G_SI, 5.0/3.0)/lal.C_SI**3.0

    def __pD_zH0(self, H0, prob, detect, checkpoint):
        """
        Detection probability over a range of redshifts and H0s,
        returned as an interpolated function.

        Parameters
        ----------
        H0 : float
            value of Hubble constant in kms-1Mpc-1

        Returns
        -------
        interpolated probabilities of detection over an array of
        luminosity distances, for a specific value of H0
        """
        lal_detectors = [lalsim.DetectorPrefixToLALDetector(name)
                                for name in self.detectors]
        
        bar =  progressbar.ProgressBar()
        for i in bar(range(checkpoint, len(self.z_array))):
            z = self.z_array[i]
            k = self.k # curvature parameter added by Mária Pálfi
            dl = self.cosmo.dl_zH0(z, k, H0) # changed by Mária Pálfi
            factor = 1+z
            network_rhosq = np.zeros(self.Nsamps)
            for n in range(self.Nsamps):
                detectors = self.dets[n]
                psd = self.psds[n]
                if detect[n] == 1:               
                    if self.full_waveform is True: 
                        hp,hc = self.simulate_waveform(factor*self.m1[n], factor*self.m2[n], dl, self.incs[n], self.phis[n])
                        rhosqs = [self.snr_squared_waveform(hp,hc,self.RAs[n],self.Decs[n],self.psis[n], 0., det, psd)
                              for det in detectors]

                    else:
                        rhosqs = [self.__snr_squared(self.RAs[n], self.Decs[n],
                              self.m1[n], self.m2[n], self.incs[n], self.psis[n],
                              det, 0.0, self.z_array[i], H0, psd)
                              for det in detectors]
                    network_rhosq[n] = np.sum(rhosqs)
                    
            survival = ncx2.sf(self.snr_threshold**2, 2*len(self.detectors), network_rhosq)
            prob[i] = np.sum(survival, 0)/self.Nsamps
            
            if self.detected_masses==True:
                self.detected[i] = np.float32(survival)
                
            not_surviving_samples = np.where(survival<=1e-6)[0] #threshold to consider event undetectable
            detect[not_surviving_samples] = 0.            
            
            if i%20==0:
                if os.path.isfile(self.path):
                    os.remove(self.path)
                checkpoint = self.checkpointing(detect,prob,i)
                pickle.dump(checkpoint, open( self.path, "wb" ))
        if os.path.isfile(self.path):
                os.remove(self.path)       
        return prob
    
    def checkpointing(self,detect,prob,i):

        return {'seed':self.seed,'detect':detect,'detected':self.detected,'prob_checkpoint':prob,'checkpoint_z':i,'z_array':self.z_array}


    def __pD_zH0_array(self, H0vec, prob_checkpoint,detect,checkpoint_z):
        """
        Function which calculates p(D|z,H0) for a range of
        redshift and H0 values

        Parameters
        ----------
        H0vec : array_like
            array of H0 values in kms-1Mpc-1

        Returns
        -------
        list of arrays?
            redshift, H0 values, and the corresponding p(D|z,H0) for a grid
        """
        return np.array([self.__pD_zH0(H0,prob_checkpoint,detect,checkpoint_z) for H0 in H0vec])

    def pD_dlH0_eval(self, dl, H0):
        """
        Returns the probability of detection at a given value of
        luminosity distance and H0.
        Note that this is slower than the function pD_zH0_eval(z,H0).

        Parameters
        ----------
        dl : float or array_like
            value(s) of luminosity distances in Mpc
        H0 : float
            H0 value in kms-1Mpc-1

        Returns
        -------
        float or array_like
            Probability of detection at the given luminosity distance and H0,
            marginalised over masses, inc, pol, and sky location
        """
        z = np.array([z_dlH0(x, H0, k) for x in dl])
        return expit(self.interp_average(z, H0))

    def pD_z_eval(self, z):
        """
        Returns the probability of detection at a given value of
        redshift. To be used with Constant_H0 option set to True only.

        Parameters
        ----------
        z : float or array_like
            value(s) of redshift

        Returns
        -------
        float or array_like
            Probability of detection at the given redshift,
            marginalised over masses, inc, pol, and sky location
        """
        return expit(self.interp_average(z))

    def pD_zH0_eval(self, z, H0):
        """
        Returns the probability of detection at a given value of
        redshift and H0.

        Parameters
        ----------
        z : float or array_like
            value(s) of redshift
        H0 : float
            H0 value in kms-1Mpc-1

        Returns
        -------
        float or array_like
            Probability of detection at the given redshift and H0,
            marginalised over masses, inc, pol, and sky location
        """
        return expit(self.interp_average(z,H0))

    def __call__(self, z, H0):
        """
        To call as function of z and H0

        Parameters
        ----------
        z : float or array_like
            value(s) of redshift
        H0 : float
            H0 value in kms-1Mpc-1

        Returns
        -------
        float or array_like
            Returns Pdet(z,H0).
        """
        return self.pD_zH0_eval(z, H0)

    def pD_distmax(self, dl, H0):
        """
        Parameters
        ----------
        dl : float or array_like
            value(s) of luminosity distances in Mpc
        H0 : float
            H0 value in kms-1Mpc-1

        Returns
        -------
        float or array_like
            Returns twice the maximum distance given corresponding
            to Pdet(dl,H0) = 0.01.
        """
        return 2.*dl[np.where(self.pD_dlH0_eval(dl, H0) > 0.01)[0][-1]]

    def __snr_squared_basic(self, RA, Dec, m1, m2, inc, psi, detector, gmst, dl):
        """
        the optimal snr squared for one detector, used for marginalising over
        sky location, inclination, polarisation, mass
        Note that this ignores the redshifting of source masses.

        Parameters
        ----------
        RA,Dec : float
            sky location of the event in radians
        m1,m2 : float
            source masses in kg
        inc : float
            source inclination in radians
        psi : float
            source polarisation in radians
        detector : str
            name of detector in network as a string (eg 'H1', 'L1')
        gmst : float
            Greenwich Mean Sidereal Time in seconds
        dl : float
            luminosity distance in Mpc

        Returns
        -------
        float
            snr squared for given parameters at a single detector
        """
        mtot = self.__mtot(m1, m2)
        mc = self.mchirp(m1, m2)
        A = self.__reduced_amplitude(RA, Dec, inc, psi, detector, gmst) * np.power(mc, 5.0/6.0) / (dl*lal.PC_SI*1e6)

        fmax = self.__fmax(mtot)
        num = self.__interpolnum[detector](fmax)

        return 4.0*A**2*num*np.power(lal.G_SI, 5.0/3.0)/lal.C_SI**3.0

    def __pD_dl_basic(self):
        """
        Detection probability over a range of distances,
        returned as an interpolated function.
        Note that this ignores the redshifting of source masses.

        Returns
        -------
        interpolated probabilities of detection over an array of luminosity
        distances, for a specific value of H0.
        """

        rho = np.zeros((self.Nsamps, len(self.dl_array)))
        for n in range(self.Nsamps):
            rhosqs = [self.__snr_squared_basic(self.RAs[n], self.Decs[n], self.m1[n], self.m2[n], self.incs[n], self.psis[n], det, 0.0, self.dl_array) for det in self.detectors]
            rho[n] = np.sum(rhosqs, 0)

        survival = ncx2.sf(self.snr_threshold**2, 2*len(self.detectors), rho)
        prob = np.sum(survival, 0)/self.Nsamps
        self.spl = splrep(self.dl_array, prob)
        return splrep(self.dl_array, prob)

    def pD_dl_eval_basic(self, dl):
        """
        Returns a probability of detection at a given luminosity distance
        Note that this ignores the redshifting of source masses.

        Parameters
        ----------
        dl : float or array_like
            value(s) of luminosity distances in Mpc

        Returns
        -------
        float or array_like
            Probability of detection at the given luminosity distance and H0,
            marginalised over masses, inc, pol, and sky location
        """
        return splev(dl, self.spl, ext=1)
