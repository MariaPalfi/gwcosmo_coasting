"""
Rachel Gray 2020
"""

class SchechterParams():
    """
    Returns the source frame Schechter function parameters for a given band.

    Note Mstar is here is M* - 5log10(h)

    ugriz parameters are from https://iopscience.iop.org/article/10.1086/375776/pdf
    (tables 1 and 2)
    B parameters are from https://doi.org/10.1046/j.1365-8711.2002.05831.x
    K parameters are from section 2.1 of
    https://iopscience.iop.org/article/10.3847/0004-637X/832/1/39
    (note paper quotes M*=-23.55 but means M*=-23.55 + 5log10(h))
    """

    def __init__(self, band, schech_alpha=None, schech_Mstar=None, schech_Mmin=None, schech_Mmax=None):
        """
        Parameters
        ----------
        band : observation band (B,K,bJ,u,g,r,i,z)
        """

        self.Mstar = None
        self.alpha = None
        self.Mmin = None
        self.Mmax = None

        self.alpha, self.Mstar, self.Mmin, self.Mmax = self.default_values(band) #initialize to default values

        if schech_alpha is not None: #change to input value if provided
            self.alpha=schech_alpha
        if schech_Mstar is not None:
            self.Mstar=schech_Mstar
        if schech_Mmin is not None:
            self.Mmin=schech_Mmin
        if schech_Mmax is not None:
            self.Mmax=schech_Mmax

    def default_values(self, band): 
        if band == 'B':
            return -1.21, -19.70, -22.96, -12.96
        elif band =='bJ':
            # From https://arxiv.org/pdf/astro-ph/0111011.pdf
            # Mmax from Abstract, of the paper above, valid for fit.
            return -1.21, -19.66, -22.00, -16.5
        elif band == 'K':
            # https://iopscience.iop.org/article/10.1086/322488/pdf
            # Mmax from  Fig 3 of the above reference
            return -1.09, -23.39, -27.0, -19.0
        elif band == 'u':                            #These values are actually u', g', r', i', zi, a.k.a. redshifted to the median redshift (0.1) of SDSS.
            return -0.92, -17.93, -21.93, -15.54 
        elif band == 'g':
            return -0.89, -19.39, -23.38, -16.10 
        elif band == 'r':
            return -1.05, -20.44, -24.26, -16.11 
        elif band == 'i':
            return -1.00, -20.82, -23.84, -17.07 
        elif band == 'z':
            return -1.08, -21.18, -24.08, -17.34 
        elif band == 'W1':
            return -1.12, -24.09, -28, -16.6 # https://iopscience.iop.org/article/10.1088/0004-637X/697/1/506/pdf Tab 3 (All 3.6)
                                             # https://arxiv.org/pdf/1702.07829.pdf (Mmin Mmax Fig2)

        else:
            raise Exception("Expected 'W1', B', 'K', 'u', 'g', 'r', 'i' or 'z' band argument")


