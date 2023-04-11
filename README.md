# gwcosmo_coasting

A package to estimate Hubble parameter of coasting cosmologies with curvature parameter k = \[-1, 0, 1\] using gravitational-wave observations. Created by Mária Pálfi (Eötvös University, Budapest, Hungary, e-mail: marika97@student.elte.hu) and Péter Raffai (Eötvös University, Budapest, Hungary). 

This package is built upon the [**gwcosmo** ](https://git.ligo.org/lscsoft/gwcosmo) package described in  [R. Gray et al. Phys. Rev. D 101, 122001](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.101.122001) and [R. Gray et al. arXiv:2111.04629](https://arxiv.org/abs/2111.04629).

If you use this code please cite this paper (link) .

# How-to install

Clone the repository with 
```
    git clone <repository>
```
The name of the repository can be copied from the git interface (top right button). If you do not have ssh key on git, please use the `https` protocol

Complete the install by following one of the two options below. The package requires Python version 3.7 or newer to run (Python version 3.10 is recommended).

The installation of the package works the same way as installation of [**gwcosmo** ](https://git.ligo.org/lscsoft/gwcosmo).

## Installing with Anaconda

You will need an [Anaconda distribution](https://www.anaconda.com/). The conda distribution is correctly initialized when, if you open your terminal, you will see the name of the python environment used. The default name is `(base)`.

Once the conda distribution is installed and activated on your machine, please follow these steps:

* Enter the cloned **gwcosmo_coasting** directory.

* Create a conda virtual environment to host gwcosmo_coasting. Use
```
conda create -n gwcosmo_coasting pyton=3.10
```
* When the virtual environment is ready, activate it with (your python distribution will change to `gwcosmo_coasting`)
```
conda activate gwcosmo_coasting
```
* Install **gwcosmo_coasting** by running 
```
pip install -e .
```
Now the **gwcosmo_coasting** package is installed. The `-e` stands for "editable" and means that your installation will automatically update when you make changes to the code.

## Installing with venv

`venv` is included in Python for versions >=3.3.

* Create a virtual environment to host **gwcosmo_coasting**. Use
```
python -m venv gwcoasting_env
```
* When the virtual environment is ready, activate it with
```
source gwcoasting_env/bin/activate
```
* Enter the cloned gwcosmo_coasting directory.
* Install **gwcosmo_coasting** by running 
```
pip install -e .
```



# Data required for running **gwcosmo_coasting**

## Gravitational-wave data
The gravitational-wave (GW) data required to the analysis are posterior samples (in hdf5 format) and skymaps (in fits format). These are publicly available through the [Gravitational Wave Open Science Center](https://www.gw-openscience.org/). The data of each releases can also be found on the following pages:

* [GWTC-1 posterior samples](https://dcc.ligo.org/LIGO-P1800370/public)

* [GWTC-1 skymaps](https://dcc.ligo.org/LIGO-P1800381/public)

* [GWTC-2 (O3a)](https://dcc.ligo.org/LIGO-P2000223-v7/public)
* [GWTC-2.1 (O3a)](https://zenodo.org/record/5117703)
* [GWTC-3 (O3b)](https://zenodo.org/record/5546663)

## Probability of detection
Another necessary input of all **gwcosmo_coasting** analyses is the gravitational-wave probability of detection ("Pdet" for short), as a function of redshift and $`H_0`$, which allows **gwcosmo_coasting** to account for gravitational-wave selection effects. The Pdets used in the analysis are available in the data folder.

The Pdets can be computed with the help of the `gwcosmo_coasting_compute_pdet` script. Running `gwcosmo_coasting_compute_pdet --help` will show all the available flags that you can pass. 

First, run the `gwcosmo_coasting_compute_pdet` script with `--constant_H0 True` and `--H0 the_value_that_you_want` and name the output files differently after the `--outputfile` flag. You can process multiple calculations parallelly.

For example, if you would like to calculate Pdets with curvature parameter $k=0$, $H_0 = 100$ (in km/s/Mpc units) for binary black holes (BBHs) and with the same mass distribution as used in the [O3 cosmology paper](https://arxiv.org/abs/2111.03604) main analysis, run the following command:


```
gwcosmo_coasting_compute_pdet --k 0 --mass_distribution BBH-powerlaw-gaussian --powerlaw_slope 3.78 --beta 0.81 --minimum_mass 4.98 --maximum_mass 112.5 --mu_g 32.27 --lambda_peak 0.03 --sigma_g 3.88 --delta_m 4.8 --b 0.5 --full_waveform True --Nsamps 150000 --constant_H0 True --snr 11.0 --detected_masses False --detectors HLV --det_combination True --seed 1000 --H0 100 --outputfile constant_run_H0_100_k_0_BBH.p
```

For binary neutron star (BNS) mergers:

```
gwcosmo_coasting_compute_pdet --k 0 --mass_distribution BNS --full_waveform True --Nsamps 50000 --constant_H0 True --snr 11.0 --detected_masses False --detectors HLV --det_combination True --seed 1000 --H0 100 --outputfile  constant_run_H0_100_k_0_BNS.p
```

And for neutron star - black hole (NSBH) mergers:

```
gwcosmo_coasting_compute_pdet --k 0 --mass_distribution NSBH-powerlaw-gaussian --powerlaw_slope 3.78 --beta 0.81 --minimum_mass 4.98 --maximum_mass 112.5 --mu_g 32.27 --lambda_peak 0.03 --sigma_g 3.88 --delta_m 4.8 --b 0.5 --full_waveform True --Nsamps 150000 --constant_H0 True --snr 11.0 --detected_masses False --detectors HLV --det_combination True --seed 1000 --H0 100 --outputfile  constant_run_H0_100_k_0_BNS.p
```

If you have the Pdets for each constant $H_0$ value you want (we used a minimum of $H_0$ = 20, a maximum of $H_0$ = 140 and 100 bins), you can combine them with the following command:

```
gwcosmo_coasting_compute_pdet --combine path_to_the_folder_of_the_constant_H0_files
```
This should be run separately for the three types of mergers (BBH, BNS, NSBH). If you do not specify an output file name, you will get `BBH-powerlaw-gaussian_alpha_3.78_beta_0.81_Mmin_4.98_Mmax_112.5_mu_32.27_lambda_0.03_sigma_3.88_delta_4.8_Nsamps150000_full_waveform_snr_11.0_k_0.p` for binary black hole mergers and for a curvature parameter of k = 0.




## Galaxy catalogues
A galaxy catalogue is required for the galaxy catalogue analysis, such as the GLADE 2.4 or the GLADE+ galaxy catalogues (see http://glade.elte.hu/). These catalogues have to be preprocessed with the scripts under `scripts_galaxy_catalogs/` folder.


# Computing the posterior on `$H_0$` for a single gravitational-wave event
Similar to the **gwcosmo** package, the main executable is `gwcosmo_coasting_single_posterior`, which calculates the posterior on $`H_0`$ for a single gravitational-wave event.

There are 4 main options:
1. **The counterpart method:** the GW data is used in conjunction with a known associated host galaxy to measure $`H_0`$.
1. **The population method:** the empty catalogue method, where no EM information is assumed, and all information on $`H_0`$ comes from chosen priors, and the mass and distance information of the GW event.
1. **The statistical method:** the GW data is used in conjunction with a galaxy catalogue which updates the redshift prior using known galaxies, and assumes uniform galaxy catalogue incompleteness across the sky-area of the GW event.
1. **The pixel method:** an improvement to the statistical method, where its applied on a pixel-by-pixel basis.

You can learn more about these on the [**gwcosmo** website](https://git.ligo.org/lscsoft/gwcosmo), these are similar in **gwcosmo_coasting**. By running
```
gwcosmo_coasting_single_posterior --help
``` 
you can check the command line arguments and their default values.


When running the counterpart, population or statistical methods, the output comes in the form of files called **eventname.npz** and **eventname_likelihood_breakdown.npz**, and a figure **eventname.png**.

**eventname.npz** contains the following: `[H0,likelihood,posterior_uniform_norm,posterior_log_norm,opts]`

**eventname_likelihood_breakdown.npz** contains `[H0, likelihood, pxG, pDG, pG, pxB, pDB, pB, pxO, pDO, pO]`, where `likelihood = (pxG / pDG) * pG + (pxB / pDB) * pB + (pxO / pDO) * pO`.

We show some examples of the EM counterpart method and the pixelated method, which were used in the paper (link). The usage of the population and statistical (quick approximation) methods are similar as in the original [**gwcosmo** package](https://git.ligo.org/lscsoft/gwcosmo).



## The EM counterpart method
In this case we know the redshift (or velocity) and its uncertanty of the EM counterpart. The counterpart recession velocity or redshift has to be  corrected for peculiar motion in advance. It assume that the GW posterior samples have already been conditioned on the line-of-sight of the EM counterpart.

### Example
We run the EM counterpart method for GW170817, the only GW which has a confident EM counterpart. The example show the case when $k = -1$.


```
gwcosmo_coasting_single_posterior --method counterpart --posterior_samples GW170817_GWTC-1.hdf5 --Pdet /path_to_Pdets/BNS_Nsamps50000_full_waveform_snr_11.0_k_-1.p --counterpart_v 3017 --counterpart_sigmav 166 --min_H0 20 --max_H0 140 --bins_H0 100 --reweight_posterior_samples True --outputfile GW170817_k-1  --redshift_evolution Madau --Lambda 4.59 --Madau_beta 2.86 --Madau_zp 2.47 
```

## The galaxy catalogue method (pixelated)

This method needs GW posterior samples, GW skymap and a galaxy catalogue. It can take into account non-uniform catalogue and makes full use of GW data.


It is recommended to use a cluster for this method to parallelise the analysis, otherwise it is very time consuming. Here, we only present how to generate and use a DAG for the pixelated method, but you can use your own computer and follow the steps described on the [**gwcosmo** website](https://git.ligo.org/lscsoft/gwcosmo).



### Example

First set the galaxy catalogue path by running

```
export GWCOSMO_CATALOG_PATH=/path/to/catalog/directory/
```

For GW150914 and $k = -1$ we can genetate a DAG:

```
cd GW150914

gwcosmo_coasting_pixel_dag --posterior_samples GW150914_GWTC-1.hdf5 --skymap GW150914_skymap.fits.gz --Pdet /path_to_Pdets/SNR11_new_cosmology/BBH-powerlaw-gaussian_alpha_3.78_beta_0.81_Mmin_4.98_Mmax_112.5_mu_32.27_lambda_0.03_sigma_3.88_delta_4.8_Nsamps150000_full_waveform_snr_11.0_k_-1.p --catalog GLADE+ --catalog_band K --min_H0 20 --max_H0 140 --bins_H0 100 --reweight_posterior_samples True --nside 32 --sky_area 0.999 --min_pixels 30 --outputfile GW150914_k-1 --Kcorrections True --redshift_evolution Madau --Lambda 4.59 --Madau_beta 2.86 --Madau_zp 2.47 --ram 5000 --seed 400 --galaxy_weighting True --search_tag ligo.prod.o3.cbc.hubble.gwcosmo
```

If you do not run on a ligo cluster, add the argument `--run_on_ligo_cluster False.`

The outputs are the following:
- **dagfile.dag**,
- **single_pixel.sub**,
- **combine_pixel.sub**,
- **eventname_indices.txt** which lists all the pixel indices which cover the event's sky area, assuming the given `--sky_area` and `--min_pixels` arguments.
- **log/** folder which will store the log files during the analysis.

By running the example command `eventname` will be `GW150914_k-1`.
Run
```
condor_submit_dag dagfile.dag
```
to parallelise the analysis for the pixels, and combine them when finished. 

Each pixel will have an **eventname_pixel_i.npz** which is in the same format as **eventname_likelihood_breakdown.npz**, but the likelihood is only for the pixel under consideration. The final result is the **eventname.npz** and the figure **eventname.png**.

**Note** that in our analysis we set `--posterior_samples_field PublicationSamples` for O3a events. We set `--posterior_samples_field C01:Mixed` for GW190521 and O3b events, and `--posterior_samples_field C01:IMRPhenomNSBH` for NSBH events GW200105_162426 and GW200115_042309.



# Combining the posteriors on $`H_0`$ from the single gravitational-wave events

In order to get the final $H_0$ posterior we have to combine the posteriors from the individual gravitational wave events. We can use the  `gwcosmo_coasting_combined_posterior` for this task.

### Example
To combine all posteriors when $k = 1$ stored in the folder `posteriors_k1`, use the command

```
gwcosmo_coasting_combined_posterior --dir posteriors_k1 --outputfile combined_all_k1
```

The outputfiles will be `combined_all_k1.npz` which contains `[H0,likelihood_comb,posterior_uniform_norm,posterior_log_norm]` and the figure `combined_all_k1.png`.
