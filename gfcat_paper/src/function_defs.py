"""
.. module:: function_defs
   :synopsis: Helper functions for the UV Ceti gPhoton notebook tutorials, used
       to re-create the data and figures used in the paper.

.. moduleauthor:: Chase Million, Scott W. Fleming
"""

import itertools
import os
from astropy.io import fits as pyfits
from astropy import wcs as pywcs
from astropy.stats import sigma_clip
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import anderson
import warnings

def angularSeparation(ra1, dec1, ra2, dec2):
    d2r = np.pi/180.
    ra2deg = 1./d2r
    d1 = dec1*d2r,
    d2 = dec2*d2r
    r1 = ra1*d2r
    r2 = ra2*d2r
    a = np.sin((d2-d1)/2.)**2.+np.cos(d1)*np.cos(d2)*np.sin((r2-r1)/2.)**2.
    r = 2*np.arcsin(np.sqrt(a))
    return r*ra2deg

def counts2flux(cps, band):
    scale = 1.4e-15 if band == 'FUV' else 2.06e-16
    return scale*cps

def counts2mag(cps, band):
    scale = 18.82 if band == 'FUV' else 20.08
    # This threw a warning if the countrate was negative which happens when
    #  the background is brighter than the source. Suppress.
    with np.errstate(invalid='ignore'):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mag = -2.5 * np.log10(cps) + scale
    return mag

def mag2counts(mag, band):
    scale = 18.82 if band == 'FUV' else 20.08
    return 10.**(-(mag-scale)/2.5)

def apcorrect1(radius, band):
    if not band in ['NUV', 'FUV']:
        print("Invalid band.")
        return
    aper = np.array([1.5, 2.3, 3.8, 6.0, 9.0, 12.8, 17.3, 30., 60., 90.])/3600.
    if radius > aper[-1]:
        return 0.
    if band == 'FUV':
        dmag = [1.65, 0.96, 0.36, 0.15, 0.1, 0.09, 0.07, 0.06, 0.03, 0.01]
    else:
        dmag = [2.09, 1.33, 0.59, 0.23, 0.13, 0.09, 0.07, 0.04, -0.00, -0.01]
        if radius > aper[-2]:
            return 0.
    if radius < aper[0]:
        return dmag[0]
    ix = np.where((aper-radius) >= 0.)
    x = [aper[ix[0][0]-1], aper[ix[0][0]]]
    y = [dmag[ix[0][0]-1], dmag[ix[0][0]]]
    m, C = np.polyfit(x, y, 1)
    return m*radius+C

def find_flare_ranges(lc, sigma=3, quiescence=None):
    """ Identify the start and stop indexes of a flare event. The range will continue backwards and forwards
        in time from the peak until either the end of the visit, or a flux that is within 1-sigma of the INFF
        is found.
    """
    tranges = [[min(lc['t0']), max(lc['t1'])]]
    if not quiescence:
        q, q_err = get_inff(lc)
    else:
        q, q_err = quiescence
    flare_ranges = []
    for trange in tranges:
        # The range excludes those points that don't have good coverage in the time bin, based on 'expt'.
        # NOTE: This assumes a 30-second bin size!!
        ix = np.where((np.array(lc['t0'].values) >= trange[0]) &
                      (np.array(lc['t0'].values) <= trange[1]) &
                      (np.array(lc['expt'].values) >= 20.0) &
                      (np.array(lc['cps'].values) -
                       sigma*np.array(lc['cps_err'].values) >= q))[0]
        # Save the points that are 3-sigma above the INFF to return.
        fluxes_3sig = ix
        if not len(ix):
            # No flares were found
            continue
        # This chunk extends flares until they are indistinguishable from
        # INFF, which we define has having two sequential fluxes that are less than
        # 1-sigma above the INFF.
        temp_ix = []
        for ix_range in find_ix_ranges(ix):
            # Set extra_part = 0.0 for the original version from Chase that did not
            # take into account errors, otherwise this is set to require fluxes be
            # greater than 1-sigma from the INFF before it stops the range extension.
            # Going backwards.
            n_in_a_row = 0
            extra_part = lc.iloc[ix_range[0]]['cps_err']
            while (lc.iloc[ix_range[0]]['cps']-extra_part >= q and ix_range[0] > 0 or (n_in_a_row < 1 and ix_range[0] > 0)):
                extra_part = lc.iloc[ix_range[0]]['cps_err']
                if (lc.iloc[ix_range[0]]['cps']-extra_part < q):
                    n_in_a_row += 1
                else:
                    n_in_a_row = 0
                if (lc.iloc[ix_range[0]]['t0'] - lc.iloc[ix_range[0]-1]['t0'] >
                        1000):
                    break
                ix_range = [ix_range[0] - 1] + ix_range
            # Going forwards.
            n_in_a_row = 0
            extra_part = lc.iloc[ix_range[-1]]['cps_err']
            while (lc.iloc[ix_range[-1]]['cps']-extra_part >= q and ix_range[-1] != len(lc)-1 or (n_in_a_row < 1 and ix_range[-1] != len(lc)-1)):
                extra_part = lc.iloc[ix_range[-1]]['cps_err']
                if (lc.iloc[ix_range[-1]]['cps']-extra_part < q):
                    n_in_a_row += 1
                else:
                    n_in_a_row = 0
                if (lc.iloc[ix_range[-1]+1]['t0']-lc.iloc[ix_range[-1]]['t0'] >
                        1000):
                    break
                ix_range = ix_range + [ix_range[-1] + 1]
            temp_ix += ix_range
        ix = np.unique(temp_ix)
        flare_ranges += find_ix_ranges(list(np.array(ix).flatten()))
    return (flare_ranges, fluxes_3sig)

def refine_flare_ranges(lc, sigma=3., makeplot=True, flare_ranges=None):
    """ Identify the start and stop indexes of a flare event after
    refining the INFF by masking out the initial flare detection indexes. """
    if not flare_ranges:
        flare_ranges, _ = find_flare_ranges(lc, sigma=sigma)
    flare_ix = list(itertools.chain.from_iterable(flare_ranges))
    quiescience_mask = [False if i in flare_ix else True for i in
                        np.arange(len(lc['t0']))]
    quiescence = ((lc['cps'][quiescience_mask] *
                   lc['expt'][quiescience_mask]).sum() /
                  lc['expt'][quiescience_mask].sum())
    quiescence_err = (np.sqrt(lc['counts'][quiescience_mask].sum()) /
                      lc['expt'][quiescience_mask].sum())
    flare_ranges, flare_3sigs = find_flare_ranges(lc,
                                                  quiescence=(quiescence,
                                                              quiescence_err),
                                                  sigma=sigma)
    flare_ix = list(itertools.chain.from_iterable(flare_ranges))
    not_flare_ix = list(set([x for x in range(len(lc['t0']))]) - set(flare_ix))
    if makeplot:
        plt.figure(figsize=(15, 3))
        plt.plot(lc['t0']-min(lc['t0']), lc['cps'], '-k')
        plt.errorbar(lc['t0'].iloc[not_flare_ix]-min(lc['t0']),
                     lc['cps'].iloc[not_flare_ix],
                     yerr=1.*lc['cps_err'].iloc[not_flare_ix], fmt='ko')
        plt.errorbar(lc['t0'].iloc[flare_ix]-min(lc['t0']),
                     lc['cps'].iloc[flare_ix],
                     yerr=1.*lc['cps_err'].iloc[flare_ix], fmt='rs')
        plt.plot(lc['t0'].iloc[flare_3sigs]-min(lc['t0']),
                 lc['cps'].iloc[flare_3sigs],
                    'ro', fillstyle='none', markersize=20)
        plt.hlines(quiescence, lc['t0'].min()-min(lc['t0']),
                   lc['t0'].max()-min(lc['t0']))
        plt.show()
    return flare_ranges, quiescence, quiescence_err

def find_ix_ranges(ix, buffer=False):
    """ Finds indexes in the range. """
    foo, bar = [], []
    for n, i in enumerate(ix):
        if len(bar) == 0 or bar[-1] == i-1:
            bar += [i]
        else:
            if buffer:
                bar.append(min(bar)-1)
                bar.append(max(bar)+1)
            foo += [np.sort(bar).tolist()]
            bar = [i]
        if n == len(ix)-1:
            if buffer:
                bar.append(min(bar)-1)
                bar.append(max(bar)+1)
            foo += [np.sort(bar).tolist()]
    return foo

def get_inff(lc, clipsigma=3, quiet=True, band='NUV',
             binsize=30.):
    """ Calculates the Instantaneous Non-Flare Flux values. """
    sclip = sigma_clip(np.array(lc['cps']), sigma=clipsigma)
    inff = np.ma.median(sclip)
    inff_err = np.sqrt(inff*len(sclip)*binsize)/(len(sclip)*binsize)
    if inff and not quiet:
        print('Quiescent at {m} AB mag.'.format(m=counts2mag(inff, band)))
    return inff, inff_err

# Alternative INFF calculation method, not used for the GJ 65 paper.
#def get_inff(lc, clipsigma=3, use_mcmc=False, quiet=True, band='NUV',
#             binsize=30.):
#    if anderson(lc['cps']).statistic < max(anderson(lc['cps']).critical_values):
#        return np.mean(lc['cps']), np.std(lc['cps'])
#    sclip = sigma_clip(lc['cps'].values,sigma_lower=3, sigma_upper=1)
#    quiescence = np.ma.median(sclip)
#    quiescence_err = np.sqrt(quiescence*len(sclip)*binsize)/(len(sclip)*binsize)
#    if quiescence and not quiet:
#        print('Quiescent at {m} AB Mag.'.format(m=counts2mag(quiescence,band)))
#    return quiescence, quiescence_err

def calculate_flare_energy(lc, frange, distance, binsize=30, band='NUV',
                           effective_widths={'NUV':729.94, 'FUV':255.45},
                           quiescence=None,aperture=17.5):
    """ Calculates the energy of a flare in erg. """
    if not quiescence:
        q, _ = get_inff(lc)
        # Convert to aperture-corrected flux
        q = mag2counts(counts2mag(q,band)-apcorrect1(aperture,band),band)
    else:
        q = quiescence[0]

    if 'cps_apcorrected' in lc.keys():
        # Converting from counts / sec to flux units.
        flare_flux = (np.array(counts2flux(
            np.array(lc.iloc[frange]['cps_apcorrected']), band)) -
                      counts2flux(q, band))
    else:
        # Really need to have aperture-corrected counts/sec.
        raise ValueError("Need aperture-corrected cps fluxes to continue.")
    # Zero any flux values where the flux is below the INFF so that we don't subtract from the total flux!
    flare_flux = np.array([0 if f < 0 else f for f in flare_flux])
    # if only the last bin has zero exposure time, zero it and proceed
    if not np.isfinite(flare_flux[-1]):
        flare_flux[-1]=0
    # if more bins than that, then bail out because this is a cursed lightcurve
    if not all(np.isfinite(flare_flux)):
        return np.nan, np.nan
    flare_flux_err = counts2flux(np.array(lc.iloc[frange]['cps_err']), band)
    tbins = (np.array(lc.iloc[frange]['t1'].values) -
             np.array(lc.iloc[frange]['t0'].values))
    # Calculate the area under the curve.
    integrated_flux = (binsize*flare_flux).sum()
    """
    GALEX effective widths from
    http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?id=GALEX/GALEX.NUV
    width = 729.94 A
    http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?id=GALEX/GALEX.FUV
    width = 255.45 A
    """
    # Convert integrated flux to a fluence using the GALEX effective widths.
    fluence = integrated_flux*effective_widths[band]
    fluence_err = (np.sqrt(((counts2flux(lc.iloc[frange]['cps_err'], band) *
                             binsize)**2).sum())*effective_widths[band])
    if not distance:
        return fluence, fluence_err
    # Convert from parsecs to cm
    distance_cm = distance * 3.086e+18

    energy = (4 * np.pi * (distance_cm**2) * fluence)
    energy_err = (4 * np.pi * (distance_cm**2) * fluence_err)
    return energy, energy_err

def is_left_censored(frange):
    """ Returns true if the light curve is cutoff on the left. """
    return 0 in frange

def is_right_censored(lc, frange):
    """ Returns true if the light curve is cutoff on the right. """
    return len(lc['t0'])-1 in frange

def peak_cps(lc, frange):
    """ Returns the peak cps in the light curve. """
    return (lc['cps'][np.argmax(np.array(lc['cps'][frange].values))],
            lc['cps_err'][np.argmax(np.array(lc['cps'][frange].values))])

def peak_time(lc, frange, stepsz=30):
    """ Return the bin start time corresponding to peak flux. """
    return lc['t0'][np.argmax(np.array(lc['cps'][frange].values))] + stepsz/2

def is_peak_censored(lc, frange):
    """ Returns true is the peak flux is the first or last point in the light
    curve. """
    return ((np.argmax(np.array(lc['cps'][frange].values)) == 0) or
            (np.argmax(np.array(lc['cps'][frange].values)) == len(lc) - 1))
