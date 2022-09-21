import os
import tqdm
import pandas as pd
import csv
import math
import numpy as np
from scipy import signal, stats
from sklearn.cluster import DBSCAN
from photutils.psf import DBSCANGroup
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle
import sys
from astropy.io import fits as pyfits
from astropy import wcs as pywcs
from astropy.visualization import simple_norm, ZScaleInterval
from rich import print
import astropy
from astroquery.simbad import Simbad
Simbad.add_votable_fields("otype")
import astropy.units as u
import time


#import pyarrow
#from pyarrow import parquet


def make_wcs(
    skypos,
    pixsz=0.000416666666666667,  # Same as the GALEX intensity maps
    imsz=[3200, 3200],  # Same as the GALEX intensity maps...
    ):
    wcs = pywcs.WCS(naxis=2)
    wcs.wcs.cdelt = np.array([-pixsz, pixsz])
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crpix = [(imsz[1] / 2.0) + 0.5, (imsz[0] / 2.0) + 0.5]
    wcs.wcs.crval = skypos
    return wcs

def read_image(fn,hdunum=0):
    # set hdu=1 for rice compressed data
    if 'rice' in fn:
        hdunum = 1
    hdu = pyfits.open(fn)
    image = hdu[hdunum].data
    exptimes, tranges = [], []
    for i in range(hdu[hdunum].header["N_FRAME"]):
        exptimes += [hdu[hdunum].header["EXPT_{i}".format(i=i)]]
        tranges += [
            [hdu[hdunum].header["T0_{i}".format(i=i)], hdu[hdunum].header["T1_{i}".format(i=i)]]
        ]
    skypos = (hdu[hdunum].header["CRVAL1"], hdu[hdunum].header["CRVAL2"])
    wcs = make_wcs(skypos,imsz=np.shape(image))
    try:
        flagmap = hdu[hdunum+1].data
        edgemap = hdu[hdunum+2].data
    except IndexError:
        flagmap = None
        edgemap = None
    hdu.close()
    return image, flagmap, edgemap, wcs, tranges, exptimes

def generate_visit_database(catdbfile='/Users/cm/GFCAT/catalog.parquet',
                            photdir = '/Users/cm/GFCAT/photom',
                            wrong_eclipse_file='/Users/cm/GFCAT/incorrectly_analyzed_eclipses.txt'):
    observations = {'eclipse': [], 'id': [],
                    'ra': [], 'dec': [],
                    'xcenter': [], 'ycenter': [],
                    'exptime': [], 'cps': [], 'cps_err': [],
                    'hasmask': [], 'hasedge': []}
    wrong_eclipses = pd.read_csv(wrong_eclipse_file)['eclipse'].values
    for i, edir in enumerate(tqdm.tqdm(os.listdir(photdir))):
        try:
            eclipse = int(edir[1:])
        except ValueError:
            continue  # not a normal eclipse directory
        if eclipse in wrong_eclipses:
            continue  # skipping accidentally processed eclipse
        if not len(os.listdir(f'{photdir}/{edir}')):
            continue  # light curves not created (for any number of reasons)
        phot = csv.DictReader(open(f'{photdir}/{edir}/{edir}-nd-30s-photom.csv'))
        expt = parse_exposure_time(f'{photdir}/{edir}/{edir}-nd-30s-exptime.csv')
        for obj in phot:
            counts = float(obj['aperture_sum'])
            observations['eclipse'] += [eclipse]
            observations['id'] += [obj['id']]
            observations['ra'] += [float(obj['ra'])]
            observations['dec'] += [float(obj['dec'])]
            observations['xcenter'] += [float(obj['xcenter'])]
            observations['ycenter'] += [float(obj['ycenter'])]
            observations['exptime'] += [expt['expt_total']]
            observations['cps'] += [counts / expt['expt_total']]
            observations['cps_err'] += [math.sqrt(counts) / expt['expt_total']]
            observations['hasmask'] += [float(obj['aperture_sum_mask']) > 0]
            observations['hasedge'] += [float(obj['aperture_sum_edge']) > 0]

    #engine = sql.create_engine(f'sqlite:///{catdbfile}', echo=False)
    #pd.DataFrame(observations).to_sql('gfcat', con=engine, if_exists='replace')  # 'append' if i!=0 else 'replace')
    #engine.execute("CREATE INDEX 'ix_gfcat' ON 'gfcat' ('ra', 'dec')")
    #engine.dispose()
    #print(f'Visit level data dumped to {catdbfile}.\n')

    VARIABLES_FOR_WHICH_DICTIONARY_COMPRESSION_IS_USEFUL = [
        "eclipse",
        "id",
        "ra",
        "dec",
        "xcenter",
        "ycenter",
        "exptime",
        "cps",
        "hasmask",
        "hasedge",
    ]
    dict_comp = VARIABLES_FOR_WHICH_DICTIONARY_COMPRESSION_IS_USEFUL.copy()

    parquet.write_table(
        pyarrow.Table.from_arrays(
            list(observations.values()), names=list(observations.keys())
        ),
        catdbfile,
        use_dictionary=dict_comp,
        version="2.6",
    )

def eliminate_dupes(variable_table):
    # Run a spatial clustering algorithm and consider variables within 1 arcmin
    #  of each other to be most likely the same source and combine them, choosing
    #  the brightest of the sources as the primary
    X = list(zip(variable_table['xcenter'],variable_table['ycenter']))#variable_table['pos']
    db = DBSCAN(eps=40,min_samples=1).fit(X) # 40 pixels ~= 1 arcmin
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    print(labels)
    varix = {'id':[]}
    for lbl in set(labels):
        dbix = np.where(labels==lbl)[0]
        if any(np.array(variable_table["cps"])[dbix] > 170):
            return [] # if there is a very bright star in the cluster, dump them all
        if len(dbix)>=12: # big cluster of variables are presumed artifacts
            continue
        elif len(dbix)>1: # small cluster -- use the one with maximum variation
            xcenters,ycenters = np.array(variable_table['xcenter']),np.array(variable_table['ycenter'])
            dist = np.sqrt((xcenters[dbix].min()-
                            xcenters[dbix].max())**2 +
                           (ycenters[dbix].min()-
                            ycenters[dbix].max())**2)
            if dist > 80: # cluster is more than 2 arcminutes across
                continue
            ix = [np.argmax(np.abs(variable_table['delta_cps'])[dbix])]
            varix['id']+=np.array(variable_table['id'])[dbix][ix].tolist() # Earlier version was missing dbix ---Fixed 220227
        else:
            varix['id']+=np.array(variable_table['id'])[dbix].tolist()
    return varix['id']

def parse_exposure_time(fn:str):
    # parse the exposure time files... quickly...
    with open(fn) as data:
        expt_data = csv.DictReader(data) # way faster to parse the file like this
        expt_rows = []
        for row in expt_data:
            expt_rows.append(
                {
                    't0':row['t0'],
                    't1':row['t1'],
                    'expt_eff':float(row['expt'])
                }
            )
    return pd.DataFrame(expt_rows).astype({'t0':'float64','t1':'float64'}).to_dict('list')

def parse_lightcurves(fn:str):
    # This is a blunt force way to suppress divide by zero warnings.
    # It's dangerous to suppress warnings. Don't do this.
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")

    expt = parse_exposure_time(fn.split('photom')[0]+'exptime.csv')
    with open(fn) as data:
        obs = csv.DictReader(data)
        lightcurves = []
        for row in obs:
            lc = {}
            try:
                lc['counts'] = np.array([float(row[f'aperture_sum_{n}']) for n in range(len(expt['t0']))])
                # NOTE: There is a bug in the version of gPhoton that generated these photometry files that
                #  switches the "mask" and "edge" columns. Bug discovered 220218.
                lc['edge_flags'] = np.array([float(row[f'aperture_sum_flag_{n}']) for n in range(len(expt['t0']))],dtype=bool)
                lc['mask_flags'] = np.array([float(row[f'aperture_sum_edge_{n}']) for n in range(len(expt['t0']))],dtype=bool)
            except ValueError:
                continue # this light curve contains no valid data
            lc['cps'] = np.array(lc['counts'] / expt['expt_eff'])
            lc['cps_err'] = np.array(np.sqrt(lc['counts']) / expt['expt_eff'])
            lc['xcenter'] = float(row['xcenter'])
            lc['ycenter'] = float(row['ycenter'])
            lc['ra'] = float(row['ra'])
            lc['dec'] = float(row['dec'])
            lightcurves+=[lc]
    return lightcurves

def is_spiky(lc:dict):
    for sigma,n_outliers in [(3,3),(2,5)]: # sigma prominence is actually ~2x
        upper_limit = (lc['cps'] + sigma * lc['cps_err'])
        lower_limit = (lc['cps'] - sigma * lc['cps_err'])
        for n in [1,2]: # number of bins to bunch up as "one" outlier
            ix = np.where((lower_limit[n:-n] - upper_limit[:-int(n*2)] > 0) &
                          (lower_limit[n:-n] - upper_limit[int(n*2):] > 0))[0]
            if len(ix) >= n_outliers - any(lc['mask_flags'][n:-1][ix]): # be a little stricter if flagged
                return True # This is a dumber test than the one above for the same thing. Seems more effective.
    return False

def screen_gfcat(eclipses,band='NUV',aper_radius=17,photdir='/Users/cm/GFCAT/photom',sigma=3,
                 cps_10p_rolloff={'NUV': 311, 'FUV': 109,}, # non-linear regime given by calpaper
                 binsz=30,
                 ):
    variables = {}
    for e in tqdm.tqdm(eclipses):
        edir = f'e{str(e).zfill(5)}'
        photpath = f'{photdir}/{edir}/{edir}-{band.lower()[0]}d-{binsz}s-photom-{str(aper_radius).replace(".","_")}.csv'
        if not os.path.exists(photpath):
            continue # there is no photometry file for this eclipse + band
        expt_fn = photpath.split('photom')[0]+'exptime.csv'
        expt = parse_exposure_time(expt_fn)
        if np.sum(expt['expt_eff'])<500:
            continue # skip the whole eclipse if there is not at least 8 min of exposure total
        lightcurves = parse_lightcurves(photpath)
        candidate_variables = []
        for i,lc in enumerate(lightcurves):
            if any(lc['edge_flags']):
                continue # skip if there is any data near the detector edge
            if any(lc['mask_flags']): #if all(lc['mask_flags'][ix]):
                continue  # skip if there is any data covered by the hotspot mask
            ix = np.where((lc['cps']!=0) & (np.isfinite(lc['cps'])))[0]
            if expt['t1'][ix[-1]] - expt['t0'][ix[0]]<500:
                continue # skip if there are not at least 8 min of exposure on target
                # This duration was chosen to eliminate a relatively high number of false positives
                # in shorter visits. It is approx. 1/3rd duration of a full MIS-depth visit.
            if len(ix)/(ix[-1]+1-ix[0])<0.75:
                continue # skip if more than a quarter of the bins are unobserved
                # NOTE: It's technically possible to have 30-second integrated flux on a source
                #  be exactly equal to zero. But it's low probability and has no meaningful effect
                #  on the variability search.
            sort_ix = np.argsort(lc["cps"][ix])
            #if len(np.where(lc["mask_flags"][ix][sort_ix][-10:])[0]) >= 5:
            #    continue # more than half of the brightest points are flagged by the hotspot mask
            # The following check has been moved into the cluster analysis because the bright stars
            # also generate false detections / variables nearby
            #if (lc["cps"][ix][sort_ix[1]] > 170):# and (lc["cps"][ix][sort_ix[-1]]>300):
            #    continue # skip if the whole visit is >14.5 AB Mag in NUV
            # The chance of one outlier low point over 50M visits is not small, generates a lot of false positives
            # The chance of two outlier low points within a visit is small.
            # So use the second-lowest point in the visit as the benchmark.
            second_min = np.sort((lc['cps']+lc['cps_err']*sigma)[ix])[1]
            outlier_ix = np.where((lc['cps'] - lc['cps_err'] * sigma)[ix] > second_min)[0]
            if len(outlier_ix) < 3:
                continue # skip if there are not 3 significant outliers using the dumbest heuristic
            if is_spiky(lc):
                continue  # skip: multiple spiky peaks, most likely contaminated by an artifact
            peak_ix, _ = signal.find_peaks(lc['cps'], prominence=3 * lc['cps_err'], distance=4)
            if len(peak_ix):
                if len(peak_ix) > 3:
                    continue  # skip multiple spiky peaks, most likely contaminated by an artifact
                # NOTE: This test for spiky behavior is a little slower than I'd like, but workable.
                #  And it does a more thorough job than the faster / simpler `is_spikey()` above.
            ad = stats.anderson(lc['cps'][ix])  # standard test of variability
            if ad.statistic <= ad.critical_values[2]:
                continue  # failed the anderson-darling test at 5%
                # NOTE: AD is the gold standard variability test, but it's relatively slow, so it
                #  has been pushed to the end of the screening heuristics
            # Whatever remains is a candidate variable
            print(i)
            candidate_variables.append({'id':i,
                                        'cps':np.median(lc['cps'][ix]),
                                        'xcenter':lc['xcenter'],'ycenter':lc['ycenter'],
                                        'delta_cps':np.min(lc['cps'][ix])-np.max(lc['cps'][ix])})
        if not len(candidate_variables):
            continue # there are no candidate variables at this point
        # Now screen out variables in clumps, which are very probably due to transient artifacts
        varix = eliminate_dupes(pd.DataFrame(candidate_variables).to_dict('list'))
        if len(varix) >= 10:
            continue  # This is a cursed eclipse --- too many "variables" --- do not believe its lies
        if len(varix) == 0:
            continue # there are no variables
        variables[e] = varix
    return variables

def generate_qa_plots(vartable:dict,band='NUV',
                      photdir='/Users/cm/GFCAT/photom',
                      plotdir='/Users/cm/GFCAT/plots',
                      cleanup=False,
                      boxsz = 200, # pixels margin, so 2x this is the width
                      rerun = False,
                        ):
    for e in tqdm.tqdm(vartable.keys()):
        if not rerun and all([os.path.exists(f'{plotdir}/e{str(e).zfill(5)}-{band}-{str(i).zfill(4)}.png') for i in vartable[e]]):
            continue # these QA plots have already been created, so skip
        edir = f'e{str(e).zfill(5)}'
        photpath = f'{photdir}/{edir}/{edir}-{band.lower()[0]}d-30s-photom.csv'
        lightcurves = parse_lightcurves(photpath)
        expt = parse_exposure_time(photpath.replace('photom.csv', 'exptime.csv'))
        cntfilename = f'{photdir}/e{str(e).zfill(5)}/e{str(e).zfill(5)}-{band[0].lower()}d-full.fits.gz'
        if not os.path.exists(cntfilename):
            cmd = f'aws s3 cp s3://dream-pool/e{str(e).zfill(5)}/e{str(e).zfill(5)}-{band[0].lower()}d-full.fits.gz {cntfilename} --quiet'
            os.system(cmd)
            if not os.path.exists(cntfilename):
                raise FileNotFoundError(f'This file has lightcurves and should definitely exist.\n{cntfilename}')
        image, flagmap, edgemap, wcs, _, _ = read_image(cntfilename)
        for i in vartable[e]:
            lc = lightcurves[i]
            imgx, imgy = lc['xcenter'],lc['ycenter'] # the image pixel coordinate of the source
            # define the bounding box for the thumbnail
            imsz = np.shape(image)

            fig = plt.figure(figsize=(17, 15))
            G = gridspec.GridSpec(4, 4)

            # make a QA image that is zoomed in
            # noting that image coordinates and numpy coordinates are flipped
            x1, x2, y1, y2 = (max(int(imgy - boxsz),0),
                              min(int(imgy + boxsz),imsz[0]),
                              max(int(imgx - boxsz),0),
                              min(int(imgx + boxsz),imsz[1]))

            ax = fig.add_subplot(G[:3,:2])
            ax.imshow(ZScaleInterval()(image[x1:x2, y1:y2]), cmap="Greys_r", origin="lower")
            ax.imshow(1 / edgemap[x1:x2, y1:y2], origin="lower", cmap="Reds_r", alpha=1)
            ax.imshow(1 / flagmap[x1:x2, y1:y2], origin="lower", cmap="Blues_r", alpha=1)
            ax.plot(boxsz, boxsz, markersize=30, color='y', lw=10, marker='o', fillstyle='none')  # marker='o')
            ax.set_xticks([])
            ax.set_yticks([])

            # make a QA image of the full detector
            ax = fig.add_subplot(G[:3,2:])
            # The cropping is here to handle very wide images created by inappropriate handling
            # of map projection distortions when initializing the image size during processing;
            # this has been fixed in the pipeline, but not for the data we have.
            x1_,x2_,y1_,y2_ = (max(int(imsz[0]/2-imsz[0]/2),0),
                               min(int(imsz[0]/2+imsz[0]/2),imsz[0]),
                               max(int(imsz[1]/2-imsz[0]/2),0),
                               min(int(imsz[1]/2+imsz[0]/2),imsz[1]))
            ax.imshow(ZScaleInterval()(image[x1_:x2_,y1_:y2_]), cmap="Greys_r", origin="lower")
            ax.imshow(1 / edgemap[x1_:x2_,y1_:y2_], origin="lower", cmap="Reds_r", alpha=1)
            ax.imshow(1 / flagmap[x1_:x2_,y1_:y2_], origin="lower", cmap="Blues_r", alpha=1)
            rect = Rectangle((y1-y1_, x1-x1_), 2*boxsz, 2*boxsz, linewidth=1, edgecolor='y', facecolor='none')
            ax.add_patch(rect)
            #ax.plot(boxsz, boxsz, markersize=30, color='y', lw=10, marker='o', fillstyle='none')  # marker='o')
            #ax.set_xlim([imgy-1000,imgy+1000])
            #ax.set_xlim([int(imsz[1]/2-imsz[0]/2),int(imsz[1]/2+imsz[0]/2)])
            #ax.set_ylim([int(imsz[0]/2-imsz[0]/2),int(imsz[0]/2+imsz[0]/2)])
            ax.set_xticks([])
            ax.set_yticks([])


            ax = fig.add_subplot(G[3,:])
            #ax.set_title(f"{edir} : {var}")
            ix = np.where(np.isfinite(lc['cps']))[0]
            ax.errorbar(np.array(expt['t0'])[ix],lc['cps'][ix],yerr=3*lc['cps_err'][ix],fmt='k-')
            ix = np.where(np.array(lc['mask_flags']))[0]
            if len(ix):
                ax.plot(np.array(expt['t0'])[ix],lc['cps'][ix],'ro')
            ax.set_xticks([])

            plt.tight_layout()
            plt.savefig(f'{plotdir}/e{str(e).zfill(5)}-{band}-{str(i).zfill(4)}.png')
            plt.close('all')
        if cleanup:
            os.system(f'rm -rf {photdir}/e{str(e).zfill(5)}/*fits*')
    return

def get_target_data(eclipse:int,index:int,band='NUV',photdir='/Users/cm/GFCAT/photom'):
    edir = f'e{str(eclipse).zfill(5)}'
    photpath = f'{photdir}/{edir}/{edir}-{band.lower()[0]}d-30s-photom.csv'
    lc = parse_lightcurves(photpath)[index]
    return lc

def get_simbad_id(skypos):
    ra, dec = skypos
    skypos_obj = astropy.coordinates.SkyCoord(ra,dec,unit='deg')
    r = 1*u.arcminute
    result_table = Simbad.query_region(skypos_obj,r)
    #try:
    #    simbad_id = result_table[0]['MAIN_ID']
    #    #this [0] index grabs the top result for the skypos within the aperture search radius.
    #except TypeError: # a TypeError will be raised if no Simbad ID is found
    #    simbad_id = '-'
    # NOTE: you will get blacklisted if you submit > 5-10 queries/sec, and this function will be used in a loop
    #time.sleep(0.5)
    return result_table[0]

def quick_summarize_visit(eclipse:int,index:int,band='NUV',photdir='/Users/cm/GFCAT/photom'):
    lc = get_target_data(eclipse,index,band=band,photdir=photdir)
    print(f'skypos:  {np.round(lc["ra"],5)}, {np.round(lc["dec"],5)}')
    print(f'eclipse: {eclipse}')
    print(f'index:   {index}')
    try:
        simbad = get_simbad_id((lc['ra'],lc['dec']))
        print(f'name:    {simbad["MAIN_ID"].decode()}')
        print(f'otype:   {simbad["OTYPE"].decode()}')
    except TypeError:
        print('**No SIMBAD entry.**')
    return

def summarize_all_visits(eclipses, band='NUV', aper_radius=17, photdir='/Users/cm/GFCAT/photom', sigma=3,
        cps_10p_rolloff={'NUV': 311, 'FUV': 109},  # non-linear regime given by calpaper
        ):
        for e in tqdm.tqdm(eclipses):
            edir = f'e{str(e).zfill(5)}'
            photpath = f'{photdir}/{edir}/{edir}-{band.lower()[0]}d-30s-photom.csv'
            if not os.path.exists(photpath):
                continue  # there is no photometry file for this eclipse + band
            expt = parse_exposure_time(photpath.replace('photom.csv', 'exptime.csv'))
            if np.sum(expt['expt_eff']) < 300:
                continue  # skip the whole eclipse if there is not at least 5 min of exposure total
            lightcurves = parse_lightcurves(photpath)
            for i, lc in enumerate(lightcurves):
                continue

