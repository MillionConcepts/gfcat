import os
import tqdm
import pandas as pd
import csv
import math
import sqlalchemy as sql
import numpy as np
from scipy import signal, stats
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from matplotlib import gridspec
import sys
from astropy.io import fits as pyfits
from astropy import wcs as pywcs
from astropy.visualization import simple_norm, ZScaleInterval

import pyarrow
from pyarrow import parquet


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

def read_image(fn):
    hdu = pyfits.open(fn)
    image = hdu[0].data
    exptimes, tranges = [], []
    for i in range(hdu[0].header["N_FRAME"]):
        exptimes += [hdu[0].header["EXPT_{i}".format(i=i)]]
        tranges += [
            [hdu[0].header["T0_{i}".format(i=i)], hdu[0].header["T1_{i}".format(i=i)]]
        ]
    skypos = (hdu[0].header["CRVAL1"], hdu[0].header["CRVAL2"])
    wcs = make_wcs(skypos,imsz=np.shape(image))
    try:
        flagmap = hdu[1].data
        edgemap = hdu[2].data
    except IndexError:
        flagmap = None
        edgemap = None
    hdu.close()
    return image, flagmap, edgemap, wcs, tranges, exptimes

# def parse_exposure_time(fn:str):
#     # parse the exposure time files... quickly...
#     expt = csv.DictReader(open(fn)) # way faster to parse the file like this
#     n,expt_total = 0,0
#     exposures = []
#     t0,t1 = [], []
#     for frame in expt:
#         n+=1
#         expt_total+=float(frame['expt'])
#         exposures+=[float(frame['expt'])]
#         t0+=[float(frame['t0'])]
#         t1+=[float(frame['t1'])]
#     return {'expt_total':expt_total,
#             'exposure_time':exposures, # should assign dtypes for speed?
#             'tstart':t0,
#             'tstop':t1}

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
    varix = {'id':[]}
    for lbl in set(labels):
        dbix = np.where(labels==lbl)[0]
        if len(dbix)>3: # cluster variables are presumed artifacts
            continue
        elif len(dbix)>1: # small cluster -- use only the brightest
            ix = [np.argmax(np.array(variable_table['cps'])[dbix])]
            varix['id']+=np.array(variable_table['id'])[ix].tolist()
        else:
            varix['id']+=np.array(variable_table['id'])[dbix].tolist()
    return varix['id']

#def make_qa_plot(lc,pixel_coords,image,flagmap,edgemap):

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

    expt = parse_exposure_time(fn.replace('photom.csv', 'exptime.csv'))
    with open(fn) as data:
        obs = csv.DictReader(data)
        lightcurves = []
        for row in obs:
            lc = {}
            try:
                lc['counts'] = np.array([float(row[f'aperture_sum_{n}']) for n in range(len(expt['t0']))])
                # NOTE: There is a bug in the version of gPhoton that generated these photometry files that
                #  switches the "mask" and "edge" columns. Discovered 220218.
                lc['edge_flags'] = np.array([float(row[f'aperture_sum_flag_{n}']) for n in range(len(expt['t0']))],dtype=bool)
                lc['mask_flags'] = np.array([float(row[f'aperture_sum_edge_{n}']) for n in range(len(expt['t0']))],dtype=bool)
            except ValueError:
                continue # this light curve contains no valid data
            lc['cps'] = np.array(lc['counts'] / expt['expt_eff'])
            lc['cps_err'] = np.array(np.sqrt(lc['counts']) / expt['expt_eff'])
            lc['xcenter'] = float(row['xcenter'])
            lc['ycenter'] = float(row['ycenter'])
            lightcurves+=[lc]
    return lightcurves

def screen_gfcat(eclipses,band='NUV',aper_radius=17,photdir='/Users/cm/GFCAT/photom',sigma=3):
    variables = {}
    for e in tqdm.tqdm(eclipses):
        edir = f'e{str(e).zfill(5)}'
        photpath = f'{photdir}/{edir}/{edir}-{band.lower()[0]}d-30s-photom.csv'
        expt = parse_exposure_time(photpath.replace('photom.csv', 'exptime.csv'))
        if np.sum(expt['expt_eff'])<300:
            continue # skip the whole eclipse if there is not at least 5 min of exposure total
        lightcurves = parse_lightcurves(photpath)
        candidate_variables = []
        for i,lc in enumerate(lightcurves):
            if any(lc['edge_flags']):
                continue # skip if there is any data near the detector edge
            ix = np.where((lc['cps']!=0) & (np.isfinite(lc['cps'])))[0]
            if expt['t1'][ix[-1]] - expt['t0'][ix[0]]<300:
                continue # skip if there are not at least 5 min of exposure on target
            if len(ix)/(ix[-1]+1-ix[0])<0.5:
                continue # skip if a majority of the bins are unobserved
                # NOTE: It's technically possible to have 30-second integrated flux on a source
                #  be exactly equal to zero. But it's low probability and has no meaningful effect
                #  on the variability search.
            # The chance of one outlier low point over 50M visits is not small, generates a lot of false positives
            # The chance of two outlier low points within a visit is small.
            # So use the second-lowest point in the visit as the benchmark.
            second_min = np.sort((lc['cps']+lc['cps_err']*sigma)[ix])[1]
            outlier_ix = np.where((lc['cps'] - lc['cps_err'] * sigma)[ix] > second_min)[0]
            if len(outlier_ix) < 3:
                continue # skip if there are not 3 significant outliers using the dumbest heuristic
            peak_ix, _ = signal.find_peaks(lc['cps'], prominence=3 * lc['cps_err'], distance=4)
            if len(peak_ix):
                if len(peak_ix) > 4:
                    continue  # skip: multiple spiky peaks, most likely contaminated by an artifact
                # NOTE: This test for spiky behavior is a little slower than I'd like, but workable.
            ad = stats.anderson(lc['cps'][ix])  # standard test of variability
            if ad.statistic <= ad.critical_values[2]:
                continue  # failed the anderson-darling test at 5%
                # NOTE: AD is the gold standard variability test, but it's relatively slow, so it
                #  has been pushed to the end of the screening heuristics
            # Whatever remains is a candidate variable
            candidate_variables.append({'id':i,'cps':np.median(lc['cps'][ix]),
                                        'xcenter':lc['xcenter'],'ycenter':lc['ycenter'],})
        if not len(candidate_variables):
            continue # there are no candidate variables at this point
        # Now screen out variables in clumps, which are very probably due to transient artifacts
        varix = eliminate_dupes(pd.DataFrame(candidate_variables).to_dict('list'))
        if len(varix) > 50:
            continue  # This is a cursed eclipse --- too many "variables" --- do not believe its lies
        variables[e] = varix
    return variables

def generate_qa_plots(vartable:dict,band='NUV',
                      photdir='/Users/cm/GFCAT/photom',
                      plotdir='/Users/cm/GFCAT/plots',
                      cleanup=False,
                      boxsz = 200 # pixels margin, so 2x this is the width
                        ):
    for e in tqdm.tqdm(vartable.keys()):
        edir = f'e{str(e).zfill(5)}'
        photpath = f'{photdir}/{edir}/{edir}-{band.lower()[0]}d-30s-photom.csv'
        lightcurves = parse_lightcurves(photpath)
        expt = parse_exposure_time(photpath.replace('photom.csv', 'exptime.csv'))
        cntfilename = f'{photdir}/e{str(e).zfill(5)}/e{str(e).zfill(5)}-{band[0].lower()}d-full.fits.gz'
        if not os.path.exists(cntfilename):
            cmd = f'aws s3 cp s3://dream-pool/e{str(e).zfill(5)}/e{str(e).zfill(5)}-{band[0].lower()}d-full.fits.gz {cntfilename}'
            os.system(cmd)
            if not os.path.exists(cntfilename):
                raise FileNotFoundError(f'This file has lightcurves and should definitely exist.\n{cntfilename}')
        image, flagmap, edgemap, wcs, _, _ = read_image(cntfilename)
        for i in vartable[e]:
            lc = lightcurves[i]
            imgx, imgy = lc['xcenter'],lc['ycenter'] # the image pixel coordinate of the source
            # define the bounding box for the thumbnail
            imsz = np.shape(image)
            # noting that image coordinates and numpy coordinates are flipped
            x1, x2, y1, y2 = (max(int(imgy - boxsz),0),
                              min(int(imgy + boxsz),imsz[0]),
                              max(int(imgx - boxsz),0),
                              min(int(imgx + boxsz),imsz[1]))

            fig = plt.figure(figsize=(12, 15))
            G = gridspec.GridSpec(4,2)
            ax = fig.add_subplot(G[0:3,:])
            ax.imshow(ZScaleInterval()(image[x1:x2, y1:y2]), cmap="Greys_r", origin="lower")
            ax.imshow(1 / edgemap[x1:x2, y1:y2], origin="lower", cmap="Reds_r", alpha=1)
            ax.imshow(1 / flagmap[x1:x2, y1:y2], origin="lower", cmap="Blues_r", alpha=1)
            ax.plot(boxsz, boxsz, markersize=30, color='y', lw=10, marker='o', fillstyle='none')  # marker='o')
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

            plt.savefig(f'{plotdir}/e{str(e).zfill(5)}-{band}-{str(i).zfill(4)}.png')
            plt.close('all')
        if cleanup:
            os.system(f'rm -rf {photdir}/e{str(e).zfill(5)}/')
    return

def screen_for_variables(eclipses=[],
                         catdbfile='/Users/cm/GFCAT/catalog.db',
                         photdir='/Users/cm/GFCAT/photom',
                         wrong_eclipse_file='/Users/cm/GFCAT/incorrectly_analyzed_eclipses.txt',
                         datadir='/Users/cm/GFCAT/data',plotdir='/Users/cm/GFCAT/data/plots',
                         aws=False):

    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")

    observations = {'eclipse': [], 'id': [],
                    'ra': [], 'dec': [],
                    'xcenter': [], 'ycenter': [],
                    'exptime': [], 'cps': [], 'cps_err': [],
                    'hasmask': [], 'hasedge': [],
                    'lc': []}
    cursed_eclipses = ['e36918', 'e45622',  # eclipses that just seem to be bad but aren't otherwise filtered
                       ]
    wrong_eclipses = pd.read_csv(wrong_eclipse_file)['eclipse'].values
    #for j, edir in enumerate(tqdm.tqdm(os.listdir(photdir))):
    for j, edir in enumerate(tqdm.tqdm(eclipses)):

        if edir in cursed_eclipses:
            continue
        try:
            eclipse = int(edir[1:])
        except ValueError:
            continue  # not a normal eclipse directory
        if eclipse in wrong_eclipses:
            continue  # skipping accidentally processed eclipse
        if not len(os.listdir(f'{photdir}/{edir}')):
            continue  # light curves not created (for any number of reasons)
        photstream = open(f'{photdir}/{edir}/{edir}-nd-30s-photom.csv')
        phot = csv.DictReader(photstream)  # DictReader is >> faster than Pandas
        expt = parse_exposure_time(f'{photdir}/{edir}/{edir}-nd-30s-exptime.csv')
        if len(expt['tstart']) < 25:
            continue  # not enough exposure time for me to care
        tix = np.where(np.array(expt['exposure_time']) > 0)
        varix = {'id': [], 'pos': [], 'cps': []}
        for obj in phot:
            visit_counts = float(obj['aperture_sum'])
            observations['eclipse'] += [eclipse]
            observations['id'] += [int(obj['id'])]
            observations['ra'] += [float(obj['ra'])]
            observations['dec'] += [float(obj['dec'])]
            observations['xcenter'] += [float(obj['xcenter'])]
            observations['ycenter'] += [float(obj['ycenter'])]
            observations['exptime'] += [expt['expt_total']]
            observations['cps'] += [visit_counts / expt['expt_total']]
            observations['cps_err'] += [math.sqrt(visit_counts) / expt['expt_total']]
            observations['hasmask'] += [float(obj['aperture_sum_mask']) > 0]
            observations['hasedge'] += [float(obj['aperture_sum_edge']) > 0]

            lc_counts = np.array([float(obj[f'aperture_sum_{i}']) for i in range(len(expt['exposure_time']))])
            lc = {'cps': lc_counts / expt['exposure_time'],
                  'cps_err': np.sqrt(lc_counts) / expt['exposure_time'],
                  't0': expt['tstart'],
                  'maskflag': np.array(
                      [bool(float(obj[f'aperture_sum_flag_{i}']) > 0) for i in range(len(expt['exposure_time']))]),
                  'edgeflag': np.array(
                      [bool(float(obj[f'aperture_sum_edge_{i}']) > 0) for i in range(len(expt['exposure_time']))])}
            observations['lc'] += [lc]

            if any((np.array(lc['edgeflag'])[1:-1])):
                continue  # if it even dips a toe into the red zone, don't bother

            if len(np.where((np.array(lc['cps'])[1:-1] == 0) &
                            (np.isnan(np.array(lc['cps'])[1:-1]) == 0))[0]) > 1:
                continue  # large number of data dropouts makes the lc too noisy

            if all((np.array(lc['maskflag'])[1:-1]) |
                   (np.array(lc['edgeflag'])[1:-1]) |
                   (np.array(lc['cps'])[1:-1] == 0) |
                   np.isnan(np.array(lc['cps'])[1:-1])):
                continue  # all of the data points are flagged or dropped

            # Check for just bulk variability on approx. 6-sigma scale
            sigma = 3
            # The chance of one outlier low point over 50M visits is not small, generates a lot of false positives
            # The chance of two outlier low points within a visit is small
            # So use the second-lowest point in the visit as the standard
            second_min = np.sort(np.unique((np.array(lc['cps']) + np.array(lc['cps_err']) * sigma)[1:-1]))[1]
            ix = np.where(
                (np.array(lc['cps']) - np.array(lc['cps_err']) * sigma)[1:-1] > second_min)
            if len(ix[0]) < 3:  # There must be three data points 6-sigma above the lowest data point
                continue

            ix, _ = signal.find_peaks(lc['cps'], prominence=3 * lc['cps_err'], distance=4)
            if len(ix):
                if len(ix) > 4:
                    continue  # multiple spiky peaks, most likely an artifact

            # AD is the gold standard variability test, but it's relatively slow so pushed to last
            ix = np.where(np.isfinite(np.array(lc['cps'])[1:-1]))
            ad = stats.anderson(np.array(lc['cps'])[ix])  # standard test of variability
            if ad.statistic <= ad.critical_values[2]:  # 5%
                continue  # failed the anderson-darling test

            # Whatever remains is a candidate variable
            varix['id'] += [int(obj['id'])]
            varix['pos'] += [(float(obj['xcenter']), float(obj['ycenter']))]
            varix['cps'] += [observations['cps'][-1]]

        if not len(varix['id']):
            continue
        varix = eliminate_dupes(varix)  # Screen variables very close to each other
        if len(varix['id']) > 50:
            continue  # This is a cursed eclipse --- too many "variables" --- do not believe its lies
        # if there are any candidate variables, get the QA image
        image_path = f'{datadir}/{edir}/{edir}-nd-full.fits.gz'
        cmd = f'aws s3 cp s3://dream-pool/{edir}/{edir}-nd-full.fits.gz {image_path}'
        os.system(cmd)
        # Read the image data
        image, flagmap, edgemap, wcs, _, _ = read_image(image_path)
        image[np.where(np.isinf(image))] = 0  # because it pops out with inf values... IDK
        for var in np.unique(varix['id']):
            # Remember that the index on observations is (eclipse + id) !!!
            ix = np.where((np.array(observations['id']) == var) & (np.array(observations['eclipse']) == eclipse))
            lc = np.array(observations['lc'])[ix][0]
            boxsz = 100
            pos = [np.array(observations['ycenter'])[ix][0],
                   np.array(observations['xcenter'])[ix][0]]
            x1, x2, y1, y2 = (int(pos[0] - boxsz), int(pos[0] + boxsz),
                              int(pos[1] - boxsz), int(pos[1] + boxsz))
            norm = simple_norm(image[x1:x2, y1:y2], 'log')
            fig = plt.figure(figsize=(12,15))
            G = gridspec.GridSpec(4,2)
            ax = fig.add_subplot(G[0:3,:])
            ax.imshow(ZScaleInterval()(image[x1:x2, y1:y2]), cmap="Greys_r", origin="lower")
            ax.imshow(1 / edgemap[x1:x2, y1:y2], origin="lower", cmap="Reds_r", alpha=1)
            ax.imshow(1 / flagmap[x1:x2, y1:y2], origin="lower", cmap="Blues_r", alpha=1)
            ax.plot(boxsz, boxsz, markersize=30, color='y', lw=10, marker='o', fillstyle='none')  # marker='o')
            ax.set_xticks([])
            ax.set_yticks([])

            ax = fig.add_subplot(G[3,:])
            ax.set_title(f"{edir} : {var}")
            ax.errorbar(np.array(lc['t0'])[1:-1],
                         np.array(lc['cps'])[1:-1],
                         yerr=3 * np.array(lc['cps_err'])[1:-1],
                         fmt='k-')
            ix = np.where(np.array(lc['maskflag'])[1:-1])
            ax.plot(np.array(lc['t0'])[1:-1][ix], np.array(lc['cps'])[1:-1][ix], 'bo')
            ax.set_xticks([])
            ax.set_ylabel('cps')
            ix = np.where(np.array(lc['edgeflag'])[1:-1])
            ax.plot(np.array(lc['t0'])[1:-1][ix], np.array(lc['cps'])[1:-1][ix], 'rx')
            fig.savefig(f'{plotdir}/{edir}-{var}.png')
            fig.close('all')
        # clean up the local data
        os.system(f'rm -rf {datadir}/{edir}')
        photstream.close()
    print('Done.')
