import numpy as np
import pyarrow.parquet as pq
import pandas as pd
from lightcurve_interface_skeleton import load_lightcurve_records, is_spiky
import os
from sklearn.cluster import DBSCAN
from scipy import signal, stats
from rich import print
import matplotlib.pyplot as plt

np.random.seed(19470622) # the birthdate of Bruno Latour

datadir = '/home/ubuntu/datadir'
if not os.path.exists(f'{datadir}/mislike_image_header_table.csv'):
    cmd =  f"aws s3 cp s3://nishapur/galex_metadata/mislike_image_header_table.csv {datadir}/."
    os.system(cmd)
header_data = pd.read_csv(f'{datadir}/mislike_image_header_table.csv')

catalog_filename = f"{datadir}/catalog_nd_daostarfinder.parquet"
if not os.path.exists(f"{datadir}/catalog_nd_daostarfinder.parquet"):
    cmd = f"aws s3 cp s3://dream-pool/indices/catalog_nd_daostarfinder.parquet {datadir}/."
    os.system(cmd)
catalog_file = pq.ParquetFile(catalog_filename)

def find_bright_stars(catalog_file,header_data,
                      n_ecl=100, # number of eclipses to check, not necessarily return
                      countrate_limit=100, # min cps in a 12.8" aperture to be bright
                      ):
    n=0
    targets=[]
    scrambled_ecl = np.random.choice(len(header_data),size=len(header_data),replace=False)
    #while len(np.unique(np.array(targets)[:,0]))<n_ecl:
    for ix in np.random.choice(len(header_data),size=n_ecl,replace=False):
        visit = header_data.iloc[ix]
        eclipse = int(visit['ECLIPSE'])
        #print(eclipse)
        if visit['EXPTIME']<1200.0:
            continue
        bright_stars = pq.read_table(catalog_filename,filters =
                                     [('aperture_sum_mask_n_51_2','=',0.0),
                                      ('aperture_sum_edge_n_51_2','=',0.0),
                                      ('aperture_sum_n_51_2','>',10000.0),
                                      ('eclipse','=',eclipse)]).to_pandas()

        cps = np.array(bright_stars['aperture_sum_n_12_8'])/visit['EXPT_0']
        ix = np.where(cps>countrate_limit)
        if not len(cps[ix]):
            #print(f'No bright stars in e{str(eclipse).zfill(5)}')
            continue
        X = list(zip(bright_stars['xcenter'].iloc[ix].tolist(),bright_stars['ycenter'].iloc[ix].tolist()))
        db = DBSCAN(eps=40,min_samples=1).fit(X)
        labels=db.labels_
        for lbl in set(labels):
            dbix = np.where(labels==lbl)[0]
            brightest_ix = np.argmax(cps[ix][dbix])
            star = bright_stars.iloc[ix].iloc[dbix].iloc[brightest_ix]
            cps_12_8,cps_51_2 = (star['aperture_sum_n_12_8']/visit['EXPT_0'],
                                 star['aperture_sum_n_51_2']/visit['EXPT_0'])
            if cps_12_8<=countrate_limit:
                continue
            # a huge discrepancy between aperture sizes probably means nearby bright stars / field
            if cps_51_2-cps_12_8 > 100:
                n+=1
                continue
            #print(eclipse,int(star['obj_id']),cps_12_8,cps_51_2,cps_51_2-cps_12_8)
            targets+=[[eclipse,int(star['obj_id'])]]
    print(n)
    return targets

targets = find_bright_stars(catalog_file,header_data,n_ecl=2000,countrate_limit=100)
print(f"{len(targets)} sources in {len(np.unique(np.array(targets)[:,0]))} eclipses")

for eclipse in np.unique(np.array(targets)[:,0]):
    fn = f'{datadir}/e{str(eclipse).zfill(5)}-30s-photom.parquet'
    if not os.path.exists(fn):
        cmd = f'aws s3 cp s3://dream-pool/e{str(eclipse).zfill(5)}/e{str(eclipse).zfill(5)}-30s-photom.parquet {datadir}/.'
        os.system(cmd)

def get_lc_summary_stats(lc):
    ad = stats.anderson(lc['cps'][np.where(np.isfinite(lc['cps']))])  # standard test of variability
    return {
        'mad':np.nanmean(np.abs(lc['cps'] - np.nanmean(lc['cps']))),
        'start_cps':np.nanmean(lc['cps'][:5]),
        'end_cps':np.nanmean(lc['cps'][-6:-1]), # exclude the last bin, which is often low-expt
        'min_cps':np.nanmin(lc['cps'][:-1]),
        'max_cps':np.nanmax(lc['cps'][:-1]),
        'mean_std':np.nanmean(lc['cps_err']),
        'ad_statistic':ad.statistic,
        'ad_critical_val_10':ad.critical_values[1], # 10%
        'ad_critical_val_05':ad.critical_values[2], # 5%
        'ad_critical_val_01':ad.critical_values[4], # 1%
        'xcenter':lc['xcenter'],
        'ycenter':lc['ycenter'],
        'is_spiky':bool(is_spiky(lc)),
    }

bright_star_table = pd.DataFrame()
for eclipse in np.unique(np.array(targets)[:, 0]):
    print(eclipse)
    fn = f'{datadir}/e{str(eclipse).zfill(5)}-30s-photom.parquet'
    aper_radius = 51.2
    lightcurves = load_lightcurve_records(fn, 'NUV', apersize=aper_radius)

    obj_ids = np.array(targets)[np.where(np.array(targets)[:,0]==eclipse)][:,1].tolist()

    variables = {}
    for lc in lightcurves:
        if lc['obj_id'] in obj_ids:
            variables[lc['obj_id']] = lc
    for obj_id in obj_ids:
        if obj_id not in variables.keys():
            print(f'{obj_id} not found in {eclipse} {band} unflagged lightcurves')

    for k in variables.keys():
        lc = variables[k]
        summary_stats = get_lc_summary_stats(lc)
        summary_stats['obj_id'] = int(k)
        summary_stats['eclipse'] = int(str(k)[-5:])
        hdr = header_data.loc[header_data['ECLIPSE'] == summary_stats['eclipse']].loc[header_data['BAND']=='NUV']
        summary_stats['CRPIX1'] = float(hdr['CRPIX1'])
        summary_stats['CRPIX2'] = float(hdr['CRPIX2'])
        bright_star_table = bright_star_table.append(pd.Series(summary_stats), ignore_index=True)
        plt.figure(figsize=(12, 1))
        plt.errorbar(range(len(lc['cps'][:-1])), lc['cps'][:-1],
                     yerr=3*lc['cps_err'][:-1], fmt='k-')
        plt.xticks([])
        plt.tight_layout()
        plt.savefig(f"{datadir}/e{str(eclipse).zfill(5)}_{k}_bs.jpg")
        print(f"{datadir}/e{str(eclipse).zfill(5)}_{k}_bs.jpg")
        plt.close('all')

bright_star_table = bright_star_table.astype({'obj_id': int, 'eclipse': int})
print(bright_star_table)

bright_star_table.to_csv('brightstar_stats.csv',index=None)
