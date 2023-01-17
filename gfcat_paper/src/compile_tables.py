import numpy as np
import pyarrow.parquet as pq
import pandas as pd
from rich import print
import warnings
import os
from lightcurve_interface_skeleton import load_lightcurve_records, load_exptime
import datetime
from astropy.time import Time

def counts2mag(cps, band):
    scale = 18.82 if band == 'FUV' else 20.08
    with np.errstate(invalid='ignore'):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mag = -2.5 * np.log10(cps) + scale
    return mag

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

flare_list = pd.read_csv('flare_table.csv')
flare_table = pd.DataFrame()
for flare in flare_list.iterrows():
    eclipse = int(flare[1]['eclipse'])

    fn = f'{datadir}/e{str(eclipse).zfill(5)}-30s-photom.parquet'
    if not os.path.exists(fn):
        cmd = f'aws s3 cp s3://dream-pool/e{str(eclipse).zfill(5)}/e{str(eclipse).zfill(5)}-30s-photom.parquet {datadir}/.'
        os.system(cmd)

    obj_id = int(flare[1]['obj_id'])
    this_star = pq.read_table(catalog_filename,filters =
                              [('eclipse','=',eclipse),
                               ('obj_id','=',obj_id)]).to_pandas()
    expt = header_data.loc[header_data['ECLIPSE']==eclipse].loc[header_data['BAND']=='NUV']['EXPT_0']
    obstart = header_data.loc[header_data['ECLIPSE']==eclipse].loc[header_data['BAND']=='NUV']['T0_0']
    GPSSECS = 315532800 + 432000
    t = obstart.tolist()[0] + GPSSECS
    dt = datetime.datetime.fromtimestamp(t)
    this_star['datetime_iso'] = dt.isoformat().split('.')[0]
    this_star['datetime_decimal'] = Time(dt,format='datetime').decimalyear

    obj_ids = this_star["obj_id"].tolist() # it should only contain one entry
    aper_radius = 12.8
    for band in ['NUV','FUV']:
        try:
            expt = expt = load_exptime(fn, band=band, exptime_only=True)
        except KeyError:
            this_star[f'{band}mag'] = None
            this_star[f'{band}mag_err_1'] = None
            this_star[f'{band}mag_err_2'] = None
            continue
        if expt.sum()==0.0:
            this_star[f'{band}mag'] = None
            this_star[f'{band}mag_err_1'] = None
            this_star[f'{band}mag_err_2'] = None
            continue
        lightcurves = load_lightcurve_records(fn, band, apersize=aper_radius)
        variables = {}
        for lc in lightcurves:
            if lc['obj_id'] in obj_ids:
                variables[lc['obj_id']] = lc
        for obj_id in obj_ids:
            if obj_id not in variables.keys():
                print(f'{obj_id} not found in e{str(eclipse).zfill(5)} {band} unflagged lightcurves')
        ix = np.where(np.isfinite(lc['cps']))
        counts = (lc['cps'][ix]*expt[ix]).sum()
        if counts==0:
            this_star[f'{band}mag'] = None
            this_star[f'{band}mag_err_1'] = None
            this_star[f'{band}mag_err_2'] = None
            continue
        cps = counts / expt[ix].sum()
        cps_err = np.sqrt(counts) / expt[ix].sum()
        mag = counts2mag(cps, band)

        mag_err_1 = mag - counts2mag(cps + cps_err, band)
        mag_err_2 = counts2mag(cps - cps_err, band) - mag  # this one is always larger
        this_star[f'{band}mag'] = mag
        this_star[f'{band}mag_err_1'] = mag_err_1
        this_star[f'{band}mag_err_2'] = mag_err_2

    flare_table = pd.concat([flare_table,
                             this_star[
                                 ["obj_id", "ra", "dec", "eclipse",
                                  "datetime_iso", "datetime_decimal",
                                  "NUVmag", "NUVmag_err_1", "NUVmag_err_2",
                                  "FUVmag", "FUVmag_err_1", "FUVmag_err_2",]]])
flare_table.to_csv('gfcat_flares.csv',index=None)
