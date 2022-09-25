from functools import partial
import json
import re
from typing import Literal, Optional, Sequence, Union

from cytoolz import frequencies
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pac
from pyarrow import parquet
from scipy import signal, stats

from gfcat_utils import eliminate_dupes
from gPhoton.types import GalexBand, Pathlike

from sklearn.cluster import DBSCAN

def bin_field_name(    
    size: float,
    band: GalexBand="NUV",
    binno: int = None, 
    plane: Literal["cnt", "mask", "edge"]="cnt"
) -> str:
    """
    format the name of a 'data' field for a specific 
    aperture size, band (default NUV), bin number 
    (None means full-depth) and plane (default counts)
    """
    col = "aperture_sum"
    if plane in ("mask", "edge"):
        col += f"_{plane}"
    if binno is not None:
        col += f"_{binno}"
    col += f"_{band[0].lower()}"
    return col + f"_{str(size).replace('.', '_')}"


def id_fields(fieldnames: Sequence[str]) -> list[str]:
    """filter a list of field/column names for 'not-data' fields"""
    return [f for f in fieldnames if 'aperture_sum' not in f]


VARIABLE_PIPE_ID_FIELDS = ('xcenter', 'ycenter', 'ra', 'dec', 'obj_id')


def data_fields(fieldnames: Sequence[str]) -> list[str]:
    """filter a list of field/column names for 'data' fields"""
    return [f for f in fieldnames if 'aperture_sum' in f]


def curve_fields(fieldnames: Sequence[str]) -> list[str]:
    """filter a list of field/column names for time-bin field names"""
    return [
        f for f in fieldnames 
        if re.match(r'aperture_sum_(flag_|edge_)?\d', f)
    ]


def sizes(fieldnames: Sequence[str]) -> list[str]:
    """return list of all aperture size strings used in a list of field/column names"""
    return sorted(
        set(map(lambda f: "_".join(f.rsplit("_")[-2:]), data_fields(fieldnames)))
    )


def bins(fieldnames: Sequence[str]) -> list[int]:
    """return list of bin numbers used in a list of field/column names"""
    time_fields = filter(
        None, map(partial(re.match, r"aperture_sum_(\d+)"), data_fields(fieldnames))
    )
    return sorted(set(map(int, [m.group(1) for m in time_fields])))


def load_unflagged(
    lightcurve_file: Pathlike, size: Optional[float]=None, band: GalexBand="NUV",
) -> pd.DataFrame:
    """
    load just the curves from a lightcurve parquet file for a specific aperture size 
    (by default the smallest) and band (by default NUV) with no counts from mask or 
    edge backplanes
    """
    file = parquet.ParquetFile(lightcurve_file)
    data = data_fields(file.schema.names)
    if size is None:
        size = sizes(data)[0]
    edge, mask = (bin_field_name(size, band, plane) for plane in ("edge", "mask"))
    bin_cols = [
        bin_field_name(size, band, binno) for binno in [None] + bins(data)
    ]
    tab = file.read(
        columns=list(VARIABLE_PIPE_ID_FIELDS) + bin_cols + [edge, mask], 
    )
    tab = tab.filter(pac.equal(pac.add(tab[edge], tab[mask]), 0))
    return tab.to_pandas(), load_exptime(tab, band)


def load_exptime(
    lightcurve_file: Union[Pathlike, pa.Table], 
    band: GalexBand = "NUV", 
    exptime_only=True
) -> Union[np.ndarray, pd.DataFrame]:
    """
    load exposure time table for a specific band from a lightcurve parquet file.
    if exptime_only is True, return just an array containing exposure times per bin;
    otherwise return a dataframe also containing bounds per bin
    """
    fieldname = f'{band.lower()}_exptime'.encode('ascii')
    if isinstance(lightcurve_file, pa.Table):
        metadata = lightcurve_file.schema.metadata
    else:
        metadata = parquet.ParquetFile(lightcurve_file).schema_arrow.metadata
    records = json.loads(metadata[fieldname].decode())
    if exptime_only is True:
        return np.array([rec['expt'] for rec in records], dtype=np.float32)
    return pd.DataFrame(records)


def lightcurve_df_to_cps(
    lightcurves: np.ndarray, exptime: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    cps = np.einsum('i,ji -> ji', 1/exptime, lightcurves)
    cps_err = np.einsum('i,ji -> ji', 1/exptime, np.sqrt(lightcurves))
    return cps, cps_err

def load_lightcurve_records(
    lightcurve_parquet: Pathlike, band: GalexBand = 'NUV', apersize: int = 12.8
) -> tuple[list[dict[str, Union[np.ndarray, float, int]]], np.ndarray]:
    """load lightcurve records from a lightcurve parquet file"""
    # dataframe containing unflagged bins from specified band and aperture size,
    # and ndarray with exptime per bin
    table, exptime = load_unflagged(lightcurve_parquet, band=band, size=apersize)
    # select time-series data out for conversion to cps
    lightcurves = table[curve_fields(table)]
    # make cps and cps_err arrays
    cps, cps_err = lightcurve_df_to_cps(lightcurves, exptime)
    # select metadata fields & convert to records
    id_records = table[list(VARIABLE_PIPE_ID_FIELDS)].to_dict(orient='records')
    # make records from each row of cps/cps_err arrays and merge w/metadata records
    return [
        id_record | {'cps': cps_vec, 'cps_err': cps_err_vec}
        for id_record, cps_vec, cps_err_vec in zip(id_records, cps, cps_err)
    ]


def is_spiky(lc: dict):
    for sigma,n_outliers in [(3,3),(2,5)]: # sigma prominence is actually ~2x
        sigma_err = sigma * lc['cps_err']
        upper_limit = lc['cps'] + sigma_err
        lower_limit = lc['cps'] - sigma_err
        for n in [1,2]: # number of bins to bunch up as "one" outlier
            ix = np.where(
                (lower_limit[n:-n] - upper_limit[:-int(n*2)] > 0)
                & (lower_limit[n:-n] - upper_limit[int(n*2):] > 0)
            )[0]
            if len(ix) >= n_outliers:
                return True # This is a dumber test than the one above for the same thing. Seems more effective.
    return False


def eliminate_dupes(variable_table, rejects):
    # Run a spatial clustering algorithm and consider variables within 1 arcmin
    #  of each other to be most likely the same source and combine them, choosing
    #  the brightest of the sources as the primary
    X = list(zip(variable_table['xcenter'], variable_table['ycenter']))  #variable_table['pos']
    db = DBSCAN(eps=40, min_samples=1).fit(X) # 40 pixels ~= 1 arcmin
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    varix = {'id': []}
    for lbl in set(labels):
        dbix = np.where(labels==lbl)[0]
        if any(np.array(variable_table["cps"])[dbix] > 170):
            for ix in dbix:
                rejects[variable_table['id'][ix]] = 'too bright (or in cluster w/too bright)'
            continue # if there is a very bright star in the cluster, dump them all
        if len(dbix)>=12: # big cluster of variables are presumed artifacts
            for ix in dbix:
                rejects[variable_table['id'][ix]] = 'deduped: cluster size > 12'
            continue
        elif len(dbix)>1: # small cluster -- use the one with maximum variation
            xcenters,ycenters = np.array(variable_table['xcenter']),np.array(variable_table['ycenter'])
            dist = np.sqrt((xcenters[dbix].min()-
                            xcenters[dbix].max())**2 +
                           (ycenters[dbix].min()-
                            ycenters[dbix].max())**2)
            if dist > 80: # cluster is more than 2 arcminutes across
                for ix in dbix:
                    rejects[variable_table['id'][ix]] = 'deduped: cluster > 2 arcmin'
                continue
            ix = [np.argmax(np.abs(variable_table['delta_cps'])[dbix])]
            varix['id']+=np.array(variable_table['id'])[dbix][ix].tolist() # Earlier version was missing dbix ---Fixed 220227
        else:
            varix['id']+=np.array(variable_table['id'])[dbix].tolist()
    return varix['id'], rejects
    
    

def screen_variables(fn:str, band='NUV', aper_radius=12.8, sigma=3, binsz=30):
    lightcurves = load_lightcurve_records(fn, band, apersize=aper_radius)
    expt = load_exptime(fn, band=band, exptime_only=False)
    if expt['expt'].sum() < 500:
        print('Short exposure.')
        return [], {}
    candidate_variables, rejects = [], {}
    for i, lc in enumerate(lightcurves):
        if not any(lc['cps'] > 0.5):
            rejects[i] = "too dim"
            continue  # too dim to be meaningful
        ix = np.where((lc['cps'] != 0) & (np.isfinite(lc['cps'])))[0]
        if expt['t1'][ix[-1]] - expt['t0'][ix[0]] < 500:
            rejects[i] = "too brief"
            continue
        if len(ix) / (ix[-1] + 1 - ix[0]) < 0.75:
            rejects[i] = "more than 1/4 bins unobserved"
            continue
        sigma_err = lc['cps_err'] * sigma
        second_min = np.sort((lc['cps'] + sigma_err)[ix])[1]
        outlier_ix = np.where(
            (lc['cps'] - sigma_err)[ix] > second_min
        )[0]
        if len(outlier_ix) < 3:
            rejects[i] = "less than 3 outliers"
            continue  # skip if there are not 3 significant outliers using the dumbest heuristic
        if is_spiky(lc):
            rejects[i] = "spiky (crude)"
            continue  # skip: multiple spiky peaks, most likely contaminated by an artifact
        peak_ix, _ = signal.find_peaks(lc['cps'], prominence=3 * lc['cps_err'], distance=4)
        if len(peak_ix) > 3:
            rejects[i] = "spiky (fine)"
            continue  # skip multiple spiky peaks, most likely contaminated by an artifact
        ad = stats.anderson(lc['cps'][ix])  # standard test of variability
        if ad.statistic <= ad.critical_values[2]:
            rejects[i] = "anderson-darling"
            continue  # failed the anderson-darling test at 5%
        candidate_variables.append(
            {
                'id': i, 
                'cps': np.median(lc['cps'][ix]), 
                'xcenter': lc['xcenter'], 
                'ycenter': lc['ycenter'],
                'delta_cps': np.min(lc['cps'][ix]) - np.max(lc['cps'][ix])
            }
        )
    if len(candidate_variables) == 0:
        return [], rejects # there are no candidate variables at this point
    # Now screen out variables in clumps, which are very probably due to transient artifacts
    varix, rejects = eliminate_dupes(pd.DataFrame(candidate_variables).to_dict('list'), rejects)
    if len(varix) >= 20:
        print("cursed eclipse")
        return [], rejects  # This is a cursed eclipse --- too many "variables" --- do not believe its lies
    if len(varix) == 0:
        print("no variables after declumping")
        return [], rejects # there are no variables
    return varix, rejects



"""
# print reasons for rejections:
eclipse = 23456
e = str(eclipse).zfill(5)
varix, rejects = screen_variables(f"e{e}/e{e}-30s-photom.parquet")
frequencies(rejects.values())
"""
