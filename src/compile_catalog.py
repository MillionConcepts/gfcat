import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
Gaia.ROW_LIMIT = 200  # Ensure the default row limit.

mdw_data = pd.read_csv('/Users/cm/github/gfcat_mdwarfs/src/mdw.csv',index_col=0)

for i in range(len(mdw_data)):
    source = mdw_data.iloc[i]
    coord = SkyCoord(source['galex_ra'], source['galex_dec'], unit=u.deg)
    radius = u.Quantity(0.2, u.deg)
    j = Gaia.cone_search_async(coord, radius)
    r = j.get_results()
    # match on the source that is closest to the estimate distance from prior catalogs... probably right...
    distance = source['cat_distance'] if np.isnan(source['distance']) else source['distance']
    distance = source['cat_distance']
    match_ix = np.argmin(np.abs(r['parallax']-distance))
    #print(r[['dist','pmra','pmdec','parallax']][match_ix])
    #print(r[['dist','pmra','pmdec','parallax']])
    print(i, distance,r['parallax'][match_ix],
          r['pmra'][match_ix],r['pmdec'][match_ix])
