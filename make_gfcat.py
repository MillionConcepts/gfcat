from gfcat_utils import *
import sqlalchemy as sql
import numpy as np

photdir = '/home/ubuntu/datadir/' # Relative path to the local disk location of the photometry data
if os.path.exists(photdir):
    print(f'There are {len(os.listdir(photdir))} processed eclipses.')
else:
    raise f"{photdir} is not available"

wrong_eclipse_file = '/home/ubuntu/incorrectly_analyzed_eclipses.txt'
try:
    wrong_eclipses = pd.read_csv(wrong_eclipse_file)['eclipse'].values
    print(f'There are {len(wrong_eclipses)} accidentally processed eclipses.')
except FileNotFoundError:
    print('The accidental eclipse file is missing.')

n_eclipses = len(os.listdir(photdir))-len(wrong_eclipses)
print(f'There are notionally {n_eclipses} eclipses in GFCAT.')

catdbfile='/home/ubuntu/catalog.db'
if not os.path.exists(catdbfile):
    # This will take like half an hour, but it's worth it.
    generate_visit_database(catdbfile=catdbfile,
                            photdir = photdir,
                            wrong_eclipse_file=wrong_eclipse_file)

# Grab a list of all processed eclipses from photometry table
engine = sql.create_engine(f'sqlite:///{catdbfile}', echo=False)
out = engine.execute(f"SELECT DISTINCT eclipse FROM gfcat ").fetchall()
engine.dispose()
eclipses = np.array(out)[:,0]

# This takes many hours.
for eclipse in eclipses:
    screen_for_variables(eclipses=[f'e{str(eclipse).zfill(5)}',],
                         catdbfile=catdbfile,
                         photdir=photdir,
                         wrong_eclipse_file=wrong_eclipse_file)
