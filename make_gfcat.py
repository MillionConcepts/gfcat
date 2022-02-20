from gfcat_utils import *

photdir = '/Users/cm/GFCAT/photom/' # Relative path to the local disk location of the photometry data
if len(os.listdir(photom)):
    print(f'There are {len(os.listdir(photdir))} processed eclipses.')
else:
    cmd = f"aws s3 sync s3://dream-pool {photdir}. --exclude '*fits*' --exclude '*parquet' --exclude '*12_8.csv' --exclude '*LowExpt' --exclude '*tar.gz' --exclude '*log.csv"
    os.system(cmd)

wrong_eclipse_file = '/Users/cm/GFCAT/gfcat/incorrectly_analyzed_eclipses.txt'
try:
    wrong_eclipses = pd.read_csv(wrong_eclipse_file)['eclipse'].values
    print(f'There are {len(wrong_eclipses)} accidentally processed eclipses.')
except FileNotFoundError:
    print('The accidental eclipse file is missing.')

eclipses = [int(e[1:]) for e in os.listdir('../photom/')]
eclipses = list(set(eclipses).difference(set(wrong_eclipses)))
print(f'There are notionally {len(eclipses)} eclipses in GFCAT.')

#catdbfile='/Users/cm/GFCAT/catalog.db'
#if not os.path.exists(catdbfile):
#    # This will take like half an hour the first time, but it's worth it.
#    generate_visit_database(catdbfile=catdbfile,
#                            photdir = photdir,
#                            wrong_eclipse_file=wrong_eclipse_file)

# Grab a list of all processed eclipses from photometry table... this might take a minute...
#engine = sql.create_engine(f'sqlite:///{catdbfile}', echo=False)
#out = engine.execute(f"SELECT DISTINCT eclipse FROM gfcat ").fetchall()
#engine.dispose()
#eclipses = np.array(out)[:,0]

#eclipses = [34413,29703,13655,29703,] # These eclipses known variables for testing
# Variable screening takes 1-2 hours depending on available iron
candidate_variables = screen_gfcat(eclipses[:10])
# QA plot generation takes 6-12 hours, dominated by retrieving full frames from S3
generate_qa_plots(candidate_variables)
