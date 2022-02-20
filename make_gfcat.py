from gfcat_utils import *

photdir = '/home/ubuntu/datadir/photom/' # Relative path to the local disk location of the photometry data
if len(os.listdir(photdir)):
    print(f'There are {len(os.listdir(photdir))} processed eclipses.')
else:
    cmd = f"aws s3 sync s3://dream-pool {photdir}. --exclude '*fits*' --exclude '*parquet' --exclude '*12_8.csv' --exclude '*LowExpt' --exclude '*tar.gz' --exclude '*log.csv' --exclude '*random*'"
    os.system(cmd)

wrong_eclipse_file = '/home/ubuntu/gfcat/incorrectly_analyzed_eclipses.txt'
try:
    wrong_eclipses = pd.read_csv(wrong_eclipse_file)['eclipse'].values
    print(f'There are {len(wrong_eclipses)} accidentally processed eclipses.')
except FileNotFoundError:
    print('The accidental eclipse file is missing.')

eclipses = [int(e[1:]) for e in os.listdir(f'{photdir}')]
eclipses = list(set(eclipses).difference(set(wrong_eclipses)))
print(f'There are notionally {len(eclipses)} eclipses in GFCAT.')

#eclipses = [34413,29703,13655,29703,] # These eclipses known variables for testing
# Variable screening takes 1-2 hours depending on available iron
candidate_variables = screen_gfcat(eclipses[:10],photdir=photdir)
# QA plot generation takes 6-12 hours, dominated by retrieving full frames from S3
plotdir = '/home/ubuntu/datadir/plots/'
generate_qa_plots(candidate_variables,photdir=photdir,plotdir=plotdir,cleanup=True)
