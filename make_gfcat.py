from lightcurve_interface_skeleton import screen_variables, load_lightcurve_records
from gfcat_utils import read_image, parse_exposure_time
import os
import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
from matplotlib.patches import Rectangle, Circle
import imageio.v2 as imageio
import matplotlib as mpl
from clize import run
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import shutil

def screen_eclipse(eclipse, photdir = '/home/ubuntu/datadir/', band = 'NUV'):
    estring = f"e{str(eclipse).zfill(5)}"
    edir = f"{photdir}{estring}"
    if not os.path.exists(edir):
        os.makedirs(edir)

    photfilename = f"{edir}/{estring}-30s-photom.parquet"
    if not os.path.exists(photfilename):
        cmd = f"aws s3 cp s3://dream-pool/{estring}/{estring}-30s-photom.parquet {edir}/."
        os.system(cmd)

    varix, rejects = screen_variables(f'{edir}/{estring}-30s-photom.parquet')

    if not len(varix):
        shutil.rmtree(edir)
        return []

    return varix

def make_qa_image(eclipse, obj_ids, step="prescreen", # or "final"
                  photdir = '/home/ubuntu/datadir/', band = 'NUV',aper_radius=12.8, cleanup=True):
    e,b = eclipse,band[0].lower()
    estring = f"e{str(eclipse).zfill(5)}"
    edir = f"{photdir}{estring}"
    print(f'Initialize QA images creation for {estring} {band}')
    photfilename = f"{edir}/{estring}-30s-photom.parquet"
    if not os.path.exists(photfilename):
        cmd = f"aws s3 cp s3://dream-pool/{estring}/{estring}-30s-photom.parquet {edir}/."
        os.system(cmd)

    try:
        lightcurves = load_lightcurve_records(photfilename, band, apersize=aper_radius)
    except KeyError:
        print(f'No {band} data available for {estring}.')
        return
    expt = parse_exposure_time(photfilename)

    variables = {}
    for lc in lightcurves:
        if lc['obj_id'] in obj_ids:
            variables[lc['obj_id']] = lc
    for obj_id in obj_ids:
        if obj_id not in variables.keys():
            print(f'{obj_id} not found in {eclipse} {band} unflagged lightcurves')
    if not len(lc):
        print(f'No matching objects in {estring} {band}')
        return

    depth = 'full' if step=='prescreen' else '30s'
    imgfilename = f"{edir}/{estring}-{band[0].lower()}d-{depth}.fits.gz"
    if not os.path.exists(imgfilename):
        cmd = f"aws s3 cp s3://dream-pool/{estring}/{estring}-{band[0].lower()}d-{depth}.fits.gz {edir}/."
        os.system(cmd)
    print(f'Reading {estring} {band} movie file.')
    imgmap, _, _, wcs, tranges, exptimes = read_image(movfilename)
    # The WCS in the movie files incorrectly uses the number of frames as an image dimension. Hack fix it here.
    wcs.wcs.crpix[0] = np.shape(imgmap)[2]/2 + 0.5
    wcs.wcs.crpix[1] = np.shape(imgmap)[1]/2 + 0.5
    imgmap[np.where(np.isinf(imgmap))] = 0  # because it pops out with inf values... IDK
    imgmap[np.where(imgmap < 0)] = 0

    for source_ix in variables.keys():
        lc = variables[source_ix]
        print(f'Initializing {source_ix} {band} QA frames.')
        curve = {band:{'t':expt['t0'],
                       'cps':lc['cps'],
                       'cps_err':lc['cps_err']}}
        min_i, max_i = np.argmin(curve[band]['cps']), np.argmax(curve[band]['cps'])

        assert len(lc['cps']) == np.shape(imgmap)[0]  # if these don't match then the gif will be out of sync

        # get the image pixel coordinates of the source via WCS
        imgpos = wcs.wcs_world2pix([[lc['ra'],lc['dec']]],1) # set the origin to FITS standard
        imgx,imgy = imgpos[0]

        # define the bounding box for the thumbnail
        imsz = np.shape(imgmap[0])

        # crop on the subframe
        # noting that image coordinates and numpy coordinates are flipped
        boxsz = 200
        x1, x2, y1, y2 = (max(int(imgy - boxsz), 0),
                          min(int(imgy + boxsz), imsz[0]),
                          max(int(imgx - boxsz), 0),
                          min(int(imgx + boxsz), imsz[1]))

        # crop on the full frame
        # The cropping is here to handle very wide images created by inappropriate handling
        # of map cos(theta) projection distortions when initializing the image size during processing;
        # this was fixed in the pipeline that generated the final run of gfcat data, so it should just
        # be returning the full image dimensions now.
        x1_, x2_, y1_, y2_ = (max(int(imsz[0] / 2 - imsz[0] / 2), 0),
                              min(int(imsz[0] / 2 + imsz[0] / 2), imsz[0]),
                              max(int(imsz[1] / 2 - imsz[0] / 2), 0),
                              min(int(imsz[1] / 2 + imsz[0] / 2), imsz[1]))

        gs = gridspec.GridSpec(nrows=3, ncols=6)  # , height_ratios=[1, 1, 2])

    if step=="prescreen": # generate faster but less informative full depth qa images
        print(f'Generating {source_ix} {band} QA image.')
        fig = plt.figure(figsize=(12, 9));
        fig.tight_layout()
        ax = fig.add_subplot(gs[:3, :])
        ax.imshow(ZScaleInterval()(frame[x1_:x2_, y1_:y2_]), origin="lower", cmap="Greys_r")
        ax.set_xticks([])
        ax.set_yticks([])
        rect = Rectangle((y1 - y1_, x1 - x1_), 2 * boxsz, 2 * boxsz, linewidth=1, edgecolor='y', facecolor='none',
                         ls='solid')
        ax.add_patch(rect)

        ax = fig.add_subplot(gs[3:, :])
        ax.vlines(curve[band]['t'][i], curve[band]['cps'][min_i] - 3 * curve[band]['cps_err'][min_i],
                  curve[band]['cps'][max_i] + 3 * curve[band]['cps_err'][max_i], ls='dotted')
        ax.scatter(curve[band]['t'][i], curve[band]['cps'][i], c='y', s=100, marker='o')
        ax.errorbar(curve[band]['t'], curve[band]['cps'],
                    yerr=curve[band]['cps_err'] * 3, fmt='k.-', label=band)
        ax.set_xlim([curve[band]['t'].min() - 30, curve[band]['t'].max() + 60])
        ax.set_xticks([])
        plt.legend()

        plt.savefig(f'{edir}/{estring}-{b}-full-{str(i).zfill(2)}-{str(source_ix).zfill(5)}.jpg', dpi=100)
        plt.close('all')

    else: # generate slower but more informative animated qa images
            print(f'Generating {source_ix} {band} QA frames.')
            for i, frame in enumerate(imgmap):  # probably eliminate the first / last frame, which always has lower exposure
                fig = plt.figure(figsize=(12, 9));
                fig.tight_layout()
                ax = fig.add_subplot(gs[:3, :3])
                ax.imshow(ZScaleInterval()(frame[x1_:x2_, y1_:y2_]),origin="lower",cmap="Greys_r")
                ax.set_xticks([])
                ax.set_yticks([])
                rect = Rectangle((y1 - y1_, x1 - x1_), 2 * boxsz, 2 * boxsz, linewidth=1, edgecolor='y', facecolor='none',
                                 ls='solid')
                ax.add_patch(rect)

                ax = fig.add_subplot(gs[:3, 3:])
                ax.imshow(ZScaleInterval()(frame[x1:x2, y1:y2]),origin="lower",cmap="Greys_r")
                ax.set_xticks([])
                ax.set_xticks([])
                ax.set_yticks([])
                circ = Circle((boxsz, boxsz), 20, linewidth=1, edgecolor='y', facecolor='none', ls='solid')
                ax.add_patch(circ)

                ax = fig.add_subplot(gs[3:, :])
                ax.vlines(curve[band]['t'][i], curve[band]['cps'][min_i] - 3 * curve[band]['cps_err'][min_i],
                          curve[band]['cps'][max_i] + 3 * curve[band]['cps_err'][max_i], ls='dotted')
                ax.scatter(curve[band]['t'][i], curve[band]['cps'][i], c='y', s=100, marker='o')
                ax.errorbar(curve[band]['t'], curve[band]['cps'],
                            yerr=curve[band]['cps_err'] * 3, fmt='k.-',label=band)
                ax.set_xlim([curve[band]['t'].min()-30,curve[band]['t'].max()+60])
                ax.set_xticks([])
                plt.legend()

                plt.savefig(f'{edir}/{estring}-{b}-30s-{str(i).zfill(2)}-{str(source_ix).zfill(5)}.jpg', dpi=100)
                plt.close('all')

            print(f'Compiling {source_ix} {band} movie.')
            n_frames = np.shape(imgmap)[0]
            # write the animated gif
            gif_fn = f'{edir}/{estring}-{str(source_ix).zfill(5)}-{b}-30s.gif'
            print(f"writing {gif_fn}")
            with imageio.get_writer(gif_fn, mode='I', fps=6) as writer:
                for i in np.arange(n_frames):
                    frame_fn = f'{edir}/{estring}-{b}-30s-{str(i).zfill(2)}-{str(source_ix).zfill(5)}.jpg'
                    image = imageio.imread(frame_fn)
                    writer.append_data(image)
                    # remove the png frames
                    os.remove(frame_fn)

    # remove the local copies of the
    #if cleanup:
    os.remove(photfilename)
    os.remove(movfilename)

def main(eclipse:int,photdir = '/home/ubuntu/datadir/', make_qa_images=True):
    estring = f"e{str(eclipse).zfill(5)}"
    edir = f"{photdir}{estring}"
    print(f'Processing {estring}')
    varix = screen_eclipse(eclipse, photdir=photdir)
    if len(varix):
        print(f'Variables found {varix}')
    else:
        print('No variables found')

    if len(varix) and make_qa_images:
        make_qa_image(eclipse,varix,band='NUV', photdir=photdir)
        try:
            make_qa_image(eclipse,varix,band='FUV', photdir=photdir)
        except KeyError:
            pass

    cmd = f"aws s3 cp {edir}/*gif s3://dream-pool/{estring}/."
    print(cmd)


# tell clize to handle command line call
if __name__ == "__main__":
    run(main)
