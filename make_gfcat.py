from lightcurve_interface_skeleton import screen_variables, load_lightcurve_records
from gfcat_utils import read_image
import os
import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
from matplotlib.patches import Rectangle, Circle
import imageio.v2 as imageio

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
        os.remove(edir)
        return []

    return varix

def make_qa_image(eclipse, obj_ids, photdir = '/home/ubuntu/datadir/', band = 'NUV',aper_radius=12.8, cleanup=True):
    e,b = eclipse,band[0].lower()
    estring = f"e{str(eclipse).zfill(5)}"
    edir = f"{photdir}{estring}"
    photfilename = f"{edir}/{estring}-30s-photom.parquet"
    lightcurves = load_lightcurve_records(photfilename, band, apersize=aper_radius)

    variables = {}
    for lc in lightcurves:
        if lc['obj_id'] in obj_ids:
            variables[lc['obj_id']] = lc
    for obj_id in obj_ids:
        if obj_id not in variables.keys():
            print(f'{obj_id} not found in {eclipse} {band} unflagged lightcurves')

    movfilename = f"{edir}/{estring}-{band[0].lower()}d-30s.fits.gz"
    if not os.path.exists(movfilename):
        cmd = f"aws s3 cp s3://dream-pool/{estring}/{estring}-{band[0].lower()}d-30s.fits.gz {edir}/."
        os.system(cmd)

    print(f'Reading {estring} movie file.')
    movmap, flagmap, edgemap, wcs, tranges, exptimes = read_image(movfilename)
    movmap[np.where(np.isinf(movmap))] = 0  # because it pops out with inf values... IDK
    movmap[np.where(movmap < 0)] = 0

    for lc in variables:
        source_ix = lc['obj_id']
        print(f'Processing {source_ix}')
        curve = {band:{'t':np.arange(len(lc['cps'])),
                        'cps':lc['cps'],
                        'cps_err':lc['cps_err']}}
        min_i, max_i = np.argmin(curve[band]['cps']), np.argmax(curve[band]['cps'])

        assert len(lc['cps']) == np.shape(movmap)[0]  # if these don't match then the gif will be out of sync
        imgx, imgy = lc['xcenter'], lc['ycenter']  # the image pixel coordinate of the source
        # define the bounding box for the thumbnail
        imsz = np.shape(movmap[0])

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
        # this was fixed in the pipeline that generated the final run of gfcat data.
        x1_, x2_, y1_, y2_ = (max(int(imsz[0] / 2 - imsz[0] / 2), 0),
                              min(int(imsz[0] / 2 + imsz[0] / 2), imsz[0]),
                              max(int(imsz[1] / 2 - imsz[0] / 2), 0),
                              min(int(imsz[1] / 2 + imsz[0] / 2), imsz[1]))

        gs = gridspec.GridSpec(nrows=4, ncols=6)  # , height_ratios=[1, 1, 2])

        print('Generating frames.')
        for i, frame in enumerate(movmap):  # probably eliminate the first / last frame, which always has lower exposure
            fig = plt.figure(figsize=(12, 9));
            fig.tight_layout()
            ax = fig.add_subplot(gs[:3, :3])
            opacity = (edgemap[i] + flagmap[i]) / 2
            # M, N, 3 or M, N, 4
            ax.imshow(edgemap[i][x1_:x2_, y1_:y2_], origin="lower", cmap="Reds", alpha=opacity[x1_:x2_, y1_:y2_])
            ax.imshow(flagmap[i][x1_:x2_, y1_:y2_], origin="lower", cmap="Blues", alpha=opacity[x1_:x2_, y1_:y2_])
            ax.imshow(np.stack([ZScaleInterval()(frame[x1_:x2_, y1_:y2_]),
                                ZScaleInterval()(frame[x1_:x2_, y1_:y2_]),
                                ZScaleInterval()(frame[x1_:x2_, y1_:y2_]),
                                1 - opacity[x1_:x2_, y1_:y2_]], axis=2), origin="lower")
            ax.set_xticks([])
            ax.set_yticks([])
            rect = Rectangle((y1 - y1_, x1 - x1_), 2 * boxsz, 2 * boxsz, linewidth=1, edgecolor='y', facecolor='none',
                             ls='solid')
            ax.add_patch(rect)

            ax = fig.add_subplot(gs[:3, 3:])
            ax.imshow(edgemap[i][x1:x2, y1:y2], origin="lower", cmap="Reds", alpha=opacity[x1:x2, y1:y2])
            ax.imshow(flagmap[i][x1:x2, y1:y2], origin="lower", cmap="Blues", alpha=opacity[x1:x2, y1:y2])
            ax.imshow(np.stack([ZScaleInterval()(frame[x1:x2, y1:y2]),
                                ZScaleInterval()(frame[x1:x2, y1:y2]),
                                ZScaleInterval()(frame[x1:x2, y1:y2]),
                                1 - opacity[x1:x2, y1:y2]], axis=2), origin="lower")
            ax.set_xticks([])
            ax.set_xticks([])
            ax.set_yticks([])
            circ = Circle((boxsz, boxsz), 20, linewidth=1, edgecolor='y', facecolor='none', ls='solid')
            ax.add_patch(circ)

            ax = fig.add_subplot(gs[3:, :])
            ax.set_xticks([])
            ax.vlines(curve[band]['t'][i], curve[band]['cps'][min_i] - 3 * curve[band]['cps_err'][min_i],
                      curve[band]['cps'][max_i] + 3 * curve[band]['cps_err'][max_i], ls='dotted')
            ax.scatter(curve[band]['t'][i], curve[band]['cps'][i], c='y', s=100, marker='o')
            ax.errorbar(curve[band]['t'], curve[band]['cps'],
                        yerr=curve[band]['cps_err'] * 3, fmt='k.-',label=band)
            plt.legend()

            plt.savefig(f'{edir}/{estring}-{b}-30s-{str(i).zfill(2)}-{str(source_ix).zfill(5)}.png', dpi=100)
            plt.close('all')

        print('Compiling movie.')
        n_frames = np.shape(movmap)[0]
        # write the animated gif
        gif_fn = f'{edir}/{estring}-{b}-30s-{str(source_ix).zfill(5)}.gif'
        print(f"writing {gif_fn}")
        with imageio.get_writer(gif_fn, mode='I', fps=6) as writer:
            for i in np.arange(n_frames):
                frame_fn = f'{edir}/{estring}-{b}-30s-{str(i).zfill(2)}-{str(source_ix).zfill(5)}.png'
                image = imageio.imread(frame_fn)
                writer.append_data(image)
                if cleanup:  # remove the png frames
                    os.remove(frame_fn)

working_directory = '/home/ubuntu/datadir/'
for eclipse in [41726]:
    varix = screen_eclipse(eclipse, photdir=working_directory)
    print(varix)
    if len(varix):
        %time make_qa_image(eclipse,varix,band='NUV', photdir=working_directory)
        try:
            %time make_qa_image(eclipse,varix,band='FUV', photdir=working_directory)
        except KeyError:
            pass


