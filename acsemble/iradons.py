import numpy as np
import skimage.transform as st
import os
import subprocess
import config
import imageio
import fnmatch
import helpers

UM_PER_CM = 1e4


def sart_n(im, n=1, imr=None, theta=None, **kwargs):
    """Perform n iterations of sart reconstruction

    Parameters
    ----------
    im : 2d ndarray of floats
        input sinogram
    n : int
        number of sart iterations to perform
    imr : 2d ndarray of floats
        initial image estimate
    theta : array-like iterable of projection angles
        If None, infer n angles are from 0 to 180 (non-inclusive), where n is the number of
        sinogram columns in im

    Returns
    -------
    2d ndarray of floats

    """
    for i in range(n):
        imr = st.iradon_sart(im, theta=theta, image=imr)
    return imr


def iradon(sino, angles):
    """Wrapper for potentially a bunch of different backprojector
    implementations, wrapped with a common call signature. Selection of
    the implementation and specific parameters for individual implementations
    are pulled from the config file.

    Parameters
    ----------
    sino : 2d ndarray of floats
        sinogram to backproject
    angles : 1d array-like
        sinogram projection angles

    Returns
    -------
    2d ndarray of float32's

    """
    implementation = config.iradon_implementation

    if implementation == 'skimage_iradon':
        im = st.iradon(sino, theta=angles,
                       circle=config.skimage_iradon_circle,
                       filter=config.skimage_iradon_filter)

    elif implementation == 'skimage_iradon_sart':
        # Clip solution to positive values. MLEM requires this.
        im = st.iradon_sart(sino, theta=angles,
                            clip=(1e-6, sino.max()*sino.shape[0]*sino.shape[1]))

    elif implementation in ('xlict_recon_mpi_fbp', 'xlict_recon_gridrec'):
        outdir = config.xlict_recon_mpi_fbp_PATH_OUTPUT_DATA     # output directory
        indir = config.xlict_recon_mpi_fbp_PATH_INPUT_TIFFS     # input directory
        base_filename = 'temp_sino.tif'

        def xtract_exe_string(filename, indir, outdir, angles, ps):
            assert config.xlict_recon_mpi_fbp_filter in {'ramp', 'shepp-logan', 'cosine',
                                                         'hamming', 'hann', 'none'}
            # lookup dict for reconstruction filter type and filter enable flag
            filter_control = {
                'ramp' : (0, 1),
                'shepp-logan' : (1, 1),
                'cosine' : (2, 1),
                'hamming' : (3, 1),
                'hann' : (4, 1),
                'none' : (0, 0),
            }

            recon_filter, filter_enable = filter_control[config.xlict_recon_mpi_fbp_filter]
            rm = {'xlict_recon_mpi_fbp':0, 'xlict_recon_gridrec':1}[implementation]
            xtract_options = \
                r'-id {indir} --sino {prg} -od {outdir} -pfct r_.tif -rmu 1 ' \
                r'-e {e_keV} --recon_method {rm} -fr {recon_filter} ' \
                r'-as {astep} --recon_filter_enabled {filter_enable} ' \
                r'-ps {ps} -corm {corm} --force_cpu {force_cpu}'.format(
                    indir = indir,
                    prg = filename,                     # read file name
                    outdir = outdir,
                    e_keV = config.energy_keV,
                    rm = rm,                            # 0 (FBP), 1 (Gridrec)
                    recon_filter = recon_filter,
                    filter_enable = filter_enable,
                    astep = angles[1] - angles[0],      # Angle step (deg)
                    ps = ps,                            # pixel size (um)
                    force_cpu = config.xlict_force_cpu,
                    corm = config.xlict_recon_mpi_corm, # centre-of-rot'n offset
                )
            return xtract_options

        sino = sino.T                   # orient sinogram axes to what X-TRACT expects
        # write to disk for XTRACT
        filename = os.path.join(indir, base_filename)
        imageio.imsave(filename, sino.astype(np.float32))

        # reconstruct (XLI/XTRACT writes the result to disk), then read result from disk
        xes = xtract_exe_string(base_filename, indir, outdir, angles, ps=1)
        PATH_XTRACT_EXE = config.xlict_recon_mpi_exe
        with open(os.devnull, 'w') as fnull:
            _ = subprocess.call('{} {}'.format(PATH_XTRACT_EXE, xes),
                                       stdout=fnull, stderr=subprocess.STDOUT)

        im = imageio.imread(os.path.join(outdir, 'r_0.tif')).astype(np.float32)
        # Reorient the reconstruction to match the convention established by other
        # reconstructors here.
        im = np.rot90(im)

        # Apply a hard circular mask to the result, to match the behaviour of skimage
        im = helpers.zero_outside_circle(im)

        # write im to the output path with name r_<nnn+1>.tif,
        # where nnn is the highest number in the existing files
        files = os.listdir(outdir)
        if 'r_0.tif' in files:
            matches = sorted(fnmatch.filter(files, 'r_[0-9][0-9][0-9].tif'))
            if not matches:
                dest = 'r_000.tif'
            else:
                # files exist matching r_nnn.tif; get the highest and add 1
                nnn = int(matches[-1][2:5])
                dest = 'r_%03d.tif' % (nnn+1)
            imageio.imsave(os.path.join(outdir, dest), im.astype(np.float32))

        im *= 5e-5  # This factor was determined by direct comparison with iradon above

    return im


if __name__ == "__main__":
    angles = np.linspace(0, 360, 180, endpoint=False)
    side = 5
    m = np.zeros((5, 5))
    m[2, 2] = 1.0
    p = st.radon(m, angles, circle=True)
    config.iradon_implementation = 'skimage_iradon'
    im1 = iradon(p, angles)
    config.iradon_implementation = 'xlict_recon_mpi_fbp'
    # config.iradon_implementation = 'xlict_recon_gridrec'
    im2 = iradon(p, angles)
    print im1
    print im2
    print 'scale_factor =', im1/im2
