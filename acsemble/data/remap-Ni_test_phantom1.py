# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import imageio
import os, sys
import subprocess
import imageio
import glob
import skimage.transform as st


# Here 1 is background, 2 is matrix, 4 is Element 1, 8 is Element 2,
# so 6 is background+matrix
VALUE_MAP = {0:1, 1:2, 2:6}
INFILE = 'Ni_test_phantom1'

# Convert inkscape svg to png.
# See http://answers.launchpad.net/inkscape/+question/208228
# Also http://www.imagemagick.org/discourse-server/viewtopic.php?t=24433

# 72 dpi gives 100x100
dpi = 72*4
os.system('convert +antialias -density {dpi} {f}.svg {f}.png'.format(dpi=dpi, f=INFILE))

im = imageio.imread(INFILE+'.png')
im = im[..., 0]
im_out = np.zeros_like(im)

# squeeze everything into the range 0->#INPUT_LEVELS-1
print 'svg->png max grey value:', im.max()
im = np.round(im / (float(im.max()) / (len(VALUE_MAP)-1)))
im = im.astype(np.uint8)
print 'remapped value range: 0 -', im.max()

# Now, remap the 0->INPUT_LEVELS-1 to the desired binary map
for k,v in VALUE_MAP.iteritems():
    im_out[im==k] = v

imageio.imsave(INFILE+'.png', im_out)

BASE = r'R:\Science\XFM\GaryRuben\projects\TMM'
CMD_BASE = r'R:\Science\XFM\GaryRuben\git_repos\tmm_model\commands'
DATA_DIR = os.path.abspath(os.path.dirname(__file__))
os.chdir(DATA_DIR)
print "In dir", os.getcwd()

# Write out the phantom matrix and elemental maps
cmd = r'python {cmd} {in1} {in2}'.format(
    cmd = os.path.join(CMD_BASE, 'pngtotiff.py'),
    in1 = os.path.join(DATA_DIR, INFILE + '.png'),
    in2 = os.path.join(DATA_DIR, INFILE + '.yaml'),
)
print cmd
status = subprocess.call(cmd, shell=True)

ims = glob.glob(INFILE + '-*.tiff')
for im in ims:
    i = imageio.imread(im)
    i = st.downscale_local_mean(i, (4,4))
    imageio.imsave(im, i)