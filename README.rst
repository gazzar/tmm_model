acsemble
=========================================================

Author: Gary Ruben

Absorption corrected reconstruction for X-ray Fluorescence Tomography.
Currently this implements a forward model for tomography at the Australian
Synchrotron XFM beamline.

Dependencies
------------
`acsemble`_ is written in Python 2.7 and depends on numpy, scipy, pandas,
matplotlib, scikit-image, xraylib [1], as well as pure-Python
modules that are pip-installable from PyPI.

Installation
------------

Notes
-----
This version developed using Anaconda Python distribution and tested on
Windows 7 64-bit.
The file tifffile.py is included under the terms of the modified BSD license and
is Copyright (c) 2008-2014, Christoph Gohlke
It was obtained from http://www.lfd.uci.edu/~gohlke/ in January 2014

Notes:
This code uses xraylib [1]
[1] T. Schoonjans et al., Spectrochim. Acta B 66, 776 (2011),
    http://github.com/tschoonj/xraylib/wiki

Version History
---------------
0.1     Initial version
