import os
import sys
PATH_HERE = os.path.abspath(os.path.dirname(__file__))
sys.path = [
    os.path.join(PATH_HERE, '..'),
    os.path.join(PATH_HERE, '..', '..'),    # include path to version.py
    ] + sys.path
