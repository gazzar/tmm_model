import appsettings
import version
import os
from appdirs import AppDirs
import textwrap
import logging


def update(args):
    globals().update(args)


def set_logger(verbose):
    logging.basicConfig(level=logging.WARNING, format="%(msg)s")
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


description = textwrap.dedent("""\
    {filename} -
    A commandline tool to generate a sinogram from a tiff or tiffs.
    Examples:
    python {filename} -a angles.txt path/golosio*.tiff
    python {filename} -t f -e 12.0 -a angles.txt path/golosio*.tiff
    """.format(filename=__file__))


def open_first(filelist):
    for f in filelist:
        if os.path.isfile(f):
            return f
    return None


def parse():
    dirs = AppDirs(version.__name__)
    configfile = 'config.yaml'
    config_filename = open_first([
        os.path.join(os.path.dirname(os.path.realpath(__file__)), configfile),
        os.path.join(dirs.user_config_dir, configfile),
        os.path.join(__file__, configfile),
        ])
    try:
        parser = appsettings.SettingsParser(
            yaml_file=open(config_filename),
            version = version.__version__,
            description = description,
        )
    except IOError:
        parser = appsettings.SettingsParser(
            version = version.__version__,
            description = description,
        )

    parser.add_argument('--filepattern', action='store',
                        help=textwrap.dedent('''e.g. a-*.tiff reads
                                              a-Ca.tiff, a-Zn.tiff, etc.'''))
    parser.add_argument('--yamlfile', action='store',
                        help='yamlfile e.g. golosio.yaml')
    parser.add_argument('-t', '--event_type', action='store', default='r',
                        choices=['a','f','r','c'],
                        help=textwrap.dedent(
                        '''event_type {a=absorption, f=fluoro,
                        r=rayleigh(default), c=compton}'''))
    parser.add_argument('-a', '--anglesfile', action='store',
                        default='angles.txt',
                        help=textwrap.dedent('''filename of textfile containing
                                              list of projection angles, e.g.
                                              angles.txt'''))
    parser.add_argument('-s', '--scale', type=float, default=10.0,
                        help='scale (um/px) (default 10.0)')
    parser.add_argument('-e', '--energy', type=float, default=15.0,
                        help='energy (keV) (default 15.0)')
    parser.add_argument('--no-in-absorption',
                        default=False, action='store_true',
                        help='disables incident absorption')
    parser.add_argument('--no-out-absorption',
                        default=False, action='store_true',
                        help='disables outgoing absorption')
    parser.add_argument('--verbose',
                        default=False, action='store_true',
                        help='status update verbosity')
    parser.add_argument('--show_progress',
                        default=False, action='store_true',
                        help='status update verbosity')

    argument_namespace = parser.parse_args()
    update(vars(argument_namespace))
    set_logger(argument_namespace.verbose)
