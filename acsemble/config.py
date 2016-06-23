from __future__ import absolute_import, division, print_function
import six
import os
import ruamel.yaml as yaml
from appdirs import AppDirs
import logging
import logging.config
import textwrap
from . import version

logger = logging.getLogger(__name__)


ENV_NAME = 'ACSEMBLE_CONFIG'
CONFIG_NAME = 'acsemble.yaml'
config_dict = {}


def config_init(local_config_file=None):
    global config_dict

    def _open_first_config(filelist):
        for f in filelist:
            try:
                if os.path.isfile(f):
                    return f
            except TypeError:
                continue
        raise OSError

    def _expand_env_var(env_var):
        """Expands (potentially nested) env vars by repeatedly applying
        `expandvars` and `expanduser` until interpolation stops having
        any effect.

        Args:
            env_var: str

        Returns:

        """
        if not env_var:
            return env_var
        while True:
            interpolated = os.path.expanduser(os.path.expandvars(str(env_var)))
            if interpolated == env_var:
                return interpolated
            else:
                env_var = interpolated

    # populate module namespace with default config settings
    config_updates = yaml.load(default())
    update_config(config_updates)

    # override default config settings if requested
    dirs = AppDirs(version.__name__)
    if ENV_NAME not in os.environ:
        try:
            config_filename = _open_first_config([
                os.path.join(os.getcwd(), CONFIG_NAME),
                os.path.join(dirs.user_config_dir, CONFIG_NAME),
            ])
            update_config_from_file(config_filename)
        except OSError:
            print('Using default config')
    else:
        update_config_from_file(_expand_env_var(os.environ[ENV_NAME]))
    # Apply commandline-specified config overrides
    if local_config_file is not None:
        try:
            update_config_from_file(local_config_file)
        except OSError:
            print('{} not found; using default config'.format(local_config_file))

    # Start the logger and recipy if enabled

    set_logger(verbose=True)
    import_recipy()


def default():
    dirs = AppDirs(version.__name__)

    default_yaml = textwrap.dedent(r"""
        %YAML 1.2
        ---
        yaml_config_version: 0.1

        # Specimen-specific settings
        # ---
        # Examples
        # energy_keV: 15.6
        # element: Zn
        # sinogram_density_scaling:
        #     Zn: 1.0e-9
        #     Mn: 1.0e-9
        # density_per_compton_count: 5.0e-2      # matrix density for Ni_test_phantom2
        # ---
        energy_keV: null
        element: null
        sinogram_density_scaling:
            null

        # For density_per_compton_count see work/20160222_1/Absorption_from_Ge_sinogram-rescaled_image.ipynb
        density_per_compton_count: null

        angles_compute: True
        angles_closed_range: False
        angles_max: 360.0

        # debug settings
        use_recipy: False
        verbose: False
        show_progress: True

        # forward model
        no_in_absorption: False
        no_out_absorption: False
        absorb_with_matrix_only: True

        # input model 200x200px of the simulated test phantom. See readme.txt in
        # R:/Science/XFM/GaryRuben/projects/TMM/work/20160405_1:
        map_pattern: 'R:/Science/XFM/GaryRuben/projects/TMM/work/20160503_1/Grain_test_phantom1-*.tiff'
        map_elements: 'R:/Science/XFM/GaryRuben/projects/TMM/work/20160422_1/Grain_test_phantom1.yaml'
        map_width_um: 4000.0

        # detector data
        detector_csv: 'R:/Science/XFM/GaryRuben/git_repos/tmm_model/acsemble/data/Maia_384C.csv'
        detector_pads: [1]

        # mlem
        save_f_images: True
        save_d_images: False
        save_g_images: False
        save_projection_images: True
        mlem_profile: False

        # ---
        # Examples
        # mlem_im_path: 'C:/temp/mlem_ims'
        # mlem_reference_image_path: 'R:/Science/XFM/GaryRuben/projects/TMM/work/20160502_3/Grain_test_phantom1-Zn.tiff'
        # mlem_mse_path: 'C:/temp/mlem_ims/mse.txt'
        # mlem_mssim_path: 'C:/temp/mlem_ims/mssim.txt'
        # ---
        mlem_im_path: null
        mlem_reference_image_path: null
        mlem_mse_path: null
        mlem_mssim_path: null
        mlem_save_similarity_metrics: True
        mlem_transpose_on_read: True
        matrix_blur: False

        # backprojector
        # one of 'skimage_iradon', 'skimage_iradon_sart', 'xlict_recon_mpi_fbp'
        iradon_implementation: 'xlict_recon_mpi_fbp'
        skimage_iradon_circle: True
        skimage_iradon_filter: null
        backprojector: 'fbp'    # one of 'fbp', 'sart_no_prior'

        # X-TRACT settings
        # ---
        # Examples
        # xlict_recon_mpi_exe: 'R:/Science/XFM/GaryRuben/software_to_keep/XLICTReconMPI/XLICTReconMPI.exe'
        # xlict_recon_mpi_fbp_PATH_INPUT_TIFFS: 'R:/Science/XFM/GaryRuben/projects/ricegrain_recon'
        # xlict_recon_mpi_fbp_PATH_OUTPUT_DATA: 'C:/Users/rub015/Desktop/ricegrain_out'
        # ---
        xlict_recon_mpi_exe: null
        xlict_recon_mpi_fbp_PATH_INPUT_TIFFS: null
        xlict_recon_mpi_fbp_PATH_OUTPUT_DATA: null
        xlict_recon_mpi_corm: 0.0
        xlict_force_cpu: 0
        # Filter type is one of 'ramp', 'shepp-logan', 'cosine', 'hamming', 'hann', 'none'
        xlict_recon_mpi_fbp_filter: 'none'
        xlict_recon_reference_fbp_filter: 'ramp'

        # logging
        logging_config:
            version: 1
            disable_existing_loggers: false
            formatters:
                standard:
                    datefmt: '%H:%M:%S'
                    format: '[%(asctime)s][%(levelname)s] %(name)s %(filename)s:%(funcName)s:%(lineno)d | %(message)s'
            handlers:
                console:
                    class: logging.StreamHandler
                    formatter: standard
                    level: INFO
                file:
                    class: logging.handlers.RotatingFileHandler
                    filename: {user_config_dir}/log.txt
                    formatter: standard
                    maxBytes: 1000000
                    backupCount: 100
                    level: INFO
            loggers:
                '':
                    handlers:
                        - console
                        - file
                    level: INFO
                    propagate: false
        """.format(user_config_dir=dirs.user_config_dir)
    )
    return default_yaml


def ensure_dir_exists(d):
    # http://stackoverflow.com/questions/273192/how-to-check-if-a-directory-exists-and-create-it-if-necessary
    if not os.path.exists(d):
        try:
            os.makedirs(d)
            print('Created log directory ' + d)
        except:
            print('Cannot create log directory ' + d)
            raise


def set_logger(verbose):
    global config_dict
    dirs = AppDirs(version.__name__)
    ensure_dir_exists(dirs.user_config_dir)

    logging.config.dictConfig(config_dict['logging_config'])
    print('logging to ' + config_dict['logging_config']['handlers']['file']['filename'])
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


def update_dict(dct, merge_dct):
    """Paul Durivage's recursive dict merge:
        https://gist.github.com/angstwad/bf22d1822c38a92ec0a9
    Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    Note, I also consulted these, which aren't sufficient.
    http://stackoverflow.com/q/3232943
    http://stackoverflow.com/q/7204805

    Args:
        dct: dict onto which the merge is executed
        merge_dct: dct merged into dct

    Returns:
        None

    """
    for k, v in six.iteritems(merge_dct):
        if k in dct and isinstance(dct[k], dict) and isinstance(merge_dct[k], dict):
            update_dict(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]


def update_config(config):
    global config_dict

    update_dict(config_dict, config)
    update_dict(globals(), config)


def update_config_from_file(filename):
    with open(filename, 'r') as config_file:
        config_updates = yaml.load(config_file.read())
        update_config(config_updates)
        print('Updating config from {}'.format(filename))


def write_config(filename):
    """ Dump a copy of the config to filename
    Arguments
    ---------
    filename - str

    """
    with open(filename, 'w') as f:
        yaml.dump(config_dict, f)


def import_recipy():
    if config_dict['use_recipy']:
        try:
            import recipy
        except ImportError:
            logger.info('Install recipy or change the use_recipy config setting to False')
