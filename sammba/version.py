# *- encoding: utf-8 -*-
"""
sammba version, required package versions, and utilities for checking
"""
# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
# X.Y
# X.Y.Z # For bugfix releases
#
# Admissible pre-release markers:
# X.YaN # Alpha release
# X.YbN # Beta release
# X.YrcN # Release Candidate
# X.Y # Final release
#
# Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
# 'X.Y.dev0' is the canonical version of 'X.Y.dev'
#
__version__ = '0.0.1'

_SAMMBA_INSTALL_MSG = 'See %s for installation information.' % (
    'http://sammba-mri.github.io/introduction.html#installation')

# This is a tuple to preserve order, so that dependencies are checked
#   in some meaningful order (more => less 'core').
REQUIRED_MODULE_METADATA = (
    ('numpy', {
        'min_version': '1.8.2',
        'required_at_installation': True,
        'install_info': _SAMMBA_INSTALL_MSG}),
    ('scipy', {
        'min_version': '0.17',
        'required_at_installation': True,
        'install_info': _SAMMBA_INSTALL_MSG}),
    ('nilearn', {
        'min_version': '0.2.0',
        'required_at_installation': True,
        'install_info': _SAMMBA_INSTALL_MSG}),
    ('nibabel', {
        'min_version': '1.2.0',
        'required_at_installation': False}),
    ('pandas', {
        'min_version': '0.13.0',
        'required_at_installation': True,
        'install_info': _SAMMBA_INSTALL_MSG}),
    ('patsy', {
        'min_version': '0.2.0',
        'required_at_installation': True,
        'install_info': _SAMMBA_INSTALL_MSG}),
    ('sklearn', {
        'min_version': '0.15.0',
        'required_at_installation': True,
        'install_info': _SAMMBA_INSTALL_MSG}),
    ('tqdm', {
        'min_version': '4.11.2',
        'required_at_installation': True,
        'install_info': _SAMMBA_INSTALL_MSG}),
    )
    
OPTIONAL_MATPLOTLIB_MIN_VERSION = '1.3.1'


def _import_module_with_version_check(
        module_name,
        minimum_version,
        install_info=None):
    """Check that module is installed with a recent enough version
    """
    from distutils.version import LooseVersion

    try:
        module = __import__(module_name)
    except ImportError as exc:
        user_friendly_info = ('Module "{0}" could not be found. {1}').format(
            module_name,
            install_info or 'Please install it properly to use sammba.')
        exc.args += (user_friendly_info,)
        raise

    # Avoid choking on modules with no __version__ attribute
    module_version = getattr(module, '__version__', '0.0.0')

    version_too_old = (not LooseVersion(module_version) >=
                       LooseVersion(minimum_version))

    if version_too_old:
        message = (
            'A {module_name} version of at least {minimum_version} '
            'is required to use sammba. {module_version} was found. '
            'Please upgrade {module_name}').format(
                module_name=module_name,
                minimum_version=minimum_version,
                module_version=module_version)

        raise ImportError(message)

    return module


def _check_module_dependencies(is_sammba_installing=False):
    """Throw an exception if sammba dependencies are not installed.
    Parameters
    ----------
    is_sammba_installing: boolean
        if True, only error on missing packages that cannot be auto-installed.
        if False, error on any missing package.
    Throws
    -------
    ImportError
    """

    for (module_name, module_metadata) in REQUIRED_MODULE_METADATA:
        if not (is_sammba_installing and
                not module_metadata['required_at_installation']):
            # Skip check only when installing and it's a module that
            # will be auto-installed.
            _import_module_with_version_check(
                module_name=module_name,
                minimum_version=module_metadata['min_version'],
install_info=module_metadata.get('install_info'))
