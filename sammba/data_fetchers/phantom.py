import numpy as np
import os
from sklearn.datasets.base import Bunch
from nilearn.datasets.utils import _get_dataset_dir, _fetch_file
from .utils import _get_dataset_descr


def fetch_phantom_dicoms_cyceron(files=range(11), data_dir=None, url=None,
                                 resume=True, verbose=1):
    """Download and load phantom DICOM images from CYCERON plateform (2018)

    Parameters
    ----------
    files : sequence of int or None, optional
        ids of files to load, default to loading all DICOM files.

    data_dir : str, optional
        Path of the data directory. Use to forec data storage in a non-
        standard location. Default: None (meaning: default)

    url: string, optional
        Download URL of the dataset. Overwrite the default URL.

    resume : bool
        whether to resumed download of a partly-downloaded file.

    verbose : int
        verbosity level (0 means no message).

    Returns
    -------
    data: sklearn.datasets.base.Bunch
        dictionary-like object, contains:

        - 'dwi' : str, path to dicom files.

        - 'multi_echo' : str, path to multi-echofiles.

        - 'description' : description about the data.

    Licence:
    -------
    CC-BY 4.0 licence.
    Please attribute this work to the UMS3408, CYCERON http://www.cyceron.fr
    """
    base_url = 'https://gitlab.com/naveau/bruker2nifti_qa/blob/master/raw/'
    dwi_dirs = ['Cyceron_DWI/20170719_075627_Lego_1_1/1/pdata/1/',
                'Cyceron_DWI/20170719_075627_Lego_1_1/2/pdata/1',
                'Cyceron_DWI/20170719_075627_Lego_1_1/2/pdata/2',
                'Cyceron_DWI/20170719_075627_Lego_1_1/3/pdata/1'
                'Cyceron_DWI/20170719_075627_Lego_1_1/3/pdata/2']
    multi_echo_dirs = [
        os.path.join('Cyceron_MultiEcho/20170720_080545_Lego_1_2',
                     '{0}/pdata/1'.format(n)) for n in range(6)]
    dcm_files = [os.path.join(f, 'dicom/EnIm1.dcm')
                 for f in dwi_dirs + multi_echo_dirs]

    dcm_files = np.array(dcm_files)[files]

    dataset_name = 'cyceron_phantom'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    target_dcms = []
    for dcm_file in dcm_files:
        target_dcm = _fetch_file(
            data_dir,
            (dcm_file, base_url + dcm_file, {'move': dcm_file}),
            verbose=verbose)
        target_dcms.append(target_dcm)

    dwi = target_dcms[:5]
    multi_echo = target_dcms[5:]
    fdescr = _get_dataset_descr(dataset_name)

    params = dict(dwi=dwi, multi_echo=multi_echo, description=fdescr)

    return Bunch(**params)
