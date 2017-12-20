
#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.
#
# This script is adapted from a similar script from the scikit-learn repository.
#
# License: 3-clause BSD

set -e

# Fix the compilers to workaround avoid having the Python 3.4 build
# lookup for g++44 unexpectedly.
export CC=gcc
export CXX=g++

create_new_venv() {
    # At the time of writing numpy 1.9.1 is included in the travis
    # virtualenv but we want to be in control of the numpy version
    # we are using for example through apt-get install
    deactivate
    virtualenv --system-site-packages testvenv
    source testvenv/bin/activate
    pip install packaging
    pip install prov
    pip install nose
    pip install doctest-ignore-unicode
    sudo apt-get install graphviz
}

print_conda_requirements() {
    # Echo a conda requirement string for example
    # "pip nose python='.7.3 scikit-learn=*". It has a hardcoded
    # list of possible packages to install and looks at _VERSION
    # environment variables to know whether to install a given package and
    # if yes which version to install. For example:
    #   - for numpy, NUMPY_VERSION is used
    #   - for scikit-learn, SCIKIT_LEARN_VERSION is used
    TO_INSTALL_ALWAYS="pip nose libgfortran=1.0=0 nomkl"
    REQUIREMENTS="$TO_INSTALL_ALWAYS"
    TO_INSTALL_MAYBE="python numpy scipy matplotlib scikit-learn pandas configparser future traits simplejson networkx packaging funcsigs click"
    for PACKAGE in $TO_INSTALL_MAYBE; do
        # Capitalize package name and add _VERSION
        PACKAGE_VERSION_VARNAME="${PACKAGE^^}_VERSION"
        # replace - by _, needed for scikit-learn for example
        PACKAGE_VERSION_VARNAME="${PACKAGE_VERSION_VARNAME//-/_}"
        # dereference $PACKAGE_VERSION_VARNAME to figure out the
        # version to install
        PACKAGE_VERSION="${!PACKAGE_VERSION_VARNAME}"
        if [ -n "$PACKAGE_VERSION" ]; then
            REQUIREMENTS="$REQUIREMENTS $PACKAGE=$PACKAGE_VERSION"
        fi
    done
    echo $REQUIREMENTS
}

create_new_conda_env() {
    # Skip Travis related code on circle ci.
    if [ -z $CIRCLECI ]; then
        # Deactivate the travis-provided virtual environment and setup a
        # conda-based environment instead
        deactivate
    fi

    # Use the miniconda installer for faster download / install of conda
    # itself
    wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh \
        -O miniconda.sh
    chmod +x miniconda.sh && ./miniconda.sh -b
    export PATH=/home/travis/miniconda2/bin:$PATH
    echo $PATH
    conda update --quiet --yes conda

    # Configure the conda environment and put it in the path using the
    # provided versions
    REQUIREMENTS=$(print_conda_requirements)
    echo "conda requirements string: $REQUIREMENTS"
    conda create -n testenv --yes $REQUIREMENTS
    source activate testenv

    if [[ "$INSTALL_MKL" == "true" ]]; then
        # Make sure that MKL is used
        conda install --quiet --yes mkl
    elif [[ -z $CIRCLECI ]]; then
        # Travis doesn't use MKL but circle ci does for speeding up examples
        # generation in the html documentation.
        # Make sure that MKL is not used
        conda remove --yes --features mkl || echo "MKL not installed"
    else
        # Make sure that MKL is not used
        conda remove --yes --features mkl || echo "MKL not installed"
    fi
}

if [[ "$DISTRIB" == "neurodebian" ]]; then
    create_new_venv
    bash <(wget -q -O- http://neuro.debian.net/_files/neurodebian-travis.sh)
    sudo apt-get install -qq python-scipy python-nose python-nibabel\
         python-sklearn python-pandas python-nilearn python-patsy\
         python-networkx python-configparser python-future python-traits\
         python-simplejson python-funcsigs python-click graphviz afni
    source /etc/afni/afni.sh
    echo "AFNI plugin path $AFNI_PLUGINPATH."
    echo "AFNI binaries installed in $(which afni)"

elif [[ "$DISTRIB" == "conda" ]]; then
    create_new_conda_env
    # Note: nibabel is in setup.py install_requires so nibabel will
    # always be installed eventually. Defining NIBABEL_VERSION is only
    # useful if you happen to want a specific nibabel version rather
    # than the latest available one.
    if [ -n "$NIBABEL_VERSION" ]; then
        pip install nibabel=="$NIBABEL_VERSION"
    fi
    # Install the latest available version of nilearn and patsy
    pip install nilearn
    pip install patsy
    # dependencies that are only available through pip
    pip install prov
    # Allow nose to ignore unicode in doctest
    pip install doctest-ignore-unicode

else
    echo "Unrecognized distribution ($DISTRIB); cannot setup travis environment."
    exit 1
fi

if [[ "$COVERAGE" == "true" ]]; then
    pip install coverage coveralls
fi

python setup.py install
