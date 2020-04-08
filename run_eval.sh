#!/bin/bash
#SBATCH --job-name=pless
#SBATCH --gres=gpu
#SBATCH --partition=gpu,gpu2,gpu6
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

#################
# configuration #
#################

# overwrite home directory for clean environment
export HOME="/srv/tmp/${USER}" # SSD on your workstation

# miniconda base dir
_conda_base_dir="${HOME}"

# conda python environment to use
_conda_env="cuda10.0_p3.7"

# python version to use
_conda_python_version="3.7"

# python packages to install:
# conda packages
_conda_install_packages="numpy matplotlib pillow=6.2.1 tqdm pyyaml cudatoolkit=10.0 cudnn pytorch torchvision"
# conda packages from a conda-channel (list of <channel>:<package>)
#_conda_channel_install_packages="conda-forge:matplotlib2tikz anaconda:cudatoolkit=10.0 anaconda:cudnn anaconda:tensorflow-gpu anaconda:cupti"
#_conda_channel_install_packages="pytorch:pytorch=1.3.1 pytorch:torchvision=0.4.2 pytorch:torchaudio"

########################
# code for environment #
########################

# make shure ${HOME} exists
mkdir -p ${HOME} || exit 1

# define custom environment
# minimal $PATH for home
export PATH=${_local_cuda_bin_path}:${_conda_base_dir}/miniconda${_conda_python_version:0:1}/bin:${HOME}/bin:/usr/local/bin:/usr/bin:/bin:/usr/local/games:/usr/games
  
# python
# install miniconda
if [ ! -d ${_conda_base_dir}/miniconda${_conda_python_version:0:1} ]; then
  if [ ! -f Miniconda2-latest-Linux-x86_64.sh ]; then
    wget https://repo.continuum.io/miniconda/Miniconda${_conda_python_version:0:1}-latest-Linux-x86_64.sh
  fi
  chmod +x ./Miniconda${_conda_python_version:0:1}-latest-Linux-x86_64.sh
  ./Miniconda${_conda_python_version:0:1}-latest-Linux-x86_64.sh -b -f -p ${_conda_base_dir}/miniconda${_conda_python_version:0:1} || { echo "ERROR during installing Miniconda" ;exit 1; }
  rm ./Miniconda${_conda_python_version:0:1}-latest-Linux-x86_64.sh
  INSTALL=${INSTALL:-true}
  if [ ! -d ${_conda_base_dir}/miniconda${_conda_python_version:0:1} ]; then
    echo "ERROR: \"${_conda_base_dir}/miniconda${_conda_python_version:0:1}\" does not exist!"
    echo "       Ths means there was a problem while installing Miniconda. Maybe too little disk space?"
    exit 1
  fi
fi
# setup virtual environment
if [ ! -d "${_conda_base_dir}/miniconda${_conda_python_version:0:1}/envs/${_conda_env}" ]; then
  conda create --yes -q -n ${_conda_env} python=${_conda_python_version}
  INSTALL=${INSTALL:-true}
fi
# activate environment
source activate ${_conda_env}
if [ "${CONDA_DEFAULT_ENV}" != "${_conda_env}" ]; then
  echo "ERROR: unable to activate conda environment \"${_conda_env}\""
  exit 1
fi
# ensure right python version
# ${_conda_python_version} matches python version installed
# for example _conda_python_version=3.5 and python 3.5.3 installed will be ok
_python_ver_installed=$(python --version 2>&1 | awk '{print $2}')
[[ ${_python_ver_installed} =~ ^${_conda_python_version}.*$ ]] || { \
    echo "python version ${_python_ver_installed} installed but ${_conda_python_version} expected"
    echo "manual change required..."
    exit 1
}
# ensure all packages are installed
if [ -n "${INSTALL}" ]; then
  conda install --yes ${_conda_install_packages}
  for _conda_channel_package in ${_conda_channel_install_packages}; do
    conda install --yes -c ${_conda_channel_package%:*} ${_conda_channel_package#*:}
  done
  for _pip_package in ${_pip_install_packages}; do
    pip install --exists-action=i ${_pip_package}
  done
  for _pip_package in ${_pip_install_whl}; do
    pip install --exists-action=i ${_pip_package}
  done
fi

# print config
echo -e "\n\nconfig:\n"
echo "HOME=${HOME}"
echo "PATH=${PATH}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "HOSTNAME=${HOSTNAME}"

####################
# START: user code #
####################

echo -e "\n\nRUN: python eval.py -d cifar10\n"
python eval.py -d cifar10

##################
# END: user code #
##################
