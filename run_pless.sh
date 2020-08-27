#!/bin/bash
#SBATCH --job-name=pless
#SBATCH --output=slurm_out/pless_sweep_wm1.25_%A_%a.out
#SBATCH --error=slurm_out/pless_sweep_wm1.25_%A_%a.err
#SBATCH --gres=gpu:GeForceGTX10606GB:1
#SBATCH --partition=gpu
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-5:1%2

PARRAY=(0 1 2 4 8 16)

#################
# configuration #
#################

# overwrite home directory for clean environment
export HOME="/clusterFS/home/student/${USER}"

# miniconda base dir
_conda_base_dir="${HOME}"

# conda python environment to use
_conda_env="cuda10.1_p3.8"

# python version to use
_conda_python_version="3.8"

# python packages to install:
# conda packages
_conda_install_packages="numpy matplotlib pillow tqdm pyyaml yaml"
# conda packages from a conda-channel (list of <channel>:<package>)
#_conda_channel_install_packages="conda-forge:matplotlib2tikz anaconda:cudatoolkit=10.0 anaconda:cudnn anaconda:tensorflow-gpu anaconda:cupti"
_conda_channel_install_packages="pytorch:pytorch pytorch:cudatoolkit=10.1 pytorch:torchvision pytorch:torchaudio"

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

echo -e "\n\nRUN: pless\n"
cd search
sleep $[ ( $RANDOM % 120 )  + 1 ]s

python imagenet_arch_search.py --path pless_sweep_wm1.25_$SLURM_ARRAY_TASK_ID --dataset speech_commands --init_lr 0.2 --train_batch_size 100 --test_batch_size 100 --target_hardware "flops" --flops_ref_value 20e6 --n_worker 4 --gpu 0 --arch_lr 4e-3 --grad_reg_loss_alpha 1 --grad_reg_loss_beta ${PARRAY[$SLURM_ARRAY_TASK_ID]} --weight_bits 8 --width_mult 1.25 
   
sleep 60s
python imagenet_run_exp.py --path pless_sweep_wm1.25_$SLURM_ARRAY_TASK_ID/learned_net --train --gpu 0

##################
# END: user code #
##################
