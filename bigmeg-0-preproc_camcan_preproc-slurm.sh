#!/bin/bash

#SBATCH --account quinna-camcan
#SBATCH --ntasks 1
#SBATCH --time 10:0
#SBATCH --mem 25G
#SBATCH --qos bbdefault
#SBATCH --array=1-643
#SBATCH -o slurm_logs/slurm-%A_%a.out
#SBATCH --constraint=sapphire

# 643
## ---------------------------------------------------------

module purge; module load bluebear
module load bear-apps/2023a
module load uv/0.6.5

#module load Python/3.11.3-GCCcore-12.3.0
#module load Tkinter/3.11.3-GCCcore-12.3.0
#module load IPython/8.14.0-GCCcore-12.3.0

#export VENV_DIR="${HOME}/virtual-environments"
#export VENV_PATH="${VENV_DIR}/osl-gammameg-${BB_CPU}"

# Create master dir if necessary
#mkdir -p ${VENV_DIR}
#echo ${VENV_PATH}

# Activate virtual environment
#source ${VENV_PATH}/bin/activate

# Check if virtual environment exists and create it if not
#if [[ ! -d ${VENV_PATH} ]]; 
#then
#	python3 -m venv --system-site-packages ${VENV_PATH}

#    # Activate virtual environment
#    source ${VENV_PATH}/bin/activate

#    # Any additional installations
#    #pip install --upgrade pip
#    pip install glmtools
#    pip install sails
#    pip install meegkit
#    pip install ${HOME}/src/oslpy
#    #pip install --upgrade numpy<=1.23
#else
#    # if venv aleady exists - just activate it
#    source ${VENV_PATH}/bin/activate
#fi

#pip install openpyxl
#pip install --upgrade mne==1.8.0

## ---------------------------------------------------------

cd /rds/homes/q/quinna/code/glm_bigmeg
source .venv/bin/activate

python bigmeg-0-preproc_camcan_preproc-slurm.py ${SLURM_ARRAY_TASK_ID}