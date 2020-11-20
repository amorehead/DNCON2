#!/bin/bash
####################### Batch Headers #########################
#SBATCH -p Lewis
#SBATCH -J cnn-train
#SBATCH -t 0-02:00
#SBATCH -p gpu3
#SBATCH --gres gpu:1
#SBATCH --mem 16
#SBATCH -N 1
#SBATCH -n 4
###############################################################

# Local project path #
#export MYLOCAL=/home/$USER/Repositories/lab_repos/DNCON2

# Remote project path #
export MYLOCAL=/home/$USER/data/DNCON2

module load miniconda3
source $MYLOCAL/venv/bin/activate

python $MYLOCAL/scripts/cnn-train.sh
