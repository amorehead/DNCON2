#!/bin/bash
####################### Batch Headers #########################
#SBATCH -p Lewis
#SBATCH -J cnn-train
#SBATCH -t 0-02:00
#SBATCH -p gpu4
#SBATCH --gres gpu:1
#SBATCH --mem 16
#SBATCH -N 1
#SBATCH -n 4
###############################################################

# Local project path #
# export my_local=/home/"$USER"/Repositories/lab_repos/DNCON2

# Remote project path #
export my_local=/home/"$USER"/data/DNCON2

module load miniconda3
# source /home/alexm/Repositories/lab_repos/DNCON2/venv/bin/activate
source /home/acmwhb/data/DNCON2/venv/bin/activate

python3 "$my_local"/scripts/cnn-train.sh
