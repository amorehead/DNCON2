#!/bin/bash
####################### Batch Headers #########################
#SBATCH -p Lewis
#SBATCH -J cnn-train
#SBATCH -t 0-02:00
#SBATCH --partition gpu3
#SBATCH --gres gpu:1
#SBATCH --ntasks-per-node 4
#SBATCH --mem 120G
#SBATCH --nodes 1
###############################################################

# Local project path #
# export my_local=/home/"$USER"/Repositories/lab_repos/DNCON2

# Remote project path #
export my_local=/home/"$USER"/data/DNCON2

module load miniconda3
source "$my_local"/venv/bin/activate

python "$my_local"/scripts/cnn-train.py "$my_local"/model-config-n-weights "stage1-60A.hdf5" "60A" "$my_local"/databases/DNCON2/features/X-1EE6-A.txt "$my_local"/test-dncon2/X-1EE6-A.rr "$my_local"/test-dncon2/feat-stg2.txt 500 200
