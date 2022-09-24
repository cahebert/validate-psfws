#!/bin/bash -l
#SBATCH --partition=kipac
#SBATCH -t 2:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mydo@stanford.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=6G
#SBATCH --output=submit.v%a.out
#SBATCH --array=6-10,22,23,25,26,27,30-39,42-53,55-60,62-89

source ~/setup.sh
conda activate wfsim

#python simtest.py --atmSeed $SLURM_ARRAY_TASK_ID --outdir sameheightPsfws --outfile outh_psfws_$SLURM_ARRAY_TASK_ID.pkl --usePsfws

python simtest.py --atmSeed $SLURM_ARRAY_TASK_ID --outdir matchSpeedRand --outfile outv_rand_$SLURM_ARRAY_TASK_ID.pkl --useRand
