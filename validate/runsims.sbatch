#!/bin/bash -l
#SBATCH --partition=kipac
#SBATCH -t 2:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mydo@stanford.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=6G
#SBATCH --output=submit.out
#SBATCH --array=6-10,22,23,25,26,27,30-39,42-170

source ~/setup.sh
conda activate wfsim

#python simtest.py --atmSeed $SLURM_ARRAY_TASK_ID --outdir sameheightPsfws --outfile outh_psfws_$SLURM_ARRAY_TASK_ID.pkl --usePsfws


python matchProfileSim.py --atmSeed $SLURM_ARRAY_TASK_ID --outdir matchProfileRand --outfile outp_rand_$SLURM_ARRAY_TASK_ID.pkl --useRand
