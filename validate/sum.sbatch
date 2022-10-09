#!/bin/bash -l
#SBATCH --partition=kipac
#SBATCH -t 2:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mydo@stanford.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6G
#SBATCH --output=sum.out

source ~/setup.sh
conda activate wfsim

python summaryStats.py --kind "psfws"
python summaryStats.py --kind "rand"
python summaryStats.py --kind "randMatch"
