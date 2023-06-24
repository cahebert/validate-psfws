#!/bin/bash -l

POOL=12 #number of cores on which to run threads
TIME=1:00:00
PART=kipac
MEM=8GB

dir='/home/groups/burchat/chebert/psfwsPaperSims/'

for i in {301..537}
do
psfSeed=$RANDOM
cat << EOF > randSim.sl 
#!/bin/bash
#SBATCH --job-name=actuallyrand-$i
#SBATCH --time=$TIME
#SBATCH -p $PART
#SBATCH -c $POOL
#SBATCH --mem-per-cpu=$MEM
#SBATCH --out=../slurmfiles/sims/actuallyRand-$i-%j.out
#SBATCH --err=../slurmfiles/sims/actuallyRand-$i-%j.err

source ~/load_modules.sh

python3 simActuallyRandomPSFs.py --outfile=actuallyRandom_$i.p --outdir=$dir --psfSeed=$psfSeed --nPool=$POOL
EOF

# Now submit the batch job
sbatch randSim.sl
done
