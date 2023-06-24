#!/bin/bash -l

POOL=12 #number of cores on which to run threads
TIME=1:00:00
PART=kipac
MEM=8GB

dir='/home/groups/burchat/chebert/psfwsPaperSims/'

# kind=psfws
# while read p; do
# arr=( $p )
# i=${arr[0]}
# psfSeed=${arr[1]}
# atmSeed=${arr[2]}
# echo $i $psfSeed $atmSeed

for i in {301..537}
do
atmSeed=$RANDOM
psfSeed=$RANDOM
for kind in psfws rand match
do
cat << EOF > launchSim.sl 
#!/bin/bash
#SBATCH --job-name=$kind-$i
#SBATCH --time=$TIME
#SBATCH -p $PART
#SBATCH -c $POOL
#SBATCH --mem-per-cpu=$MEM
#SBATCH --out=../slurmfiles/sims/$kind-$i-%j.out
#SBATCH --err=../slurmfiles/sims/$kind-$i-%j.err

source ~/load_modules.sh

python3 wfsim.py --outfile=$kind$i.p --outdir=$dir --atmSeed=$atmSeed --psfSeed=$psfSeed --i=$i --kind=$kind
EOF

# Now submit the batch job
sbatch launchSim.sl
done
done  #<fail_atm_psf.txt
