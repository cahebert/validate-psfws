#!/bin/bash -l

POOL=12 #number of cores on which to run threads
TIME=1:00:00
PART=kipac
MEM=4GB

for kind in actualRand psfws rand match
do
cat << EOF > sbatch_get_summary.sl 
#!/bin/bash
#SBATCH --job-name=$kind-sum-%j
#SBATCH --time=$TIME
#SBATCH -p $PART
#SBATCH -c $POOL
#SBATCH --mem-per-cpu=$MEM
#SBATCH --out=../slurmfiles/$kind-sum-%j.out
#SBATCH --err=../slurmfiles/$kind-sum-%j.err

source ~/load_modules.sh

python3 polarSumStats.py $kind --nPool=$POOL
python3 getAtmSummaries.py $kind --nPool=$POOL
EOF

# Now submit the batch job
sbatch sbatch_get_summary.sl
done
