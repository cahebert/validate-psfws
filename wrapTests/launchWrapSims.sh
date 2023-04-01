#!/bin/bash -l

POOL=12 #number of cores on which to run threads
TIME=3:00:00
PART=bigmem #kipac
MEM=24GB

for SEED in {23453..23462} #..23462}
do
for TEST in full #split wrap
do
cat << EOF > sbatch_run_sim.sl 
#!/bin/bash
#SBATCH --job-name=sim-$SEED-%j
#SBATCH --time=$TIME
#SBATCH -p $PART
#SBATCH -c $POOL
#SBATCH --mem-per-cpu=$MEM
#SBATCH --out=../slurmfiles/sim-$TEST-$SEED-%j.out
#SBATCH --err=../slurmfiles/sim-$TEST-$SEED-%j.err

source ~/load_modules.sh

python3 testWrapping.py --atmSeed=$SEED --nPool=$POOL --outfile=$TEST$SEED".p" --testtype=$TEST
EOF

# Now submit the batch job
sbatch sbatch_run_sim.sl
done
done
