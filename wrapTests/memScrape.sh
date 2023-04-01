#!/usr/bin

rm summary.txt

for TEST in full split wrap
do
echo $TEST | cat >> summary.txt
for SEED in {23413..23462}
do

jobid=$(ls ../slurmfiles/*.out | grep $SEED | grep $TEST | cut -f 4 -d '-' | cut -f 1 -d '.')
memused=$(seff $jobid | grep "Memory Utilized" | cut -f 3 -d ' ')
walltime=$(seff $jobid | grep "Job Wall-clock time" | cut -f 4 -d ' ')

echo $jobid" "$SEED" "$memused" "$walltime | cat >> summary.txt

done
done
