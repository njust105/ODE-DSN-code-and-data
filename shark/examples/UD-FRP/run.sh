#!/bin/bash

# output directory
mkdir -p Outputs
rm Outputs/* -rf
mkdir -p Outputs/log
INPUTFILE=micro.i

MAX_PROCESSES=4
# N: processors for each task
N=1
TOTOL_TASKS=800

start_process() {
    data_file="paths/$i.csv"
    if [ -f $data_file ]; then
        echo "Running with data_file = $data_file"
        end_time=$(awk -F',' 'END {print $1}' "$data_file")
        args="csvfile=$data_file Outputs/file_base=Outputs/$i Outputs/exodus=false Outputs/csv=true Outputs/color=false end_time=$end_time"
        mpiexec -n $N ../../shark-opt -i $INPUTFILE $args >>Outputs/log/$i.txt &
    fi
}

i=0

while true; do

    RUNNING_PROCESSES=$(pgrep -c "^shark-opt$")

    if [ $RUNNING_PROCESSES -lt $MAX_PROCESSES ]; then
        start_process
        i=$((i + 1))
    fi
    sleep 1
    if [ $i -ge $TOTOL_TASKS ]; then
        break
    fi
done

wait
