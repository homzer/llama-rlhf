#!/bin/bash

FILE_PATH=$1
if [ -f "$FILE_PATH" ]; then
  export _MASTER_ADDR=$(hostname -I 2>/dev/null | awk '{print $1}')
  MPI_ARGS="--bind-to none --map-by slot --hostfile $FILE_PATH --mca btl_tcp_if_include eth0 --mca oob_tcp_if_include eth0"
else
  export _MASTER_ADDR="127.0.0.1"
  MPI_ARGS="--bind-to none --map-by slot --np 1"
fi

RUN_SHELL=$2
mpirun -v --allow-run-as-root \
  $MPI_ARGS \
  --mca plm_rsh_agent "ssh -p 22" \
  -x PATH -x _MASTER_ADDR \
  bash $RUN_SHELL