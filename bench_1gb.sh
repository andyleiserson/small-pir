#!/bin/bash

set -exo pipefail

grep -m 1 '^model name[[:space:]]*:' /proc/cpuinfo
grep -m 1 '^cpu MHz[[:space:]]*:' /proc/cpuinfo

git rev-parse --short HEAD

configs=(
    final_1gb_q30_q_cost
    final_1gb_q30_query_sz
    final_1gb_q30_t_cost
    final_1gb_q30_t_comm
    final_1gb_q30_compute
    final_1gb_q56_q_cost
    final_1gb_q56_t_cost
    final_1gb_q56_compute
    final_1gb_q56_a
)
#    final_1gb_q56_query_sz
#    final_1gb_q56_t_comm

for config in ${configs[@]}; do
    cargo test --release -- --nocapture --include-ignored $config >& $config.out
done
