#!/bin/bash

for net_path in $(find results/slurm_out -name net.config); do
    /usr/bin/python3 plot_net.py -i $net_path
done
