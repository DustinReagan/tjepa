#!/bin/bash

while read line; do
    if [[ $line == *"/envs/"* ]]; then
        env_name=$(basename "$line")
        echo "Checking environment: $env_name"
        package_list=$(conda list -n $env_name | grep pytorch)
        if [[ $package_list == *"pytorch"* ]]; then
            echo "PyTorch installed in $env_name"
        else
            echo "PyTorch not found in $env_name"
        fi
    fi
done < <(conda env list)