#!/bin/bash

# SERVERS=(\
#     agra.diamond.cs.cmu.edu \
#     briolette.diamond.cs.cmu.edu \
#     cullinan.diamond.cs.cmu.edu \
#     dresden.diamond.cs.cmu.edu \
#     indore.diamond.cs.cmu.edu \
#     kimberly.diamond.cs.cmu.edu \
#     patiala.diamond.cs.cmu.edu \
#     transvaal.diamond.cs.cmu.edu \
# )

SERVERS=($(cat $(dirname $0)/SERVER_LIST))
# echo $SERVERS

for i in ${!SERVERS[@]}; do 
    cookiecutter -s ${SERVERS[$i]} -u http://localhost:5873/proxy/$(($i+1))of${#SERVERS[@]}/localhost:5873/collection/VIRAT
done;
#  http://localhost:5873/proxy/2of8/localhost:5873/collection/VIRAT