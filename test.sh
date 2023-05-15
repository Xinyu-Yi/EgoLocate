#!/bin/bash


for ((j=0; j<9; j+=1))
do
    for ((i=0; i<45; i+=1))
    do
        echo "----------------------------- TotalCapture seq $i - $j -----------------------------"
        timeout 240 python test_totalcapture.py --run $i $j
    done
done
