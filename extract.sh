#!/bin/bash

awk '{print $2,$3}' orca2.txt | sed 's/[^0-9.-]//g' > cores
awk '{print $8,$9}' orca2.txt | sed 's/[^0-9.-]//g' > percentage
paste cores percentage > data.txt
