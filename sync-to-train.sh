#!/bin/bash

# Sync datasets and checkpoints both ways

rsync -av ~/sandbox/inputctl/dataset/ sean-gallagher@intuition.local:/home/sean-gallagher/sandbox/inputctl/dataset/
rsync -av sean-gallagher@intuition.local:/home/sean-gallagher/sandbox/inputctl/dataset/ ~/sandbox/inputctl/dataset/