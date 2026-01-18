#!/bin/bash

# Sync code to the GPU machine
rsync -av ~/sandbox/inputctl/ sean-gallagher@intuition.local:/home/sean-gallagher/sandbox/inputctl/ --exclude dataset --exclude '*.venv' --exclude target --exclude node_modules --delete

# Sync datasets and checkpoints both ways

rsync -av ~/sandbox/inputctl/dataset/ sean-gallagher@intuition.local:/home/sean-gallagher/sandbox/inputctl/dataset/
rsync -av sean-gallagher@intuition.local:/home/sean-gallagher/sandbox/inputctl/dataset/ ~/sandbox/inputctl/dataset/