#!/bin/bash

for i in {10..350..10}
do
    bash /raid/haoran/Project/PartDiffusion/PartDiffusion/SDFusion/scripts/test/test_ply2shape-128.sh hinge-ply2shape-plyrot-scale3-lr0.00001 0 200000 $i
done