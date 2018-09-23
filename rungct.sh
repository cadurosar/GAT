#!/bin/bash

for s in softmax sum; do
  for ia in None relu; do
    for id in 0.0 0.5; do
      for rd in 0.0 0.5; do
        for ld in 0.0 0.5; do
          # 16x1 and 8x8
          python runner.py -m gct -hu 16 -n 1 1 -b true -a relu -ia $ia -id $id -rd $rd -ld $ld -s $s -p 100 -ub true
          python runner.py -m gct -hu 8 -n 8 1 -b true -a relu -ia $ia -id $id -rd $rd -ld $ld -s $s -p 100 -ub true
        done
      done
    done
  done
done
