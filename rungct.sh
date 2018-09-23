#!/bin/bash

# gct-16-1
for s in softmax sum; do
  for ia in None relu; do
  python runner.py -m gct -hu 16 -n 1 1 -b true -a relu -ia $ia -id 0.5 -rd 0.5 -ld 0.5 -s $s -p 100 -ub true -t $1
  done
done

# gct-8-8
for s in softmax sum; do
  for ia in None relu; do
  python runner.py -m gct -hu 8 -n 8 1 -b true -a relu -ia $ia -id 0.5 -rd 0.5 -ld 0.5 -s $s -p 100 -ub true -t $1
  done
done
