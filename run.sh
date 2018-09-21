# model
 # gcn experimental setup (Kipf)
 # gat experimental setup (Veličković)
 # with bias
 # with left dropout
 # with right dropout
 # with left and right droupout

# gct-16-1
python runner.py -m gct -hu 16 -n 1 1 -b 0 -a relu -ia None -rd 0.0 -ld 0.0 -s softmax -p 10 -ub 0
python runner.py -m gct -hu 16 -n 1 1 -b 1 -a relu -ia None -rd 0.5 -ld 0.5 -s softmax -p 100 -ub 1
# gct-8-8
python runner.py -m gct -hu 8 -n 8 1 -b 0 -a relu -ia None -rd 0.5 -ld 0.5 -s softmax -p 10 -ub 1
python runner.py -m gct -hu 8 -n 8 1 -b 1 -a relu -ia None -rd 0.5 -ld 0.5 -s softmax -p 100 -ub 1
python runner.py -m gct -hu 8 -n 8 1 -b 1 -a relu -ia relu -rd 0.5 -ld 0.5 -s softmax -p 100 -ub 1
python runner.py -m gct -hu 8 -n 8 1 -b 1 -a elu -ia None -rd 0.5 -ld 0.5 -s softmax -p 100 -ub 1
python runner.py -m gct -hu 8 -n 8 1 -b 1 -a elu -ia elu -rd 0.5 -ld 0.5 -s softmax -p 100 -ub 1

# gat-16-1
python runner.py -m gat -hu 16 -n 1 1 -b 0 -a relu -ia None -rd 0.0 -ld 0.0 -s None -p 10 -ub 0
python runner.py -m gat -hu 16 -n 1 1 -b 1 -a relu -ia None -rd 0.5 -ld 0.5 -s None -p 100 -ub 1
# gat-8-8
python runner.py -m gat -hu 8 -n 8 1 -b 0 -a relu -ia None -rd 0.5 -ld 0.5 -s None -p 10 -ub 1
python runner.py -m gat -hu 8 -n 8 1 -b 1 -a relu -ia None -rd 0.5 -ld 0.5 -s None -p 100 -ub 1

# gcn-16-1
python runner.py -m gcn -hu 16 -n 1 1 -b 0 -a relu -ia None -rd 0.0 -ld 0.0 -s None -p 10 -ub 0
python runner.py -m gcn -hu 16 -n 1 1 -b 1 -a relu -ia None -rd 0.0 -ld 0.0 -s None -p 100 -ub 0
python runner.py -m gcn -hu 16 -n 1 1 -b 1 -a relu -ia None -rd 0.0 -ld 0.0 -s None -p 100 -ub 1
python runner.py -m gcn -hu 16 -n 1 1 -b 1 -a relu -ia None -rd 0.0 -ld 0.5 -s None -p 100 -ub 1
python runner.py -m gcn -hu 16 -n 1 1 -b 1 -a relu -ia None -rd 0.5 -ld 0.0 -s None -p 100 -ub 1
python runner.py -m gcn -hu 16 -n 1 1 -b 1 -a relu -ia None -rd 0.5 -ld 0.5 -s None -p 100 -ub 1
python runner.py -m gcn -hu 16 -n 1 1 -b 1 -a elu -ia None -rd 0.5 -ld 0.5 -s None -p 100 -ub 1
# gcn-8-8
python runner.py -m gcn -hu 8 -n 8 1 -b 0 -a relu -ia None -rd 0.5 -ld 0.5 -s None -p 10 -ub 1
python runner.py -m gcn -hu 8 -n 8 1 -b 1 -a relu -ia None -rd 0.5 -ld 0.5 -s None -p 100 -ub 1
