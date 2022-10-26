import numpy as np
import math
signal_num = 0
one_bit_trans = 0
total_qi = 0
for idx, line in enumerate(open("input.dat", 'r')):
    item = line.rstrip()
    split_item = item.split()
    if idx == 0:
        signal_num = int(split_item[0])
        trans_time = np.zeros(signal_num)
        period_time = np.zeros(signal_num)
    elif idx == 1:
        one_bit_trans = float(split_item[0])
    else:
        trans_time[int(split_item[0])] = float(split_item[1])
        period_time[int(split_item[0])] = float(split_item[2])

for i in range(signal_num):
    block_time = np.max(trans_time[i:])
    high_priority_signal = trans_time[:i]
    LHS = block_time
    while 1:
        RHS = block_time
        for j in range(len(high_priority_signal)):
            RHS += math.ceil((one_bit_trans + LHS)/period_time[j])*high_priority_signal[j]
        if RHS == LHS:
            print("signal: %s response time: %s"%(i, (RHS + trans_time[i])))
            break
        elif RHS >= LHS:
            LHS = RHS
        else:
            print("error in message %s"% (i))
            break