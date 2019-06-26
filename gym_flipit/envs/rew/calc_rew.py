import math

def LM_benefit(found_FM,p1_moved,p1_cost,p0_LM,p1_LM,tick,p0_moves):
    if found_FM and p1_moved:
        gain = 0 if p0_LM < p1_LM else [x for x in p0_moves if x < tick and x >= p1_LM][0]
        return gain - p1_cost
    else:
        return 0

def exponential(found_FM,p1_moved,p1_cost,p0_LM,p1_moves,tick,configs):
    if not found_FM or not p1_moved:
        return 0
    elif p0_LM >= p1_moves[-2]: 
        tau = tick-p0_LM
        return math.exp((configs['b']-tau)/configs['a']) - p1_cost 
    else:
        return - p1_cost

def reciprocal(found_FM,p1_moved,p0_LM,p1_moves,p1_cost,tick):
    if not found_FM or not p1_moved:
        return 0
    elif p0_LM >= p1_moves[-2]:
        tau = tick-p0_LM
        return p1_cost*(1/tau)
    else:
        return - p1_cost

def constant_reciprocal(found_FM,p1_moved,p0_LM,p1_moves,p1_cost,upper_lim,tick):
    if not found_FM or not p1_moved:
        return 0
    elif p0_LM >= p1_moves[-2]:
        tau = tick-p0_LM
        return upper_lim*(1/tau)
    else:
        return - p1_cost

def constant(found_FM,p1_moved,p0_LM,p1_moves,p1_cost,val,tick):
    if not found_FM or not p1_moved:
        return 0
    elif p0_LM >= p1_moves[-2]:
        return val
    else:
        return -p1_cost

def constant_minus_cost(found_FM,p1_moved,p0_LM,p1_moves,p1_cost,val,tick):
    constant_minus_cost_norm(1)

def constant_minus_cost_norm(found_FM,p1_moved,p0_LM,p1_moves,p1_cost,val,tick,c):
    if not found_FM or not p1_moved:
        return 0
    elif p0_LM >= p1_moves[-2]:
        return (val-p1_cost)/c
    else:
        return -p1_cost

def exp_cost(found_FM,p1_moved,p0_LM,p1_moves,p1_cost,val,tick):
    if not found_FM or not p1_moved:
        return 0
    elif p0_LM >= p1_moves[-2]:
        return val
    else:
        return -p1_cost**1.3

def LM_avg(found_FM,p1_moved,p0_LM,p1_moves,p1_cost,val,tick):
    if not found_FM or not p1_moved:
        return 0
    elif p0_LM >= p1_moves[-2]:
        return val-(tick-p0_LM)-p1_cost
    else:
        return -p1_cost
