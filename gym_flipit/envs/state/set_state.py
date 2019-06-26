def opp_LM(p0_LM,p1_moved,found_FM,tick,state):
    if p0_LM == 0 or (not found_FM and not p1_moved):
        state = -1
    elif p1_moved: # p1 played this turn
        state = tick - p0_LM
    else: # p1 did not play this turn
        state += 1
    return state

def own_LM(p1_moved,state):
    if p1_moved: # p1 played this turn
        state = 0
    else: # p1 did not play this turn
        state += 1
    return state

def composite(p0_LM,p1_moved,found_FM,tick,state):
    return (own_LM(p1_moved,state[0]), opp_LM(p0_LM,p1_moved,found_FM,tick,state[1]))
