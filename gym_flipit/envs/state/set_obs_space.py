from gym import spaces

def opp_LM(duration):
    return spaces.Discrete(duration)

def own_LM(duration):
    return spaces.Discrete(duration)

def composite(duration):
    return spaces.Tuple((own_LM(duration),opp_LM(duration)))