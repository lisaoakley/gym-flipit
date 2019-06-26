import gym
from gym import error, utils, spaces
from gym.utils import seeding
import random
from gym_flipit.envs.strategies import periodic, exponential, uniform, normal
from gym_flipit.envs.state import reset_state,set_obs_space,set_state
from gym_flipit.envs.rew import calc_rew

class FlipitEnv(gym.Env):
    """
    Description:
        FLIPIT: The Game of "Stealthy Takeover" is a security game invented by
        Marten van Dijk, Ari Juels, Alina Oprea, and Ronald L. Rivest. In the
        game, players compete for a shared resource, gaining benefit for amount
        of time in control of the resource, minus the cost to take control. The
        game is "stealthy" meaning that players do not immediately get to know
        when their opponent has played. The goal of the game is to maximize total
        benefit.

    Source:
        van Dijk, M., Juels, A., Oprea, A. et al. J Cryptol (2013) 26: 655. https://doi.org/10.1007/s00145-012-9134-5

    Actions:
        Type: Discrete(2)
        Num Action
        0   Do not play
        1   Play (take control of the resource and learn opponent's last move time)

    Observation:
    (1) 'opp_LM'
        Type: Discrete(<game-duration>)
        Description: The observation is the amount of time since the last *known*
                     opponent move. Note: the observation might be greater than
                     the time since the last move of the opponent because there
                     is no way to know the opponent's move time without playing.

    (2) 'own_LM'
        Type: Discrete(<game-duration>)
        Description: The observation is the amount of time since the player's 
                     own last move.

    (3) 'composite'
        Type: Tuple(Discrete(<game-duration>),Discrete(<game-duration>))
        Description: tuple of the form (own_LM,opp_LM).

    Reward:
    (1) 'constant_minus_cost_norm'
        Constant positive reward minus move cost, normalized by some constant c.
        Requires rew_configs['val'] and rew_configs['c']
        If you did not move this turn, reward is 0
        Otherwise,
        If the most recent game move was your own, reward is -(your move cost)
        If the most recent game move was the opponents, reward is 
            (rew_configs['val'] - your move cost)/rew_configs['c']
        example:
        assume both players have move cost = 5, val = 25, c = 5
        player 0 plays at: [3,20]
        player 1 plays at: [5,8]
        if player 1 plays again at time = 12, the reward will be -2
        if player 1 plays again at time = 25, the reward will be (25 - 5)/5 = 4


    (2) 'LM_benefit'
        If you did not move this turn, reward is 0
        Otherwise,
        If the most recent game move was your own, reward is -(your move cost).
            (note that moving twice consecutively does not provide new information,
            nor does it give you any additional time in control)
        If the most recent game move was the opponent's, reward is the (time in
            control since your last move) - (your move cost).
        example:
        assume both players have move cost = 2
        player 0 plays at: [3,20]
        player 1 plays at: [5,8]
        if player 1 plays again at time = 12, the reward will be -2
        if player 1 plays again at time = 25, the reward will be 20 - 8 - 2 = 10

    (3) 'exponential'
        Requires rew_configs['a'] and rew_configs['b']
        If you did not move this turn, reward is 0
        Otherwise,
        If the most recent game move was your own, reward is -(your move cost).
            (note that moving twice consecutively does not provide new information,
            nor does it give you any additional time in control)
        If the most recent game move was the opponent's, reward is
            e^{(rew_configs['b']-observation)/rew_configs['a']} - (your move cost).
        example:
        assume both players have move cost = 2
        player 0 plays at: [3,20]
        player 1 plays at: [5,8]
        if player 1 plays again at time = 12, the reward will be -2
        if player 1 plays again at time = 25, the reward will be e^{(100-5)/20} - 2


    Starting State:
        Player 0 has control at tick = 0

    Episode Termination:
        tick reaches duration of game

    Opponents:
        There are various renewal strategies defined in FLIPIT: The Game of "Stealthy Takeover".
        The default is "periodic" with a period (delta) of 10. You can change the
        opponent strategy by running:
               env.config(p0=<strategy>,p0_configs={<config-1>:<val>,...})
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,state_type='opp_LM',rew_type='constant_minus_cost_norm',rew_configs={'val':10,c:'5'},p0='periodic',p0_configs={'delta':10},duration=1000,p0_move_cost=1,p1_move_cost=5):
        self.config(state_type,rew_type,rew_configs,p0,p0_configs,duration,p0_move_cost,p1_move_cost)

    def config(self,state_type,rew_type,rew_configs,p0,p0_configs,duration=100,p0_move_cost=1,p1_move_cost=5):
        self.duration = duration
        self.state_type = state_type
        self.set_obs_space()
        self.action_space = spaces.Discrete(2)
        self.rew_configs = rew_configs
        self.rew_type = rew_type
        self.p0_configs = p0_configs
        if p0 == 'periodic':
            self.p0 = periodic.Periodic()
        if p0 == 'exponential':
            self.p0 = exponential.Exponential()
        if p0 == 'uniform':
            self.p0 = uniform.Uniform()
        if p0 == 'normal':
            self.p0 = normal.Normal()
        self.player_move_costs = [p0_move_cost, p1_move_cost]
        self.reset()
    
    def reset(self):
        self.p0.config(self.p0_configs)
        self.player_moves = [[0], [0]]
        self.player_total_gain = [0,0]
        self.player_total_move_cost = [0,0]
        self.p0_next_move = self.p0.first_move()
        self.controller = 0
        self.tick = 0
        self.reset_state()
        self.found_FM = False
        return self.state

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        self.tick += 1
        # if both players play, defender gets control
        if action == 1 and self.tick == self.p0_next_move:
            action = 0
        #p0 plays according to its strategy
        if self.tick == self.p0_next_move:
            self.move(0)
            self.p0_next_move = self.p0.move(self.tick)
        #p1 plays if action is 1
        if action == 1:
            self.move(1)
            if self.get_LM(0) > 0:
                self.found_FM = True
        #update output values
        self.set_state()
        reward = self.calc_rew()
        done = self.tick >= self.duration
        self.player_total_gain[self.controller] += 1
        return self.state, reward, done, {'true_action':action}

    def render(self):
        return

    def calc_rew(self):
        if self.rew_type == 'exponential':
            return calc_rew.exponential(self.found_FM,self.moved(1),self.player_move_costs[1],self.get_LM(0),self.player_moves[1],self.tick,self.rew_configs)
        elif self.rew_type == 'LM_benefit':
            return calc_rew.LM_benefit(self.found_FM,self.moved(1),self.player_move_costs[1],self.get_LM(0),self.get_LM(1),self.tick,self.player_moves[0])
        elif self.rew_type == 'reciprocal':
            return calc_rew.reciprocal(self.found_FM,self.moved(1),self.get_LM(0),self.player_moves[1],self.player_move_costs[1],self.tick)
        elif self.rew_type == 'constant_reciprocal':
            return calc_rew.constant_reciprocal(self.found_FM,self.moved(1),self.get_LM(0),self.player_moves[1],self.player_move_costs[1],self.rew_configs['upper_lim'],self.tick)
        elif self.rew_type == 'constant':
            return calc_rew.constant(self.found_FM,self.moved(1),self.get_LM(0),self.player_moves[1],self.player_move_costs[1],self.rew_configs['val'],self.tick)
        elif self.rew_type == 'constant_minus_cost':
            return calc_rew.constant_minus_cost(self.found_FM,self.moved(1),self.get_LM(0),self.player_moves[1],self.player_move_costs[1],self.rew_configs['val'],self.tick)
        elif self.rew_type == 'constant_minus_cost_norm':
            return calc_rew.constant_minus_cost(self.found_FM,self.moved(1),self.get_LM(0),self.player_moves[1],self.player_move_costs[1],self.rew_configs['val'],self.tick,self.rew_configs['c'])
        elif self.rew_type == 'exp_cost':
            return calc_rew.exp_cost(self.found_FM,self.moved(1),self.get_LM(0),self.player_moves[1],self.player_move_costs[1],self.rew_configs['val'],self.tick)
        elif self.rew_type == 'LM_avg':
            return calc_rew.LM_avg(self.found_FM,self.moved(1),self.get_LM(0),self.player_moves[1],self.player_move_costs[1],self.rew_configs['val'],self.tick)
        else:
            raise NotImplementedError
    
    def set_obs_space(self):
        if self.state_type == 'opp_LM':
            self.observation_space = set_obs_space.opp_LM(self.duration)
        elif self.state_type == 'own_LM':
            self.observation_space = set_obs_space.own_LM(self.duration)
        elif self.state_type == 'composite':
            self.observation_space = set_obs_space.composite(self.duration)
        else:
            raise NotImplementedError

    def reset_state(self):
        if self.state_type == 'opp_LM':
            self.state = reset_state.opp_LM()
        elif self.state_type == 'own_LM':
            self.state = reset_state.own_LM()
        elif self.state_type == 'composite':
            self.state = reset_state.composite()
        else:
            raise NotImplementedError

    def set_state(self):
        if self.state_type == 'opp_LM':
            self.state = set_state.opp_LM(self.get_LM(0),self.moved(1),self.found_FM,self.tick,self.state)
        elif self.state_type == 'own_LM':
            self.state = set_state.own_LM(self.moved(1),self.state)
        elif self.state_type == 'composite':
            self.state = set_state.composite(self.get_LM(0),self.moved(1),self.found_FM,self.tick,self.state)
        else:
            raise NotImplementedError

    def get_LM(self,player):
        return self.player_moves[player][-1]
    
    def moved(self,player):
        return self.tick == self.player_moves[player][-1]

    def move(self, player):
        self.player_moves[player].append(self.tick)
        self.player_total_move_cost[player] += self.player_move_costs[player]
        self.controller = player

    '''
    Calculate benefit
    '''
    def calc_benefit(self, player):
        return self.player_total_gain[player] - self.player_total_move_cost[player]

    def calc_avg_benefit(self, player):
        return self.calc_benefit(player, self.tick) / self.tick
