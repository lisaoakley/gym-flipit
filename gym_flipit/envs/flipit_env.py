import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random
from gym_flipit.envs.strategies import periodic, exponential, uniform

class FlipitEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, p0='periodic', p0_configs={'delta':10}, duration=100, p0_move_cost=1, p1_move_cost=5):
        self.config(p0, p0_configs, duration, p0_move_cost, p1_move_cost)

    def config(self, p0, p0_configs, duration=100, p0_move_cost=1, p1_move_cost=5):
        self.tick = 0
        self.state = 0
        self.action_space = spaces.Discrete(2)
        # duration only used to bound possible states, is there a better way to do this?
        self.duration = duration
        self.observation_space = spaces.Discrete(self.duration)

        self.p0_configs = p0_configs
        if p0 == 'periodic':
            self.p0 = periodic.Periodic()
        if p0 == 'exponential':
            self.p0 = exponential.Exponential()
        if p0 == 'uniform':
            self.p0 = uniform.Uniform()

        self.p0.config(self.p0_configs)

        self.player_moves = [[0, self.p0.first_move()], [0]]
        self.player_move_costs = [p0_move_cost, p1_move_cost]



    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        self.tick += 1
        #p0 plays if necessary
        p0_next_move = self.player_moves[0][-1]
        if self.tick == p0_next_move:
            self.player_moves[0].append(self.p0.move(p0_next_move))

        #p1 plays if action is taken
        p0_LM = 0
        reward = 0
        if action == 0:
            self.state += 1
        else:
            p1_LM = self.player_moves[1][-1]
            p0_LM = list(filter(lambda x: x < self.tick, self.player_moves[0]))[-1]
            self.player_moves[1].append(self.tick)
            self.state = self.tick - p0_LM
            gain = 0 if p0_LM < p1_LM else [x for x in self.player_moves[0] if x < self.tick and x >= p1_LM][0]
            reward = gain - self.player_move_costs[1]

        # state indicates time since opponent's last known move
        observation = self.state

        done = self.tick >= self.duration
        info = {'p0_moves':self.player_moves[0], 'p1_moves':self.player_moves[1], 'p0_LM':p0_LM}

        return observation, reward, done, info

    def reset(self):
        self.p0.config(self.p0_configs)
        self.player_moves = [[0, self.p0.first_move()], [0]]
        self.tick = 0
        self.state = 0
        return self.state

    def render(self, mode='human'):
        return

    # Some strategies require extra information about the opponent.
    def p0_pdf(self, x):
        return self.p0.pdf(x)

    def p0_cdf(self, x):
        return self.p0.cdf(x)

    '''
    Calculate benefit
    '''
    def calc_gain(self):
        control = 0
        gain = [0,0]
        moves = self.player_moves
        for i in range(self.tick):
            if i in moves[1-control]:
                control = 1-control
            gain[control] += 1
        return gain

    def calc_total_cost(self):
        return len(self.player_moves[0])*self.player_move_costs[0], len(self.player_moves[1])*self.player_move_costs[1]

    def calc_benefit(self, player):
        return self.calc_gain()[player] - self.calc_total_cost()[player]

    def calc_avg_benefit(self, player):
        return self.calc_benefit(player, self.tick) / self.tick
