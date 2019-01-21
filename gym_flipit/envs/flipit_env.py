import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random
from gym_flipit.envs.strategies import periodic, exponential

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

        self.p0.config(self.p0_configs)
        self.p0_move_cost = p0_move_cost
        self.p0_moves = [0, self.p0.first_move()]

        self.p1_move_cost = p1_move_cost
        self.p1_moves = [0]



    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        self.tick += 1
        #p0 plays if necessary
        p0_next_move = self.p0_moves[-1]
        if self.tick == p0_next_move:
            self.p0_moves.append(self.p0.move(p0_next_move))

        #p1 plays if action is taken
        p0_LM = 0
        reward = 0
        if action == 0:
            self.state += 1
        else:
            p1_LM = self.p1_moves[-1]
            p0_LM = list(filter(lambda x: x < self.tick, self.p0_moves))[-1]
            self.p1_moves.append(self.tick)
            self.state = self.tick - p0_LM
            gain = 0 if p0_LM < p1_LM else [x for x in self.p0_moves if x < self.tick and x >= p1_LM][0]
            reward = gain - self.p1_move_cost

        # state indicates time since opponent's last known move
        observation = self.state

        done = self.tick >= self.duration
        info = {'p0_moves':self.p0_moves, 'p1_moves':self.p1_moves, 'p0_LM':p0_LM}

        return observation, reward, done, info

    def reset(self):
        self.p0.config(self.p0_configs)
        self.p0_moves = [0, self.p0.first_move()]
        self.p1_moves = [0]
        self.tick = 0
        self.state = 0
        return self.state

    def p0_pdf(self, x):
        return self.p0.pdf(x)

    def p0_cdf(self, x):
        return self.p0.cdf(x)

    def render(self, mode='human'):
        return
