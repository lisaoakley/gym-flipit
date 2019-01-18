import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random

class Periodic():
    def config(self,configs):
        self.delta = configs['delta']
        self.phase = random.randint(1,self.delta)

    def first_move(self):
        return self.phase

    def move(self, LM):
        return LM + self.delta


class FlipitEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, duration=100, p0='periodic', p0_configs={'delta':10}, p0_move_cost=10, p1_move_cost=10):
        # duration only used to bound possible states, is there a better way to do this?
        self.duration = duration
        self.tick = 0
        self.state = 0
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(self.duration)
        
        self.p0_configs = p0_configs
        if p0 == 'periodic':
            self.p0 = Periodic()
        self.p0.config(self.p0_configs)
        
        self.p0_move_cost = p0_move_cost
        self.p0_moves = [0, self.p0.first_move()]


        self.p1_move_cost = p1_move_cost
        self.p1_moves = []
        
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        self.tick += 1

        #p0 plays if necessary
        p0_LM = self.p0_moves[-1]
        if self.tick == p0_LM:
            self.p0_moves.append(self.p0.move(p0_LM))
        
        #p1 plays if action is taken
        if action == 0:
            self.state += 1
        else:
            self.p1_moves.append(self.tick)
            self.state = self.tick - list(filter(lambda x: x < self.tick, self.p0_moves))[-1]
            
        # state indicates time since opponent's last known move
        observation = self.state

        #TODO: decide reward (local benefit?)
        # should final reward be total benefit?
        reward = 0
        done = self.tick >= self.duration
        info = {'p0_moves':self.p0_moves, 'p1_moves':self.p1_moves}

        return observation, reward, done, info

    def reset(self):
        self.p0_moves = [0, p0.first_move()]
        self.p1_moves = []
        self.tick = 0
        self.state = 0

    def render(self, mode='human'):
        return