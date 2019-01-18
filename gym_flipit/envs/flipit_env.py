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

    def __init__(self, duration=100, p0='periodic', p0_configs={'delta':10}, p0_move_cost=1, p1_move_cost=1):
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
        self.p1_moves = [0]
        
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        self.tick += 1

        #p0 plays if necessary
        p0_next_move = self.p0_moves[-1]
        if self.tick == p0_next_move:
            self.p0_moves.append(self.p0.move(p0_next_move))
        
        reward = 0

        #p1 plays if action is taken
        p0_LM = 0
        if action == 0:
            self.state += 1
        else:
            p1_LM = self.p1_moves[-1]
            p0_LM = list(filter(lambda x: x < self.tick, self.p0_moves))[-1]
            
            self.p1_moves.append(self.tick)
            self.state = self.tick - p0_LM

            #moves = len([x for x in self.p1_moves if p1_LM <= x and self.tick >= x ])
            gain = 0 if p0_LM < p1_LM else p0_LM - p1_LM
            reward = gain - self.p1_move_cost
            
        # state indicates time since opponent's last known move
        observation = self.state



        done = self.tick >= self.duration
        info = {'p0_moves':self.p0_moves, 'p1_moves':self.p1_moves, 'p0_LM':p0_LM}

        return observation, reward, done, info

    def reset(self):
        self.p0_moves = [0, self.p0.first_move()]
        self.p1_moves = [0]
        self.tick = 0
        self.state = 0
        return self.state

    def render(self, mode='human'):
        return