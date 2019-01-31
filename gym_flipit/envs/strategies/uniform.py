import random
import scipy as sp
import scipy.integrate as integrate
import scipy.stats as stats
import numpy as np

class Uniform():
    def config(self,configs):
        self.delta = configs['d']
        self.u = configs['u']
        self.strategy = 'uniform'

    def first_move(self):
        return self.move(0)

    def move(self, LM):
        mv = round(LM + random.uniform(self.delta - self.u/2, self.delta + self.u/2))
        if mv == LM:
            mv += 1
        return mv

    def pdf(self, x):
        if x >= self.delta - (self.u/2) and x <= self.delta + (self.u/2):
            return 1/self.u
        return 0

    def cdf(self, tau):
        if tau < self.delta - self.u / 2:
            return 0
        if tau >= self.delta - (self.u / 2) and tau <= self.delta + (self.u/2):
            return (tau - (self.delta - self.u/2)) / self.u
        if tau > self.delta + self.u/2:
            return 1
