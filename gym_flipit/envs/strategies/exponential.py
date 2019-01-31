import random
import scipy as sp
import scipy.integrate as integrate
import scipy.stats as stats
import numpy as np

class Exponential():
    def config(self,configs):
        self.lambd = configs['lambd']
        self.strategy = 'exponential'

    def first_move(self):
        return self.move(0)

    def move(self, LM):
        mv = round(LM + random.expovariate(self.lambd))
        if mv == LM:
            mv += 1
        return mv

    def pdf(self, x):
        return stats.expon.pdf(x, scale = 1 / self.lambd)

    def cdf(self, tau):
        return integrate.quad(self.pdf, -np.inf, tau)[0]
