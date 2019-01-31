import random
from scipy.stats import norm
import math

class Periodic():
    def config(self,configs):
        self.delta = configs['delta']
        self.phase = random.randint(1,self.delta)
        self.strategy = 'periodic'

    def first_move(self):
        return self.phase

    def move(self, LM):
        return LM + self.delta

    def pdf(self, x):
        return norm(self.delta+1,.1).pdf(x)

    def cdf(self, tau):
        return norm(self.delta,.1).cdf(tau)
