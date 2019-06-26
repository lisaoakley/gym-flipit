import random
import scipy as sp
import scipy.integrate as integrate
import scipy.stats as stats
import numpy as np

class Normal():
    def config(self,configs):
        self.std_dev = configs['std_dev']
        self.mean = configs['mean']
        self.strategy = 'normal'

    def first_move(self):
        return self.move(0)

    def move(self, LM):
        mv = round(LM + np.random.normal(loc=self.mean,scale=self.std_dev))
        if mv == LM:
            mv += 1
        return mv

    def pdf(self, x):
        return stats.norm.pdf(x,loc=self.mean,scale=self.std_dev)

    def cdf(self, tau):
        return integrate.quad(self.pdf, -np.inf, tau)[0]
