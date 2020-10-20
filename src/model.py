import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class FNNeuron:
    def __init__(self, dt, niter):
        self.dt = dt
        self.niter = niter
        self.a = 0.5
        self.b = np.random.random()
        self.r = 0.1
        self.I_ext = np.random.random()
        self.v = np.random.random()
        self.w = np.random.random()
        self.v_hist = np.zeros(self.niter)
        self.w_hist = np.zeros(self.niter)
        self.t = np.arange(self.niter)*self.dt

    def set_niter(self, n):
        self.niter = n
        self.v_hist = np.zeros(self.niter)
        self.w_hist = np.zeros(self.niter)
        self.t = np.arange(self.niter)*self.dt

    def set_dt(self, dt):
        self.dt = dt
        self.t = np.arange(self.niter)*self.dt    

    def set_b(self, b):
        self.b = b

    def set_I_ext(self, I):
        self.I_ext = I
   
    def set_v(self, v):
        self.v = v

    def set_w(self, w):
        self.w = w
 
    def f(self, v):
        return v*(self.a-v)*(v-1)

    def dv(self, v, w):
        return self.dt*(self.f(v) - w + self.I_ext)

    def dw(self, v, w):
        return self.dt*(self.b*v - self.r*w)
    
    def reset(self):
        self.b = np.random.random()
        self.r = 0.1
        self.a = 0.5
        self.I_ext = np.random.random()
        self.v = np.random.random()
        self.w = np.random.random()
        self.v_hist = np.zeros(self.niter)
        self.w_hist = np.zeros(self.niter)
        self.t = np.arange(self.niter)*self.dt
    
    def plot(self, figname):
        fig, axes = plt.subplots(3, 1, figsize = (5, 15))
        axes[0].plot(self.v_hist, self.w_hist)
        axes[0].set_xlabel('voltage')
        axes[0].set_ylabel('recovery variable')
        axes[0].set_title('phase plot')
        axes[1].plot(self.t, self.v_hist)
        axes[1].set_xlabel('time')
        axes[1].set_ylabel('voltage')
        axes[1].set_title('voltage vs time')
        axes[2].plot(self.t, self.w_hist)
        axes[2].set_xlabel('time')
        axes[2].set_ylabel('recovery variable')
        axes[2].set_title('recovery variable vs time') 
        fig.savefig(figname)
        plt.clf()
        plt.close('all')
    
    def __call__(self, perturb = False):
        for i in range(self.niter):
            self.v = self.v + self.dv(self.v, self.w)
            self.w = self.w + self.dw(self.v, self.w)
            if(i>0):
                if self.niter/i == 2 and perturb:      
                    print('perturbed')
                    self.set_v(0.001)
            self.v_hist[i] = self.v
            self.w_hist[i] = self.w
    
        
