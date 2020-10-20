import numpy as np
from src.model import FNNeuron
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy import signal

def exp1():
    exp_plot_dir = 'images/exp1'
    dt = 0.001
    niter = int(10e4)
    b = 0.25
    I_ext = 0.1
    fn = FNNeuron(
        dt, 
        niter   
    )
    v = np.random.random()
    w = np.random.random()
    b_ = np.arange(10)*0.1
    I_ = np.arange(30)*0.1
    for b in tqdm(b_):
        for I_ext in I_:
            fn.set_b(b)
            fn.set_v(v)
            fn.set_w(w)
            fn.set_I_ext(I_ext)
            image_name = 'v_{val_v:.4f}_w_{val_w:.4f}_b_{val_b:.4f}_dt_{val_dt:.4f}_I_ext_{val_I:.4f}_niter_{n}.png'.format(
                val_v = v,
                val_w = w,
                val_b = b, 
                val_dt = dt, 
                val_I = I_ext, 
                n = niter
            )
            if not os.path.exists(exp_plot_dir):
                os.mkdir(exp_plot_dir)
            fn()
            fn.plot(os.path.join(exp_plot_dir, image_name))
            fn.reset()

def exp2():
    exp_plot_dir = 'images/exp2'
    dt = 0.001
    niter = int(10e4)
    b = -0.5
    I_ext = 0 
    fn = FNNeuron(
        dt, 
        niter   
    )
    num_exp = 50
    v = np.random.normal(0, 1, num_exp)
    w = np.random.normal(0, 1, num_exp)
    fig, axes = plt.subplots(1, 1,figsize =  (5, 5))
    for i in tqdm(range(num_exp)):
        fn.set_b(b)
        fn.set_v(v[i])
        fn.set_w(w[i])
        fn.set_I_ext(I_ext)
        fn()
        axes.plot(fn.v_hist, fn.w_hist)
        axes.set_xlabel('voltage')
        axes.set_ylabel('recovery variable')
        axes.set_title('phase plot')
        fn.reset()
    if not os.path.exists(exp_plot_dir):
        os.mkdir(exp_plot_dir)
    image_name = 'case_1a_phase_plot_num_iter_{num}_b_{val_b:.4f}_dt_{val_dt:.4f}_I_ext_{val_I:.4f}_niter_{n}.png'.format(
        num = num_exp,
        val_b = b,
        val_dt = dt, 
        val_I = I_ext,
        n = niter
    )
    fig.savefig(os.path.join(exp_plot_dir, image_name))

def exp3():
    exp_plot_dir = 'images/exp3'
    dt = 0.001
    niter = int(10e4)
    b = 0
    I_ext = 0 
    fn = FNNeuron(
        dt, 
        niter   
    )    
    V = np.arange(-10, 20)*0.1
    w = 0
    for v in tqdm(V):
        fn.set_b(b)
        fn.set_v(v)
        fn.set_w(w)
        fn.set_I_ext(I_ext)
        image_name = 'case_1b_v_{val_v:.4f}_w_{val_w:.4f}_b_{val_b:.4f}_dt_{val_dt:.4f}_I_ext_{val_I:.4f}_niter_{n}.png'.format(
            val_v = v,
            val_w = w,
            val_b = b,
            val_dt = dt,
            val_I = I_ext,
            n = niter
        )
        exp = 'b_{val_b:.4f}'.format(val_b=b)
        dir = os.path.join(exp_plot_dir, exp)
        if not os.path.exists(dir):
            os.mkdir(os.path.join(dir))
        fn()
        fn.plot(os.path.join(dir,image_name))
        fn.reset()

def exp4():
    I_ext = np.arange(115, 285, 2)*0.01
    b = 0.4
    v = np.random.normal(0, 1)
    w = np.random.normal(0, 1)
    exp_plot_dir = 'images/exp4'
    dt = 0.001
    niter = int(10e5) 
    fn = FNNeuron(
        dt,
        niter
    )
    def is_periodic(samples, tol):
        m = tol*max(samples)
        t = (max(samples) - min(samples)) <= 0.25*max(samples)
        return all(m <= d for d in samples) and t
    
    osc_I = []
    for I in I_ext:
        fn.set_b(b)
        fn.set_v(v)
        fn.set_w(w)
        fn.set_I_ext(I)
        image_name = 'case_2a_v_{val_v:.4f}_w_{val_w:.4f}_b_{val_b:.4f}_dt_{val_dt:.4f}_I_ext_{val_I:.4f}_niter_{n}.png'.format(
            val_v = v,
            val_w = w,
            val_b = b,
            val_dt = dt,
            val_I = I,
            n = niter
        ) 
        if not os.path.exists(exp_plot_dir):
            os.mkdir(os.path.join(exp_plot_dir))
        fn()
        peaks, _ = signal.find_peaks(fn.v_hist)
        heights = [fn.v_hist[p] for p in peaks]
        print('\n')
        print(I)
        val = is_periodic(heights[1:], 0.75)
        print(val)
        if val:
            osc_I.append(I)
        fn.plot(os.path.join(exp_plot_dir, 'phase_'+image_name))
        fn.reset()
    return osc_I

def exp5():
    exp_plot_dir = 'images/exp5'
    dt = 0.001
    niter = int(10e4)
    b = 0.4
    I_ext = 1.55 
    fn = FNNeuron(
        dt, 
        niter   
    )   
    num_exp = 50
    v = np.random.normal(0, 1, num_exp)
    w = np.random.normal(0, 1, num_exp)
    fig, axes = plt.subplots(1, 1,figsize =  (5, 5)) 
    for i in tqdm(range(num_exp)):
        fn.set_b(b)
        fn.set_v(v[i])
        fn.set_w(w[i])
        fn.set_I_ext(I_ext)
        fn()
        axes.plot(fn.v_hist, fn.w_hist, 'b')
        axes.set_xlabel('voltage')
        axes.set_ylabel('recovery variable')
        axes.set_title('phase plot')
        fn.reset()
    fn.set_niter(int(10e5))
    fn.set_b(b)
    fn.set_v(np.random.normal(0, 1))
    fn.set_w(np.random.normal(0, 1))
    fn.set_I_ext(0.5) 
    fn(True)
    axes.plot(fn.v_hist, fn.w_hist, 'r')
    axes.set_xlabel('voltage')
    axes.set_ylabel('recovery variable')
    axes.set_title('phase plot')
    fn.reset() 
    if not os.path.exists(exp_plot_dir):
        os.mkdir(exp_plot_dir)
    image_name = 'case_2ba_phase_plot_num_iter_{num}_b_{val_b:.4f}_dt_{val_dt:.4f}_I_ext_{val_I:.4f}_niter_{n}.png'.format(
        num = num_exp,
        val_b = b,
        val_dt = dt, 
        val_I = I_ext,
        n = niter
    )   
    fig.savefig(os.path.join(exp_plot_dir, image_name))

exp5()  
#I = exp4()
#exp3()
#exp2()
#exp1()
