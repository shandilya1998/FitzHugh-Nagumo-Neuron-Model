# FN Neuron Model Simulation Experiments
The following were experiments conducted to understand the dynamics of FN Neuron model.
## Experiment 1
The phase plots were plotted for values of b ranging from 0 to 1 and I\_ext ranging from 0 to 3.
As the values of I\_ext increased for a fixed value of b, the neuron varied from a a fixed point to a limit cycle and then again a fixed point behaviour. 
As the value of b increased, the value of I\_ext for which a limit cycle bevaviour was observed also increases.


## Experiment 2
Phase plane for I\_ext=0 were plotted for the following values of b
- 0
- 0.4
- 0.8
- -0.5

The following is the plot for b = 0.

![case1](images/exp2/case_1a_phase_plot_num_iter_100_b_0.0000_dt_0.0010_I_ext_0.0000_niter_100000.png)

The following is the plot for b = 0.4.

![case2](images/exp2/case_1a_phase_plot_num_iter_100_b_0.4000_dt_0.0010_I_ext_0.0000_niter_100000.png)

The following is the plot for b = 0.8.

![case3](images/exp2/case_1a_phase_plot_num_iter_100_b_0.8000_dt_0.0010_I_ext_0.0000_niter_100000.png)

The following is the plot for b = -0.5.

![case4](images/exp2/case_1a_phase_plot_num_iter_50_b_-0.5000_dt_0.0010_I_ext_0.0000_niter_100000.png)


## EXPERIMENT 3
Phase plot, W vs t and V vs t plots for the following initial states.
- b = 0.8
- b = 0.4
The value of v is varied between -1 and 2 by steps of 0.1 to obtain different plots

## EXPERIMENT 4
The range of I\_ext is computed for which the neuron model exhibits limit cycle behaviour.
First approximate values I1 and I2 are calculated using brute forece plotting of phase plots and V vs t curves, based on observed values exact range is then calculated programmatically.
Oscillations are determined by the ensuring that all peaks are a minimum of a 75% the maximum peak value observed and the distance between the minimum and the maximum peaks are atleast 25% the maximum peak value.
These values were obtained by repeated experimentation with the possible thresholds and observation of the plots obtained.
For b = 0.4, I1 = 1.23 and I2 = 2.8, such that for I1\<I\_ext\<I2, the neuron model exhibits limit cycle behaviour.


