# This is the monte-carlo fitting procedure for the two-step exchange where the T1 relaxation is same for the different states but the populations are different.
# The reference is Perrin, C.L., and T.J. Dwyer. 1990. Application of two-dimensional NMR to kinetics of chemical exchange. Chem. Rev. 90:935–967.

# you will get the parameters at the end of the run in the terminal
# you will also get the plots of the parameters with standard deviation and the output file for statistical analysis


import numpy as np
import math
import matplotlib.pyplot as plt
import random as rand
import pylab

file_name = "output.dat"
with open(file_name, 'w') as text_file:
    text_file.write('#parameters key: [fraction, rate]')
    text_file.close()


# This is the data file. The data file is arranged as xaxis (mixing time) and yaxis (ratio of sum(cross)/sum(dia))
data = "exsy_BD.dat" # Change the name of the file for different data. The data should be in two columns; first one is xaxis.
xaxis, yaxis = np.loadtxt(data, unpack=True)
err = 0.01

# print(xaxis)
# print(yaxis)

# define the equation for the fitting here
def mod_eval(x,p):
    x = np.array(x)
    xa = p[0]
    k = p[1]
    
    xb = 1-xa
    b = (np.exp(k*x)+1)/(np.exp(k*x)-1)
    
    return 4*xa*xb/(b+(xa-xb)**2)


# define the loglikelihood calculation here
def LogLikelihood(params):
    frac_xa = params[0]
    rate_k = params[1]

    value = mod_eval(xaxis, p=[frac_xa,rate_k])

    chi_squared = sum(((np.array(value)-np.array(yaxis))**2)/(2*(err**2)))

    return chi_squared*-1.0

# initial paramaeters
params_0 = [0.2, 50.0]

# increament steps which needs to be randomized
d_m = 0.01
d_c = 10.0

def del_m():
    del_m = rand.uniform(-d_m, d_m)
    return del_m
def del_c():
    del_c = rand.uniform(-d_c, d_c)
    return del_c

accepted = 0

# Metropolis-Hastings algorithm

while(accepted < 50000):           # number of iterations
    old_params = params_0
    old_loglik = LogLikelihood(params_0)    

#print(old_loglik)
    new_params = [-1., -1.]
    
    while(new_params[0] < 0.):
        new_params[0] = old_params[0]+del_m()
    while(new_params[1] < 0.):
        new_params[1] = old_params[1]+del_c()

    newloglik = LogLikelihood(new_params)

    if (newloglik > old_loglik):
        accepted_params = str(new_params)
        with open(file_name, 'a') as text_file:
            text_file.write('\n' + accepted_params[1:(len(accepted_params)-1)])
            text_file.close
        params_0 = new_params
        accepted = accepted+1
        #print(accepted)
    else:
        u = rand.uniform(0.0,1.0)
        if (u < math.exp(newloglik - old_loglik)):
            accepted_params = str(new_params)
            with open(file_name, 'a') as text_file:
                text_file.write('\n' + accepted_params[1:(len(accepted_params)-1)])
                text_file.close            
            params_0 = new_params
            accepted = accepted+1
            #print(accepted)
        else:
            params_0=old_params

        
def mcmc_plot(param_file):
    params = np.loadtxt(param_file, delimiter=',', unpack=True)
    
    xar = np.array(params[0])
    Frac = np.round(np.mean(xar, axis=0), decimals=2)
    stand = np.round(np.std(xar, axis=0), decimals=2) 
    print('Fraction_A = ',Frac, ';', 'Standard deviation = ',stand)
    print('Fraction_B = ',1-Frac, ';', 'Standard deviation = ',stand) 
    fig1 = plt.figure()
    plt.hist(params[0], bins=40, range=[min(params[0]), max(params[0])], facecolor='g')
    #plt.legend(loc='upper left')
    #plt.title('Fraction-A')
    plt.ylabel('counts')
    plt.xlabel('Fraction-A')     
    plt.ylim(0,10000)          # You can change the y limit (min, max) according to your data
    plt.xlim(0.15,0.25)        # You can change the y limit (min, max) according to your data
    pylab.savefig('fraction_xa_hist.png', dpi=300)
    plt.close()

    xar1 = np.array(params[1])
    Frac1 = str(np.round(np.mean(xar1, axis=0), decimals=0))
    stand1 = str(np.round(np.std(xar1, axis=0), decimals=0))
    print('Exchange rate = ',Frac1, 's-1', ';', 'Standard Deviation = ',stand1)
    fig2 = plt.figure()
    plt.hist(params[1], bins=40, range=[min(params[1]), max(params[1])], facecolor='g')
    #plt.legend(loc='upper left')
    #plt.title('Exchange Rate')
    plt.ylabel('counts')
    plt.xlabel('Exchange Rate $s^{-1}$')
    plt.ylim(0,10000)       # You can change the y limit (min, max) according to your data
    plt.xlim(30,60)         # You can change the y limit (min, max) according to your data
    pylab.savefig('rate_hist.png', dpi=300)
    plt.close()

    
mcmc_plot('output.dat')