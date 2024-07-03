# -*- coding: utf-8 -*-
"""
November 2023
Author: Grace Capek, Garand Group, University of Wisconsin - Madison
Questions and suggestions welcome, please contact at gcapek@wisc.edu

This is a simple program to plot grid-based Mathieu a-q stability diagrams 
from the matrix solutions of the Mathieu-Hill equations for a linear quadrupole.
The user should adjust the duty cycle and a-q range for their experiment.
Important references: Konenkov et al 2002, Brabeck et al 2014

"""

#%% Import the necessary libraries
import numpy as np                
import matplotlib.pyplot as plt
import functools as ft

#%% User-adjustable variables: X and Y duty cycles in the form of t1 and t3, a and q ranges for stability plot
### Adjust the # of points in the a-q arrays for a less pixelated image
### Note that it runs a little slow when you go higher than a 100x matrix - she's working, I promise

t1 = 0.540                        # HIGH duty cycle for the X rods as a fraction of period T
t3 = 0.460                        # HIGH duty cycle for the Y rods as a fraction of period T

# Note that when a = 0 you divide by zero in the math, so I start at very small a to avoid this
# (if I mathed this wrong, somebody please let me know, thanks)
# Also note that if you change the a-q range, you may need to adjust the plot scaling in line 135
a = np.linspace(0.001,0.5,100)    # array of Mathieu a values 
q = np.linspace(0.001,1,100)      # array of Mathieu q values

#%% Small calculations from user-input variables
qq,aa = np.meshgrid(q,np.flip(a))   # grid of a and q values to use for the calculation later
t2 = np.absolute(1-t1-t3)           # calculate t2, the time that the quadrupolar potential is zero
tau1 = t1*np.pi                     # calculate tau for all 3 time points during the square wave period
tau2 = t2*np.pi
tau3 = t3*np.pi

#%% Define the functions which calculate the transfer matrices
### Inputs are tau and f values, output is the 4by4 matrix M

# create a function to calculate the matrix when f > 0
def transM_pos(f,t):
    m11 = np.cos(t*np.sqrt(f))
    m12 = (1/np.sqrt(f))*np.sin(t*np.sqrt(f))
    m21 = -np.sqrt(f)*np.sin(t*np.sqrt(f))
    m22 = np.cos(t*np.sqrt(f))
    M = [[m11,m12],[m21,m22]]
    return M
# create a function to calculate the matrix when f < 0
def transM_neg(f,t):
    m11 = np.cosh(t*np.sqrt(-f))
    m12 = (1/np.sqrt(-f))*np.sinh(t*np.sqrt(-f))
    m21 = np.sqrt(-f)*np.sinh(t*np.sqrt(-f))
    m22 = np.cosh(t*np.sqrt(-f))
    M = [[m11,m12],[m21,m22]]
    return M

#%% Define the functions which calculate the transmission matrices and their traces for X and Y
### Inputs are the arrays of Mathieu a and q values, output is the trace of the matrix M

# Function to calculate X dimension traces
def X_trace(q,a):
    # Define f values for X dimension
    f1x = a + 2*q
    f2x = a
    f3x = a - 2*q
    # Calculate the transmission matrices from functions defined above
    if f1x > 0:
        M1x = transM_pos(f1x,tau1)
    else:
        M1x = transM_neg(f1x,tau1)
    if f2x > 0:
        M2x = transM_pos(f2x,tau2)
    else:
        M2x = transM_neg(f2x,tau2)
    if f3x > 0:
        M3x = transM_pos(f3x,tau3)
    else:
        M3x = transM_neg(f3x,tau3)  
    # Multiply all 3 matrices by each other to get the total transmission matrix
    MX_total = ft.reduce(np.dot,[M1x,M2x,M3x])
    MX_trace = np.absolute(np.trace(MX_total))  # Take the trace of that matrix
    return MX_trace

# Function to calculate Y dimension traces
def Y_trace(q,a):
    # Define f values for Y dimension
    f1y = -a - 2*q
    f2y = -a
    f3y = -a + 2*q
    # Calculate the transmission matrices from functions defined above
    if f1y > 0:
        M1y = transM_pos(f1y,tau1)
    else:
        M1y = transM_neg(f1y,tau1)
    if f2y > 0:
        M2y = transM_pos(f2y,tau2)
    else:
        M2y = transM_neg(f2y,tau2)
    if f3y > 0:
        M3y = transM_pos(f3y,tau3)
    else:
        M3y = transM_neg(f3y,tau3)
    # Multiply all 3 matrices by each other to get the total transmission matrix
    MY_total = ft.reduce(np.dot,[M1y,M2y,M3y])
    MY_trace = np.absolute(np.trace(MY_total))  # Take the trace of that matrix
    return MY_trace

#%% Iterate through the a and q matrices and calculate the trace of the transmission matrix
### for each a,q combination in both dimensions

# Initialize matrices for the X and Y traces
X_traces = np.zeros(np.shape(qq))
Y_traces = np.zeros(np.shape(qq))
# For the specified a-q range, calculate the X and Y traces and put them in the matrices
for i in range(len(q)):
    for j in range(len(q)):
        X_traces[i,j] = X_trace(qq[i,j],aa[i,j])
        Y_traces[i,j] = Y_trace(qq[i,j],aa[i,j])
# Initialize a grid to plot later
grid = np.zeros(np.shape(qq))
# Assign integer values for each grid item based on X and Y stability
# Ions only have a stable trajectory in that dimension if the trace < 2
for i in range(len(q)):
    for j in range(len(q)):
        if X_traces[i,j] < 2 and Y_traces[i,j] > 2:     # X stable, Y unstable assigned to 1
            grid[i,j] = 1
        if X_traces[i,j] < 2 and Y_traces[i,j] < 2:     # X stable, Y stable assigned to 2
            grid[i,j] = 2
        if X_traces[i,j] > 2 and Y_traces[i,j] < 2:     # X unstable, Y stable assigned to 3
            grid[i,j] = 3

#%% Plot the grid of integer values based on the stability conditions defined above
plt.imshow(grid, cmap = 'Greens', extent=[0.001,1,0.001,0.5])
plt.xlabel("q", fontsize = 14)
plt.ylabel("a", fontsize = 14)
# Title contains the duty cycles as high X:Y in %
plt.title(str(t1*100) + ":" + str(t3*100) + " Mathieu Stability Diagram", fontsize = 14)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
# Uncomment the next line if you want to export a higher res pic
#plt.savefig("54_46_a-q.png",dpi=1200)




        
        

      
                        
    
    

