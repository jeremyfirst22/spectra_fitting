#!/usr/bin/env python 

import numpy as np 

def single_gaussian(x, a,b,c, C) :
    return a*np.exp(-(x-b)**2/(2*c**2)) + C
def double_gaussian(x, a,b,c, d,e,f, C) :
    return a*np.exp(-(x-b)**2/(2*c**2)) + d*np.exp(-(x-e)**2/(2*f**2)) + C
def triple_gaussian(x, a,b,c, d,e,f, g,h,i, C) :
    return a*np.exp(-(x-b)**2/(2*c**2)) + d*np.exp(-(x-e)**2/(2*f**2))+ g*np.exp(-(x-h)**2/(2*i**2)) + C
def four_gaussians(x, a,b,c, d,e,f, g,h,i, j,k,l, C) :
    return a*np.exp(-(x-b)**2/(2*c**2)) + d*np.exp(-(x-e)**2/(2*f**2))+ g*np.exp(-(x-h)**2/(2*i**2)) + j*np.exp(-(x-k)**2/(2*l**2)) + C
def five_gaussians(x, a,b,c, d,e,f, g,h,i, j,k,l, m,n,o, C) :
    return a*np.exp(-(x-b)**2/(2*c**2)) + d*np.exp(-(x-e)**2/(2*f**2))+ g*np.exp(-(x-h)**2/(2*i**2)) + j*np.exp(-(x-k)**2/(2*l**2))\
            + m*np.exp(-(x-n)**2/(2*o**2)) + C
def six_gaussians(x, a,b,c, d,e,f, g,h,i, j,k,l, m,n,o, p,q,r, C) :
    return a*np.exp(-(x-b)**2/(2*c**2)) + d*np.exp(-(x-e)**2/(2*f**2))+ g*np.exp(-(x-h)**2/(2*i**2)) + j*np.exp(-(x-k)**2/(2*l**2))\
            + m*np.exp(-(x-n)**2/(2*o**2)) + p*np.exp(-(x-q)**2/(2*r**2)) + C
def seven_gaussians(x, a,b,c, d,e,f, g,h,i, j,k,l, m,n,o, p,q,r, s,t,u, C) :
    return a*np.exp(-(x-b)**2/(2*c**2)) + d*np.exp(-(x-e)**2/(2*f**2))+ g*np.exp(-(x-h)**2/(2*i**2)) + j*np.exp(-(x-k)**2/(2*l**2))\
            + m*np.exp(-(x-n)**2/(2*o**2)) + p*np.exp(-(x-q)**2/(2*r**2)) + s*np.exp(-(x-t)**2/(2*u**2)) + C
def eight_gaussians(x, a,b,c, d,e,f, g,h,i, j,k,l, m,n,o, p,q,r, s,t,u, v,w,y, C) :
    return a*np.exp(-(x-b)**2/(2*c**2)) + d*np.exp(-(x-e)**2/(2*f**2))+ g*np.exp(-(x-h)**2/(2*i**2)) + j*np.exp(-(x-k)**2/(2*l**2))\
            + m*np.exp(-(x-n)**2/(2*o**2)) + p*np.exp(-(x-q)**2/(2*r**2)) + s*np.exp(-(x-t)**2/(2*u**2)) + v*np.exp(-(x-w)**2/(2*y**2)) + C

def matchFunctionName(i) :
    if i == 1 :
        func = single_gaussian
    elif i == 2 :
        func = double_gaussian
    elif i == 3 :
        func = triple_gaussian
    elif i == 4 :
        func = four_gaussians
    elif i == 5 :
        func = five_gaussians
    elif i == 6 :
        func = six_gaussians
    elif i == 7 :
        func = seven_gaussians
    elif i == 8 :
        func = eight_gaussians
    else :
        print "ERROR: Only up to 8 guassian peaks currently supported"
        print "\t You have chosed %s peaks to be fitted" %i
        sys.exit(2)
    return func

