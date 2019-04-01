#!/usr/bin/env python 

import numpy 

def single_gaussian(x, a,b,c) :
    return a*numpy.exp(-(x-b)**2/(2*c**2))
def double_gaussian(x, a,b,c, d,e,f) :
    return a*numpy.exp(-(x-b)**2/(2*c**2)) + d*numpy.exp(-(x-e)**2/(2*f**2))
def triple_gaussian(x, a,b,c, d,e,f, g,h,i) :
    return a*numpy.exp(-(x-b)**2/(2*c**2)) + d*numpy.exp(-(x-e)**2/(2*f**2))+ g*numpy.exp(-(x-h)**2/(2*i**2))
def four_gaussians(x, a,b,c, d,e,f, g,h,i, j,k,l) :
    return a*numpy.exp(-(x-b)**2/(2*c**2)) + d*numpy.exp(-(x-e)**2/(2*f**2))+ g*numpy.exp(-(x-h)**2/(2*i**2)) + j*numpy.exp(-(x-k)**2/(2*l**2))
def five_gaussians(x, a,b,c, d,e,f, g,h,i, j,k,l, m,n,o) :
    return a*numpy.exp(-(x-b)**2/(2*c**2)) + d*numpy.exp(-(x-e)**2/(2*f**2))+ g*numpy.exp(-(x-h)**2/(2*i**2)) + j*numpy.exp(-(x-k)**2/(2*l**2))\
            + m*numpy.exp(-(x-n)**2/(2*o**2))
def six_gaussians(x, a,b,c, d,e,f, g,h,i, j,k,l, m,n,o, p,q,r) :
    return a*numpy.exp(-(x-b)**2/(2*c**2)) + d*numpy.exp(-(x-e)**2/(2*f**2))+ g*numpy.exp(-(x-h)**2/(2*i**2)) + j*numpy.exp(-(x-k)**2/(2*l**2))\
            + m*numpy.exp(-(x-n)**2/(2*o**2)) + p*numpy.exp(-(x-q)**2/(2*r**2))
def seven_gaussians(x, a,b,c, d,e,f, g,h,i, j,k,l, m,n,o, p,q,r, s,t,u) :
    return a*numpy.exp(-(x-b)**2/(2*c**2)) + d*numpy.exp(-(x-e)**2/(2*f**2))+ g*numpy.exp(-(x-h)**2/(2*i**2)) + j*numpy.exp(-(x-k)**2/(2*l**2))\
            + m*numpy.exp(-(x-n)**2/(2*o**2)) + p*numpy.exp(-(x-q)**2/(2*r**2)) + s*numpy.exp(-(x-t)**2/(2*u**2))
def eight_gaussians(x, a,b,c, d,e,f, g,h,i, j,k,l, m,n,o, p,q,r, s,t,u, v,w,y) :
    return a*numpy.exp(-(x-b)**2/(2*c**2)) + d*numpy.exp(-(x-e)**2/(2*f**2))+ g*numpy.exp(-(x-h)**2/(2*i**2)) + j*numpy.exp(-(x-k)**2/(2*l**2))\
            + m*numpy.exp(-(x-n)**2/(2*o**2)) + p*numpy.exp(-(x-q)**2/(2*r**2)) + s*numpy.exp(-(x-t)**2/(2*u**2)) + v*numpy.exp(-(x-w)**2/(2*y**2))

def find_reasonable_bounds(data,numpeaks) :
    amin = 0
    amax = 2*numpy.max(data[:,1])  ##max fit is twice height of data
    bmin = numpy.min(data[:,0])
    bmax = numpy.max(data[:,0])
    cmin = 0 
    cmax = bmax - bmin          #max standard deviation is width of data

    mins = numpy.tile([amin,bmin,cmin],numpeaks)
    maxs = numpy.tile([amax,bmax,cmax],numpeaks)

    bounds = (mins,maxs)

    return bounds


