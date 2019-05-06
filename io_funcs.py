import os
from os import sys 
import numpy as np

import matplotlib.pyplot as plt 

from gaussian_funcs import *
from spectra_transformations import find_nearest

def read_data(fileName) :
    if not os.path.isfile(fileName) :
        print "Error: %s does not exist"%(fileName)
        sys.exit(2)
    try :
        data = np.genfromtxt(fileName)
    except IOError :
        print "Error: Failed to import data from %s"%fileName
    return data

def write_fit_data(data,popt, opts) :
    basename = os.path.splitext(opts.inputFileName)[0]
    fileName = basename + ".out"

    numpeaks = len(popt) / 3
    #xs = np.linspace(np.min(data[:,0]),np.max(data[:,0]),500) 
    xs = data[:,0]
    ys = data[:,1]

    func = matchFunctionName(len(popt)/ 3)
    combinedFit = func(xs,*popt)
    fits = []

    avg, std = weighted_avg_and_std(data[:,0],data[:,1])
    fwhm = full_width_half_max(data[:,0], data[:,1], opts)

    with open(fileName,'w') as f :
        f.write('#\n')
        f.write("# Mean vibrational frequency: %.3f +/- %.3f. FWHM: %.3f \n"%(avg, std, fwhm) )
        f.write('# Fit to %i gaussians:\n'%numpeaks)
        f.write('# fit(x) = a*e^(-(x-b)^2 / (2c^2))\n')
        f.write('#\n')
        for n in range(numpeaks) :
            a, b, c = popt[n*3:n*3+3]
            C = popt[-1]
            fits.append( single_gaussian(xs, a,b,c, C))
            f.write('#Gaussian #%i: a = %5.3f\tb = %5.3f\tc = %5.3f\n'%(n+1,a,b,c))
        f.write('#\n')

        fits = np.array(fits)
        f.write("#%7s\t%8s\t%8s"%("x-axis:", "Spectrum:","Fit:") )
        for i in range(numpeaks) :
            f.write("\t%6s%2i"%("Gauss:",i))
        f.write('\n')

        for i in range(len(xs)) :
            f.write("%8.3f\t%8.3f\t%8.3f"%(xs[i], ys[i], combinedFit[i]) )
            for n in range(numpeaks) :
                f.write("\t%8.3f"%fits[n,i])
            f.write('\n')

    xs = np.linspace(np.min(data[:,0]),np.max(data[:,1]),500)
    fit = func(xs,*popt)
    return 0

def plot_fits(data,popt, opts) :
    basename = os.path.splitext(opts.inputFileName)[0]
    filename = basename + ".fit.png"

    fig, ax = plt.subplots(1)

    xs = np.linspace(np.min(data[:,0]),np.max(data[:,0]),500)
    func = matchFunctionName(len(popt )/ 3)
    fit = func(xs, *popt)

    ax.scatter(data[:,0], data[:,1],marker='o',s=5,color='r',zorder=5)
    ax.plot(xs,fit,'k-',linewidth=4,zorder=1)

    if len(popt) / 3 > 1 :
        for i in range(len(popt)/3) :
            params = popt[i*3:i*3+3]
            params = np.append(params, popt[-1])
            fiti = single_gaussian(xs, *params)
            ax.plot(xs,fiti,'b--',linewidth=2,zorder=3)

    fig.savefig(filename,format='png')
    return 0

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))

def full_width_half_max(x, y, opts) : 
    if opts.debug : print "Entering full_width_half_max"

    peakIndex = np.argmax(y) 

    if opts.debug : print "Peak height = %.3f" %y[peakIndex]
    if opts.debug : print "Peak location = %.3f" %x[peakIndex]

    minIndex = find_nearest(y[:peakIndex],y[peakIndex] * 0.5, opts)   #half peak from below peak
    maxIndex = find_nearest(y[peakIndex:],y[peakIndex] * 0.5, opts) + peakIndex  #half peak from above peak

    if opts.debug : print "Half peak min = %.3f at %.3f" %(y[minIndex], x[minIndex]) 
    if opts.debug : print "Half peak max = %.3f at %.3f" %(y[maxIndex], x[maxIndex]) 

    if opts.debug : print "Half peaks: %.3f  %.3f"%(x[minIndex], x[maxIndex]) 

    return np.abs(x[maxIndex] - x[minIndex]) #width at half peak 


