#!/usr/bin/env python 

import matplotlib.pyplot as plt  
import numpy as np 
#from scipy.optimize import curve_fit 
#from scipy import interpolate
#from scipy.interpolate import UnivariateSpline
#from scipy.interpolate import interp1d
#from scipy.spatial import ConvexHull
import os 
from os import sys
from scipy.signal import savgol_filter
import argparse

from gaussian_funcs import * 
from io_funcs import * 
from spectra_transformations import * 

def parse_command_line_opts(argv) : 
    parser = argparse.ArgumentParser(
            description="Fit spectrum to Gaussian.", 
            epilog="Example: Provide a working example" , 
            ) 

    parser.add_argument('inputFileName', help="Name of input file. Typically text file of spectrum in two columns (Wavelength & Absorbance)." ) 
    parser.add_argument('--verbose', action='store_true', help="Print additional information") 
    parser.add_argument('--debug', action='store_true', help="Print everything debugging") 
    parser.add_argument('--overwrite', action='store_true', help="Force overwriting of an already existing outfile") 
    parser.add_argument('-n', '--numPeaks', dest='numPeaks', type=int, choices=range(1,9), default=1, help="Number of gaussian components to fit to spectrum") 
    parser.add_argument('--full-analysis', dest='doFullAnalysis', action='store_true', help="Fit spectrum to all allowed number of gaussians. Plots residuals of fits to file") 
    parser.add_argument('-p', '--plot', dest='doPlot', action='store_true', help="Plot baseline correction and gaussian fit to file") 
    parser.add_argument('-c', '--cut',dest='cuts', type=int, nargs=2, metavar=('min', 'max'), help="Cut spectrum before baseline correction and fitting to provided 'min' and 'max'")  
    parser.add_argument('-b', '--baseline', dest='baseline', type=str, choices=['rubberband', 'spline-fit'], help="Baseline correct spectrum using specified method") 
    parser.add_argument('-s', '--smooth', dest='doSmooth', action='store_true') 
    parser.add_argument('--smooth-factor', dest='smoothFactor', action='store', default=1, type=int, help="Window (in number of data points) to smooth. Must be >2.") 
    parser.add_argument('-m', '--normalize', dest='doNormalize', action='store_true', help="Normalize maximum of spectrum to 1") 
    parser.add_argument('-o', '--outfile', dest='doOutfile', action='store_true', help="Write corrected spectrum, fit, and gaussian components to file") 
    parser.add_argument('-e', '--peak-edge',dest='peak', type=float, nargs=2, metavar=('minPeak', 'maxPeak'), help="Provide edge of peaks for baseline fit. Inside this range is considered not the baseline") 

    args = parser.parse_args() 
    args.doBaseLineCorrect = False 
    args.doCut = False 
    args.guessPeak = False
    if args.cuts is not None : 
        args.doCut = True 
    if args.peak is not None : 
        args.guessPeak = True 
    if args.baseline is not None : 
        args.doBaseLineCorrect = True 
    if args.debug : 
        args.verbose = True 

    if args.verbose : 
        parser.print_help() 
    if args.debug : 
        print "Command line arguments:" , args 

    if args.doOutfile and os.path.isfile(os.path.splitext(args.inputFileName)[0] + ".out") \
            and not args.overwrite : 
        print "ERROR: %s file already exists. Cowardly refusing to overwrite data. Use '--overwrite' flag to override" %(os.path.splitext(args.inputFileName)[0] + ".out") 
        sys.exit() 


    return args 

def main(argv) : 
    myOpts = parse_command_line_opts(argv) 

    data = read_data(myOpts.inputFileName) 

    if myOpts.doCut : 
        data = cut_peak(data, min(myOpts.cuts) , max(myOpts.cuts), myOpts) 

    if myOpts.doSmooth : 
        data = smooth_data(data, myOpts) 

    if myOpts.doBaseLineCorrect : 
        if myOpts.baseline == "rubberband" : 
            baseline = rubberband(data[:,0], data[:,1], myOpts) 
        elif myOpts.baseline == "spline-fit" : 
            baseline = spline_fitting(data[:,0],data[:,1], myOpts) 
        else : 
            print "ERROR: Baseline fitting not recognized. Argument for '--baseline' must be either 'rubberband' or 'spline-fit'"
            print "\t\tbaseline = %s"%myOpts.baseline 
            sys.exit(2) 
        data[:,1] -=  baseline

    if myOpts.doNormalize : 
        data = normalize(data) 

    avg, std = weighted_avg_and_std(data[:,0],data[:,1]) 
    fwhm = full_width_half_max(data[:,0], data[:,1], myOpts) 

    print "Mean vibrational frequency: %5.3f +/- %.3f. FWHM: %.3f"%(avg, std,fwhm) 

    popt, pcov = fit_data(data,myOpts.numPeaks,myOpts.debug) 

    for i in range(myOpts.numPeaks) : 
        print "Component %i: a = %5.3f\tb = %5.3f\tc = %5.3f"\
                %(i+1,popt[i*3],popt[i*3+1], 2.35482*popt[i*3+2]) 

    if myOpts.doPlot : 
        plot_fits(data,popt, myOpts) 
    
    if myOpts.doOutfile : 
        write_fit_data(data,popt, myOpts)

    if myOpts.doFullAnalysis : 
        if myOpts.verbose : print "Starting Full analysis"
        data = normalize(data) 

        f1, ax1 = plt.subplots(1,1) 
        f2, ax2 = plt.subplots(1,1) 
        for i in range(1,8+1) : 
            if myOpts.verbose : print "Calculating residuals with %i gaussians"%i
            func = matchFunctionName(i) 
            if myOpts.debug : print "\t",func

            popt, pcov = fit_data(data, i,myOpts.debug) 
            if myOpts.debug : print i, len(popt) 

            resid = data[:,1] - func(data[:,0],*popt) 
            ax2.scatter(i, np.sqrt(np.sum(resid**2) / len(resid)),color='k' ) 

            ax1.plot(data[:,0],resid,label="%i gaussians"%i) 

        ax1.legend(loc=2,fontsize='xx-small') 
        ax2.set_yscale('log') 
        basename = os.path.splitext(myOpts.inputFileName)[0]
        f1.savefig(basename+'.residuals.png',format='png') 
        f2.savefig(basename+'.sumofsquares.png',format='png') 

if __name__ == "__main__":
   main(sys.argv[1:])


