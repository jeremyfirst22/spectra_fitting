#!/usr/bin/env python 

import matplotlib.pyplot as plt  
import numpy as np 
from scipy.optimize import curve_fit 
import os 
from os import sys
import getopt

from gaussian_funcs import * 

def usage() :
    ##TODO: Need to actually write a help function
    print "Help me"

###
#  Class for holding all command line options used for program. 
###
class optStruct : 
    #### 
    ## Initialize default command line options 
    #### 
    overwrite = False               ##Bool: Permission to write to files that already exist. 
    verbose = False                 ##Bool: Print information about fits and parameters. 
    doPlot = False                  ##Bool: Plot fits and components
    doOutfile = False               ##Bool: Write fits and components to text file
    #plotFileName = "/dev/null"      ##String: PNG file name to save plot as 
    #outputFileName = "/dev/null"    ##String: txt file name to save fits as. "*.out" recommended. 
    inputFileName = "/dev/null"     ##String: txt file from which to read data. Data must be two-column format. 
    numPeaks = 1                    ##Int:    Number of gaussian components to fit to input data
    doNormalize = False             ##Bool: Flag to normalize the data by peak height to 1. 
    doFullAnalysis = False          ##Bool: Run a series of fits with 1->8 components. Plot residuals for comparison. 
    debug = False                   ##Bool: Print an annoying amount of information for debugging purposes. 
    doBaseLineCorrect = False       ##Bool: Do base line correction. (Inteded to correct for water peak)

def parse_command_line_opts(argv) :
    myOpts = optStruct   ##Struct to hold command line options to be passed back to main(). 

    ##TODO: Perhaps switch to argparse library for parsing command line options. 
    ##      Seems to support optional arguments, whereas getopt does not. 
    try : 
        opts, args = getopt.getopt(sys.argv[1:], "hvdi:on:p", ["help", "verbose", "debug","input=", "output", "numPeaks=","plot", "overwrite", "normalize", "full-analysis", "baseline"] )   
    except getopt.GetoptError as err : 
        print str(err) 
        usage() 
        sys.exit(2) 
    ###
    #  Parse command line options
    ###
    for o, a in opts :
        if o in ('-v','--verbose') : 
            myOpts.verbose = True
        elif o in ('-d','--debug') : 
            myOpts.debug = True
            myOpts.verbose = True 
        elif o in ("-h", "--help") : 
            usage() 
            sys.exit() 
        elif o in ("-i", "--input") : 
            if not os.path.isfile(a) : 
                print "ERROR: Input file \'%s\' not found."%a 
                usage 
                sys.exit(2) 
            myOpts.inputFileName = a 
        elif o in ("-o", "--output") : 
            myOpts.doOutfile = True 
        elif o in ("-n", "--numPeaks") : 
            try : 
                myOpts.numPeaks = int(a) 
            except ValueError : 
                print "ERROR: %s argument accepts an integer value. "%o
                print "\tYou supplied %s"%(a)
                sys.exit(2)
            if myOpts.numPeaks not in range(8+1) : 
                print "ERROR: Only up to 8 gaussians so far are supported. "
                print "\tYou requested %i"%myOpts.numPeaks
                sys.exit(2) 
        elif o in ("-p", "--plot"): 
            myOpts.doPlot = True 
        elif o == "--overwrite" : 
            myOpts.overwrite = True
        elif o == "--normalize" : 
            myOpts.doNormalize = True
        elif o == "--full-analysis" : 
            myOpts.doFullAnalysis = True
        elif o == "--baseline" : 
            myOpts.doBaseLineCorrect = True
        else : 
            print "ERROR: Unrecognized option %s"%o 
            sys.exit(2) 

    ###
    #  Check command line options to make sure they are reasonable
    ###
    if myOpts.inputFileName == "/dev/null" : 
        print "ERROR: Input file option required."
        usage() 
        sys.exit(2) 
    if myOpts.doOutfile : 
        basename = os.path.splitext(opts.inputFileName)[0]
        outputFileName = basename + ".out" 
        if not myOpts.overwrite and os.path.isfile(myOpts.outputFileName) :
            print "ERROR: Output file \'%s\' exists already, and permission to overwrite not given."%myOpts.outputFileName
            print "Please delete the file, or give the \'--overwrite\' option" 
            sys.exit(2) 
        if outputFileName == myOpts.inputFileName : 
            print "ERROR: Cowardly refusing to overwrite %s with output data. Check file names."%myOpts.inputFileName
            sys.exit(2)

    ###
    #  Print command line options if verbose
    ### 
    if myOpts.verbose  : 
        print "verbose                   = %r"%myOpts.verbose 
        print "overwrite                 = %s"%myOpts.overwrite 
        print "input file                = %s"%myOpts.inputFileName 
        #print "output file               = %s"%myOpts.outputFileName 
        #print "plot file                 = %s"%myOpts.plotFileName 
        print "number of gaussians to fit= %i"%myOpts.numPeaks 

    return myOpts

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

def read_data(fileName) :
    if not os.path.isfile(fileName) :
        print "Error: %s does not exist"%(fileName)
        sys.exit()
    try :
        data = np.genfromtxt(fileName)
    except :
        print "Error: Failed to import data from %s"%fileName
    return data

def fit_data(data,i,debug) :
    if debug : print "Entering fit_data." 
    x = data[:,0] 
    y = data[:,1] 
    assert len(x) == len(y) 

    bounds = find_reasonable_bounds(data,i) 

    func = matchFunctionName(i) 
    if debug : print "\t", func

    popt, pcov = curve_fit(func, x,y   ,bounds=bounds,maxfev = 100000  ) 
    return popt, pcov

def normalize(data) : 
    xs = data[:,0] 
    ys = data[:,1] 
    ys /= np.max(ys) 
    return np.array([xs,ys]).T


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
        for i in range(len(popt[::3])) : 
            params = popt[i*3:i*3+3]
            fiti = single_gaussian(xs, *params)
            ax.plot(xs,fiti,'b--',linewidth=2,zorder=3) 
    
    fig.savefig(filename,format='png') 
    return 0

def write_fit_data(data,popt,fileName) : 
    numpeaks = len(popt) / 3 
    xs = np.linspace(np.min(data[:,0]),np.max(data[:,0]),500) 
    func = matchFunctionName(len(popt)/ 3) 
    combinedFit = func(xs,*popt) 
    fits = [] 

    with open(fileName,'w') as f :
        f.write('#\n') 
        f.write('# Fit to %i gaussians:\n'%numpeaks) 
        f.write('# fit(x) = a*e^(-(x-b)^2 / (2c^2))\n') 
        f.write('#\n') 
        for n in range(numpeaks) :
            a, b, c = popt[n*3:n*3+3] 
            fits.append( single_gaussian(xs, a,b,c)) 
            f.write('#Gaussian #%i: a = %5.3f\tb = %5.3f\tc = %5.3f\n'%(n+1,a,b,c)) 
        f.write('#\n') 

        fits = np.array(fits) 
        f.write("#%7s\t%8s"%("x-axis:", "Fit:") ) 
        for i in range(numpeaks) : 
            f.write("\t%6s%2i"%("Gauss:",i))  
        f.write('\n') 

        for i in range(len(xs)) : 
            f.write("%8.3f\t%8.3f"%(xs[i], combinedFit[i]) ) 
            for n in range(numpeaks) : 
                f.write("\t%8.3f"%fits[n,i]) 
            f.write('\n') 

    xs = np.linspace(np.min(data[:,0]),np.max(data[:,1]),500) 
    fit = func(xs,*popt) 
    return 0

def closest_minima(arr , center) : 
    assert 0 < center and center < len(arr) 

    lowerMin = center 
    for i in range(center-1, 0, -1) : 
        if arr[i] < arr[lowerMin] : 
            lowerMin = i 
        else : break

    upperMin = center 
    for i in range(center+1, len(arr)) : 
        if arr[i] < arr[upperMin] : 
            upperMin = i 
        else : break
    
    return lowerMin, upperMin


def baseline_correct(data,opts) : 
    if opts.verbose : print "Entering base line correction function" 

    x,y = data[:,0],data[:,1]


    ## Fit 5th order polynomial 
    guessOrder = 5 

    z = np.polyfit(x,y,guessOrder)
    fit = np.poly1d(z) 
    if opts.verbose : 
        print z 

    cutFactor = 3
    cutsOkay = False 

    for cutFactor in range(3, 0, -1) : 
        residuals = np.sqrt((y - fit(x))**2) 
        peak = np.argmax(residuals) 
        lowMin, highMin = closest_minima(residuals, peak) 
        cutMin = peak - cutFactor*(peak - lowMin) 
        cutMax = peak + cutFactor*(highMin - peak) 
        if opts.verbose : 
            print "Min. of signal: %.2f"%x[lowMin]
            print "Max. of signal: %.2f"%x[highMin]

        if not (peak - cutMin) < cutMin and (cutMax - peak) < (len(x) - cutMax) : 
            print (peak-cutMin) 
            print cutMin
            print (cutMax - peak) 
            print (len(x) - cutMax) 
            print "WARNING: Not enough data to fit with factor of %i times nearest minima. "%cutFactor
            print "\tReducing factor and trying again" 
        else : 
            cutsOkay = True 
            break 
    if not cutsOkay : 
        print "ERROR: Unable to find a fit factor small enough to fit data. Try using data with larger bounds" 
        sys.exit() 


    if opts.doPlot : 
        basename = os.path.splitext(opts.inputFileName)[0]
        findFitFile = basename + ".find_baseline.png"

        figFindFit, axFindFit = plt.subplots(1) 

        axFindFit.plot(x,np.sqrt((y - fit(x))**2), label="Residuals of %i order polynomial fit"%(guessOrder)) 
        axFindFit.axvline(x[lowMin], linestyle='--',color='k', label="Closest min of residuals") 
        axFindFit.axvline(x[highMin], linestyle='--',color='k') 
        axFindFit.axvline(x[peak], linestyle='--',color='b',label="Max of residual, should be peak") 
        axFindFit.axvline(x[cutMin], linestyle='--',color='r', label="%i x min of residual"%(cutFactor))  
        axFindFit.axvline(x[cutMax], linestyle='--',color='r') 

        axFindFit.legend(loc=1,fontsize='xx-small') 

        figFindFit.savefig(findFitFile, format='png') 

    cutX = np.delete(x, np.s_[cutMin:cutMax]) 
    cutY = np.delete(y, np.s_[cutMin:cutMax]) 


    fitOrder = 5 
    cutZ = np.polyfit(cutX,cutY,fitOrder)
    cutFit = np.poly1d(cutZ) 

    if opts.doPlot : 
        basename = os.path.splitext(opts.inputFileName)[0]
        fitFile = basename + ".baseline_correction.png"

        figFit , axFit = plt.subplots(1) 

        axFit.scatter(x,y,s=1,label="Raw spectra")
        axFit.plot(x,fit(x), label="%i order guess fit"%guessOrder) 
        axFit.plot(x,cutFit(x),'g--',label="%i order fit to data outside red dashed"%fitOrder) 

        #axFit.axvline(x[lowMin], linestyle='--',color='k', label="Closest min of residuals") 
        #axFit.axvline(x[highMin], linestyle='--',color='k') 
        axFit.axvline(x[peak], linestyle='--',color='b',label="Peak found by max of guess residuals") 

        axFit.axvline(x[cutMin], linestyle='--',color='r',label="Bounds of fitting region") 
        axFit.axvline(x[cutMax], linestyle='--',color='r') 

        axFit.legend(loc=1,fontsize='xx-small') 

        figFit.savefig(fitFile,format='png') 

    corrected = data 
    corrected[:,1] -= cutFit(corrected[:,0]) 

    return corrected

def baseline_correct_minimization(data,opts) : 
    if opts.debug : print "Entering base line correction function" 

    tolerance = 1e-7
    goodFitResidual = 1e-5 

    x,y = data[:,0],data[:,1]

    ## Fit 5th order polynomial 
    guessOrder = 5 

    z, _, _, _, _ = np.polyfit(x,y,guessOrder, full=True)
    fit = np.poly1d(z) 
    if opts.debug : 
        print "Parameters for guess polynomial: ", z 


    residuals = np.sqrt((y - fit(x))**2) 
    peak = np.argmax(residuals) 

    maxSignalRatio = 0.5
    maxIterations = peak * maxSignalRatio
    if (len(data) - peak)*maxSignalRatio < maxIterations : maxIterations = (len(data) - peak)*maxSignalRatio
    if opts.debug : print "Max iterations are %i" %maxIterations 

    i = 0 
    lastResidual = 1e10
    cutsOkay = False 
    while not cutsOkay and i < maxIterations :  
        i += 1 
        if opts.debug : print "Fitting with %i window"%i , 

        cutMin = peak - i 
        cutMax = peak + i 

        cutX = np.delete(x, np.s_[cutMin:cutMax]) 
        cutY = np.delete(y, np.s_[cutMin:cutMax]) 

        fitOrder = 4
        cutZ, _ ,  rank, _, _  = np.polyfit(cutX,cutY,fitOrder,full=True)

        cutFit = np.poly1d(cutZ) 
        residuals = (np.sum((cutY - cutFit(cutX))**2) )

        if opts.debug : 
            if not peak == np.argmax(np.sqrt((y - cutFit(x))**2)) : print "Updating peak!"
        peak = np.argmax(np.sqrt((y - cutFit(x))**2)) 


        if opts.debug : 
            print "Residuals: %.15f\tRank: %i"%(residuals , rank)  , 

        if opts.debug : print "Residual diff: %15.8f"%(lastResidual - residuals) 

        if lastResidual < residuals : 
            print "ERROR: Baseline fit diverging. Try using a larger range spectrum."
            sys.exit() 

        if lastResidual - residuals < tolerance  : 
            cutsOkay = True 
            if residuals > goodFitResidual : print "WARNING: Baseline fit may not be a good fit. Check fitting files" 
        else : lastResidual = residuals 

    if not cutsOkay : 
        print "WARNING: Acceptable baseline not found within maxIterations. Try using a larger range of spectra"
        print "\tUse baseline fitted spectra with caution" 

    if opts.verbose : 
        print "Baseline Cutting bounds: ", x[cutMin], "\t", x[cutMax]
        print "Baseline polynomial coefficients: ", cutZ

    cutFit = np.poly1d(cutZ) 

    if opts.doPlot : 
        fig, ax = plt.subplots(1,1) 

        basename = os.path.splitext(opts.inputFileName)[0]
        fitFile = basename + ".baseline_correction.png"

        ax.scatter(x,y,s=1,label="Raw data") 
        ax.axvline(x[peak], linestyle='--', color='k', label="Guess for highest signal") 
        ax.axvline(x[cutMin], linestyle='--',color='r', label="bounds of cut") 
        ax.axvline(x[cutMax], linestyle='--',color='r') 
        ax.plot(x, cutFit(x), 'b--', label="%i order polynomial fit"%fitOrder) 

    corrected = data 
    corrected[:,1] -= cutFit(corrected[:,0]) 

    if opts.doPlot : 
        ax.plot(corrected[:,0], corrected[:,1],'k',label="Baseline corrected") 

        ax.legend(loc=2,fontsize='xx-small') 
        fig.savefig(fitFile,format='png') 

    return corrected


def main(argv) : 
    myOpts = parse_command_line_opts(argv) 

    data = read_data(myOpts.inputFileName) 

    if myOpts.doBaseLineCorrect : 
        data = baseline_correct_minimization(data,myOpts) 
        #data = baseline_correct(data,myOpts) 

    if myOpts.doNormalize : 
        data = normalize(data) 

    popt, pcov = fit_data(data,myOpts.numPeaks,myOpts.debug) 
    #if myOpts.verbose : 
    for i in range(myOpts.numPeaks) : 
        print "Component %i: a = %5.3f\tb = %5.3f\tc = %5.3f"\
                %(i+1,popt[i*3],popt[i*3+1],popt[i*3+2]) 

    if myOpts.doPlot : 
        plot_fits(data,popt, myOpts) 
    
    if myOpts.doOutfile : 
        write_fit_data(data,popt,myOpts.outputFileName)   

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


