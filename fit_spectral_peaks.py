#!/usr/bin/env python 

import matplotlib.pyplot as plt  
import numpy as np 
from scipy.optimize import curve_fit 
from scipy import interpolate
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
import os 
from os import sys
from scipy.signal import savgol_filter
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
    inputFileName = "/dev/null"     ##String: txt file from which to read data. Data must be two-column format. 
    numPeaks = 1                    ##Int:    Number of gaussian components to fit to input data
    doNormalize = False             ##Bool: Flag to normalize the data by peak height to 1. 
    doFullAnalysis = False          ##Bool: Run a series of fits with 1->8 components. Plot residuals for comparison. 
    debug = False                   ##Bool: Print an annoying amount of information for debugging purposes. 
    doBaseLineCorrect = False       ##Bool: Do baseline correction. (Inteded to correct for water peak)
    doCut = False 
    doSmooth = False 
    baseline = "False"

def parse_command_line_opts(argv) :
    myOpts = optStruct   ##Struct to hold command line options to be passed back to main(). 

    ##TODO: Perhaps switch to argparse library for parsing command line options. 
    ##      Seems to support optional arguments, whereas getopt does not. 
    try : 
        opts, args = getopt.getopt(sys.argv[1:], "hvdi:on:pmfb:cs", ["help", "verbose", "debug","input=", "output", "numPeaks=","plot", "overwrite", "normalize", "full-analysis", "baseline=", "--cut", "--smooth" ] )   
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
        elif o in ("-m", "--normalize") : 
            myOpts.doNormalize = True
        elif o == "--full-analysis" : 
            myOpts.doFullAnalysis = True
        elif o in ("-c", "--cut") : 
            myOpts.doCut = True
        elif o in ("-s", "--smooth") : 
            myOpts.doSmooth = True
        elif o in ("-b", "--baseline") : 
            myOpts.doBaseLineCorrect = True
            myOpts.baseline = a 
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
        basename = os.path.splitext(myOpts.inputFileName)[0]
        outputFileName = basename + ".out" 
        if not myOpts.overwrite and os.path.isfile(outputFileName) :
            print "ERROR: Output file \'%s\' exists already, and permission to overwrite not given."%outputFileName
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
        print "number of gaussians to fit= %i"%myOpts.numPeaks 
        print "doBaseline                = %r"%myOpts.doBaseLineCorrect
        print "baseline type             = %r"%myOpts.baseline 

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
        sys.exit(2)
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
        for i in range(len(popt)/3) : 
            params = popt[i*3:i*3+3]
            params = np.append(params, popt[-1])
            fiti = single_gaussian(xs, *params)
            ax.plot(xs,fiti,'b--',linewidth=2,zorder=3) 
    
    fig.savefig(filename,format='png') 
    return 0

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

    with open(fileName,'w') as f :
        f.write('#\n') 
        f.write("# Mean vibrational frequency: %.3f +/- %.3f \n"%(avg, std) ) 
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

def baseline_correct_minimization(data,opts) : 
    if opts.debug : print "Entering base line correction function" 

    tolerance = 1e-5 
    goodFitResidual = 1e-5 

    x,y = data[:,0],data[:,1]
    print len(x) 

    ## Fit 5th order polynomial 
    maxSignalRatio = 0.65
    guessOrder = 4 
    fitOrder = 4

    z, _, _, _, _ = np.polyfit(x,y,guessOrder, full=True)
    fit = np.poly1d(z) 
    if opts.debug : 
        print "Parameters for guess polynomial: ", z 

    residuals = np.sqrt((y - fit(x))**2) 
    peak = np.argmax(residuals) 

    if opts.doPlot : 
        fig, axarr = plt.subplots(2,1) 
        ax1 = axarr[0]
        ax2 = axarr[1]

        basename = os.path.splitext(opts.inputFileName)[0]
        guessFile = basename + ".guess_signal.png"

        ax1.scatter(x,y,s=1) 
        ax1.plot(x,fit(x)) 

        ax2.plot(x,residuals) 
        ax2.axvline(x[peak]) 

        fig.savefig(guessFile) 

    maxIterations = peak * maxSignalRatio
    print maxIterations, (len(data) - peak)*maxSignalRatio 
    if (len(data) - peak)*maxSignalRatio < maxIterations : maxIterations = (len(data) - peak)*maxSignalRatio
    if opts.debug : print "Max iterations are %i" %maxIterations 

    i = 0 
    lastResidual = 1e10
    cutsOkay = False 
    plt.show() 
    while not cutsOkay and i < maxIterations :  
        i += 1 
        if opts.debug : print "Fitting with %i window"%i , 

        cutMin = peak - i 
        cutMax = peak + i 

        cutX = np.delete(x, np.s_[cutMin:cutMax]) 
        cutY = np.delete(y, np.s_[cutMin:cutMax]) 

        cutZ, _ ,  rank, _, _  = np.polyfit(cutX,cutY,fitOrder,full=True)
        cutFit = np.poly1d(cutZ) 

        #cutFit = interpolate.InterpolatedUnivariateSpline(cutX, cutY)
        #cutFit = UnivariateSpline(cutX, cutY)
        #cutFit.set_smoothing_factor(0.00001) 

        #plt.close() 
        #plt.scatter(x,y,s=1) 
        #plt.axvline(x[cutMin], linestyle='--',color='r', label="bounds of cut") 
        #plt.axvline(x[cutMax], linestyle='--',color='r') 
        #plt.plot(x,cutFit(x)) 
        #plt.show() 

        residuals = (np.sum((cutY - cutFit(cutX))**2) )
        plt.scatter(i, np.log10(residuals) ) 

        if opts.debug : 
            if not peak == np.argmax(np.sqrt((y - cutFit(x))**2)) : print "Updating peak!"
        #peak = np.argmax(np.sqrt((y - cutFit(x))**2)) 


        if opts.debug : 
            print "Residuals: %.15f\tRank: %i"%(residuals , rank)  , 

        if opts.debug : print "Residual diff: %15.8f"%(lastResidual - residuals) 

        if lastResidual < residuals : 
            print "ERROR: Baseline fit diverging. Try using a larger range spectrum."
            sys.exit(2) 

        if residuals < tolerance : #lastResidual - residuals < tolerance  : 
            #cutsOkay = True 
            if residuals > goodFitResidual : print "WARNING: Baseline fit may not be a good fit. Check fitting files" 
        else : lastResidual = residuals 
    plt.axhline(np.log10(tolerance)) 
    print np.log10(tolerance) 
    plt.show() 

    if not cutsOkay : 
        print "WARNING: Acceptable baseline not found within maxIterations. Try using a larger range of spectra"
        print "\tUse baseline fitted spectra with caution" 

    if opts.verbose : 
        print "cutMin: ", cutMin, "\tcutMax: ", cutMax, "\tLength of x: ", len(x) 
        print "Baseline Cutting bounds: ", x[cutMin], "\t", x[cutMax]
        print "Baseline polynomial coefficients: ", cutZ

    cutFit = np.poly1d(cutZ) 

    if opts.doPlot : 
        fig, axarr = plt.subplots(2,1) 
        ax1 = axarr[0] 
        ax2 = axarr[1] 

        basename = os.path.splitext(opts.inputFileName)[0]
        fitFile = basename + ".baseline_correction.png"

        ax1.scatter(x,y,s=1,label="Raw data") 
        ax1.axvline(x[peak], linestyle='--', color='k', label="Guess for highest signal") 
        ax1.axvline(x[cutMin], linestyle='--',color='r', label="bounds of cut") 
        ax1.axvline(x[cutMax], linestyle='--',color='r') 
        ax1.plot(x, cutFit(x), 'b--', label="%i order polynomial fit"%fitOrder) 

    corrected = data 
    corrected[:,1] -= cutFit(corrected[:,0]) 

    if opts.doPlot : 
        ax2.axvline(x[peak], linestyle='--', color='k', label="Guess for highest signal") 
        ax2.axvline(x[cutMin], linestyle='--',color='r', label="bounds of cut") 
        ax2.axvline(x[cutMax], linestyle='--',color='r') 
        ax2.plot(corrected[:,0], corrected[:,1],'k',label="Baseline corrected") 

        fig.legend(loc=2,fontsize='xx-small') 
        fig.savefig(fitFile,format='png') 

    return corrected

def spline_fitting(x,y,opts) : 
    tolerance = 1e-6

    guessSpl = UnivariateSpline(x, y)

    #plt.plot(x, guessSpl(x), 'k--') 

    peak = np.argmax((guessSpl(x) - y )**2) 
    if x[peak] > 2170 or x[peak] < 2130 : 
        print "WARNING: Guess of peak outside of 2130-2170 cm^-1. Double check fitting file and be sure we have found the correct peak" 

    maxSignalRatio = 0.50 
    maxwidth = int(round((len(x) - peak) * maxSignalRatio))   ##distance to maximum signal
    if maxwidth > peak * maxSignalRatio : 
        maxwidth = int(round(peak * maxSignalRatio))         ##distance to minimum signal 

    cutMin = peak - maxwidth
    cutMax = peak + maxwidth 

    cutX = np.delete(x, np.s_[cutMin:cutMax]) 
    cutY = np.delete(y, np.s_[cutMin:cutMax]) 

    if opts.debug : print "Max width of peak = %i" %maxwidth

    spl = UnivariateSpline(cutX,cutY) #,s= 0.1) 
    if opts.debug : 
        residual = np.sum((spl(x) - y)**2)
        print "Residual of baseline = %f"%residual 

    if opts.doPlot : 
        fig, axarr = plt.subplots(2,1) 
        axarr[0].scatter(x,y, s = 1) 
        axarr[0].axvline(x[peak], linestyle='--', color='k', label="Guess of peak position") 
    #    axarr[0].axvline(x[cutMin], linestyle='--',color='r', label="bounds of cut") 
    #    axarr[0].axvline(x[cutMax], linestyle='--',color='r') 
        axarr[0].plot(x,spl(x),'k--', label="Baseline spline fit"  ) 

        axarr[1].plot(x, y - spl(x))  

        basename = os.path.splitext(opts.inputFileName)[0]
        fig.savefig(basename+'.spline_fitting.pdf', format='pdf') 

    return spl(x) 

def rubberband(x, y, opts):
    # Find the convex hull
    if opts.verbose : "Print using rubberband baseline correction method" 

    v = ConvexHull(np.array(zip(x, y))).vertices
    if opts.debug : 
        print "Vertices of Convex Hull: "
        for point in v : 
            print "(%5i, %5i)"%(x[point], y[point]) 

    # Rotate convex hull vertices until they start from the lowest one
    v = np.roll(v, -v.argmin())

    if opts.doPlot : 
        fig, axarr = plt.subplots(2,1) #,figsize=(6,12)) 
        axarr[0].scatter(x,y, s = 1) 
        for point in v : 
            axarr[0].scatter(x[point],y[point],s=15,color='r')

    # Leave only the ascending part
    if opts.debug: print v 
    v = v[:v.argmax()+1]

    if opts.doPlot : 
        for point in v : 
            axarr[0].scatter(x[point],y[point],s=25,color='b')
        axarr[0].plot(x,np.interp(x, x[v], y[v]) , 'k--', label="Convex hull (rubberband) fit")  

#        axarr[1].plot(x, y - np.interp(x, x[v], y[v])) 
        spl = UnivariateSpline(x[v], y[v])
        axarr[1].plot(x, y - spl(x) ) 

        basename = os.path.splitext(opts.inputFileName)[0]
        fig.savefig(basename+'.rubberband_fitting.pdf', format='pdf') 

    ##Spline fit to vertices
    spl = UnivariateSpline(x[v], y[v])
    return spl(x)


    # Create baseline using linear interpolation between vertices
    #return np.interp(x, x[v], y[v])

def find_nearest(array, value):
    array = np.asarray(array)
    idx = np.argmin(np.abs(array - value))
    #print value, array[idx] 
    return idx

def cut_peak(data, minX, maxX, opts) : 
    if opts.debug : print "Now entering cut_peak function" 
    x,y = data[:,0], data[:,1]

    cutMin = find_nearest(x, minX) 
    cutMax = find_nearest(x, maxX) 
    
    if opts.debug : print "Cutting from %.2f to %.2f" %(cunMin, cutMax) 

    cutX = x[cutMin:cutMax]
    cutY = y[cutMin:cutMax]

    data = np.array([cutX, cutY]).T

    return data 

def smooth_data(data, opts) : 
    smoothed = savgol_filter(data[:,1], 5, 2)
    data[:,1] = smoothed 
    return data

def weighted_avg_and_std(values, weights):
    """ 
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))

def main(argv) : 
    myOpts = parse_command_line_opts(argv) 

    data = read_data(myOpts.inputFileName) 

    if myOpts.doCut : 
        data = cut_peak(data, 2145, 2180, myOpts) 

    #plt.close() 
    #plt.plot(data[:,0], data[:,1]) 
    #plt.show() 

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

    #plt.close() 
    #plt.plot(data[:,0], data[:,1]) 
    #plt.show() 

    if myOpts.doSmooth : 
        data = smooth_data(data, myOpts) 

    if myOpts.doNormalize : 
        data = normalize(data) 

    if myOpts.doCut : 
        data = cut_peak(data, 2140, 2180, myOpts) 

    avg, std = weighted_avg_and_std(data[:,0],data[:,1]) 

    print "Mean vibrational frequency: %5.3f +/- %.3f"%(avg, std) 

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


