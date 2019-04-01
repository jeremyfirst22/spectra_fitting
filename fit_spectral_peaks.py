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
    plotFileName = "/dev/null"      ##String: PNG file name to save plot as 
    outputFileName = "/dev/null"    ##String: txt file name to save fits as. "*.out" recommended. 
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
        opts, args = getopt.getopt(sys.argv[1:], "hvdi:o:n:p:", ["help", "verbose", "debug","input=", "output=", "numPeaks=","plotfile=", "overwrite", "normalize", "full-analysis", "baseline"] )   
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
            myOpts.outputFileName = a 
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
        elif o in ("-p", "--plotfile"): 
            myOpts.doPlot = True 
            myOpts.plotFileName = a 
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
        if not myOpts.overwrite and os.path.isfile(myOpts.outputFileName) :
            print "ERROR: Output file \'%s\' exists already, and permission to overwrite not given."%myOpts.outputFileName
            print "Please delete the file, or give the \'--overwrite\' option" 
            sys.exit(2) 
        if myOpts.outputFileName == myOpts.inputFileName : 
            print "ERROR: Cowardly refusing to overwrite %s with output data. Check file names."%myOpts.inputFileName
            sys.exit(2)
    if myOpts.doPlot : 
        if not myOpts.plotFileName.split('.')[-1] == "png" : 
            print "ERROR: Plotting to files only currently supported in PNG format." 
            print "You have requested to plot to a file with extention: %s"%(myOpts.plotFileName.split('.')[-1]) 
            sys.exit(2) 
        if not myOpts.overwrite and os.path.isfile(myOpts.plotFileName) :
            print "ERROR: Plot file \'%s\' exists already, and permission to overwrite not given."%myOpts.plotFileName
            print "Please delete the file, or give the \'--overwrite\' option" 
            sys.exit(2) 

    ###
    #  Print command line options if verbose
    ### 
    if myOpts.verbose  : 
        print "verbose                   = %r"%myOpts.verbose 
        print "overwrite                 = %s"%myOpts.overwrite 
        print "input file                = %s"%myOpts.inputFileName 
        print "output file               = %s"%myOpts.outputFileName 
        print "plot file                 = %s"%myOpts.plotFileName 
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


def plot_fits(data,popt,fileName) : 
    xs = np.linspace(np.min(data[:,0]),np.max(data[:,0]),500) 
    func = matchFunctionName(len(popt )/ 3) 
    fit = func(xs, *popt) 

    plt.scatter(data[:,0], data[:,1],marker='o',s=25,color='r',zorder=5) 
    plt.plot(xs,fit,'k-',linewidth=4,zorder=1) 

    if len(popt) / 3 > 1 : 
        for i in range(len(popt[::3])) : 
            params = popt[i*3:i*3+3]
            fiti = single_gaussian(xs, *params)
            plt.plot(xs,fiti,'b--',linewidth=2,zorder=3) 
    
    plt.savefig(fileName,format='png') 
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

def baseline_correct(data,verbose) : 
    if verbose : print "Entering base line correction function" 

    x,y = data[:,0],data[:,1]

    z = np.polyfit(x,y,5)
    fit = np.poly1d(z) 
    if verbose : 
        print z 

    plt.scatter(x,y,s=1)
    plt.plot(x,fit(x)) 

    plt.show() 

    sys.exit() 

    return 0



def main(argv) : 
    myOpts = parse_command_line_opts(argv) 

    data = read_data(myOpts.inputFileName) 

    if myOpts.doBaseLineCorrect : 
        baseline_correct(data,myOpts.verbose) 

    if myOpts.doNormalize : 
        data = normalize(data) 

    popt, pcov = fit_data(data,myOpts.numPeaks,myOpts.debug) 
    if myOpts.verbose : 
        for i in range(myOpts.numPeaks) : 
            print "Component %i: a = %5.3f\tb = %5.3f\tc = %5.3f"\
                    %(i+1,popt[i*3],popt[i*3+1],popt[i*3+2]) 

    if myOpts.doPlot : 
        plot_fits(data,popt,myOpts.plotFileName) 
    
    if myOpts.doOutfile : 
        write_fit_data(data,popt,myOpts.outputFileName)   

    if myOpts.doFullAnalysis : 
        if myOpts.verbose : print "Starting Full analysis"
        data = normalize(data) 
        plt.clf() 
        plt.close() 

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

        ax1.legend() 
        ax2.set_yscale('log') 
        basename = os.path.splitext(myOpts.inputFileName)[0]
        f1.savefig(basename+'.residuals.png',format='png') 
        f2.savefig(basename+'.sumofsquares.png',format='png') 

if __name__ == "__main__":
   main(sys.argv[1:])


