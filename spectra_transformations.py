from scipy.interpolate import UnivariateSpline
from scipy.spatial import ConvexHull
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import numpy as np 
import matplotlib.pyplot as plt 
import os 
from os import sys 

from gaussian_funcs import matchFunctionName

def normalize(data, opts) :
    xs = data[:,0]
    ys = data[:,1]

    ys /= np.max(ys)
    return np.array([xs,ys]).T

def spline_fitting(x,y,opts) :
    flipped = False 
    if x[0] > x[-1] : 
        if opts.debug : print "\tUnivariateSpline only accepts increasing x order. Flipping x and y arrays\n" 
        x = np.flip(x,0) 
        y = np.flip(y,0) 
        flipped = True 
    assert x[0] < x[-1] 

    if opts.guessPeak : 
        minPeak = find_nearest(x, np.min(opts.peak), opts) 
        maxPeak = find_nearest(x, np.max(opts.peak), opts) 
    else : 
        guessSpl = UnivariateSpline(x, y)
        peak = np.argmax((guessSpl(x) - y )**2)
        if x[peak] > 2170 or x[peak] < 2130 :
            print "WARNING: Guess of peak outside of 2130-2170 cm^-1. Double check fitting file and be sure we have found the correct peak"
        maxSignalRatio = 0.50
        maxwidth = int(round((len(x) - peak) * maxSignalRatio))   ##distance to maximum signal
        if maxwidth > peak * maxSignalRatio :
            maxwidth = int(round(peak * maxSignalRatio))         ##distance to minimum signal 
        minPeak= peak - maxwidth
        maxPeak= peak + maxwidth

    cutX = np.append(x[:minPeak], x[maxPeak:]) 
    cutY = np.append(y[:minPeak], y[maxPeak:]) 

    spl = UnivariateSpline(cutX,cutY, k=5) #,s= 0.1) 
    if opts.debug :
        residual = np.sum((spl(x) - y)**2)
        print "Residual of baseline = %f"%residual

    if opts.doPlot :
        fig, axarr = plt.subplots(2,1)
        axarr[0].scatter(x,y, s = 1)
        #axarr[0].axvline(x[peak], linestyle='--', color='k', label="Guess of peak position")
        axarr[0].axvline(x[minPeak], linestyle='--',color='r', label="bounds of cut") 
        axarr[0].axvline(x[maxPeak], linestyle='--',color='r') 
        axarr[0].plot(x,spl(x),'k--', label="Baseline spline fit"  )

        axarr[1].plot(x, y - spl(x))
        axarr[1].axvline(x[minPeak], linestyle='--',color='r', label="bounds of cut") 
        axarr[1].axvline(x[maxPeak], linestyle='--',color='r') 

        basename = os.path.splitext(opts.inputFileName)[0]
        fig.savefig(basename+'.spline_fitting.pdf', format='pdf')
    if opts.debug : print spl(x) 

    if flipped : x = np.flip(x,0) ##Flip x values back to match original data
    return spl(x)




def rubberband(x, y, opts):
    flipped = False 
    if x[0] > x[-1] : 
        if opts.debug : print "\tUnivariateSpline only accepts increasing x order. Flipping x and y arrays\n" 
        x = np.flip(x,0) 
        y = np.flip(y,0) 
        flipped = True 
    assert x[0] < x[-1] 
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

    if flipped : x = np.flip(x) ##Flip x values back to match original data
    return spl(x)

    # Create baseline using linear interpolation between vertices
    #return np.interp(x, x[v], y[v])

def find_nearest(array, value, opts):
    if opts.debug : print "Now entering find_nearest function" 

    array = np.asarray(array)
    idx = np.argmin(np.abs(array - value))
    if opts.debug : print "\tFound %.3f at %.3f. Looking for %.3f" %(array[idx],idx, value)
    return idx

def cut_spectrum(data, minX, maxX, opts) :
    if opts.debug : print "Now entering cut_spectrum function"
    if opts.debug : print "\tCutting from %8.3f to %8.3f" %(minX, maxX) 
    x,y = data[:,0], data[:,1]

    cutMin = find_nearest(x, minX, opts)
    cutMax = find_nearest(x, maxX, opts)
    
    if cutMin > cutMax : 
        if opts.debug : print "\tMin index > Max index. Flipping"
        temp = cutMin 
        cutMin = cutMax 
        cutMax = temp 

    if opts.debug : print "Cutting from %.2f to %.2f" %(x[cutMin], x[cutMax])

    cutX = x[cutMin:cutMax]
    cutY = y[cutMin:cutMax]

    if opts.debug : print "\tLength of cutX: %i\tLength of cutY: %i\n" %(len(cutX), len(cutY) ) 

    data = np.array([cutX, cutY]).T
    if opts.debug : print "Length of cut spectrum %i\n" %(len(data)) 

    return data

def smooth_data(data, opts) :
    smoothed = savgol_filter(data[:,1], opts.smoothFactor, 2)
    data[:,1] = smoothed
    return data

def find_reasonable_bounds(data,numpeaks) :
    amin = 0
    amax = 2*np.max(data[:,1])  ##max fit is twice height of data
    bmin = np.min(data[:,0])
    bmax = np.max(data[:,0])
    cmin = 0
    cmax = bmax - bmin          #max standard deviation is width of data

    mins = np.tile([amin,bmin,cmin],numpeaks)
    mins = np.append(mins, -1)
    maxs = np.tile([amax,bmax,cmax],numpeaks)
    maxs = np.append(maxs, +1)

    bounds = (mins,maxs)

    return bounds

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



####
#   This is kept in case I want to re-implement it. Right now it is not functional. 
###
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


