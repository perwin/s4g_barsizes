# code for performing and evaluating fits

import numpy as np
from scipy.optimize import curve_fit
import astrostat




# Functions for fits: returning one or array of values given input
# x (potentially multiple parts) and individual parameter values
# For use with scipy.optimize.curve_fit

def flin( x, a, b ):
    """Linear function of x
    
    Parameters
    ----------
    x : float or ndarray or list of float
        x values (independent variable)
    a, b : float
        parameters for the model (intercept, slope)
    
    Returns
    -------
    yy : ndarray of float
        array of y values
    """
    return a + b*x


def fbrokenlin( x, a1, b1, x_brk, b2 ):
    """Broken-linear function of x
    
    Parameters
    ----------
    x : float or ndarray or list of float
        x values (independent variable)
    a1, b1, x_brk, b2 : float
        parameters for the model
    
    Returns
    -------
    yy : ndarray of float
        array of y values
    
    The model is
        y = a1 + b1*x   for x < x_brk
        y = a2 + b2*x   for x > x_brk
    Note that a2 is computed from the other parameters (it's not an independent
    parameter, bcs. both equations have to be equal when x=x_brk)
    """
    a2 = a1 + (b1 - b2)*x_brk
    npts = len(x)
    yy = []
    for x_i in x:
        if x_i < x_brk:
            y_i = a1 + b1*x_i
        else:
            y_i = a2 + b2*x_i
        yy.append(y_i)
    return np.array(yy)


def fmulti_lin_brokenlin_old( X, a, b, a1, b1, x_brk, b2 ):
    """Composite function which add linear fit (a, b) to broken-linear
    fit (rest of parameters)

	*** THIS IS THE OLDER, INCORRECT VERSION *** 
	
    Parameters
    ----------
    X : tuple of x1, x2 
        x1 : 1D numpy array of predictor using linear fit (e.g., log R_e)
        x2 : 1D numpy array of predictor using broken-linear fit (e.g., log M_star)
    a, b, a1, b1, x_brk, b2 : float
        parameters for the model

    Returns
    -------
    yy : ndarray of float
        array of y values
    
    The model is
        y = a + b*x1 + a1 + b1*x2   for x < x_brk
        y = a + b*x1 + a2 + b2*x2   for x > x_brk
    Note that a2 is computed from the other parameters
    """
    # unpack the two independent variables
    # e.g., x1 = log(R_e), x2 = log(M_star)
    x1,x2 = X
    
    a2 = a1 + (b1 - b2)*x_brk
    npts = len(x1)
    yy = []
    for i in range(npts):
        x1_i = x1[i]
        x2_i = x2[i]
        if x2_i < x_brk:
            y_i = a + b*x1_i + a1 + b1*x2_i
        else:
            y_i = a + b*x1_i + a2 + b2*x2_i
        yy.append(y_i)
    return np.array(yy)


def fmulti_lin_brokenlin( X, a1, b, b1, x_brk, b2 ):
    """Composite function which add linear fit (a, b) to broken-linear
    fit (rest of parameters)

    Parameters
    ----------
    X : tuple of x1, x2 
        x1 : 1D numpy array of predictor using linear fit (e.g., log R_e)
        x2 : 1D numpy array of predictor using broken-linear fit (e.g., log M_star)
    a, b, a1, b1, x_brk, b2 : float
        parameters for the model

    Returns
    -------
    yy : ndarray of float
        array of y values
    
    The model is
        y = a1 + b*x1 + b1*x2   for x < x_brk
        y = a2 + b*x1 + b2*x2   for x > x_brk
    Note that a2 is computed from the other parameters (it's not an independent
    parameter, bcs. both equations have to be equal when x=x_brk)
    """
    # unpack the two independent variables
    # e.g., x1 = log(R_e), x2 = log(M_star)
    x1,x2 = X
    
    a2 = a1 + (b1 - b2)*x_brk
    npts = len(x1)
    yy = []
    for i in range(npts):
        x1_i = x1[i]
        x2_i = x2[i]
        if x2_i < x_brk:
            y_i = a1 + b*x1_i + b1*x2_i
        else:
            y_i = a2 + b*x1_i + b2*x2_i
        yy.append(y_i)
    return np.array(yy)


def fmulti_binary( X, a, b, a1, b1, x_brk, b2 ):
    """Like fmulti_lin_brokenlin, but only computes the linear fit when the 
    value ofthe first data array is > 0

    Parameters
    ----------
    X : tuple of x1, x2 
        x1 : 1D numpy array of predictor using linear fit (e.g., log barsize)
            For galaxies without bars, log barsize must = -inf
        x2 : 1D numpy array of predictor using broken-linear fit (e.g., log M_star)
    a, b, a1, b1, x_brk, b2 : float
        parameters for the model

    Returns
    -------
    yy : ndarray of float
        array of y values
    
    The model is
        y = B*(a + b*x1) + a1 + b1*x2   for x < x_brk
        y = B*(a + b*x1) + a2 + b2*x2   for x > x_brk
    where B = 1 for barred galaxy (defined as x1 > 0) and 0 for unbarred
    
    Note that a2 is computed from the other parameters
    """
    # unpack the two independent variables
    # e.g., x1 = log(barsize), x2 = log(M_star)
    x1,x2 = X
    
    a2 = a1 + (b1 - b2)*x_brk
    npts = len(x1)
    yy = []
    for i in range(npts):
        x1_i = x1[i]
        x2_i = x2[i]
        if x1_i > -90.0:
            barTerm = a + b*x1_i
        else:
            barTerm = 0.0
        if x2_i < x_brk:
            y_i = barTerm + a1 + b1*x2_i
        else:
            y_i = barTerm + a2 + b2*x2_i
        yy.append(y_i)
    return np.array(yy)



def logLikelihood( x, y, errs, fitFn, params, debug=False ):
    """
    Computes the log likelihood of the model described by fitFn, with
    model parameters in params, given data described by x (independent
    variable[s]) and y. Assumes chi^2 statistics.
    
    Parameters
    ----------
    x : ndarray of float (or tuple/list of two such arrays)
        data -- independent variable values
    y : ndarray of float
        data -- dependent variable values
    errs : ndarray of float
        Gaussian sigma values describing uncertainties on y values
    fitFn : function
        function defining the model; will be called as fitFn(x, *params)
    params : ndarray of float
        parameter values for the model
    
    Returns
    -------
    logLikelihood : float
        = -0.5 * chi^2 from comparing computed model values with data
    """
    y_model = fitFn(x, *params)
    weights = 1.0 / errs**2
    resid2 = (y - y_model)**2
    chi2 = np.sum(weights * resid2)
    if debug:
        print("chi^2 = %g" % chi2)
    return -0.5 * chi2


def PrintParams( params, prefix="", mode="linear" ):
    """
    Pretty-printing of parameter values
    
    Parameters
    ----------
    params : ndarray of float
        parameter values for the model
    prefix : str
        optional prefix to go at the head of printed output lines
    mode : str
        what type of model the parameters are for 
        one of ["linear", "broken-linear", "composite", or "binary"]
    
    """
    if mode == "linear":
        alpha, beta = params
        print(prefix + "alpha, beta = [%g, %g]" % (alpha, beta))
    elif mode == "broken-linear":
        alpha1, beta1, x_break, beta2 = params
        alpha2 = alpha1 + (beta1 - beta2)*x_break
        print("alpha1, beta1, alpha2, beta2, x_break = ")
        print("[%.3f, %.3f, %.3f, %.3f, %.3f]" % (alpha1, beta1, alpha2, beta2, x_break))
    elif mode == "composite":
        alpha1, beta, beta1, x_break, beta2 = params
        alpha2 = alpha1 + (beta1 - beta2)*x_break
        print("alpha1, beta, alpha2, beta1, beta2, x_break = ")
        txt = "[%.3f, %.3f, %.3f, " % (alpha1, beta, alpha2)
        txt += "%.3f, %.3f, %.3f]" % (beta1, beta2, x_break)
        print(txt)
    elif mode in "binary":
        alpha, beta, alpha1, beta1, x_break, beta2 = params
        alpha2 = alpha1 + (beta1 - beta2)*x_break
        print("alpha, beta, alpha1, beta1, alpha2, beta2, x_break = ")
        txt = "[%.3f, %.3f, %.3f, " % (alpha, beta, alpha1)
        txt += "%.3f, %.3f, %.3f, %.3f]" % (beta1, alpha2, beta2, x_break)
        print(txt)

        
def DoFit( x, y, errs, ii, p0, mode="linear", doPrint=True ):
    """
    Fit model to data (x,y), returning best-fit parameters and AIC.
    
    Parameters
    ----------
    x : ndarray of float (or tuple/list of two such arrays)
        data -- independent variable values
    y : ndarray of float
        data -- dependent variable values
    errs : ndarray of float
        Gaussian sigma values describing uncertainties on y values
    ii : list of int
        indices into x and y, specifying a particular subsample
        (only x[ii] and y[ii] will be used for the fit)
    p0 : ndarray of float
        initialparameter values for the model
    mode : str
        specifies which model to fit to the data
        One of ["linear", "broken-linear", "composite", "binary"]
    doPrint : bool
        If True, the best-fitting parameter values and the corresponding AIC
        are printed at the end of the fit

    Returns
    -------
    (p_bestfit, aic) : tuple of (ndarray of float, float)
        p_bestfit = ndarray of best-fitting parameter values
        aic = AIC (Akaike Information Criterion) for best-fit parameters
    """
    if mode == "linear":
        func = flin
        xx = x[ii]
    elif mode == "broken-linear":
        func = fbrokenlin
        xx = x[ii]
    elif mode == "composite":
        func = fmulti_lin_brokenlin
        xx1,xx2 = x
        xx1_sub = xx1[ii]
        xx2_sub = xx2[ii]
        xx = [xx1_sub,xx2_sub]
    elif mode == "binary":
        func = fmulti_binary
        xx1,xx2 = x
        xx1_sub = xx1[ii]
        xx2_sub = xx2[ii]
        xx = [xx1_sub,xx2_sub]
    yy = y[ii]
    if errs is not None:
        ee = errs[ii]
    nParams = len(p0)
    
    pp, pcov = curve_fit(func, xx, yy, p0=p0, sigma=ee)
    ll = logLikelihood(xx, yy, ee, func, pp)
    aic = astrostat.AICc(ll, nParams, len(xx))
    
    if doPrint:
        PrintParams(pp, "   ", mode)
        print("   AIC = %g" % aic)
    return (pp, aic)




def ParameterUncertainties( x, y, errs, ii, p0, mode="linear", nIterations=100 ):
    """
    Estimate uncertainties for fitting model to data (x,y), using bootstrap
    resampling.
    
    Parameters
    ----------
    x : ndarray of float (or tuple/list of two such arrays)
        data -- independent variable values
    y : ndarray of float
        data -- dependent variable values
    errs : ndarray of float
        Gaussian sigma values describing uncertainties on y values
    ii : list of int
        indices into x and y, specifying a particular subsample
        (only x[ii] and y[ii] will be used for the fit)
    p0 : ndarray of float
        initialparameter values for the model
    mode : str
        specifies which model to fit to the data
        One of ["linear", "broken-linear", "composite", "binary"]
    nIterations : int
        Number of rounds of bootstrap resampling to do

    Returns
    -------
    paramIntervals : list of (float, float)
        list of (lower_limit,upper_limit) for parameters, where lower_limit
        and upper_limit mark the confidence interval for the distribution
        of a given parameter's bootstrap values.
        This has the same number of tuples as the length of p0, with the
        same ordering.
    """
    if mode == "linear":
        func = flin
        xx = x[ii]
    elif mode == "broken-linear":
        func = fbrokenlin
        xx = x[ii]
    elif mode == "composite":
        func = fmulti_lin_brokenlin
        xx1,xx2 = x
        xx1_sub = xx1[ii]
        xx2_sub = xx2[ii]
        xx = [xx1_sub,xx2_sub]
    elif mode == "binary":
        func = fmulti_binary
        xx1,xx2 = x
        xx1_sub = xx1[ii]
        xx2_sub = xx2[ii]
        xx = [xx1_sub,xx2_sub]
    yy = y[ii]
    if errs is not None:
        ee = errs[ii]
    
    nData = len(yy)
    nParams = len(p0)
    paramsArray = []
    for i in range(nParams):
        paramsArray.append([])
    
    pp, pcov = curve_fit(func, xx, yy, p0=p0, sigma=ee)

    indices = np.arange(0, nData)
    nFailed = 0
    for n in range(nIterations):
        # generate bootstrap sample
        try:
            i_bootstrap = np.random.choice(indices, nData, replace=True)
            if type(xx) in [tuple,list]:
                xx_b = (xx[0][i_bootstrap], xx[1][i_bootstrap])
            else:
                xx_b = xx[i_bootstrap]
            yy_b = yy[i_bootstrap]
            sigma_b = ee[i_bootstrap]
            pnew, pcov = curve_fit(func, xx_b, yy_b, p0=pp, sigma=sigma_b)
            for i in range(nParams):
                paramsArray[i].append(pnew[i])
        except RuntimeError:
            # couldn't get a proper fit, so let's discard this sample and try again
            nFailed += 1
            pass

    paramIntervals = []
    for i in range(nParams):
        paramIntervals.append(astrostat.ConfidenceInterval(paramsArray[i]))
    
    if nFailed > 0:
        print("\tParameterUncertainties: %d failed iterations" % nFailed)
    return paramIntervals


