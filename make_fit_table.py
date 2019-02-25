#!/usr/bin/env python
#
# Python code to generate table of log(a_vis)--log(Mstar) fits for S4G bar-size data
# Also generate table of log(a_vis)--log(R_e), log(a_vis)--log(h), and
# log(a_vis) vs log(R_e) + log(Mstar) and log(a_vis) vs log(h) + log(Mstar) fits


from __future__ import print_function

import os, sys, optparse, random, pickle
import numpy as np
import scipy.optimize
import astrostat

baseDir = "/Users/erwin/Documents/Working/Projects/Project_BarSizes/"
baseDirPaper = "/Users/erwin/Documents/Working/Paper-s4gbars/"

sys.path.append(baseDir[:-1])
import datasets
s4gdata = datasets.s4gdata
nDisksTot = datasets.nDisksTot

savedFitsFile = baseDirPaper + "saved_fits.pkl"
tableFile = baseDirPaper + "table_fits.tex"
tableFile_Re = baseDirPaper + "table_fits_Reh.tex"


keyList = ['D25_m8.5', 'D25_m8.5to11', 'D30_m9', 'D30_m9to11', 'D30_m9to11_Reh', 'all']
sampleNames = {'D30_m9to11': "Parent Spiral", 'D30_m9to11_Reh': "Main Spiral",
			'Re_D30_m9to11': "Main Spiral", 'h_D30_m9to11': "Main Spiral"}
distances = {'D25_m8.5': "25", 'D25_m8.5to11': "25",
			'D30_m9': "30", 'D30_m9to11': "30",
			'all': "---", 'Re_D30_m9to11': "30", 'D30_m9to11_Reh': "30", 'h_D30_m9to11': "30"}
fitRanges = {'D25_m8.5': r"$\geq 8.5$", 'D25_m8.5to11': "8.5--11",
			'D30_m9': r"$\geq 9$", 'D30_m9to11': "9--11",
			'all': "All", 'Re_D30_m9to11': "9--11", 'D30_m9to11_Reh': "9--11",
			'h_D30_m9to11': "9--11"}
texNames = {'D25_m8.5': r"$D \leq 25$, $\logmstar \geq 8.5$",
			'D25_m8.5to11': r"$D \leq 25$, $\logmstar = 8.5$--11",
			'D30_m9': r"$D \leq 30$, $\logmstar \geq 9$",
			'D30_m9to11': r"$D \leq 30$, $\logmstar = 9$--11",
			'D30_m9to11_Reh': r"$D \leq 30$, $\logmstar = 9$--11",
			'all': "All", 'Re_D30_m9to11': r"$D \leq 30$, $\logmstar = 9$--11",
			'h_D30_m9to11': r"$D \leq 30$, $\logmstar = 9$--11"}
varNames = {'Re_D30_m9to11': r"$\log \re$", 'h_D30_m9to11': r"$\log h$"}



# Stuff for LaTeX tables:

tableHeader = r"""\begin{table*}
\begin{minipage}{126mm}
\caption{Fits to Bar Size versus Stellar Mass}
\label{tab:bar-size-fits}
\renewcommand{\arraystretch}{1.5}
\begin{tabular}{lccccc}
\hline
Sample & $\alpha_{1}$ & $\beta_{1}$ & $\alpha_{2}$  & $\beta_{2}$ & $\log \: (M_{\rm brk} / \Msun)$ \\
  (1)  & (2)          & (3)         & (4)           & (5)         &        (6)  \\
\hline
"""

tableEnd = r"""\hline
\end{tabular}

\medskip

Results of broken-linear fits to $\log \avis$ as a function of \logmstar{} (see
Eqn.~\ref{eq:barsize1}).
Parameter uncertainties are based on 2000 rounds of bootstrap resampling.

\end{minipage}
\end{table*}
"""

tableHeader_Reh = r"""\begin{table}
\caption{Fits to Bar Size versus \re{} and $h$}
\label{tab:bar-size-fits-reh}
\begin{tabular}{lrccc}
\hline
Predictor    & Fit range    & $D_{\rm max}$ & $\alpha$ & $\beta$ \\
             & (\logmstar)  &  (Mpc)        &              & \\
  (1)        &  (2)         & (3)           & (4)          & (5) \\
\hline
"""

tableEnd_Reh = r"""\hline
\end{tabular}

\medskip

Results of linear fits to bar size $\log \avis$ as a function of \logre{} and $\log h$.
Parameter uncertainties are based on 2000 rounds of bootstrap resampling.

\end{table}
"""

templateLine = r"{0:15} & {1:25s} & {2:25s} & {3:25s} & {4:25s} & {5:25s} \\"

templateLine_Reh = r"{0:10s} & {1:2s} & {2:10s} & {3:25s} & {4:25s} \\"

# Indices defining different subsets of barred-galaxy sample:
ii_spirals = [i for i in range(nDisksTot) if s4gdata.t_s4g[i] > -0.5]
ii_barred = [i for i in range(nDisksTot) if s4gdata.sma[i] > 0]
ii_unbarred = [i for i in range(nDisksTot) if s4gdata.sma[i] <= 0]

ii_all_limited1 = [i for i in range(nDisksTot) if s4gdata.dist[i] <= 25]
ii_barred_limited1 = [i for i in ii_all_limited1 if i in ii_barred]
ii_unbarred_limited1 = [i for i in ii_all_limited1 if i not in ii_barred]
ii_all_limited1_m8_5 = [i for i in ii_all_limited1 if s4gdata.logmstar[i] >= 8.5]
ii_barred_limited1_m8_5 = [i for i in ii_all_limited1_m8_5 if i in ii_barred]
ii_barred_limited1_m8_5to11 = [i for i in ii_barred_limited1_m8_5 if s4gdata.logmstar[i] <= 11]

ii_all_limited2 = [i for i in range(nDisksTot) if s4gdata.dist[i] <= 30]
ii_barred_limited2 = [i for i in ii_all_limited2 if i in ii_barred]
ii_unbarred_limited2 = [i for i in ii_all_limited2 if i not in ii_barred]
ii_all_limited2_m9 = [i for i in ii_all_limited2 if s4gdata.logmstar[i] >= 9]
ii_barred_limited2_m9 = [i for i in ii_all_limited2_m9 if i in ii_barred]
ii_barred_limited2_m9to11 = [i for i in ii_barred_limited2_m9 if s4gdata.logmstar[i] <= 11]

# R_e-related
ii_all_lim2m9to11_nonzero_Re = [i for i in range(nDisksTot) if s4gdata.Re_kpc[i] > 0 and s4gdata.logmstar[i] >=9 and s4gdata.logmstar[i] <= 11]
ii_barred_lim2m9to11_nonzero_Re = [i for i in ii_barred_limited2_m9to11 if s4gdata.Re_kpc[i] > 0]
ii_all_limited2_Re = [i for i in ii_all_limited2 if s4gdata.logmstar[i] >= 9 and s4gdata.logmstar[i] <= 11 and s4gdata.Re_kpc[i] > 0]
ii_barred_limited2_Re = [i for i in ii_all_limited2_Re if i in ii_barred]
ii_unbarred_limited2_Re = [i for i in ii_all_limited2_Re if i not in ii_barred]

# Spirals with R_e *and* h *and* logMstar = 9--11:
ii_all_Reh = [i for i in ii_spirals if s4gdata.logmstar[i] >= 9 and s4gdata.logmstar[i] <= 11 and s4gdata.Re_kpc[i] > 0 and s4gdata.h_kpc[i] > 0]
ii_barred_Reh = [i for i in ii_all_Reh if i in ii_barred]
ii_unbarred_Reh = [i for i in ii_all_Reh if i not in ii_barred]

# Spirals (D < 30 Mpc) with R_e *and* h *and* logMstar = 9--11:
ii_all_limited2_Reh = [i for i in ii_all_limited2 if s4gdata.logmstar[i] >= 9 and s4gdata.logmstar[i] <= 11 and s4gdata.Re_kpc[i] > 0 and s4gdata.h_kpc[i] > 0]
ii_barred_limited2_Reh = [i for i in ii_all_limited2_Reh if i in ii_barred]
ii_unbarred_limited2_Reh = [i for i in ii_all_limited2_Reh if i not in ii_barred]


xx1 = s4gdata.logmstar[ii_barred_limited1_m8_5]
xx1b = s4gdata.logmstar[ii_barred_limited1_m8_5to11]
xx2 = s4gdata.logmstar[ii_barred_limited2_m9]
xx2b = s4gdata.logmstar[ii_barred_limited2_m9to11]
xx_all = s4gdata.logmstar[ii_barred]

logre = np.log10(s4gdata.Re_kpc)
logh = np.log10(s4gdata.h_kpc)
log_barsize = np.log10(s4gdata.sma_dp_kpc2)

# deprojected bar sizes
yy_dp1 = np.log10(s4gdata.sma_dp_kpc2[ii_barred_limited1_m8_5])
yy_dp1b = np.log10(s4gdata.sma_dp_kpc2[ii_barred_limited1_m8_5to11])
yy_dp2 = np.log10(s4gdata.sma_dp_kpc2[ii_barred_limited2_m9])
yy_dp2b = np.log10(s4gdata.sma_dp_kpc2[ii_barred_limited2_m9to11])
yy_dp_all = np.log10(s4gdata.sma_dp_kpc2[ii_barred])

# R_e and h subsamples
# logre2b = logre[ii_barred_lim2m9to11_nonzero_Re]
# yy_re2b = log_barsize[ii_barred_lim2m9to11_nonzero_Re]
xx_reh2b = s4gdata.logmstar[ii_barred_limited2_Reh]
logre2b = logre[ii_barred_limited2_Reh]
logh2b = logh[ii_barred_limited2_Reh]
yy_reh2b = log_barsize[ii_barred_limited2_Reh]





def linear_func( params, x ):
	alpha = params[0]
	beta = params[1]
	vals = alpha + beta*x
	return vals
	
def broken_linear_func( params, logmstar ):
	alpha_1 = params[0]
	beta_1 = params[1]
	mstar_break = params[2]
	beta_2 = params[3]
	alpha_2 = alpha_1 + (beta_1 - beta_2)*mstar_break
	npts = len(logmstar)
	vals = []
	for i in range(npts):
		if logmstar[i] < mstar_break:
			value = alpha_1 + beta_1*logmstar[i]
		else:
			value = alpha_2 + beta_2*logmstar[i]
		vals.append(value)
	return vals


def f2( X, a, b, a1, b1, x_brk, b2 ):
    """Given an input consisting of the 2-element datamtuple or list X (which consists of 2 1-D
    arrays) this function computes the sum of linear fit (using parameters a,b) to the first data 
    array and a broken-linear fit (using rest of parameters) to the second data array.
    
        X : tuple of x1, x2
            x1 : 1D numpy array of predictor using linear fit (e.g., log R_e)
            x2 : 1D numpy array of predictor using broken-linear fit (e.g., log M_star)
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

def multivar_func( params, xvals ):
	# xvals[0] = logmstar
	# xvals[1] = logre or logh
	# params = [a_rad, b_rad, a1_mass, b1_mass, m_break, b2_mass]
	return f2(xvals, *params)


def meritfunc(params, x, y, theFunc):
	return (theFunc(params, x) - y)

def bootstrap_errors_linear( params0, x, y, meritFn, mainFn, nIterations=100 ):
	alpha_array = []
	beta_array = []
	p1, s = scipy.optimize.leastsq(meritFn, params0, args=(x, y, mainFn))
	
	nPts = len(x)
	orderedIndices = list(range(nPts))
	for n in range(nIterations):
		# generate bootstrap sample
		new_indices = [random.randint(0, nPts - 1) for i in orderedIndices]
		x_new = np.array([x[i] for i in new_indices])
		y_new = np.array([y[i] for i in new_indices])
		# do fit and store params
		pnew, s = scipy.optimize.leastsq(meritfunc, params0, args=(x_new, y_new, linear_func))
		alpha_array.append(pnew[0])
		beta_array.append(pnew[1])

	a_low,a_high = astrostat.ConfidenceInterval(alpha_array)
	b_low,b_high = astrostat.ConfidenceInterval(beta_array)
	
	return ((a_low,a_high), (b_low,b_high))

def bootstrap_errors_broken_linear( params0, x, y, meritFn, mainFn, nIterations=100 ):
	alpha1_array = []
	beta1_array = []
	mstar_break_array = []
	alpha2_array = []
	beta2_array = []
	p1, s = scipy.optimize.leastsq(meritFn, params0, args=(x, y, mainFn))
	
	nPts = len(x)
	orderedIndices = list(range(nPts))
	for n in range(nIterations):
		# generate bootstrap sample
		new_indices = [random.randint(0, nPts - 1) for i in orderedIndices]
		x_new = np.array([x[i] for i in new_indices])
		y_new = np.array([y[i] for i in new_indices])
		# do fit and store params
		pnew, s = scipy.optimize.leastsq(meritfunc, params0, args=(x_new, y_new, broken_linear_func))
		alpha1_array.append(pnew[0])
		beta1_array.append(pnew[1])
		mstar_break_array.append(pnew[2])
		beta2_array.append(pnew[3])
		alpha2_array.append(pnew[0] + (pnew[1] - pnew[3])*pnew[2])

	a1_low,a1_high = astrostat.ConfidenceInterval(alpha1_array)
	b1_low,b1_high = astrostat.ConfidenceInterval(beta1_array)
	mbrk_low,mbrk_high = astrostat.ConfidenceInterval(mstar_break_array)
	a2_low,a2_high = astrostat.ConfidenceInterval(alpha2_array)
	b2_low,b2_high = astrostat.ConfidenceInterval(beta2_array)
	
	return ((a1_low,a1_high), (b1_low,b1_high), (a2_low,a2_high), (b2_low,b2_high), 
			(mbrk_low,mbrk_high))

def bootstrap_errors_multivar( params0, x1, x2, y, meritFn, mainFn, nIterations=100 ):
	alpha1_array = []
	beta1_array = []
	mstar_break_array = []
	alpha2_array = []
	beta2_array = []
	alpha_rad_array = []
	beta_rad_array = []
	p1, s = scipy.optimize.leastsq(meritFn, params0, args=([x1,x2], y, mainFn))
	
	nPts = len(x1)
	orderedIndices = list(range(nPts))
	for n in range(nIterations):
		# generate bootstrap sample
		new_indices = [random.randint(0, nPts - 1) for i in orderedIndices]
		x1_new = np.array([x1[i] for i in new_indices])
		x2_new = np.array([x2[i] for i in new_indices])
		y_new = np.array([y[i] for i in new_indices])
		# do fit and store params
		pnew, s = scipy.optimize.leastsq(meritFn, params0, args=([x1_new,x2_new], y_new, 
										mainFn))
		alpha_rad,beta_rad, alpha1, beta1, m_break, beta2 = pnew
		alpha_rad_array.append(alpha_rad)
		beta_rad_array.append(beta_rad)
		alpha1_array.append(alpha1)
		beta1_array.append(beta1)
		mstar_break_array.append(m_break)
		beta2_array.append(beta2)
		alpha2_array.append(alpha1 + (beta1 - beta2)*m_break)

	a1_low,a1_high = astrostat.ConfidenceInterval(alpha1_array)
	b1_low,b1_high = astrostat.ConfidenceInterval(beta1_array)
	mbrk_low,mbrk_high = astrostat.ConfidenceInterval(mstar_break_array)
	a2_low,a2_high = astrostat.ConfidenceInterval(alpha2_array)
	b2_low,b2_high = astrostat.ConfidenceInterval(beta2_array)
	arad_low,arad_high = astrostat.ConfidenceInterval(alpha_rad_array)
	brad_low,brad_high = astrostat.ConfidenceInterval(beta_rad_array)
	
	return ( (arad_low,arad_high), (brad_low,brad_high), (a1_low,a1_high), (b1_low,b1_high), 
			(a2_low,a2_high), (b2_low,b2_high), (mbrk_low,mbrk_high) )




def ComputeAndSaveFits( nIters=100 ):
	fitDict = {}
	p0 = [-2.5, 0.3, 10.1, 0.5]
	
	print("Calculating bar-size--logmstar fits with %d rounds of bootstrap resampling..." % nIters)
	# Do the fits
	(pp_dp, s) = scipy.optimize.leastsq(meritfunc, p0, args=(xx2b, yy_dp2b, broken_linear_func))
	print("Best-fit params: ", pp_dp)
	alpha1,beta1,mstar_break,beta2 = pp_dp
	alpha2 = alpha1 + (beta1 - beta2)*mstar_break
	((a1_low,a1_high), (b1_low,b1_high), (a2_low,a2_high), (b2_low,b2_high), 
			(mbrk_low,mbrk_high)) = bootstrap_errors_broken_linear(pp_dp, xx2b, yy_dp2b, meritfunc,
			broken_linear_func, nIters)
	fitDict['D30_m9to11'] = ((alpha1,alpha1 - a1_low,a1_high - alpha1), 
						(beta1,beta1 - b1_low,b1_high - beta1),
						(alpha2,alpha2 - a2_low,a2_high - alpha2), 
						(beta2,beta2 - b2_low,b2_high - beta2),
						(mstar_break,mstar_break - mbrk_low,mbrk_high - mstar_break))

	(pp_dp, s) = scipy.optimize.leastsq(meritfunc, p0, args=(xx_reh2b, yy_reh2b, broken_linear_func))
	print("Best-fit params: ", pp_dp)
	alpha1,beta1,mstar_break,beta2 = pp_dp
	alpha2 = alpha1 + (beta1 - beta2)*mstar_break
	((a1_low,a1_high), (b1_low,b1_high), (a2_low,a2_high), (b2_low,b2_high), 
			(mbrk_low,mbrk_high)) = bootstrap_errors_broken_linear(pp_dp, xx2b, yy_dp2b, meritfunc,
			broken_linear_func, nIters)
	fitDict['D30_m9to11_Reh'] = ((alpha1,alpha1 - a1_low,a1_high - alpha1), 
						(beta1,beta1 - b1_low,b1_high - beta1),
						(alpha2,alpha2 - a2_low,a2_high - alpha2), 
						(beta2,beta2 - b2_low,b2_high - beta2),
						(mstar_break,mstar_break - mbrk_low,mbrk_high - mstar_break))

	print("Calculating bar-size--log(R_e) fits with %d rounds of bootstrap resampling..." % nIters)
	# Do the fits
	plin0 = [0.0, 1.0]
	(pp_re2b, s) = scipy.optimize.leastsq(meritfunc, plin0, args=(logre2b, yy_reh2b, linear_func))
	alpha,beta = pp_re2b
	((a_low,a_high), (b_low,b_high)) = bootstrap_errors_linear(pp_re2b, logre2b, yy_reh2b,
										meritfunc, linear_func, nIters)
	fitDict['Re_D30_m9to11'] = ((alpha,alpha1 - a_low,a_high - alpha), 
						(beta,beta - b_low,b_high - beta))

	print("Calculating bar-size--log(h) fits with %d rounds of bootstrap resampling..." % nIters)
	# Do the fits
	plin0 = [0.0, 1.0]
	(pp_re2b, s) = scipy.optimize.leastsq(meritfunc, plin0, args=(logh2b, yy_reh2b, linear_func))
	alpha,beta = pp_re2b
	((a_low,a_high), (b_low,b_high)) = bootstrap_errors_linear(pp_re2b, logh2b, yy_reh2b,
										meritfunc, linear_func, nIters)
	fitDict['h_D30_m9to11'] = ((alpha,alpha1 - a_low,a_high - alpha), 
						(beta,beta - b_low,b_high - beta))


	# IN PROGESS...
	print("Calculating bar-size--log(R_e) + log(Mstar) fits with %d rounds of bootstrap resampling..." % nIters)
	# Do the fits
	#    [a_rad, b_rad, a1_mass, b1_mass, m_break, b2_mass]
	pmulti0 = [0, 0.5, 0, 0.0, 10.0, 0.5]
	X = [logre2b,xx_reh2b]
	(pp_re2b, s) = scipy.optimize.leastsq(meritfunc, pmulti0, args=(X, yy_reh2b, multivar_func))
	alpha_rad,beta_rad, alpha1, beta1, mstar_break, beta2 = pp_re2b
	alpha2 = alpha1 + (beta1 - beta2)*mstar_break
	print("Best-fit params: ", pp_re2b)
	xx = bootstrap_errors_multivar(pp_re2b, logre2b, xx_reh2b, yy_reh2b,
										meritfunc, multivar_func, nIters)
	( (arad_low,arad_high), (brad_low,brad_high), (a1_low,a1_high), (b1_low,b1_high), 
			(a2_low,a2_high), (b2_low,b2_high), (mbrk_low,mbrk_high) ) = xx
	fitDict['Re+logmstar_D30_m9to11'] = ( (alpha_rad,alpha_rad - arad_low,arad_high - alpha_rad),
						(beta_rad,beta_rad - brad_low,brad_high - beta_rad),
						(alpha1,alpha1 - a1_low,a1_high - alpha1), 
						(beta1,beta1 - b1_low,b1_high - beta1),
						(alpha2,alpha2 - a2_low,a2_high - alpha2), 
						(beta2,beta2 - b2_low,b2_high - beta2),
						(mstar_break,mstar_break - mbrk_low,mbrk_high - mstar_break) )
	
 	print("ComputeAndSaveFits:")
 	print(fitDict["Re+logmstar_D30_m9to11"])

	pickleFile = savedFitsFile
	pickle.dump(fitDict, open(pickleFile, "wb"))
	
	print("Done.")
	
	return fitDict



def GetPrecomputedFits( ):
	pkl_file = open(savedFitsFile, "rb")
	fitDict = pickle.load(pkl_file)
	pkl_file.close()
	
	return fitDict


def MakeLatexVal( inputVals ):
	f, e_low, e_high = inputVals
	return "$%.2f^{+%.2f}_{-%.2f}$" % (f, e_low, e_high)
	
def MakeTable( fitResults ):
	"""Generate and save LaTeX table with fit parameters for log(bar-size) vs
	logMstar [broken-linear fit]
	
	Currently = Table 1 in paper.
	"""
	print("MakeTable:")
	
	outf = open(tableFile, 'w')
	outf.write(tableHeader)
	for fitName in ["D30_m9to11", "D30_m9to11_Reh"]:
		sampleName = sampleNames[fitName]
		thisFitResults = fitResults[fitName]
		alpha1Txt = MakeLatexVal(thisFitResults[0])
		beta1Txt = MakeLatexVal(thisFitResults[1])
		alpha2Txt = MakeLatexVal(thisFitResults[2])
		beta2Txt = MakeLatexVal(thisFitResults[3])
		mbrkTxt = MakeLatexVal(thisFitResults[4])
		outLine = templateLine.format(sampleName, alpha1Txt, beta1Txt, alpha2Txt, beta2Txt, mbrkTxt)
		outf.write(outLine + "\n")
	outf.write(tableEnd)
	outf.close()


def MakeReHTable( fitResults ):
	"""Generate and save LaTeX table with fit parameters for log(bar-size) vs
	log(R_e) and vs log(h) [linear fits]
	
	Currently = Table 2 in paper.
	"""
	print("MakeTable:")
	
	outf = open(tableFile_Re, 'w')
	outf.write(tableHeader_Reh)
	for fitName in ['Re_D30_m9to11', 'h_D30_m9to11']:
		varName = varNames[fitName]
		fr = fitRanges[fitName]
		D = distances[fitName]
		thisFitResults = fitResults[fitName]
		print(thisFitResults)
		alpha1Txt = MakeLatexVal(thisFitResults[0])
		beta1Txt = MakeLatexVal(thisFitResults[1])
		outLine = templateLine_Reh.format(varName, fr, D, alpha1Txt, beta1Txt)
		outf.write(outLine + "\n")
	# R_e fits
	fr = fitRanges['Re_D30_m9to11']
	D = distances['Re_D30_m9to11']
	outf.write(tableEnd_Reh)
	outf.close()


def main(argv=None):

	usageString = "%prog [options] blahblahblah\n"
	parser = optparse.OptionParser(usage=usageString, version="%prog ")


	parser.add_option("--recalculate", action="store_true", dest="recalculate",
					  default=False, help="Re-run fits and bootstrapping")
	parser.add_option("--niters", dest="nIterations", type="int",
					  default=100, help="Number of bootstrap iterations")
	
	(options, args) = parser.parse_args(argv)

	# args[0] = name program was called with
	# args[1] = first actual argument, etc.

	if options.recalculate is False and os.path.exists(savedFitsFile):
		fitResults = GetPrecomputedFits()
	else:
		fitResults = ComputeAndSaveFits(options.nIterations)

	MakeTable(fitResults)
	MakeReHTable(fitResults)


if __name__ == '__main__':
	
	main(sys.argv)
