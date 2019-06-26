#!/usr/bin/env python
#
# Python code to generate tables of best-fit parameter values:
# log(a_vis)--log(Mstar) fits for S4G bar-size data.
# Also generate table of log(a_vis)--log(R_e), log(a_vis)--log(h), and
# log(a_vis) vs log(R_e) + log(Mstar) and log(a_vis) vs log(h) + log(Mstar) fits


from __future__ import print_function

import os, sys, optparse, random, pickle
import numpy as np
import scipy.optimize
import astrostat

baseDir = "/Users/erwin/Documents/Working/Projects/Project_BarSizes/"
baseDirPaper = "/Users/erwin/Documents/Working/Papers-s4gbars/"

sys.path.append(baseDir[:-1])
import datasets
s4gdata = datasets.s4gdata
nDisksTot = datasets.nDisksTot

#savedFitsFile = baseDirPaper + "saved_fits.pkl"
savedFitsFile = baseDir + "bestfit_parameters.txt"
savedUndertaintiesFile = baseDir + "bestfit_parameter_uncertainties.txt"

tableFile = baseDirPaper + "table_fits.tex"
tableFile_Re = baseDirPaper + "table_fits_Reh.tex"


keyList = ['D25_m8.5', 'D25_m8.5to11', 'D30_m9', 'D30_m9to11', 'D30_m9to11_Reh', 'all']
sampleNames = {'barsize-vs-Mstar_parent_brokenlin': "Parent Spiral", 
			'barsize-vs-Mstar_brokenlin': "Main Spiral",
			'barsize-vs-Re_lin_Reh': "Main Spiral", 
			'barsize-vs-h_lin_Reh': "Main Spiral"}
varNames = {'barsize-vs-Re_lin_Reh': r"$\log \re$", 'barsize-vs-h_lin_Reh': r"$\log h$"}



# Stuff for LaTeX tables:

tableHeader = r"""\begin{table*}
\begin{minipage}{126mm}
\caption{Fits to Bar Size versus Stellar Mass}
\label{tab:bar-size-fits}
\renewcommand{\arraystretch}{1.5}
\begin{tabular}{lcccc}
\hline
Sample & $\alpha_{1}$ & $\beta_{1}$ & $\beta_{2}$ & $\log \: (M_{\rm brk} / \Msun)$ \\
  (1)  & (2)          & (3)         & (4)         &        (5)  \\
\hline
"""

tableEnd = r"""\hline
\end{tabular}

\medskip

Results of broken-linear fits to $\log \avis$ as a function of \logmstar{} (see
Eqn.~\ref{eq:barsize1}; parameter $\alpha_{2} = \alpha_{1} + (\beta_{1} - \beta_{2}) M_{\rm brk}$).
Parameter uncertainties are based on 2000 rounds of bootstrap resampling.

\end{minipage}
\end{table*}
"""

tableHeader_Reh = r"""\begin{table}
\caption{Fits to Bar Size versus \re{} and $h$}
\label{tab:bar-size-fits-reh}
\begin{tabular}{lcc}
\hline
Predictor    & $\alpha$ & $\beta$ \\
  (1)        & (2)          & (3) \\
\hline
"""

tableEnd_Reh = r"""\hline
\end{tabular}

\medskip

Results of linear fits to bar size $\log \avis$ as a function of \logre{} and $\log h$.
Parameter uncertainties are based on 2000 rounds of bootstrap resampling.

\end{table}
"""

templateLine = r"{0:15} & {1:25s} & {2:25s} & {3:25s} & {4:25s} \\"
templateLine_Reh = r"{0:10s} & {1:25s} & {2:25s} \\"



def MakeValLimitsTriplet( value, blob ):
	"""Given a central value and a string including the lower and upper bounds,
	returns a triplet of value, -err, +err
	
	blob = string in one of the following formats:
		['   (-1.0135,-0.2338',
		 ' (0.0509,0.1322',
		 ' (10.0960,10.2437',
		 ' (0.5136,0.6842)']
	"""
	blob = blob.strip().lstrip("(").rstrip(")")
	v1,v2 = blob.split(",")
	return (value, value - float(v1), float(v2) - value)

def GetPrecomputedFits( ):
	fitNames = []
	fitDict1 = {}
	fitDict_final = {}
	# get best-fit parameters
	dlines = [line for line in open(savedFitsFile) if line[0] != '#']
	for line in dlines:
		pp = line.split(":")
		fitName = pp[0]
		parameters = [float(p) for p in pp[1].split()]
		fitDict1[fitName] = parameters
		fitNames.append(fitName)
	# get parameter confidence intervals
	dlines = [line for line in open(savedUndertaintiesFile) if line[0] != '#']
	for line in dlines:
		pp = line.split(":")
		fitName = pp[0]
		pairs = [p for p in pp[1].split("),")]
		nVals = len(pairs)
		fitName_base = fitName.split("_confint")[0]
		bestFitValues = fitDict1[fitName_base]
		valueErrsTriplets = []
		for i in range(nVals):
			value = bestFitValues[i]
			pairText = pairs[i]
			f, e_low, e_high = MakeValLimitsTriplet(value, pairText)
			valueErrsTriplets.append([f, e_low, e_high])
		fitDict_final[fitName_base] = valueErrsTriplets
	
	return fitDict_final


def MakeLatexVal( inputVals ):
	f, e_low, e_high = inputVals
	return "$%.2f^{+%.2f}_{-%.2f}$" % (f, e_low, e_high)
	
def MakeMstarTable( fitResults ):
	"""Generate and save LaTeX table with fit parameters for log(bar-size) vs
	logMstar [broken-linear fit]
	
	Currently = Table 1 in paper.
	"""
	print("MakeMstarTable:")
	
	outf = open(tableFile, 'w')
	outf.write(tableHeader)
	# should have fits for parent-spiral and main-spiral ("barsize-vs-Mstar_brokenlin") samples
	for fitName in ["barsize-vs-Mstar_parent_brokenlin", "barsize-vs-Mstar_brokenlin"]:
		sampleName = sampleNames[fitName]
		thisFitResults = fitResults[fitName]
		alpha1Txt = MakeLatexVal(thisFitResults[0])
		beta1Txt = MakeLatexVal(thisFitResults[1])
#		alpha2Txt = MakeLatexVal(thisFitResults[2])
		mbrkTxt = MakeLatexVal(thisFitResults[2])
		beta2Txt = MakeLatexVal(thisFitResults[3])
		outLine = templateLine.format(sampleName, alpha1Txt, beta1Txt, beta2Txt, mbrkTxt)
		outf.write(outLine + "\n")
	outf.write(tableEnd)
	outf.close()


def MakeReHTable( fitResults ):
	"""Generate and save LaTeX table with fit parameters for log(bar-size) vs
	log(R_e) and vs log(h) [linear fits]
	
	Currently = Table 2 in paper.
	"""
	print("MakeReHTable:")
	
	outf = open(tableFile_Re, 'w')
	outf.write(tableHeader_Reh)
	for fitName in ['barsize-vs-Re_lin_Reh', 'barsize-vs-h_lin_Reh']:
		varName = varNames[fitName]
		thisFitResults = fitResults[fitName]
		print(thisFitResults)
		alpha1Txt = MakeLatexVal(thisFitResults[0])
		beta1Txt = MakeLatexVal(thisFitResults[1])
		outLine = templateLine_Reh.format(varName, alpha1Txt, beta1Txt)
		outf.write(outLine + "\n")
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

	fitResults = GetPrecomputedFits()

	MakeMstarTable(fitResults)
	MakeReHTable(fitResults)


if __name__ == '__main__':
	
	main(sys.argv)
