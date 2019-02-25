# Python code for assembling S$G-based local bar-size and fraction dataset
# 
# 	ListDataFrame with
# 	name, M_star, B-V_tc, g-r_tc, a_max_obs[arcsec, kpc], amax_dp[arcsec, kpc], distance,
# 	distance_source, inclination
# 		distance_source = direct (Cepheids, SBF, TRGB, etc), T-F, redshift	
# 
# 	Two separate datasets:
# 	1. All S4G from DG16 (unbarred and barred)
#      This should be the 1334 galaxies with valid M_star and distances in DG16,
#      out of the 1445 galaxies in their low-inc disk sample
# 	
# 	2. Our local galaxies meeting S0--Sd, D < 25 Mpc, i = 45--70
# 		ED16b
# 		bardata_i40-70_smaller.txt

import math
import scipy.interpolate
#import scipy.stats
import numpy as np

import astro_utils, angles
import datautils as du

baseDir = "/Users/erwin/Documents/Working/Projects/Project_BarSizes/"

# Data tables (retrieved via Vizier) for Salo+2015, Herrera-Endoqui+2015, and 
# Diaz-Garcia+2016 
tableDir = "/Beleriand/Astronomy Papers/"
tableSalo15_1 = tableDir + "salo+15_tables/table1.dat"
tableSalo15_6 = tableDir + "salo+15_tables/table6.dat"
tableSalo15_7 = tableDir + "salo+15_tables/table7.dat"
tableHE15_2 = tableDir + "herrera-endoqui+15_tables/table2.dat"
tableDG16_A1 = tableDir + "diaz-garcia+16_tables/tablea1.dat"
tableDG16_A3 = tableDir + "diaz-garcia+16_tables/tablea3.dat"
# General S4G table, downloaded from IPAC on 11 Feb 2017
tableS4G = baseDir + "spitzer.s4gcat22704.tbl.txt"
# Consolandi+2016 SDSS data for Virgo Supercluster, etc.
tableC16_1 = tableDir + "consolandi+16_table1/table1.dat"

# Galaxy Zoo 2 bar sizes and extra data
tableGZ2_barsizes = baseDir+'GalaxyZoo2_barlengths_alldata.txt'

# Virgo and Fornax Cluster members
virgoNameFile = baseDir+"virgo-cluster-members.txt"
fornaxNameFile = baseDir+"fornax-cluster-members.txt"




# Cleaned-up data for computing spline interpolation of f(B-v) vs B_tc
# (fraction of galaxies with HyperLeda B-V_tc values as function of B_tc
#x_Btc = [7, 8.25, 8.75, 9.25, 9.75, 10.25, 10.75, 11.25, 11.75, 12.25, 12.75, 13.25, 13.75, 14.25, 14.75, 15.25, 15.75, 16.25, 17]
#y_fBmV = [1.0, 1.0, 1.0, 1.0, 1.0, 0.95, 0.9565217391304348, 0.84, 0.7692307692307693, 0.6145833333333334, 0.45535714285714285, 0.3434343434343434, 0.15841584158415842, 0.23076923076923078, 0.23404255319148937, 0.125, 0.05, 0.01, 0.0]
x_Btc = [7.0, 8.25, 8.75, 9.25, 9.75, 10.25, 10.75, 11.25, 11.75, 12.25, 12.75, 13.25, 13.75, 14.25, 14.75, 15.25, 15.75, 16.25]
y_fBmV = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9722222222222222, 0.8840579710144928, 0.8125, 0.6222222222222222, 0.5632183908045977, 0.4074074074074074, 0.2727272727272727, 0.3442622950819672, 0.2978723404255319, 0.10714285714285714, 0.01, 0.0]

fBmV_akimaspline = scipy.interpolate.Akima1DInterpolator(x_Btc, y_fBmV)



hdrText = """# This table summarizes useful information about galaxies in the Parent Disc Sample of
# Erwin (2017, in prep).
#
# Except for the following columns, values are from Herrera-Endoqui et al. (2015: A&A, 582, A86)
#    dist, logmstar -- distance in Mpc and log of galaxy stellar mass (in solar masses), from 
#       Munoz-Mateos et al. (2015: ApJS, 219, 3)
#    bar_strength -- Buta et al. (2015, ApJS, 217, 32)
#    A2, A4 -- Diaz-Garcia et al. (2016a, 587, A160)
#    Re, Re_kpc, h_kpc -- Salo et al. (2015, ApJS, 219, 4)
#    R25, etc.; t_s4g -- main S4G table
#    B_tc, BmV_tc, m21c, W_gas, t_leda -- HyperLeda (as of 29 March 2017)
#    gmr_tc -- total g-r color, based on BmV_tc as described in paper
#    weight_BmVtc -- B_t-based weights for galaxy colors (see paper)
#    w25, w30, w40 -- V/V_max weights assuming D_max = 25, 30, or 40 Mpc
#    M_HI, logfgas -- based on dist + m21c, as described in paper
#    inclination -- Munoz-Mateos et al. (2015)
#    sma_dp_kpc2 -- deprojected bar semi-major axis (in kpc) using bar size and
#       PA from Herrera-Endoqui et al. and galaxy inclination and PA from Munoz-Mateos et al.
#
# "No data" values:
# For the following columns "no data" is indicated by the value -99:
#    BmV_tc, gmr_tc, m21c, M_HI, logfgas
# For the following columns "no data" is indicated by the value 0:
#    sma, sma_kpc, sma_ell_kpc, sma_dp_kpc, sma_dp_kpc2, sma_ell_dp_kpc2
#
"""

# -99 -- BmV_tc, gmr_tc, m21c, M_HI, logfgas
# 0 -- sma, sma_kpc, sma_ell_kpc, sma_dp_kpc, sma_dp_kpc2, sma_ell_dp_kpc2
colNames_to_save = ["name", "logmstar", "dist", "B_tc", "BmV_tc", "weight_BmVtc", "gmr_tc",
			"m21c", "M_HI", "logfgas", "w25", "w30", "w40", "sma", "sma_kpc", "sma_ell_kpc", 
			"sma_dp_kpc", "sma_dp_kpc2", "sma_ell_dp_kpc2", "bar_strength", "A2", "A4",
			"inclination", "R25", "R25_5", "R25_kpc", "R25_5_kpc", "R25c_kpc", "Re", "Re_kpc",
			"h_kpc",  "W_gas", "t_s4g", "t_leda"]
def WriteTableToFile( dataFrame, outFilename ):
	nEntries = len(dataFrame.name)
	hdrLine = "#"
	for cname in colNames_to_save:
		hdrLine += " " + cname
	
	outf = open(outFilename, 'w')
	outf.write(hdrText)
	outf.write(hdrLine + "\n")
	for i in range(nEntries):
		newLine = ""
		for cname in colNames_to_save:
			newLine += str(dataFrame[cname][i]) + " "
		outf.write(newLine + "\n")
	outf.close()
	
	

def VmaxWeight( distance, R_25, R_25_limit=30.0, maxSurveyDist=30.0 ):
	"""Returns W = V_tot / V_max, where V_tot is the total survey volume
	(out to a distance of maxSurveyDist in Mpc) and V_max is volume out to 
	distance = distMax in Mpc, with distMax = maximum distance object could 
	have been observed at, assuming an R_25 limiting radius of R_25_limit 
	and an observed radius of R_25 (both in arcsec).
	
	If V_tot > V_max (i.e., the galaxy would have been observed regardless of
	distance), then W = 1
	
	For S4G, R25_limit = 30 arcsec
	"""
	V_tot = maxSurveyDist**3
	distMax = distance * (R_25 / R_25_limit)
	V_max = distMax**3
	if V_max > V_tot:
		return 1.0
	else:
		return (V_tot / V_max)



def BmV_to_gmr( BmV ):
	"""Relation from Cook+2014 (2014MNRAS.445..890C) to transform
	galaxy B-V color to g-r.
	"""
	ii_bad = [i for i in range(len(BmV)) if BmV[i] == -99.0]
	gmr = 1.12*BmV - 0.18
	gmr[ii_bad] = -99.0
	return gmr


def IsDisk( htypeText ):
	"""Simple function to take morphological type code from Buta+2015 and
	determine if it refers to a disk galaxy (no Sm or Im)
	"""
	# simple way to get only proper disks: anything with "I" or "m" in text
	if (htypeText.find("I") >= 0) or (htypeText.find("m") >=0):
		return False
	else:
		return True


def GetConsolandiDataFromLine( dline ):
	sdssName = dline[:20].strip()
	vccName = dline[21:25].strip()
	ngcName = dline[34:38].strip()
	ra = float(dline[39:48])
	dec = float(dline[49:58])
	gmag = float(dline[59:64])
	imag = float(dline[65:70])
	gmi = float(dline[71:76])
	logMstar = float(dline[77:81])
	return (ngcName, vccName, sdssName, ra, dec, gmag, imag, gmi, logMstar)

def GetConsolandiData( fileName=tableC16_1 ):
	dlines = [line for line in open(fileName) if len(line) > 1 and line[0] != "#"]
	consolandiDict = {}
	for dline in dlines:
		x = GetConsolandiDataFromLine(dline)
		if x[0] != "":
			gname = "NGC" + x[0]
		elif x[1] != "":
			gname = "VCC" + x[1]
		else:
			gname = x[2]
		consolandiDict[gname] = x[3:]
	return consolandiDict





# *** Read in main tables

# Main S4G table (from IPAC)
ds4g = du.ReadCompositeTable(tableS4G, columnRow=71, dataFrame=True)
nS4GMain = len(ds4g.name)
s4gdict = { ds4g.name[i]: [ds4g.t[i], 0.5 * ds4g.sma1_25p5[i], 3 * 10**ds4g.logd25[i], 
			ds4g.c31_1[i], ds4g.c42_1[i], ds4g.vrad[i], ds4g.dmean[i]] for i in range(nS4GMain) }

# Read in Herrera-Endoqui+2015 and Diaz-Garcia+2016 tables: 1344 galaxies
#    This is where we get M_star, distance -- also Hubble type?
# Diaz-Garcia+2016 tables
d16 = du.ReadCompositeTable(tableDG16_A1, columnRow=74, dataFrame=True)
d16a3 = du.ReadCompositeTable(tableDG16_A3, columnRow=26, dataFrame=True)

# Herrera-Endoqui+2015 Table 2: 2387 lines (one for each feature in each galaxy)
# 1146 galaxies with bars
#    This is where we get bar measurements (e.g., sma, sma_ell, sma_ell_dp)
#    Other potentially useful stuff: sma, sma_dp [latter only exists if ell-based
#    measurements also exist]
# HType
d15 = du.ReadCompositeTable(tableHE15_2, columnRow=50, dataFrame=True)
# dict mapping galaxy names to bar sizes d15.sma_ell_dp[i]
d15dict = { d15.Name[i]: [d15.sma[i], d15.sma_dp[i], d15.sma_ell[i], d15.sma_ell_dp[i], 
			d15.PA[i], d15.quality[i], d15.HType[i]] 
			for i in range(len(d15.Name)) if d15.feature[i] == "bar" }
d15barnames = list(d15dict.keys())

# For reference: barred galaxies in Herrera-Endoqui+2015 that are *not* in
# Diaz-Garcia+2016
gnames_HE15_not_in_DG16 = [gname for gname in d15barnames if gname not in d16.Galaxy]


# *** DEFINITION OF "GOOD D16" SUBSAMPLE

# Lists of galaxies to exclude for various reasons: 22 galaxies total in parent S4G
# get names of galaxies with dubious distances (IMPORTANT because some of these can
# produce very large 1/Vmax weights)
dubiousDistNames = [ds4g.name[i] for i in range(nS4GMain) if ds4g.dmean[i] == 0 and ds4g.vrad[i] <= 500.0]
# get names of galaxies with D_25 < 1 arcmin
badR25Names = [ds4g.name[i] for i in range(nS4GMain) if ds4g.logd25[i] < 1.0]
badGalaxyNames = set(dubiousDistNames + badR25Names)

# construct list of indices for "good" galaxies (those with valid M_star and distance): 1334 galaxies
#    index into the *parent* (1344-galaxy) D16 sample, above
# 1. Exclude galaxies with bad M_star and/or bad distances ==> 1334 galaxies left in sample
ii_d16parent_goodnames1 = [i for i in range(len(d16.Galaxy)) if d16['M*'][i] > 0 and d16['Dist'][i] > 0]
# 2. Exclude dubiousDistNames [defined above] ==> 1322 galaxies left in sample
ii_d16parent_goodnames = [i for i in ii_d16parent_goodnames1 if d16.Galaxy[i] not in badGalaxyNames]
d16_goodnames = [d16.Galaxy[i] for i in ii_d16parent_goodnames]
nDisksTot = nD16good = len(ii_d16parent_goodnames)


# *** Start constructing data vectors:

# Get names, Mstar, distances, etc., from Diaz-Garcia+16
d16_good_logmstar = [math.log10(d16['M*'][i]) for i in ii_d16parent_goodnames]
d16_good_dist = [d16['Dist'][i] for i in ii_d16parent_goodnames]
d16_good_VHI = [d16['VHI'][i] for i in ii_d16parent_goodnames]
d16_good_DM_ratio = [d16['Mh/M*'][i] for i in ii_d16parent_goodnames]


# Get B_tc and B-V values for S4G galaxies (from HyperLeda)
# 1334 galaxies (d16_goodnames) in file, 616 with B-V_tc values
dlines = [line for line in open(baseDir + "s4g_goodnames_ledadata.txt") if len(line) > 1 and line[0] != "#"]
BmV_e = []
BmV_tc =[]
B_tc = []
m21c = []
r_25c = []
W_gas = []
t_leda = []
#                 name                  |      type       |        t        |     logd25      |       bve       |      vmaxg      |      logdc      |       btc       |      bvtc       |      m21c       |

i_logd25 = 3
i_bve = 4
i_vmaxg = 5
i_logdc = 6
i_btc = 7
i_bvtc = 8
i_m21c = 9

for dline in dlines:
	pp = dline.split("|")
	gname = pp[0].strip()
	if gname in d16_goodnames:
		btc = float(pp[i_btc])
		B_tc.append(btc)
		try:
			bmv_e = float(pp[i_bve])
		except ValueError:
			bmv_e = -99.0
		try:
			bmv_tc = float(pp[i_bvtc])
		except ValueError:
			bmv_tc = -99.0
		BmV_e.append(bmv_e)
		BmV_tc.append(bmv_tc)
		try:
			W = float(pp[i_vmaxg])
		except:
			W = -99.0
		W_gas.append(W)
		try:
			m21 = float(pp[i_m21c])
		except ValueError:
			m21 = -99.0
		m21c.append(m21)
		t_leda.append(float(pp[2]))
	
		# radii in arc sec
		logdc = float(pp[i_logdc])
		r_25c.append(3 * 10**logdc)

d16_good_Btc = np.array(B_tc)
d16_good_BmV_e = np.array(BmV_e)
d16_good_BmV_tc = np.array(BmV_tc)
d16_weights_BmVtc = 1.0 / fBmV_akimaspline(d16_good_Btc)
good_gmr_tc = BmV_to_gmr(d16_good_BmV_tc)
d16_good_m21c = np.array(m21c)
# compute M_HI and log(M_HI/M_star)
d16_good_MHI = np.array([astro_utils.HIMass(d16_good_m21c[i],d16_good_dist[i],1) for i in range(nD16good)])
# M_baryon = 1.4*M_HI + Mstar
d16_good_logMbaryon = np.array([np.log10(1.4*d16_good_MHI[i] + 10**d16_good_logmstar[i]) for i in range(nD16good)])
d16_good_logfgas = np.array([np.log10(d16_good_MHI[i]) - d16_good_logmstar[i] for i in range(nD16good)])
# Redefine bad values to be = -99
ii_bad_HI = [i for i in range(nD16good) if d16_good_logfgas[i] > 10]
d16_good_logMbaryon[ii_bad_HI] = -99.0
d16_good_logfgas[ii_bad_HI] = -99.0

# Get NED extinction values
dlines = [line for line in open(baseDir + "s4g_goodnames_neddata.txt") if len(line) > 1 and line[0] != "#"]
A_B = []
A_V =[]
for dline in dlines:
	pp = dline.split("|")
	gname = pp[0].strip()
	if gname in d16_goodnames:
		A_B.append(float(pp[1]))
		A_V.append(float(pp[2]))
A_B = np.array(A_B)
A_V = np.array(A_V)

# Extinction correction for B-V_e value
d16_good_BmV_ec = d16_good_BmV_e - (A_B - A_V)


d16a3dict = { d16a3.Galaxy[i]: [d16a3.A2[i], d16a3.A4[i]] for i in range(len(d16a3.Galaxy)) }

# collect or determine bar values (including deprojected values)
d16_bar_sma = []
d16_bar_sma_kpc = []
d16_bar_sma_ell_kpc = []
d16_bar_sma_dp_kpc = []
d16_bar_pa = []
d16_bar_quality = []
d16_bar_strength = []
d16_bar_a2 = []
d16_bar_a4 = []
for i in range(nD16good):
	gname = d16_goodnames[i]
	if gname in d15barnames:
		sma, sma_dp, sma_ell, smal_ell_dp, bar_pa, quality, HType = d15dict[gname]
		try:
			a2, a4 = d16a3dict[gname]
		except KeyError:
			a2 = a4 = -999.999
		d16_bar_a2.append(a2)
		d16_bar_a4.append(a4)
		d16_bar_pa.append(bar_pa)
		distMpc = d16_good_dist[i]
		kpcScale = astro_utils.pcperarcsec(distMpc)/1e3
		sma_kpc = kpcScale * sma
		# apply pixel-scale correction for sma_ell
		sma_ell_kpc = kpcScale * sma_ell * 0.75
		sma_dp_kpc = kpcScale * sma_dp
		d16_bar_quality.append(int(quality))
		if HType.find("SB") > -1:
			d16_bar_strength.append(1)
		else:
			d16_bar_strength.append(2)
	else:
		sma = sma_dp = sma_kpc = sma_ell_kpc = sma_dp_kpc = 0.0
		d16_bar_pa.append(-1000)
		d16_bar_quality.append(0)
		d16_bar_strength.append(3)
		d16_bar_a2.append(-999.999)
		d16_bar_a4.append(-999.999)
	d16_bar_sma.append(sma)
	d16_bar_sma_kpc.append(sma_kpc)
	d16_bar_sma_ell_kpc.append(sma_ell_kpc)
	d16_bar_sma_dp_kpc.append(sma_dp_kpc)


# Salo+2015 Table 1 (for diskPA, ell)
# Note that "PGC052336" has an empty row
salo15t1 = du.ReadCompositeTable(tableSalo15_1, columnRow=31, dataFrame=True)
# Table 6 -- single-Sersic fits -- has numerous blank lines where no fit succeeded
def GetRe( line ):
	if len(line) < 40:
		return 0.0
	else:
		return float(line.split()[5])

# make sure to get the *largest* disk-scale-length component value (for the
# 20 or 30 cases where a galaxy was fit by two or more "D" components)
def GetSalo15ScaleLengths( filename=tableSalo15_7 ):
	dlines = [line for line in open(filename) if len(line) > 1 and line[0] != "#"]
	
	gnameList = []
	dlinesDict = {}
	for line in dlines:
		pp = line.split("|")
		gname = pp[1].strip()
		if gname not in gnameList:
			gnameList.append(gname)
			dlinesDict[gname] = [line]
		else:
			dlinesDict[gname].append(line)
			
	scaleLengthDict = {}
	for gname in gnameList:
		thisGalaxyLines = dlinesDict[gname]
		hvals = []
		for line in thisGalaxyLines:
			pp = line.split("|")
			component = pp[5].strip()
			function = pp[6].strip()
			if (component == "D") and (function == "expdisk"):
				hvals.append(float(pp[22]))
		if len(hvals) > 0:
			scaleLengthDict[gname] = max(hvals)
		
	return scaleLengthDict

def GetSalo15BtoT( filename=tableSalo15_7 ):
	dlines = [line for line in open(filename) if len(line) > 1 and line[0] != "#"]
	
	gnameList = []
	BotTDict = {}
	bulgeFound = False
	for line in dlines:
		pp = line.split("|")
		gname = pp[1].strip()
		component = pp[5].strip()
		function = pp[6].strip()
		if gname not in gnameList:
			bulgeFound = False
			# new galaxy!
			gnameList.append(gname)
			if (component == "B") and (function == "sersic"):
				BotTDict[gname] = float(pp[7])
				bulgeFound = True
		elif not bulgeFound:
			if (component == "B") and (function == "sersic"):
				BotTDict[gname] = float(pp[7])
				bulgeFound = True
	return BotTDict


# Get global, single-Sersic R_e from Table 6
salo15_Re = [GetRe(line) for line in open(tableSalo15_6) if line[0] != "#"]
# Get exp-disk scale length (if it exists) from Table 7
dd_s15t7 = GetSalo15ScaleLengths()
s15t7_gnames = list(dd_s15t7.keys())
salo15_h = [dd_s15t7[gname] if gname in s15t7_gnames else 0.0  for gname in salo15t1.Name ]
# Get Sersic-based B/T (if it exists) from Table 7
dd_s15t7 = GetSalo15BtoT()
s15t7_gnames = list(dd_s15t7.keys())
salo15_BtoT = [dd_s15t7[gname] if gname in s15t7_gnames else 0.0  for gname in salo15t1.Name ]

# dict mapping galaxy names to disk orientation params
salo15dict = { salo15t1.Name[i]: [salo15t1.PA[i], salo15t1.Ell[i], salo15_Re[i], salo15_h[i],
				salo15_BtoT[i]] for i in range(len(salo15t1.Name)) }
salo15names = list(salo15dict.keys())

# Add R_25.5, R_25, inclinations and deprojected bar sma [using visual length]
R25_5 = []
R25 = []
R25_5kpc = []
R25kpc = []
r_25c_kpc = []
Re = []
Re_kpc = []
h_kpc = []
inclinations = []
d16_bar_sma_ell_dp_kpc2 = []
d16_bar_sma_dp_kpc2 = []
d16_bar_deltaPA_dp = []
T_s4g = []
c31 = []
c42 = []
BtoT = []
d16_good_Vrad = []
d16_good_dmean = []
d16_weights_vvmax_dmax25 = []
d16_weights_vvmax_dmax30 = []
d16_weights_vvmax_dmax40 = []
for i in range(nD16good):
	gname = d16_goodnames[i]
	diskPA, ellipticity, r_e, h, BtoT_s15 = salo15dict[gname]
	inclination = angles.ifrome(ellipticity)
	inclinations.append(inclination)
	distMpc = d16_good_dist[i]
	kpcScale = astro_utils.pcperarcsec(distMpc)/1e3
	t_s4g, r25_5, r25, c3_1, c4_2, vrad, dmean = s4gdict[gname]
	d16_weights_vvmax_dmax25.append(VmaxWeight(distMpc, r25, maxSurveyDist=25.0))
	d16_weights_vvmax_dmax30.append(VmaxWeight(distMpc, r25, maxSurveyDist=30.0))
	d16_weights_vvmax_dmax40.append(VmaxWeight(distMpc, r25, maxSurveyDist=40.0))
	d16_good_Vrad.append(vrad)
	d16_good_dmean.append(dmean)
	T_s4g.append(t_s4g)
	R25.append(r25)
	R25_5.append(r25_5)
	R25kpc.append(r25 * kpcScale)
	R25_5kpc.append(r25_5 * kpcScale)
	r_25c_kpc.append(r_25c[i] * kpcScale)
	Re.append(r_e)
	Re_kpc.append(r_e * kpcScale)
	h_kpc.append(h * kpcScale)
	c31.append(c3_1)
	c42.append(c4_2)
	BtoT.append(BtoT_s15)
	if gname in d15barnames:
		barPA = d16_bar_pa[i]
		deprojFactor = angles.deprojectr(barPA - diskPA, inclination, 1)
		deltaPA_dp = angles.deprojectpa(barPA - diskPA, inclination)
		sma_kpc_dp = d16_bar_sma_kpc[i] * deprojFactor
		sma_ell_kpc_dp = d16_bar_sma_ell_kpc[i] * deprojFactor
		d16_bar_sma_dp_kpc2.append(sma_kpc_dp)
		d16_bar_deltaPA_dp.append(deltaPA_dp)
		if d16_bar_sma_ell_kpc[i] > 0:
			d16_bar_sma_ell_dp_kpc2.append(sma_ell_kpc_dp)
		else:
			d16_bar_sma_ell_dp_kpc2.append(0.0)
	else:
		d16_bar_sma_dp_kpc2.append(0.0)
		d16_bar_sma_ell_dp_kpc2.append(0.0)
		d16_bar_deltaPA_dp.append(-99.0)

# 1/V_max weights
d16_weights_vvmax_dmax25 = np.array(d16_weights_vvmax_dmax25)
d16_weights_vvmax_dmax30 = np.array(d16_weights_vvmax_dmax30)
d16_weights_vvmax_dmax40 = np.array(d16_weights_vvmax_dmax40)


# Add Consolandi+2016 SDSS mags, colors, logMstar
consolandiDict = GetConsolandiData()
consolandiNames = list(consolandiDict.keys())
gmag = []
imag = []
gmi = []
logMstar_c16 = []
for j in range(nD16good):
	gname = d16_goodnames[j]
	if gname in consolandiNames:
		g, i, gmi_color = consolandiDict[gname][2:5]
	else:
		g = i = gmi_color = -99.0
	gmag.append(g)
	imag.append(i)
	gmi.append(gmi_color)


# Add environment coding
virgoNames = [line.strip() for line in open(virgoNameFile) if len(line) > 1 and line[0] != "#"]
fornaxNames = [line.strip() for line in open(fornaxNameFile) if len(line) > 1 and line[0] != "#"]
environment = []
for gname in d16_goodnames:
	if gname in virgoNames:
		environment.append("Virgo")
	elif gname in fornaxNames:
		environment.append("Fornax")
	else:
		environment.append("field")


# construct final ListDataFrame object
# Notes: R25 = mu_B = 25 radius from main S4G table at IPAC
#        R25_5 = mu_3.6 = 25.5 radius
#        R25c_kpc = extinction-corrected mu_B = 25 radius from HyperLeda
dataList = [ np.array(d16_goodnames), np.array(d16_good_logmstar), np.array(d16_good_dist),
			np.array(d16_good_Vrad), np.array(d16_good_Btc), np.array(d16_good_BmV_tc), 
			np.array(d16_weights_BmVtc), np.array(d16_good_BmV_ec), 
			np.array(good_gmr_tc), np.array(gmi), np.array(d16_good_m21c),
			np.array(d16_good_MHI), np.array(d16_good_logMbaryon), np.array(d16_good_logfgas),
			np.array(d16_weights_vvmax_dmax25), np.array(d16_weights_vvmax_dmax30),
			np.array(d16_weights_vvmax_dmax40), np.array(d16_bar_sma), np.array(d16_bar_sma_kpc), 
			np.array(d16_bar_sma_ell_kpc), np.array(d16_bar_sma_dp_kpc),
			np.array(d16_bar_sma_dp_kpc2), np.array(d16_bar_sma_ell_dp_kpc2), 
			np.array(d16_bar_deltaPA_dp),
			np.array(d16_bar_strength), np.array(d16_bar_quality), 
			np.array(d16_bar_a2), np.array(d16_bar_a4), np.array(W_gas), np.array(inclinations),
			np.array(R25), np.array(R25_5), np.array(R25kpc), np.array(R25_5kpc),
			np.array(r_25c_kpc), np.array(Re), np.array(Re_kpc), np.array(h_kpc), np.array(d16_good_VHI), 
			np.array(d16_good_DM_ratio), np.array(d16_good_dmean),
			np.array(c31), np.array(c42), np.array(BtoT), np.array(T_s4g), np.array(t_leda), 
			np.array(environment) ]
colNames = ["name", "logmstar", "dist", "Vrad", "B_tc", "BmV_tc", "weight_BmVtc", "BmV_ec", "gmr_tc", "gmi", 
			"m21c", "M_HI", "logMbaryon", "logfgas", "w25", "w30", "w40", "sma", "sma_kpc", "sma_ell_kpc", 
			"sma_dp_kpc", "sma_dp_kpc2", "sma_ell_dp_kpc2", "deltaPA_bar_dp", "bar_strength", "quality",
			"A2", "A4", "W_gas", "inclination", "R25", "R25_5", "R25_kpc", "R25_5_kpc", "R25c_kpc", 
			"Re", "Re_kpc", "h_kpc", "VHI", "DM_ratio", "dmean", 
			"c31", "c42", "BtoT", "t_s4g", "t_leda", "environment"]
s4gdata = du.ListDataFrame(dataList, colNames)


# index vector for all galaxies with bar feature (including those without sma_ell and
# deprojected sizes) = 749 galaxies
#    bars with quality = 1 -- 305 galaxies
#    bars with quality = 1 or 2 = 702 galaxies
ii_allbars = [i for i in range(nD16good) if s4gdata.sma[i] > 0]
ii_q1bars = [i for i in ii_allbars if s4gdata.quality[i] == 1]
ii_q12bars = [i for i in ii_allbars if s4gdata.quality[i] in [1,2]]



# GZoo bar sizes and related info (logMstar, z, etc.)
h11barsizes = du.ReadCompositeTable(tableGZ2_barsizes, columnRow=0, dataFrame=True)

