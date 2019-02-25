# definitions of various samples and subsamples from S4G
import sys

projDir = "/Users/erwin/Documents/Working/Projects/Project_BarSizes/"
sys.path.append(projDir)
import datasets

s4gdata = datasets.s4gdata
nDisksTotal = len(s4gdata.name)



# general subsamples: all barred, all unbarred, all spirals
ii_barred = [i for i in range(nDisksTotal) if s4gdata.sma[i] > 0]
ii_unbarred = [i for i in range(nDisksTotal) if s4gdata.sma[i] <= 0]
ii_spirals = [i for i in range(nDisksTotal) if s4gdata.t_s4g[i] > -0.5]


# Limited subsample 2: spirals with D < 30 Mpc -- 856 galaxies: 483 barred, 373 unbarred
ii_all_limited2 = [i for i in ii_spirals if s4gdata.dist[i] <= 30]
ii_barred_limited2 = [i for i in ii_all_limited2 if i in ii_barred]
ii_unbarred_limited2 = [i for i in ii_all_limited2 if i not in ii_barred]

# Parent Spiral Sample: limited subsample 2 + logMstar = 9--11:
ii_all_limited2_m9to11 = [i for i in ii_all_limited2 if s4gdata.logmstar[i] >= 9 and s4gdata.logmstar[i] <= 11]
ii_barred_limited2_m9to11 = [i for i in ii_all_limited2_m9to11 if i in ii_barred]



# Spirals with R_e *and* h
ii_all_Reh = [i for i in ii_spirals if s4gdata.Re_kpc[i] > 0 and s4gdata.h_kpc[i] > 0]
ii_barred_Reh = [i for i in ii_all_Reh if i in ii_barred]
ii_unbarred_Reh = [i for i in ii_all_Reh if i not in ii_barred]


# Main Spiral Sample: Spirals with D < 30 Mpc, valid R_e *and* h, *and* logMstar = 9--11:
ii_all_Reh_m9to11 = [i for i in ii_spirals if s4gdata.logmstar[i] >= 9 and s4gdata.logmstar[i] <= 11 and s4gdata.Re_kpc[i] > 0 and s4gdata.h_kpc[i] > 0]
ii_barred_Reh_m9to11 = [i for i in ii_all_Reh_m9to11 if i in ii_barred]
ii_unbarred_Reh_m9to11 = [i for i in ii_all_Reh_m9to11 if i not in ii_barred]

ii_all_limited2_Reh = [i for i in ii_all_Reh if s4gdata.dist[i] <= 30]
ii_barred_limited2_Reh = [i for i in ii_all_limited2_Reh if i in ii_barred]
ii_unbarred_limited2_Reh = [i for i in ii_all_limited2_Reh if i not in ii_barred]

ii_all_lim2m9to11_Reh = [i for i in ii_all_limited2_Reh if s4gdata.logmstar[i] >= 9 and s4gdata.logmstar[i] <= 11]
ii_barred_lim2m9to11_Reh = [i for i in ii_all_lim2m9to11_Reh if i in ii_barred]
ii_unbarred_lim2m9to11_Reh = [i for i in ii_all_lim2m9to11_Reh if i not in ii_barred]

# useful aliases which are more directly descriptive
ii_all_D30 = ii_all_limited2
ii_all_D30_m9to11 = ii_all_limited2_m9to11
ii_barred_D30_m9to11 = ii_barred_limited2_m9to11
ii_barred_D30_m9to11_Reh = ii_barred_lim2m9to11_Reh
