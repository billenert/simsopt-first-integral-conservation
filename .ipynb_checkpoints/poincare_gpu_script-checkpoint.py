'''
Run using https://github.com/billenert/simsopt_gpu_saw_testing, git checkout the simsopt_orm_simplified_gpu_saw_wip branch.

Run time on Jupyter NoteBook Login Node: 118 seconds
'''

import logging
import sys
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt

from simsopt.field.boozermagneticfield import (
        BoozerRadialInterpolant,
        InterpolatedBoozerField,
        ShearAlfvenHarmonic,
        ShearAlfvenWavesSuperposition
        )
from simsopt._core.util import parallel_loop_bounds
from simsopt.field.tracing import (
        MaxToroidalFluxStoppingCriterion,
        MinToroidalFluxStoppingCriterion,
        IterationStoppingCriterion,
        trace_particles_boozer_perturbed
)
from simsopt.util.constants import (
        ALPHA_PARTICLE_MASS as MASS,
        ALPHA_PARTICLE_CHARGE as CHARGE,
        FUSION_ALPHA_PARTICLE_ENERGY as ENERGY
)
from booz_xform import Booz_xform
from stellgap import AE3DEigenvector, saw_from_ae3d
import stellgap as sg
from scipy import integrate
from matplotlib.cm import ScalarMappable
import simsoptpp as sopp
import matplotlib as mpl

start_time = time.time()

# ----------- PARAMS ------------

filename = 'boozmn_precise_QH.nc'
ic_folder = 'initial_conditions'
helicity = -1

saw_omega = 32935.0
print("omega=", saw_omega)
s = np.linspace(0.05,0.95,30)
saw_srange = (s[0], s[-1], len(s))
saw_m = 17
saw_n = -17

def sah_profile(s):
    """SAH amplitude profile"""
    s = np.asarray(s)
    sc, lw, rw, h, p, off = 0.475541, 0.068345, 0.154077, 4.124030, 1.442968, 0.026951
    result = np.zeros_like(s, dtype=float)
    result = np.where(s <= sc, 
                      h * np.exp(-((sc - s)/lw)**p), 
                      h * np.exp(-((s - sc)/rw)**p))
    amplitude = 1000.0
    return (amplitude*(result + off) * s * (1-s))

saw_phihats = sah_profile(s)
saw_nharmonics = 1


# ---------- FIELD -----------

t1 = time.time()
equil = Booz_xform()
equil.verbose = 0
equil.read_boozmn(filename)
nfp = equil.nfp

bri = BoozerRadialInterpolant(
    equil=equil,
    order=3,
    no_K=True,
    N=nfp*helicity
)

degree = 3
srange = (0, 1, 15)
thetarange = (0, np.pi, 15)
zetarange = (0, 2*np.pi/nfp, 15)

field = InterpolatedBoozerField(
    bri,
    degree=3,
    srange=(0, 1, 15),
    thetarange=(0, np.pi, 15),
    zetarange=(0, 2*np.pi/nfp, 15),
    extrapolate=True,
    nfp=nfp,
    stellsym=True,
    initialize=['modB','modB_derivs']
)

# Evaluate error in interpolation
print('Error in |B| interpolation', 
    field.estimate_error_modB(1000),
    flush=True)

VELOCITY = np.sqrt(2*ENERGY/MASS)

# set up GPU interpolation grid
def gen_bfield_info(field, srange, trange, zrange):

	s_grid = np.linspace(srange[0], srange[1], srange[2])
	theta_grid = np.linspace(trange[0], trange[1], trange[2])
	zeta_grid = np.linspace(zrange[0], zrange[1], zrange[2])

	quad_pts = np.empty((srange[2]*trange[2]*zrange[2], 3))
	for i in range(srange[2]):
		for j in range(trange[2]):
			for k in range(zrange[2]):
				quad_pts[trange[2]*zrange[2]*i + zrange[2]*j + k, :] = [s_grid[i], theta_grid[j], zeta_grid[k]]


	field.set_points(quad_pts)
	G = field.G()
	iota = field.iota()
	diotads = field.diotads()
	I = field.I()
	modB = field.modB()
	J = (G + iota*I)/(modB**2)
	maxJ = np.max(J) # for rejection sampling

	psi0 = field.psi0

	# Build interpolation data
	modB_derivs = field.modB_derivs()

	dGds = field.dGds()
	dIds = field.dIds()

	quad_info = np.hstack((modB, modB_derivs, G, dGds, I, dIds, iota, diotads))
	quad_info = np.ascontiguousarray(quad_info)

	return quad_info, maxJ, psi0

# generate grid with 15 simsopt grid pts
n_grid_pts = 15
srange = (0, 1, 3*n_grid_pts+1)
trange = (0, np.pi, 3*n_grid_pts+1)
zrange = (0, 2*np.pi/nfp, 3*n_grid_pts+1)
quad_info, maxJ, psi0 = gen_bfield_info(field, srange, trange, zrange)


# ------------- INITIAL CONDITIONS ---------------------

s_init = np.loadtxt(f's_perturbed_init_poinc.txt', ndmin=1)
theta_init = np.loadtxt(f'chis_perturbed_init_poinc.txt', ndmin=1)
zeta_init = np.loadtxt(f'chis_perturbed_init_poinc.txt', ndmin=1)
vpar_init = np.loadtxt(f'vpar_perturbed_init_poinc.txt', ndmin=1)

points = np.zeros((s_init.size, 3))
points[:, 0] = s_init
points[:, 1] = theta_init
points[:, 2] = zeta_init
points = np.ascontiguousarray(points)
vpar_init = np.ascontiguousarray(vpar_init)
nparticles = len(points)

zetas = [0]
Phim = 17
Phin = -17
omega = saw_omega
omegan = omega/(Phin - Phim*helicity*nfp)
omegas = [omegan]

last_time = sopp.poincare_plotting(
	quad_pts=quad_info, 
	srange=srange,
	trange=trange,
	zrange=zrange, 
	stz_init=points,
	m=MASS, 
	q=CHARGE, 
	vtang=vpar_init, 
    mus=np.zeros(nparticles),
	tmax=1e-2, 
	tol=1e-8, 
	psi0=psi0, 
	nparticles=nparticles,
	saw_srange=saw_srange,
	saw_m=saw_m,
	saw_n=saw_n,
	saw_phihats=saw_phihats,
	saw_omega=saw_omega,
	saw_nharmonics=saw_nharmonics, dt_save=1e-6, MAX_PUNCTURES=1000, zetas=zetas, omegas=omegas)

# POSTPROCESSING: REMOVE ALL ZEROS

last_time = np.reshape(last_time, (-1, 5))
last_time[:2] = last_time[:2] % (2 * np.pi)

p_hits = []
running = []

for i in range(0, len(last_time)):
    if (not np.any(last_time[i]) and len(running) > 0):
        # print(i)
        p_hits.append(running)
        running = []
        continue
    if np.any(last_time[i]):
        running.append(last_time[i])
        if i % 1000 == 999:
            # print(i)
            p_hits.append(running)
            running = []
cmap = mpl.colormaps['tab20']

plt.figure(1, figsize=(10, 8))
plt.xlabel(r'$\chi$', fontsize=14)
plt.ylabel(r'$s$', fontsize=14)
plt.xlim([0, 2 * np.pi])
plt.ylim([0, 1])
counter = 0
for p in p_hits:
    hits = np.asarray(p)
    s_hits = hits[:, 0]
    helicity = -1
    chi_hits = (hits[:, 1] - nfp * helicity * hits[:, 2]) % (2 * np.pi)
    theta_hits = (hits[:, 1] + 2*np.pi) % (2 * np.pi) - np.pi
    plt.scatter(chi_hits, s_hits, s=2, edgecolors='none', marker='o', color=cmap(counter))
    counter += 1

plt.tight_layout()
plt.savefig('poincare_map.pdf', dpi=300)

print(time.time() - start_time, "seconds taken")