import numpy as np

D_venus = 108.2e9  # Distance moyenne de Vénus au Soleil en km
D_terre = 149.6e9  # Distance moyenne de la Terre au Soleil en km
R_soleil = 696340  # Rayon du Soleil en km

Cn2 = 1e-12        # structure constant Cn²
Los = 100          # outer scale (m)
Lis = 0.01         # inner scale (m)
c = 3e8
f = 10e9
lamb = c/f
k0  = 2*np.pi/lamb 

# dérivés
Kos = 2*np.pi / Los
km  = 5.92 / Lis

kb = 1.38e-23  # constante de Boltzmann en J/K
epsil = 0.5
epsil_0 = 8.854e-12  # permittivité du vide en F/m
re = 2.818e-15      # rayon classique de l'électron (m)

L0 = 2000e3  # échelle extérieure de turbulence en mètres
kappa_0 = 2*np.pi / L0

Axial_r = 35
Q_nu = 2 * Axial_r/np.pi
nu = 3.9
a0 = 8.75e74 # coefficient d'irrégularité
ar = 35 # rayon de l'antenne
gam = 90