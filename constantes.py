import numpy as np

D_venus = 108.2e9  # Distance moyenne de Vénus au Soleil en m
D_terre = 149.6e9  # Distance moyenne de la Terre au Soleil en m
R_soleil = 696340e3  # Rayon du Soleil en m

Cn2 = 1e-12        # structure constant Cn²
Los = 100     # outer scale (m)
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

L0 = 2e7  # échelle extérieure de turbulence en mètres
kappa_0 = 2*np.pi / L0

Axial_r = 35
Q_nu = 2 * Axial_r/np.pi
nu = 3.9 #indice spectral de la turbulence
a0 = 8.75e74 # coefficient d'irrégularité
ar = 35 # rayon de l'antenne
gam = 90
omega = 2.7*10**(-6)  # vitesse angulaire du soleil en rad/s

b = 0.4832 # facteur d'éapproximation de la fonction Airy

# Table 2 (Ho et al., 2008) — valeurs RMS
_TABLE2 = {
    "SEP_deg": np.array([0.25, 0.30, 0.40, 0.50, 0.60, 0.70]),
    "S_deg":  np.array([41.8, 15.3,  3.2,  0.9,  0.340, 0.146]),   # deg
    "X_deg":  np.array([ 3.1,  1.1,  0.237,0.069,0.026, 0.011]),   # deg (mdeg convertis)
    "Ka_deg": np.array([0.245,0.090,0.019,0.0054,0.0020,0.0008]),  # deg (mdeg convertis)
}

# Fréquences porteuses typiques DSN (tu peux ajuster si besoin)
FREQ = {
    "S": 2.3e9,    # Hz
    "X": 8.4e9,    # Hz (ou 8.6e9 selon tes choix)
    "Ka": 32.0e9,  # Hz
}