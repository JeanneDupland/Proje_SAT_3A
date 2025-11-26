import numpy as np

D_venus = 108.2e6  # Distance moyenne de Vénus au Soleil en km
D_terre = 149.6e6  # Distance moyenne de la Terre au Soleil en km
R_soleil = 696340  # Rayon du Soleil en km

Cn2 = 1e-12        # structure constant Cn²
Los = 100          # outer scale (m)
Lis = 0.01         # inner scale (m)
c= 3e8
f = 10e9
lamb = c/f
k0  = 2*np.pi/lamb 

# dérivés
Kos = 2*np.pi / Los
km  = 5.92 / Lis