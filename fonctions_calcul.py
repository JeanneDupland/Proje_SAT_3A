# Nom du module à discuter
import numpy as np
import constantes as c
import matplotlib.pyplot as plt
import scipy.integrate as int
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm
from scipy.special import gamma, factorial



def distance_terre_venus(angle_terre, angle_venus):
    """
    Calcule la distance entre la Terre et Vénus en fonction de leurs angles respectifs autour du Soleil.
    
    :param angle_terre: Angle de la Terre autour du Soleil en radians.
    :param angle_venus: Angle de Vénus autour du Soleil en radians.
    :return: Distance entre la Terre et Vénus en km.
    """
    x_terre = c.D_terre * np.cos(angle_terre)
    y_terre = c.D_terre * np.sin(angle_terre)
    
    x_venus = c.D_venus * np.cos(angle_venus)
    y_venus = c.D_venus * np.sin(angle_venus)
    
    distance = np.sqrt((x_terre - x_venus)**2 + (y_terre - y_venus)**2)
    return distance

def SEP_dist(SEP):
    """
    Calcule la distance du rayon au soleil en fonction de l'angle sun earth probe (SEP).
    """
    Dterre = c.D_terre
    Rsoleil = c.R_soleil
    return Dterre * np.sin(np.radians(SEP)) 

def densite_elec(r):
    """
    Calcule la densité électronique en fonction de la distance r au centre du Soleil.
    """
    rs = c.R_soleil
    return 2.21e14/(r/rs)**6 + 1.55e14/(r/rs)**2.3

def S2D_vonKarman(kx, kz):
    k_perp2 = kx**2 + kz**2
    return 0.055 * c.Cn2 * (k_perp2 + c.Kos**2)**(-4/3)

def F_filter(ky, kz, xR):
    k_perp = (ky**2 + kz**2)
    integrand = lambda u: 1 - np.cos(xR*u*(k_perp * (1 - u)) / c.k0**2)
    val, _ = int.quad(integrand, 0.0, 1.0)
    return 0.5 * val

def sigma_log_amplitude_2D(xR):
    """
    Variance de log-amplitude pour une propagation sur la distance xR
    en utilisant le spectre 2D de von Karman.
    """
    def integrand(kz):
        return S2D_vonKarman(0.0, kz) * F_filter(0.0, kz, xR)

    val, _ = int.quad(integrand, -np.inf, np.inf)
    prefactor = 2 * np.pi * c.k0**2 * xR

    return prefactor * val

def link_budget(P_rx, B, T):
    """
    Calcule le bilan de liaison SNR.
    """
    N = 10*np.log10(c.kb * T * B)
    SNR = P_rx - N
    return SNR

def delta_epsilon(r):
    """
    Variation de permittivité : Δε = - re * λ² * δN
    """
    N = densite_elec(r) * 1e6  # conversion de cm⁻³ à m⁻³
    deltaN = N - np.mean(N)
    return - c.re * c.lamb**2 * deltaN

def delta_theta(L):
    """
    Variation de l'angle de phase.
    """
    val, _ = int.quad(lambda r: delta_epsilon(r), 0.1, L)
    phi = c.k0/2 * val
    return val

def I1 (nu, ar):
    """
    Calcul de l'intégrale I1 pour la variance angulaire.
    """
    return ar**(nu-4) * (2**(3-nu) * gamma(2 - nu/2) * gamma(nu - 1)) / (gamma(nu/2)**2 * gamma(1 + nu/2))

def I2 (SEP):
    """
    Calcul de l'intégrale I2 pour la variance angulaire.
    """
    # 1e-18 / np.sin(np.radians(SEP))**11 * ((np.sin(SEP)**9 * np.cos(SEP))/10 + (9 * np.sin(SEP)**7 * np.cos(SEP))/80 + (63 * np.sin(SEP)**5 * np.cos(SEP))/480 + (315 * np.sin(SEP)**3 *np.cos(SEP))/1920 + (945 * np.sin(SEP) * np.cos(SEP))/3840 + (945 * (np.pi/2 - SEP))/3840)
    print(2*c.a0**2/c.D_terre**11)
    return 2 * c.a0**2 / (c.D_terre * np.sin(np.radians(SEP)))**11 * ((np.sin(SEP)**9 * np.cos(SEP))/10 + (9 * np.sin(SEP)**7 * np.cos(SEP))/80 + (63 * np.sin(SEP)**5 * np.cos(SEP))/480 + (315 * np.sin(SEP)**3 *np.cos(SEP))/1920 + (945 * np.sin(SEP) * np.cos(SEP))/3840 + (945 * (np.pi/2 - SEP))/3840)

def I3 ():
    """
    Calcul de l'intégrale I3 pour la variance angulaire.
    """
    gam_rad = np.radians(c.gam)
    return np.pi/2 * (2 + (c.Axial_r**2 - 1) * np.sin(gam_rad)**2)/(1 + (c.Axial_r**2 -1) * np.sin(gam_rad)**2)**(2/3)


def angular_variance(SEP):
    """
    Variance angulaire due aux fluctuations de densité électronique.
    """
    valI1 = I1()
    valI2 = I2(SEP)
    valI3 = I3()
    return 0.5 * c.re**2 * c.lamb**4 * c.Q_nu * c.kappa_0**(c.nu-3) * valI1 * valI2 * valI3

