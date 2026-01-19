# Nom du module à discuter
import numpy as np
import constantes as c
import matplotlib.pyplot as plt
import scipy.integrate as int
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm
from scipy.special import gamma, hyperu



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

def F_filter(kx, kz, xR):
    k_perp2 = kx**2 + kz**2
    def integrand(u):
        return 1 - np.cos(xR * u * (1 - u) * k_perp2 / c.k0**2)
    val, _ = int.quad(integrand, 0.0, 1.0, limit=200)
    return 0.5 * val

def sigma_log_amplitude_2D(xR):
    """
    Variance de log-amplitude pour une propagation sur la distance xR
    (formule cohérente avec le papier)
    """
    def integrand(kz):
        return (
            S2D_vonKarman(0.0, kz)
            * F_filter(0.0, kz, xR))

    val, _ = int.quad(integrand, 0.0, np.inf, limit=500)
    prefactor = 2 * np.pi * c.k0**2 * xR

    return prefactor * val

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
    SEP = np.radians(SEP)
    return 1*10**(-18)/np.sin(SEP)**11 * ((np.sin(SEP)**9 * np.cos(SEP))/10 + (9 * np.sin(SEP)**7 * np.cos(SEP))/80 \
                                      + (63 * np.sin(SEP)**5 * np.cos(SEP))/480 + (315 * np.sin(SEP)**3 *np.cos(SEP))/1920 \
                                        + (945 * np.sin(SEP) * np.cos(SEP))/3840 + (945 * (np.pi/2 - SEP))/3840)
    
def I3 (Vs, r):
    """
    Calcul de l'intégrale I3 pour la variance angulaire.
    """
    gam_rad = np.arctan(-Vs/(c.omega*r))
    return np.pi/2 * (2 + (c.Axial_r**2 - 1) * np.sin(gam_rad)**2)/(1 + (c.Axial_r**2 -1) * np.sin(gam_rad)**2)**(2/3)


def angular_variance(SEP):
    """
    Variance angulaire due aux fluctuations de densité électronique.
    """
    valI1 = I1(3.9, c.ar)
    valI2 = I2(SEP)
    r = SEP_dist(SEP)
    valI3 = I3(200e3, r)
    return 0.5 * c.re**2 * c.lamb**4 * c.Q_nu * c.kappa_0**(c.nu-3) * valI1 * valI2 * valI3

def intA(SEP):
    """
    Intégrale IntA pour le spectre angulaire.
    """
    SEP = np.radians(SEP)
    return c.a0**2 * c.Q_nu/(np.pi*(c.D_terre * np.sin(SEP))**7)\
              * (np.sin(SEP)**5*np.cos(SEP)/6 + \
                 5/6*((np.sin(SEP)**3*np.cos(SEP))/4 + \
                    3/4*((np.sin(SEP)*np.cos(SEP))/2 +(np.pi/2-SEP)/2) )  )

def spectre_phase(f, SEP, vent_p = 100e3):
    """
    Spectre de phase des fluctuations de densité électronique.
    """
    IntA = intA(SEP)
    omega = 2 * np.pi * f
    omega_s = vent_p/(c.b* c.ar) 
    W_phi = 4*np.pi**4*c.re**2 * c.lamb**2* vent_p**(13/12) * omega **(-25/12) * np.exp(-omega**2/omega_s**2)*gamma(1/2)*hyperu(1/2,-1/24,omega**2/omega_s**2) * IntA
    return W_phi

def spectre_doppler(f, SEP, vent_p = 100e3):
    """
    Spectre Doppler des fluctuations de densité électronique.
    """
    omega = 2 * np.pi * f
    W_phi = spectre_phase(f, SEP, vent_p)
    return (omega**2/(2*np.pi))**2*W_phi

def serie_temp_doppler(seed, omega_pos, Wfd_om_pos, fs = 200, T = 200, vent_p = 100e3):
    """
    Série temporelle Doppler des fluctuations de densité électronique.
    """
    rng = np.random.default_rng(seed) # Tirage des phases aléatoires
    N = fs * T  # Nombre de points dans la série temporelle
    df = fs/N # pas 
    f_pos = np.arange(0, N//2 + 1) * df  # Fréquences positives de FFT
    grille_omega = 2*np.pi * f_pos
    W_int = np.interp(grille_omega, omega_pos, Wfd_om_pos, left=0, right=0)  # Interpolation du spectre Doppler
    S_pos = 2*np.pi * W_int  # Densité spectrale de puissance
    X = np.zeros(N//2 +1, dtype=complex) # Initialisation des coefficients de FFT
    k = np.arange(1, N//2)  # Indices pour les fréquences positives (excluant DC et Nyquist)
    sigma = np.sqrt(0.5*S_pos[k]*df)  # Écart-type pour les fréquences positives
    X[k] = sigma * (rng.normal(size = k.size)) + 1j*rng.normal(size = k.size)  # Coefficients FFT pour fréquences positives
    X[0] = 0.0  # Composante DC
    X[N//2] = np.sqrt(S_pos[N//2]*df) * rng.normal()  # Composante de Nyquist
    X_full = np.concatenate([X, np.conj(X[-2:0:-1])])  # FFT complète en utilisant la symétrie hermitienne
    fD = np.fft.ifft(X_full).real * N  # Série temporelle Doppler
    t = np.arange(N) / fs  # Axe temporel
    return t, fD