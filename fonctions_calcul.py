# Nom du module à discuter
from random import seed
import numpy as np
import constantes as c
import matplotlib.pyplot as plt
import scipy.integrate as integ
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm
from scipy.special import gamma, hyperu

def distance_terre_venus(angle_terre, angle_venus):
    """
    Calcule la distance entre la Terre et Vénus en fonction de leurs angles respectifs autour du Soleil.
    
    Angle_terre : Angle de la terre autour du soleil en radians
    Angle_venus : Angle de Vénus autour du soleil en radians
    """
    
    x_terre = c.D_terre * np.cos(angle_terre)
    y_terre = c.D_terre * np.sin(angle_terre)
    
    x_venus = c.D_venus * np.cos(angle_venus)
    y_venus = c.D_venus * np.sin(angle_venus)
    
    distance = np.sqrt((x_terre - x_venus)**2 + (y_terre - y_venus)**2)
    return distance

def SEP_dist(SEP):
    """
    Calcule la distance au soleil en fonction de l'angle sun earth probe (SEP).
    SEP : Angle sun earth probe en degrés
    """
    Dterre = c.D_terre
    Rsoleil = c.R_soleil
    return Dterre * np.sin(np.radians(SEP)) 

def densite_elec(r):
    """
    Calcule la densité électronique en fonction de la distance r au centre du Soleil.
    r : distance au centre du Soleil en mètres
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
    val, _ = integ.quad(integrand, 0.0, 1.0, limit=200)
    return 0.5 * val

def sigma_log_amplitude_2D(xR):
    """
    Variance de log-amplitude pour une propagation sur la distance xR
    (formule cohérente avec le papier)
    xR : distance de propagation en mètres
    """
    def integrand(kz):
        return (
            S2D_vonKarman(0.0, kz)
            * F_filter(0.0, kz, xR))

    val, _ = integ.quad(integrand, 0.0, np.inf, limit=500)
    prefactor = 2 * np.pi * c.k0**2 * xR

    return prefactor * val

def delta_epsilon(r):
    """
    Variation de permittivité : Δε = - re * λ² * δN
    r : distance au centre du Soleil en mètres
    Expression codée du début de papier de Ho, pas utile pour le projet mais laissée pour référence.
    """
    N = densite_elec(r) * 1e6  # conversion de cm⁻³ à m⁻³
    deltaN = N - np.mean(N)
    return - c.re * c.lamb**2 * deltaN

def delta_theta(L):
    """
    Variation de l'angle de phase.
    L : distance de propagation en mètres
    Expression codée du début de papier de Ho, pas utile pour le projet mais laissée pour référence.
    """
    val, _ = integ.quad(lambda r: delta_epsilon(r), 0.1, L)
    phi = c.k0/2 * val
    return val

def I1 (nu, ar):
    """
    Calcul de l'intégrale I1 pour la variance angulaire.
    nu : indice spectral de la turbulence
    ar = rayon de l'antenne en mètres
    """
    return ar**(nu-4) * (2**(3-nu) * gamma(2 - nu/2) * gamma(nu - 1)) / (gamma(nu/2)**2 * gamma(1 + nu/2))

def I2 (SEP):
    """
    Calcul de l'intégrale I2 pour la variance angulaire.
    SEP : angle sun earth probe en degrés
    """
    SEP = np.radians(SEP)
    return 1*10**(-18)/np.sin(SEP)**11 * ((np.sin(SEP)**9 * np.cos(SEP))/10 + (9 * np.sin(SEP)**7 * np.cos(SEP))/80 \
                                      + (63 * np.sin(SEP)**5 * np.cos(SEP))/480 + (315 * np.sin(SEP)**3 *np.cos(SEP))/1920 \
                                        + (945 * np.sin(SEP) * np.cos(SEP))/3840 + (945 * (np.pi/2 - SEP))/3840)
    
def I3 (Vs, r):
    """
    Calcul de l'intégrale I3 pour la variance angulaire.
    Vs = vitesse radiale du vent solaire en m/s
    r = distance au soleil en mètres
    """
    gam_rad = np.arctan(-Vs/(c.omega*r))
    return np.pi/2 * (2 + (c.Axial_r**2 - 1) * np.sin(gam_rad)**2)/(1 + (c.Axial_r**2 -1) * np.sin(gam_rad)**2)**(2/3)

def angle_rms_table(SEP_deg, band="X", out="mdeg"):
    """
    Angle d'arrivée RMS interpolé à partir de la Table 2 (Ho et al. 2008).
    
    SEP_deg : float ou array
    band : "S", "X", "Ka"
    out : "deg" | "mdeg" | "rad"
    """
    band = band.strip()
    key = {"S": "S_deg", "X": "X_deg", "Ka": "Ka_deg"}[band]
    
    x = c._TABLE2["SEP_deg"]
    y_deg = c._TABLE2[key]

    SEP = np.asarray(SEP_deg, dtype=float)

    # Interpolation en log (souvent plus stable car variations sur plusieurs ordres de grandeur)
    # On interpole log10(y) en fonction de SEP, puis on revient en linéaire.
    logy = np.log10(y_deg)
    y_interp_deg = 10 ** np.interp(SEP, x, logy, left=np.nan, right=np.nan)

    if out == "deg":
        return y_interp_deg
    if out == "mdeg":
        return 1e3 * y_interp_deg
    if out == "rad":
        return np.deg2rad(y_interp_deg)
    raise ValueError("out doit être 'deg', 'mdeg' ou 'rad'")


def angular_variance_band(SEP_deg, band="X", C_norm=1.0):
    """
    Variante de angular_variance() où lambda dépend de la bande (S/X/Ka).
    Nécessite I1, I2, I3, SEP_dist déjà définies dans ton projet.
    """
    C_norm=1.e27 # Ajustement de la normalisation pour correspondre aux données expérimentales
    band = band.strip()
    f = c.FREQ[band]                 # Hz
    lamb_band = c.c / f              # m  (ici c = vitesse lumière dans ton module constantes)

    valI1 = I1(3.9, c.ar)            # si tu veux utiliser nu/ar du module : I1(nu, ar)
    valI2 = I2(SEP_deg)
    r = SEP_dist(SEP_deg)
    valI3 = I3(200e3, r)

    return 0.5 * C_norm * c.re**2 * lamb_band**4 * c.Q_nu * c.kappa_0**(c.nu - 3) * valI1 * valI2 * valI3



def angle_rms_model_band(SEP_deg, band="X", C_norm=1.0):
    """
    RMS angle d'arrivée depuis le modèle, avec fréquence dépendant de la bande.
    """
    dq2_u = angular_variance_band(SEP_deg, band=band, C_norm=C_norm)
    dq_rms_rad = 1.414 * np.sqrt(dq2_u)

    return 1e3 * np.rad2deg(dq_rms_rad)



def intA(SEP):
    """
    Intégrale IntA pour le spectre angulaire.
    SEP : angle sun earth probe en degrés
    Expression similaire de I2 mais venant de Ho 2010 et utilisée dans le spectre de phase.
    """
    SEP = np.radians(SEP)
    return c.a0**2 * c.Q_nu/(np.pi*(c.D_terre * np.sin(SEP))**7)\
              * (np.sin(SEP)**5*np.cos(SEP)/6 + \
                 5/6*((np.sin(SEP)**3*np.cos(SEP))/4 + \
                    3/4*((np.sin(SEP)*np.cos(SEP))/2 +(np.pi/2-SEP)/2) )  )

def spectre_phase(f, SEP, vent_p = 100e3):
    """
    Spectre de phase des fluctuations de densité électronique.
    f : fréquence des fluctuations en Hz
    SEP : angle sun earth probe en degrés
    vent_p : vitesse du vent solaire en m/s
    """
    IntA = intA(SEP)
    omega = 2 * np.pi * f
    omega_s = vent_p/(c.b* c.ar) 
    W_phi = 4*np.pi**4*c.re**2 * c.lamb**2* vent_p**(13/12) * omega **(-25/12) * np.exp(-omega**2/omega_s**2)*gamma(1/2)*hyperu(1/2,-1/24,omega**2/omega_s**2) * IntA
    return W_phi

def spectre_doppler(f, SEP, vent_p = 100e3):
    """
    Spectre Doppler des fluctuations de densité électronique.
    f : fréquence des fluctuations en Hz
    SEP : angle sun earth probe en degrés
    vent_p : vitesse du vent solaire en m/s
    """
    omega = 2 * np.pi * f
    W_phi = spectre_phase(f, SEP, vent_p)
    return omega**2/(2*np.pi)**2*W_phi

def serie_temp_doppler(seed, omega_pos, Wfd_om_pos, fs = 200, T = 20000, vent_p = 100e3, lowpass_fc=10, count_time=0.2):
    """
    Série temporelle Doppler des fluctuations de densité électronique.
    seed : graine pour la génération aléatoire
    omega_pos : grille de fréquences angulaires positives en rad/s
    Wfd_om_pos : spectre Doppler évalué sur la grille omega_pos utilisation de la fonction spectre_doppler()
    fs : fréquence d'échantillonnage en Hz
    T : durée totale de la série temporelle en secondes
    vent_p : vitesse du vent solaire en m/s
    """
    C_norm=1.e-30 # Ajustement de la normalisation pour correspondre aux données expérimentales
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

    return t, C_norm*fD

def reconstruction_phase_deg(fD, fs=200, remove_dc=True, phi0_deg=0.0):
    """
    Reconstruction de la phase (en degrés) à partir des résidus Doppler fD(t) en Hz.
    fD : deuxième sortie de la fonction serie_temp_doppler(), série temporelle Doppler en Hz
    fs : fréquence d'échantillonnage en Hz
    remove_dc : booléen pour indiquer si la composante continue doit être retirée
    phi0_deg : phase initiale en degrés
    """
    fD = np.asarray(fD, dtype=float)
    dt = 1.0 / fs

    if remove_dc:
        fD = fD - np.mean(fD)

    # Hz -> cycles via intégration, puis cycles -> degrés (×360)
    phi_deg = phi0_deg + 360.0 * np.cumsum(fD) * dt
    return phi_deg

def Bilan_liaison(SEP, Latm, Liono, Lpoint, Lpol,f_psd, P_t, G_t, G_r, vent_p = 100e3):
    """ 
    Calcul du bilan de liaison.
    SEP : angle sun earth probe en degrés
    Latm : pertes atmosphériques en dB fixée à 2 dB
    Liono : pertes ionosphériques en dB fixée à 1 dB
    Lpoint : pertes de pointage en dB fixée à 1 dB
    Lpol : pertes de polarisation en dB fixée à 1 dB
    f_psd : fréquence porteuse en Hz
    P_t : puissance transmise en dBW fixée à 20 dBW
    G_t : gain de l'antenne émettrice en dBi fixée à 35 dBi
    G_r : gain de l'antenne réceptrice en dBi fixée à 35 dBi
    vent_p : vitesse du vent solaire en m/s
    """
    distance = SEP_dist(SEP)
    L_fs = 20 * np.log10((4 * np.pi * distance * 1e3) / c.lamb)  # Conversion km à m
    Ltot = L_fs + Latm + Liono + Lpoint + Lpol 
    return P_t + G_t + G_r - Ltot
