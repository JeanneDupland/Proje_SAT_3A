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

def plot_distance_terre_venus():
    """
    Trace la distance entre la Terre et Vénus en fonction du temps sur une période donnée.
    """
    jours = np.linspace(0, 365, 1000)  # Période d'un an en jours
    angle_terre = (2 * np.pi / 365) * jours  # Angle de la Terre
    angle_venus = (2 * np.pi / 225) * jours  # Angle de Vénus (période orbitale de Vénus ~225 jours)
    
    distances = distance_terre_venus(angle_terre, angle_venus)
    
    plt.figure(figsize=(10, 6))
    plt.plot(jours, distances / 1e6)  # Convertir km en millions de km pour l'affichage
    plt.title("Distance entre la Terre et Vénus au cours d'une année")
    plt.xlabel("Jours")
    plt.ylabel("Distance (millions de km)")
    plt.grid()
    plt.show()

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

def plot_densite_elec():
    """
    Trace la densité électronique en fonction de la distance au centre du Soleil.
    """
    r = np.linspace(c.R_soleil, 4 * c.R_soleil, 1000)  # De la surface du Soleil à 20 rayons solaires
    densites = densite_elec(r)
    
    plt.figure(figsize=(10, 6))
    plt.plot(r / c.R_soleil, densites)
    plt.yscale('log')
    plt.title("Densité électronique en fonction de la distance au centre du Soleil")
    plt.xlabel("Distance (rayons solaires)")
    plt.ylabel("Densité électronique (m⁻³)")
    plt.grid()
    plt.show()

def plot_densite_elec_SEP():
    """
    Trace la densité électronique en fonction de la distance au centre du Soleil.
    """
    SEP = np.linspace(0.01, 30, 1000)  
    r = SEP_dist(SEP)
    densites = densite_elec(r)
    
    plt.figure(figsize=(10, 6))
    plt.plot(SEP, densites)
    plt.yscale('log')
    plt.title("Densité électronique en fonction de l'angle soleil-terre-sonde (SEP)")
    plt.xlabel("Angle SEP (Degré)")
    plt.ylabel("Densité électronique (m⁻³)")
    plt.grid()
    plt.show()

def plot_densite_elec_SEP_precis():
    """
    Trace la densité électronique en fonction de la distance au centre du Soleil de -5° à 5°
    """
    SEP = np.linspace(0.01, 5, 100)  
    r = SEP_dist(SEP)
    densites = densite_elec(r)
    
    plt.figure(figsize=(10, 6))
    plt.plot(SEP, densites)
    plt.yscale('log')
    plt.title("Densité électronique en fonction de l'angle soleil-terre-sonde (SEP)")
    plt.xlabel("Angle SEP (Degré)")
    plt.ylabel("Densité électronique (m⁻³)")
    plt.grid()
    plt.show()

def repre_dens_soleil():
    """
    Représente la densité électronique autour du Soleil avec une colormap rouge -> jaune clair.
    """
    # --- Création de la colormap personnalisée ---
    couleurs = ["#400000", "#800000", "#CC0000", "#FF5500", "#FFAA00", "#FFE066", "#FFFFCC"]
    cmap_custom = LinearSegmentedColormap.from_list("rouge_jaune", couleurs)

    r = np.linspace(c.R_soleil, 4 * c.R_soleil, 400)
    theta = np.linspace(0, 2 * np.pi, 400)
    R, Theta = np.meshgrid(r, theta)
    Z = densite_elec(R)

    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)

    plt.figure(figsize=(8, 8))
    plt.pcolormesh(X, Y, Z, shading='auto', norm=LogNorm(), cmap=cmap_custom)
    plt.colorbar(label='Densité électronique (m⁻³)')
    plt.title("Densité électronique autour du Soleil")
    plt.xlabel("X (km)")
    plt.ylabel("Y (km)")
    plt.axis('equal')
    plt.show()

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

def plot_sigma_log_amplitude_2D():
    """
    Trace la variance de log-amplitude en fonction de la distance de propagation xR.
    """
    xR_values = np.linspace(0, 40e3, 1000) 
    sigma_values = [sigma_log_amplitude_2D(xR) for xR in xR_values]

    plt.figure(figsize=(10, 6))
    plt.plot(xR_values*10**(-3), sigma_values)
    plt.title("Variance de log-amplitude en fonction de la distance de propagation")
    plt.xlabel("Distance de propagation xR (m)")
    plt.ylabel("Variance de log-amplitude σ²")
    plt.grid()
    plt.show()

def link_budget(P_rx, B, T):
    """
    Calcule le bilan de liaison SNR.
    """
    N = 10*np.log10(c.kb * T * B)
    SNR = P_rx - N
    return SNR

def plot_link_budget(B, T):
    """
    Trace le bilan de liaison SNR en fonction de la puissance reçue.
    """
    P_rx_dBm = np.linspace(-150, 0, 1000)  # Puissance reçue en dBm
    SNR_values = [link_budget(P, B, T) for P in P_rx_dBm]

    plt.figure(figsize=(10, 6))
    plt.plot(P_rx_dBm, SNR_values)
    plt.title("Bilan de liaison SNR en fonction de la puissance reçue")
    plt.xlabel("Puissance reçue P_rx (dBm)")
    plt.ylabel("SNR (dB)")
    plt.grid()
    plt.show()

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

def plot_delta_theta():
    """
    Trace la variation de l'angle de phase en fonction de la distance L.
    """
    L_values = np.linspace(0, 1e7, 1000)  # Distance en mètres
    delta_theta_values = [delta_theta(L) for L in L_values]

    plt.figure(figsize=(10, 6))
    plt.plot(L_values*1e-3, delta_theta_values)
    plt.title("Variation de l'angle de phase en fonction de la distance L")
    plt.xlabel("Distance L (km)")
    plt.ylabel("Variation de l'angle de phase Δθ (radians)")
    plt.grid()
    plt.show()

def I1 (nu, ar):
    """
    Calcul de l'intégrale I1 pour la variance angulaire.
    """
    return ar**(nu-4) * (2**(3-nu) * gamma(2 - nu/2) * gamma(nu - 1)) / (gamma(nu/2)**2 * gamma(1 + nu/2))

def plot_I1():
    """
    Trace de l'intégrale I1 en fonction de l'exposant spectral nu.
    """
    ar = np.linspace(10, 40, 1000)  # Rayon de l'antenne en mètres
    nu = np.linspace(3.1, 3.9, 5)  # Exposant spectral ν
    
    I1_values_1 = [I1(nu[0], a) for a in ar]
    I1_values_2 = [I1(nu[1], a) for a in ar]
    I1_values_3 = [I1(nu[2], a) for a in ar]
    I1_values_4 = [I1(nu[3], a) for a in ar]
    I1_values_5 = [I1(nu[4], a) for a in ar]    
    plt.figure(figsize=(10, 6))
    plt.plot(ar, I1_values_1, label=f"ν={nu[0]}")
    plt.plot(ar, I1_values_2, label=f"ν={nu[1]}")
    plt.plot(ar, I1_values_3, label=f"ν={nu[2]}")
    plt.plot(ar, I1_values_4, label=f"ν={nu[3]}")
    plt.plot(ar, I1_values_5, label=f"ν={nu[4]}")
    plt.legend()    
    plt.title("Intégrale I1 en fonction de l'exposant spectral ν")
    plt.xlabel("Exposant spectral ν")
    plt.yscale('log')
    plt.ylabel("Intégrale I1")
    plt.grid()
    plt.show()

def I2 (SEP):
    """
    Calcul de l'intégrale I2 pour la variance angulaire.
    """
    # 2 * c.a0**2 / (c.D_terre * np.sin(np.radians(SEP)))**11 * ((np.sin(SEP)**9 * np.cos(SEP))/10 + (9 * np.sin(SEP)**7 * np.cos(SEP))/80 + (63 * np.sin(SEP)**5 * np.cos(SEP))/480 + (315 * np.sin(SEP)**3 *np.cos(SEP))/1920 + (945 * np.sin(SEP) * np.cos(SEP))/3840 + (945 * (np.pi/2 - SEP))/3840)
    return 1e-18 / np.sin(np.radians(SEP))**11 * ((np.sin(SEP)**9 * np.cos(SEP))/10 + (9 * np.sin(SEP)**7 * np.cos(SEP))/80 + (63 * np.sin(SEP)**5 * np.cos(SEP))/480 + (315 * np.sin(SEP)**3 *np.cos(SEP))/1920 + (945 * np.sin(SEP) * np.cos(SEP))/3840 + (945 * (np.pi/2 - SEP))/3840)

def plot_I2():
    """
    Trace de l'intégrale I2 en fonction de l'angle SEP.
    """
    SEP = np.linspace(0.1, 5, 1000)  # Angle SEP en degrés
    I2_values = [I2(s) for s in SEP]

    plt.figure(figsize=(10, 6))
    plt.plot(SEP, I2_values)
    plt.title("Intégrale I2 en fonction de l'angle SEP")
    plt.xlabel("Angle SEP (degrés)")
    plt.ylabel("Intégrale I2")
    plt.grid()
    plt.show()

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

def plot_angular_variance():
    """
    Trace la variance angulaire en fonction de l'angle SEP.
    """
    SEP_values = np.linspace(0.1, 5, 1000)  # Angle SEP en degrés
    variance_values = [angular_variance(SEP) for SEP in SEP_values]

    plt.figure(figsize=(10, 6))
    plt.plot(SEP_values, variance_values)
    plt.title("Variance angulaire en fonction de l'angle SEP")
    plt.xlabel("Angle SEP (degrés)")
    plt.ylabel("Variance angulaire (radians²)")
    plt.yscale('log')
    plt.grid()
    plt.show()