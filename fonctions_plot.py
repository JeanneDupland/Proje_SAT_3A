import numpy as np
import constantes as c
import matplotlib.pyplot as plt
import scipy.integrate as int
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm
from scipy.special import gamma, factorial
import fonctions_calcul as fc

def plot_distance_terre_venus():
    """
    Trace la distance entre la Terre et Vénus en fonction du temps sur une période donnée.
    """
    jours = np.linspace(0, 365, 1000)  # Période d'un an en jours
    angle_terre = (2 * np.pi / 365) * jours  # Angle de la Terre
    angle_venus = (2 * np.pi / 225) * jours  # Angle de Vénus (période orbitale de Vénus ~225 jours)
    
    distances = fc.distance_terre_venus(angle_terre, angle_venus)
    
    plt.figure(figsize=(10, 6))
    plt.plot(jours, distances / 1e6)  # Convertir km en millions de km pour l'affichage
    plt.title("Distance entre la Terre et Vénus au cours d'une année")
    plt.xlabel("Jours")
    plt.ylabel("Distance (millions de km)")
    plt.grid()
    plt.show()

def plot_densite_elec():
    """
    Trace la densité électronique en fonction de la distance au centre du Soleil.
    """
    r = np.linspace(c.R_soleil, 4 * c.R_soleil, 1000)  # De la surface du Soleil à 20 rayons solaires
    densites = fc.densite_elec(r)
    
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
    r = fc.SEP_dist(SEP)
    densites = fc.densite_elec(r)
    
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
    r = fc.SEP_dist(SEP)
    densites = fc.densite_elec(r)
    
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
    Z = fc.densite_elec(R)

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

def plot_sigma_log_amplitude_2D():
    """
    Trace la variance de log-amplitude en fonction de la distance de propagation xR.
    """
    xR_values = np.linspace(0, 40e3, 100) 
    sigma_values = [75.7 * fc.sigma_log_amplitude_2D(xR) for xR in xR_values]

    plt.figure(figsize=(10, 6))
    plt.plot(xR_values, sigma_values)
    plt.title("Variance de log-amplitude en fonction de la distance de propagation")
    plt.xlabel("Distance de propagation xR (m)")
    plt.ylabel("Variance de log-amplitude σ²")
    plt.grid()
    plt.show()

def plot_link_budget(B, T):
    """
    Trace le bilan de liaison SNR en fonction de la puissance reçue.
    """
    P_rx_dBm = np.linspace(-150, 0, 1000)  # Puissance reçue en dBm
    SNR_values = [fc.link_budget(P, B, T) for P in P_rx_dBm]

    plt.figure(figsize=(10, 6))
    plt.plot(P_rx_dBm, SNR_values)
    plt.title("Bilan de liaison SNR en fonction de la puissance reçue")
    plt.xlabel("Puissance reçue P_rx (dBm)")
    plt.ylabel("SNR (dB)")
    plt.grid()
    plt.show()

def plot_delta_theta():
    """
    Trace la variation de l'angle de phase en fonction de la distance L.
    """
    L_values = np.linspace(0, 1e7, 1000)  # Distance en mètres
    delta_theta_values = [fc.delta_theta(L) for L in L_values]

    plt.figure(figsize=(10, 6))
    plt.plot(L_values*1e-3, delta_theta_values)
    plt.title("Variation de l'angle de phase en fonction de la distance L")
    plt.xlabel("Distance L (km)")
    plt.ylabel("Variation de l'angle de phase Δθ (radians)")
    plt.grid()
    plt.show()

def plot_I1():
    """
    Trace de l'intégrale I1 en fonction de l'exposant spectral nu.
    """
    ar = np.linspace(10, 40, 1000)  # Rayon de l'antenne en mètres
    nu = np.linspace(3.1, 3.9, 5)  # Exposant spectral ν
    
    I1_values_1 = [fc.I1(nu[0], a) for a in ar]
    I1_values_2 = [fc.I1(nu[1], a) for a in ar]
    I1_values_3 = [fc.I1(nu[2], a) for a in ar]
    I1_values_4 = [fc.I1(nu[3], a) for a in ar]
    I1_values_5 = [fc.I1(nu[4], a) for a in ar]    
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

def plot_I2():
    """
    Trace de l'intégrale I2 en fonction de l'angle SEP.
    """
    SEP = np.linspace(0.1, 5, 1000)  # Angle SEP en degrés
    I2_values = [fc.I2(s) for s in SEP]

    plt.figure(figsize=(10, 6))
    plt.plot(SEP, I2_values)
    plt.title("Intégrale I2 en fonction de l'angle SEP")
    plt.xlabel("Angle SEP (degrés)")
    plt.ylabel("Intégrale I2")
    plt.yscale('log')
    plt.grid()
    plt.show()

def plot_I3():
    """
    Trace de l'intégrale I3 en fonction de la distance r.
    """
    r = np.linspace(c.R_soleil, 4 * c.R_soleil, 1000)  # De la surface du Soleil à 4 rayons solaires
    Vs = [100e3, 150e3, 200e3, 250e3, 300e3]  # Vitesse radiale en m/sc
    I3_values_0 = [fc.I3(Vs[0], radius) for radius in r]
    I3_values_1 = [fc.I3(Vs[1], radius) for radius in r]
    I3_values_2 = [fc.I3(Vs[2], radius) for radius in r]
    I3_values_3 = [fc.I3(Vs[3], radius) for radius in r]
    I3_values_4 = [fc.I3(Vs[4], radius) for radius in r]

    plt.figure(figsize=(10, 6))
    plt.plot(r / c.R_soleil, I3_values_0, label=f"Vs={Vs[0]/1e3} km/s")
    plt.plot(r / c.R_soleil, I3_values_1, label=f"Vs={Vs[1]/1e3} km/s")
    plt.plot(r / c.R_soleil, I3_values_2, label=f"Vs={Vs[2]/1e3} km/s")
    plt.plot(r / c.R_soleil, I3_values_3, label=f"Vs={Vs[3]/1e3} km/s")
    plt.plot(r / c.R_soleil, I3_values_4, label=f"Vs={Vs[4]/1e3} km/s")
    plt.legend()    
    plt.title("Intégrale I3 en fonction de la distance r")
    plt.xlabel("Distance r (rayons solaires)")
    plt.ylabel("Intégrale I3")
    plt.grid()
    plt.show()
 
def plot_angular_variance():
    """
    Trace la variance angulaire en fonction de l'angle SEP.
    """
    SEP_values = np.linspace(0.1, 5, 1000)  # Angle SEP en degrés
    variance_values = [fc.angular_variance(SEP) for SEP in SEP_values]

    plt.figure(figsize=(10, 6))
    plt.plot(SEP_values, variance_values)
    plt.title("Variance angulaire en fonction de l'angle SEP")
    plt.xlabel("Angle SEP (degrés)")
    plt.ylabel("Variance angulaire (radians²)")
    plt.yscale('log')
    plt.grid()
    plt.show()

def plot_angular_variance():
    """
    Trace la variance angulaire en fonction de l'angle SEP.
    """
    SEP_values = np.linspace(0.1, 5, 1000)  # Angle SEP en degrés
    variance_values = [fc.angular_variance(SEP) for SEP in SEP_values]

    plt.figure(figsize=(10, 6))
    plt.plot(SEP_values, variance_values)
    plt.title("Variance angulaire en fonction de l'angle SEP")
    plt.xlabel("Angle SEP (degrés)")
    plt.ylabel("Variance angulaire (radians²)")
    plt.yscale('log')
    plt.grid()
    plt.show() 

def plot_RMS():
    SEP_values = np.linspace(0.1, 5, 1000)  # Angle SEP en degrés
    variance_values = [np.rad2deg(1.414*np.sqrt(fc.angular_variance(SEP)))*10**3 for SEP in SEP_values]
    plt.figure(figsize=(10, 6))
    plt.plot(SEP_values, variance_values)
    plt.title("Angle d'arrivée RMS en fonction de l'angle SEP")
    plt.xlabel("Angle SEP (degrés)")
    plt.ylabel("Angle d'arrivée (mdegrés)")
    plt.yscale('log')
    plt.grid()
    plt.show() 

def plot_spectre_phase():
    f = [10**(-4), 10**(-3), 10**(-2), 10**(-1), 1]  # Fréquence en Hz
    SEP = np.linspace(0.1, 5, 1000)  # Angle SEP en degrés
    vent_p = 100e3  # Vitesse de propagation en m/s
    W_phi_values1 = [fc.spectre_phase(f[0], SEP_angle, vent_p) for SEP_angle in SEP]
    W_phi_values2 = [fc.spectre_phase(f[1], SEP_angle, vent_p) for SEP_angle in SEP]
    W_phi_values3 = [fc.spectre_phase(f[2], SEP_angle, vent_p) for SEP_angle in SEP]
    W_phi_values4 = [fc.spectre_phase(f[3], SEP_angle, vent_p) for SEP_angle in SEP]
    W_phi_values5 = [fc.spectre_phase(f[4], SEP_angle, vent_p) for SEP_angle in SEP]
    plt.figure(figsize=(10, 6))
    plt.plot(SEP, W_phi_values1, label=f"f={f[0]:.1e} Hz")
    plt.plot(SEP, W_phi_values2, label=f"f={f[1]:.1e} Hz")
    plt.plot(SEP, W_phi_values3, label=f"f={f[2]:.1e} Hz")
    plt.plot(SEP, W_phi_values4, label=f"f={f[3]:.1e} Hz")
    plt.plot(SEP, W_phi_values5, label=f"f={f[4]:.1e} Hz")
    plt.title("Spectre de phase en fonction de l'angle SEP")
    plt.xlabel("Angle SEP (degrés)")
    plt.ylabel("Spectre de phase Wφ")
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.show()

def plot_spectre_doppler():
    f = [10**(-4), 10**(-3), 10**(-2), 10**(-1), 1]  # Fréquence en Hz
    SEP = np.linspace(0.1, 5, 1000)  # Angle SEP en degrés
    vent_p = 100e3  # Vitesse de propagation en m/s
    W_phi_values1 = [fc.spectre_doppler(f[0], SEP_angle, vent_p) for SEP_angle in SEP]
    W_phi_values2 = [fc.spectre_doppler(f[1], SEP_angle, vent_p) for SEP_angle in SEP]
    W_phi_values3 = [fc.spectre_doppler(f[2], SEP_angle, vent_p) for SEP_angle in SEP]
    W_phi_values4 = [fc.spectre_doppler(f[3], SEP_angle, vent_p) for SEP_angle in SEP]
    W_phi_values5 = [fc.spectre_doppler(f[4], SEP_angle, vent_p) for SEP_angle in SEP]

    plt.figure(figsize=(10, 6))
    plt.plot(SEP, W_phi_values1, label=f"f={f[0]:.1e} Hz")
    plt.plot(SEP, W_phi_values2, label=f"f={f[1]:.1e} Hz")
    plt.plot(SEP, W_phi_values3, label=f"f={f[2]:.1e} Hz")
    plt.plot(SEP, W_phi_values4, label=f"f={f[3]:.1e} Hz")
    plt.plot(SEP, W_phi_values5, label=f"f={f[4]:.1e} Hz")
    plt.title("Spectre Doppler en fonction de l'angle SEP")
    plt.xlabel("Angle SEP (degrés)")
    plt.ylabel("Spectre Doppler WfD")
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.show()

def plot_serie_temporelle_doppler():
    seed = 0
    SEP_list = [0.5, 1, 2, 3, 4]
    vent_p = 100e3
    Cnorm = 1e-33
    fs = 200
    T = 200

    # PSD sur une grille dense
    f_fluct = np.logspace(-4, 1, 600)        # Hz
    omega_pos = 2*np.pi*f_fluct              # rad/s

    plt.figure(figsize=(10, 6))

    for i, SEP in enumerate(SEP_list):
        Wfd_om_pos = np.array([fc.spectre_doppler(ff, SEP, vent_p) for ff in f_fluct], float)
        Wfd_om_pos = np.nan_to_num(Wfd_om_pos, nan=0.0, posinf=0.0, neginf=0.0)
        Wfd_om_pos = np.maximum(Wfd_om_pos, 0.0)

        t, fD = fc.serie_temp_doppler(seed+i, omega_pos, Wfd_om_pos, fs=fs, T=T)

        # Affiche juste 30 s pour lisibilité
        nwin = 30*fs
        plt.plot(t[:nwin], Cnorm*fD[:nwin], label=f"SEP={SEP}°")

    plt.title("Série temporelle Doppler $f_D(t)$ pour différents SEP")
    plt.xlabel("Temps (s)")
    plt.ylabel("$f_D(t)$ (Hz)")
    plt.legend()
    plt.grid()
    plt.show()
          