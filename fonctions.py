# Nom du module à discuter
import numpy as np
import constantes as c
import matplotlib.pyplot as plt
import scipy.stats as stats

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
    r = np.linspace(c.R_soleil, 20 * c.R_soleil, 1000)  # De la surface du Soleil à 20 rayons solaires
    densites = densite_elec(r)
    
    plt.figure(figsize=(10, 6))
    plt.plot(r / c.R_soleil, densites)
    plt.yscale('log')
    plt.title("Densité électronique en fonction de la distance au centre du Soleil")
    plt.xlabel("Distance (rayons solaires)")
    plt.ylabel("Densité électronique (cm⁻³)")
    plt.grid()
    plt.show()

def plot_densite_elec_SEP():
    """
    Trace la densité électronique en fonction de la distance au centre du Soleil.
    """
    SEP = np.linspace(0.01, 90, 1000)  
    r = SEP_dist(SEP)
    densites = densite_elec(r)
    
    plt.figure(figsize=(10, 6))
    plt.plot(SEP, densites)
    plt.yscale('log')
    plt.title("Densité électronique en fonction de l'angle soleil-terre-sonde (SEP)")
    plt.xlabel("Angle SEP (Degré)")
    plt.ylabel("Densité électronique (cm⁻³)")
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
    plt.ylabel("Densité électronique (cm⁻³)")
    plt.grid()
    plt.show()

def repre_dens_soleil():
    """
    Représente la densité électronique autour du Soleil.
    Le soleil est représenté au centre et la densité est représentée grâce à un gradient de couleurs.
    """
    r = np.linspace(c.R_soleil, 20 * c.R_soleil, 400)
    theta = np.linspace(0, 2 * np.pi, 400)
    R, Theta = np.meshgrid(r, theta)
    Z = densite_elec(R)

    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)

    plt.figure(figsize=(8, 8))
    plt.pcolormesh(X, Y, Z, shading='auto', norm=stats.lognorm(), cmap='inferno')
    plt.colorbar(label='Densité électronique (cm⁻³)')
    plt.title("Densité électronique autour du Soleil")
    plt.xlabel("X (km)")
    plt.ylabel("Y (km)")
    plt.axis('equal')
    plt.show()