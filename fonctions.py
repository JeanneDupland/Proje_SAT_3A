# Nom du module à discuter
import numpy as np
import constantes as c
import matplotlib.pyplot as plt

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