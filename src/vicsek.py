"""
Vicsek — 1.7.4
==============
(Alexis Peyroutet et Antoine Royer)

Licence
-------
La totalité du code est soumis à la licence GNU General Public Licence v3.0

Description
-----------
Implémentation d'un modèle de Viscek en Python. Permet de simuler un groupe d'agents en deux ou
trois dimensions.

Cette adaptation se distingue du modèle de Vicsek par plusieurs aspects :
 – gestion d'agents attractifs, répulsifs et d'obstacles ;
 – gestion de l'espace (fermé ou torique) ;
 – les agents ont un angle de vue limité.

Exemples
--------
En important le module comme suit :

>>> import vicsek as vk

Pour créer un groupe de 20 agents avec les paramètres par défaut :

>>> mon_groupe = vk.group_generator(20)

Générer une animation de 100 images:

>>> mon_groupe.compute_animation(100)

Faire évoluer le groupe de 100 pas sans garder d'images : 

>>> mon_groupe.run(100)
"""
import math
import random

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from matplotlib import animation
from matplotlib.collections import PatchCollection


__version__ = "1.7.4"


# Classes
class Agent:
    """
    Simule un agent avec sa position, sa vitesse, et le bruit associé qui traduit sa tendance naturelle à suivre le groupe ou pas.

    Paramètres
    ---------
    position : np.array
        Vecteur position, sous forme d'un tableau numpy de deux ou trois entiers.
    speed : np.array
        Direction du vecteur vitesse sous la forme d'un tableau de trois entiers.
    velocity : float
        Norme de la vitesse.
    noise : float
        Taux de bruit, qui traduit une déviation aléatoire sur la direction de la vitesse.
        Si ``noise=0``, l'agent ne s'écartera jamais de sa trajectoire.
        Si ``noise=1``, l'agent aura une trajectoire très bruitée
    sight : float
        Distance à laquelle l'agent voit les autres.
    field_sight : float, optionnel
        Angle du cône de vision de l'agent en radian.
    agent_type : int, optionnel
        Type d'agent
        0 : agent normal
        1 : agent répulsif
        2 : agent leader
    fear : float, optionnel
        Sensibilité aux agents répulsifs.
        Si ``fear=0`` l'agent sera complètement insensible aux agents répulsifs.
        Si ``fear=1`` l'agent aura une sensibilité maximale

    Attributs
    ---------
    max_velocity : float
        norme maximale de la vitesse
    """
    def __init__(self, position: np.array, speed: np.array, velocity: float, noise: float, sight: float, field_sight: float=math.pi/2, agent_type: int=0, fear: float=1):
        self.position = position.copy()
        self.speed = speed.copy()
        self.velocity = velocity
        self.noise = noise
        self.sight = sight
        self.field_sight = field_sight / 2
        self.agent_type = agent_type
        self.fear = fear

        self.max_velocity = self.velocity
        if agent_type == 1:
            self.max_velocity *= 2

    def __str__(self):
        """Affiche l'agent avec ses paramètres."""
        return f"Agent(\n\tposition={list(self.position)},"\
                f"\n\tspeed={list(self.speed)},"\
                f"\n\tvelocity={self.velocity},"\
                f"\n\tnoise={self.noise},"\
                f"\n\tsight={self.sight},"\
                f"\n\tfield_sight={self.field_sight * 360 / math.pi},"\
                f"\n\tagent_type={self.agent_type}"\
                f"\n\tfear={self.fear}\n)"

    __repr__ = __str__

    def __sub__(self, agent):
        """
        Retourne la distance entre les deux agents.

        Paramètres
        ---------
        agent : Agent
            Un autre agent.

        Signature
        ---------
        out : float
            Distance entre self et agent.
        """
        return norm(self.position - agent.position)

    __rsub__ = __sub__

    def __eq__(self, agent):
        """
        Teste l'égalité entre deux agents.

        Paramètres
        ---------
        agent : Agent
            Un autre agent.

        Signature
        ---------
        out : bool
            True si les deux agents ont la même position, False sinon.
        """
        return list(self.position) == list(agent.position)

    def copy(self):
        """Renvoie une copie profonde de l'agent."""
        return Agent(self.position.copy(), self.speed.copy(), self.velocity, self.noise, self.sight, self.field_sight, self.agent_type, self.fear)

    def get_color(self):
        """Renvoie la couleur de l'agent en fonction de son orientation."""
        if self.agent_type == 3:
            return (0, 0, 1), 0
        angle = np.angle(self.speed[0] + 1j * self.speed[1]) % (2 * math.pi)
        angle = (180 * angle) / math.pi
        return COLOR_MAP[math.floor(angle) % 360], angle

    def next_step(self, neighbours: list, dim: int, length: int, step: float=0.5):
        """
        Fait évoluer l'agent d'un pas temporel en fonction de ses voisins.

        Paramètres
        ---------
        neighbours : list
            Liste des agents voisins.
        dim : int
            Dimension de l'espace.
        length : int
            Longueur caractéristique de l'espace.
        step : float, optionnel
            Pas de temps considéré pour les équations différentielles.
        """
        length //= 2
        average_speed = 0
        average_velocity = 0
        nb_neighbours = 0

        for agent in neighbours:
            if agent.agent_type != 3:
                average_velocity += agent.velocity
                nb_neighbours += 1

            if agent.agent_type == 0:
                if self.agent_type != 1:
                    average_speed += agent.speed
                else:
                    average_speed += (1 / (self - agent)) * (agent.position - self.position)
                    average_velocity += agent.velocity / 4

            elif agent.agent_type == 1:
                if self.agent_type != 1:
                    average_speed += self.fear * len(neighbours) * (self.position - agent.position)
                else:
                    average_speed += agent.speed

            elif agent.agent_type == 2:
                if self.agent_type != 1:
                    average_speed += 5 * agent.speed
                else:
                    average_speed += (1 / (self - agent)) * (agent.position - self.position)
                    average_velocity += agent.velocity / 4

            elif agent.agent_type == 3:
                average_speed += len(neighbours) * 100 * (self.position - agent.position)

        average_speed /= nb_neighbours
        average_velocity /= nb_neighbours
        noise_min = - self.noise / 2
        noise_max = self.noise / 2

        self.position += self.velocity * step * self.speed
        self.speed = average_speed + ((noise_max - noise_min) * np.random.random(dim) + noise_min)
        self.speed /= norm(self.speed)

        if average_velocity > self.max_velocity:
            average_velocity = self.max_velocity
        self.velocity = average_velocity

        for i in range(dim):
            if self.position[i] > length:
                self.position[i] = -length
            elif self.position[i] < -length:
                self.position[i] = length


class Group:
    """
    Simule un groupe d'agents, permet de le faire évoluer et de l'afficher.

    Paramètres
    ---------
    agents : list
        Liste des agents du groupe.
    length : int, optionnel
        Longueur caractéristique de l'espace.
    dim : int, optionnel
        Dimension de l'espace considéré (2 ou 3).

    Attributs
    ---------
    density : float
        Densité d'agents dans l'espace (nombre agent / longueur ** dimension).
    dead_agents : list
        Liste des agents touchés par un agent répulsif.
    """
    def __init__(self, agents: list, length: int=50, dim: int=2):
        self.agents = agents
        self.dead_agents = []
        self.nb_agents = len(agents)
        self.length = length

        if dim not in (2, 3):
            raise DimensionError("dim must be 2 or 3")
        self.dimension = dim

        for agent in agents:
            if len(agent.position) != dim or len(agent.speed) != dim:
                raise DimensionError("dimension of agents don't match")

        self.density = self.nb_agents / (self.length ** self.dimension)

    def __getitem__(self, index: int):
        """
        Renvoie l'agent d'indice donné.

        Paramètres
        ---------
        index : int
            Indice de l'agent.

        Signature
        ---------
        out : Agent
            Agent du groupe à l'indice donné.
        """
        return self.agents[index]

    def copy(self):
        """Renvoie une copie profonde du groupe."""
        return Group([agent.copy() for agent in self.agents], self.length, self.dimension)

    def add_agent(self, agent: Agent):
        """
        Permet d'ajouter un agent au groupe.
        C'est un copie profonde de l'agent qui est ajoutée au groupe.

        Paramètres
        ----------
        agent : Agent
            Agent à ajouter.
        """
        if len(agent.position) != self.dimension or len(agent.speed) != self.dimension:
            raise DimensionError("dimension of agent doesn't match")
        self.agents.append(agent.copy())
        self.nb_agents += 1
        self.density = self.nb_agents / (self.length ** self.dimension)

    def get_neighbours(self, targeted_agent: Agent, dmin: int, check_field: bool=True,
                check_wall: bool=True):
        """
        Calcule les voisins d'un agent en fonction de la distance et de l'angle.

        Paramètres
        ---------
        targeted_agent : Agent
            Agent servant de référence pour le calcul de distance.
        dmin : int
            Distance minimale à partir de laquelle l'agent sera compté comme voisin.
        check_field : bool, optionnel
            Vérification de l'angle de vue. Si ``check_field=False`` les agents voient à 360°.
        check_wall : bool, optionnel
            Vérification des murs. Si ``check_wall=False``, l'espace est considéré torique.

        Signature
        ---------
        agents : list
            liste des agents voisins
        """
        length = self.length / 2

        if self.dimension == 2:
            wall_agents = [
                Agent(position=np.array([length, targeted_agent.position[1]]),
                    speed=np.zeros(2),
                    velocity=0,
                    noise=0,
                    sight=0,
                    field_sight=0,
                    agent_type=3),
                Agent(position=np.array([-length, targeted_agent.position[1]]),
                    speed=np.zeros(2),
                    velocity=0,
                    noise=0,
                    sight=0,
                    field_sight=0,
                    agent_type=3),
                Agent(position=np.array([targeted_agent.position[0], length]),
                    speed=np.zeros(2),
                    velocity=0,
                    noise=0,
                    sight=0,
                    field_sight=0,
                    agent_type=3),
                Agent(position=np.array([targeted_agent.position[0], length]),
                    speed=np.zeros(2),
                    velocity=0,
                    noise=0,
                    sight=0,
                    field_sight=0,
                    agent_type=3),
            ]
        else:
            wall_agents = [
                Agent(position=np.array([length, targeted_agent.position[1], targeted_agent.position[2]]),
                    speed=np.zeros(3),
                    velocity=0,
                    noise=0,
                    sight=0,
                    field_sight=0,
                    agent_type=3),
                Agent(position=np.array([-length, targeted_agent.position[1], targeted_agent.position[2]]),
                    speed=np.zeros(3),
                    velocity=0,
                    noise=0,
                    sight=0,
                    field_sight=0,
                    agent_type=3),
                Agent(position=np.array([targeted_agent.position[0], -length, targeted_agent.position[2]]),
                    speed=np.zeros(3),
                    velocity=0,
                    noise=0,
                    sight=0,
                    field_sight=0,
                    agent_type=3),
                Agent(position=np.array([targeted_agent.position[0], length, targeted_agent.position[2]]),
                    speed=np.zeros(3),
                    velocity=0,
                    noise=0,
                    sight=0,
                    field_sight=0,
                    agent_type=3),
                Agent(position=np.array([targeted_agent.position[0], targeted_agent.position[1], -length]),
                    speed=np.zeros(3),
                    velocity=0,
                    noise=0,
                    sight=0,
                    field_sight=0,
                    agent_type=3),
                Agent(position=np.array([targeted_agent.position[0], targeted_agent.position[1], length]),
                    speed=np.zeros(3),
                    velocity=0,
                    noise=0,
                    sight=0,
                    field_sight=0,
                    agent_type=3),
            ]

        agents = []
        if check_wall:
            total_agents = self.agents + wall_agents
        else:
            total_agents = self.agents

        if not check_field or self.dimension == 3:
            agents = [agent for agent in total_agents if (targeted_agent - agent) <= dmin]

        else:
            dead_index = []
            for index, agent in enumerate(total_agents):
                if targeted_agent.agent_type == 1 and not agent.agent_type in (1, 3) and (targeted_agent - agent) < self.length / 25 and index < self.nb_agents:
                    self.dead_agents.append(agent.copy())
                    dead_index.append(index)
                    self.nb_agents -= 1

                if agent != targeted_agent:
                    pos = agent.position - targeted_agent.position
                    angle_spd = np.angle(targeted_agent.speed[0] + 1j * targeted_agent.speed[1]) % 2 * math.pi
                    angle_pos = np.angle(pos[0] + 1j * pos[1]) % (2 * math.pi)
                    if agent.agent_type != 3:
                        if (targeted_agent - agent) <= dmin and (agent.agent_type == 1 or abs(angle_spd - angle_pos) <= targeted_agent.field_sight):
                            agents.append(agent)
                    elif (targeted_agent - agent) <= self.length / 25:
                        agents.append(agent)

                else: agents.append(agent)

            for index in dead_index:
                self.agents.pop(index)

        return agents

    def get_agents_arguments(self):
        """Retourne un tuple de tableaux numpy contenant les positions et les vitesses de tous les agents du groupe."""
        positions = np.zeros((self.nb_agents, self.dimension))
        speeds = np.zeros((self.nb_agents, self.dimension))

        for index, agent in enumerate(self.agents):
            positions[index] = agent.position
            speeds[index] = agent.velocity * agent.speed

        return positions, speeds

    def compute_figure(self):
        """Génère une figure matplotlib avec le groupe d'agents sous forme d'un nuage de points en deux ou trois dimensions."""
        fig = plt.figure()
        if self.dimension == 2:
            plot_axes = plt.axes()
            sight_wedges = []

            for agent in self.agents:
                color, dir_angle = agent.get_color()
                sight_angle = (180 * agent.field_sight) / math.pi

                if agent.agent_type:
                    size, color = 7, (0, 0, 0)
                else:
                    size = 5
                plt.scatter(agent.position[0], agent.position[1], s=size, color=color)

                wedge = mpatches.Wedge((agent.position[0], agent.position[1]), agent.sight,
                            dir_angle + 360 - sight_angle, dir_angle + sight_angle, ec="black")
                sight_wedges.append(wedge)

            plot_axes.add_collection(PatchCollection(sight_wedges, alpha=0.3))

            plot_axes.axes.set_xlim(-self.length / 2, self.length / 2)
            plot_axes.axes.set_ylim(-self.length / 2, self.length / 2)

        return fig

    def show(self):
        """Affiche le groupe d'agent."""
        self.compute_figure()
        plt.show()

    def compute_animation(self, frames: int=20, interval: int=100, filename: str="vicsek", check_field: bool=True, check_wall: bool=False, sight: bool=False, step: float=0.5):
        """
        Génère une animation.

        Paramètres
        ---------
        frames : int, optionnel
            Nombre d'images voulues dans l'animation.
        interval : int, optionnel
            Intervale temporel entre deux frames de l'animation en ms.
        filename : str, optionnel
            Nom du fichier de sortie GIF.
        check_field : bool, optionnel
            Vérification de l'angle de vue. Si ``check_field=False`` les agents voient à 360°.
        check_wall : bool, optionnel
            Vérification des murs. Si ``check_wall=False``, l'espace est considéré torique.
        sight : bool, optionnel
            Affichage du cône de vision de chaque agent.
        step : float, optionnel
            Pas temporel pris pour les équations différentielles.
        """
        def aux(frame_index, plot_axes, sight: bool=True):
            progress_bar(frame_index, frames, finished="exportation GIF en cours")

            sight_wedges = []
            plot_data = []
            size = 5

            if self.dimension == 2:
                for agent in self.agents:
                    if agent.agent_type != 3:
                        agent.next_step(
                                self.get_neighbours(agent, agent.sight, check_field, check_wall),
                                self.dimension, self.length, step)

                    color, dir_angle = agent.get_color()
                    sight_angle = (180 * agent.field_sight) / math.pi

                    if agent.agent_type:
                        size, color = 7, (0, 0, 0)
                    else:
                        size = 5

                    plot_data.append(plot_axes.scatter(agent.position[0], agent.position[1],
                            s=size, color=color))

                    if sight or agent.agent_type == 1:
                        wedge = mpatches.Wedge((agent.position[0], agent.position[1]), agent.sight,
                                dir_angle + 360 - sight_angle, dir_angle + sight_angle,
                                ec="none")
                        sight_wedges.append(wedge)

                plot_data.append(plot_axes.add_collection(PatchCollection(sight_wedges, alpha=0.3)))

            else:
                for agent in self.agents:
                    if agent.agent_type != 3:
                        agent.next_step(self.get_neighbours(agent, agent.sight, check_field),
                                self.dimension, self.length, step)
                    sight_angle = (180 * agent.field_sight) / math.pi

                    if agent.agent_type:
                        size, color = 7, (1, 0, 0)
                    else:
                        size, color = 5, (0, 0, 0)

                    plot_data.append(plot_axes.scatter(agent.position[0], agent.position[1], agent.position[2], s=size, color=color))
                    plot_data.append(plot_axes.quiver(agent.position[0], agent.position[1],
                            agent.position[2], agent.velocity * agent.speed[0],
                            agent.velocity * agent.speed[1], agent.velocity * agent.speed[2],
                            color=color))

            return plot_data

        images = []

        fig = plt.figure()
        if self.dimension == 2:
            plot_axes = plt.axes()
            plot_axes.axes.set_xlim(-self.length / 2, self.length / 2)
            plot_axes.axes.set_ylim(-self.length / 2, self.length / 2)
        else:
            plot_axes = plt.axes(projection="3d")
            plot_axes.axes.set_xlim3d(-self.length / 2, self.length / 2)
            plot_axes.axes.set_ylim3d(-self.length / 2, self.length / 2)
            plot_axes.axes.set_zlim3d(-self.length / 2, self.length / 2)

        for findex in range(frames):
            plot_data = aux(findex, plot_axes, sight)
            images.append(plot_data)

        ani = animation.ArtistAnimation(fig, images, interval=interval)
        ani.save(filename + ".gif")

    def run(self, steps: int=20, check_field: bool=True, check_wall: bool=True, step: float=0.5):
        """
        Fait avancer le groupe d'agent sans gérérer d'animations.

        Paramètres
        ---------
        steps : int, optionnel
            Nombre de pas dont il faut faire avancer le groupe.
        check_field : bool, optionnel
            Vérification de l'angle de vue. Si ``check_field=False`` les agents voient à 360°.
        check_wall : bool, optionnel
            Vérification des murs. Si ``check_wall=False``, l'espace est considéré torique.
        step : float, optionnel
            Pas temporel pris pour les équations différentielles.
        """
        for index in range(steps):
            progress_bar(index, steps)
            for agent in self.agents:
                if agent.agent_type != 3:
                    agent.next_step(
                                self.get_neighbours(agent, agent.sight, check_field, check_wall),
                                self.dimension, self.length, step)

    def order_parameter(self):
        """Renvoie le paramètre d'alignement."""
        speed = np.zeros(self.dimension)
        velocity = 0
        for agent in self.agents:
            agent_speed = agent.velocity * agent.speed

            speed += agent_speed
            velocity += norm(agent_speed)

        return (1 / velocity) * norm(speed)


class DimensionError(Exception):
    """Erreur de dimension."""


# Fonctions
def rarray(dim: int, minimum: float, maximum: float):
    """
    Retourne un tableau numpy de taille donnée rempli de nombres aléatoire pris entre les bornes
    communiquées.
    
    Paramètres
    ----------
    dim : int
        Dimension du tableau.
    minimum : float
        Valeur minimum du tableau.
    maximum : float
        Valeur maximum du tableau.

    Signature
    ---------
    out : np.array
        Tableau numpy de nombres aléatoires.
    """
    return minimum + (maximum - minimum) * np.random.random(dim)


def agent_generator(position: tuple=(-25, 25), speed: tuple=(-2, 2), noise: tuple=(0, 1), sight: tuple=(5, 10), field_sight: tuple=(math.pi/4, math.pi/2), agent_type: int=0, fear: tuple=(0, 1), dim: int=2):
    """
    Retourne un agent généré aléatoirement.

    Paramètres
    ---------
    position : tuple, optionnel
        Valeurs limites de la position.
    speed : tuple, optionnel
        Valeurs limites de la vitesse.
    noise : tuple, optionnel
        Valeurs limites du bruit de l'agent.
    sight : tuple, optionnel
        Valeurs limites de la portée de la vue des agents.
    field_sight : tuple, optionnel
        Valeurs limites du champ de vision des agents.
    agent_type : int, optionnel
        Type de l'agent
        0 : agent normal
        1 : agent répulsif
        2 : agent leader
        (3 : mur)
    fear : tuple, optionnel
        Valeurs limites de la peur de l'agent aux agents répulsifs.
    dim : int, optionnel
        Dimension de l'espace, peut être 2 ou 3.

    Signature
    ---------
    agent : Agent
        Agent généré dans la limite des paramètres donnés.
    """
    if noise == -1:
        noise = random.random()
    if fear == -1:
        fear = random.random()
    agent = Agent(
        position=rarray(dim, position[0], position[1]),
        speed=np.zeros(dim),
        velocity=0,
        noise=(noise[1] - noise[0]) * random.random() + noise[0],
        sight=(sight[1] - sight[0]) * random.random() + sight[0],
        field_sight=(field_sight[1] - field_sight[0]) * random.random() + field_sight[0],
        agent_type=agent_type,
        fear=(fear[1] - fear[0]) * random.random() + fear[0]
    )

    velocity = 0
    while velocity < 1e-3:
        agent.speed = rarray(dim, speed[0], speed[1])
        velocity = norm(agent.speed)

    agent.speed /= velocity
    agent.velocity = velocity
    return agent.copy()


def group_generator(nb_agents: int, position: tuple=(-25, 25), speed: tuple=(-2, 2), noise: tuple=(0, 1), sight: tuple=(5, 10), field_sight: tuple=(math.pi/4, math.pi/2), fear: tuple=(0, 1), length: int=50, dim: int=2):
    """
    Retourne un groupe d'agents normaux générés aléatoirement dans les limites données.

    Paramètres
    ---------
    nb_agents : int
        Nombre d'agents à générer pour le groupe.
    position : tuple, optionnel
        Valeurs limites de la position.
    speed : tuple, optionnel
        Valeurs limites de la vitesse.
    noise : tuple, optionnel
        Valeurs limites du bruit de l'agent.
    sight : tuple, optionnel
        Valeurs limites de la portée de la vue des agents.
    field_sight : tuple, optionnel
        Valeurs limites du champ de vision des agents.
    fear : tuple, optionnel
        Valeurs limites de la peur de l'agent aux agents répulsifs.
    length : int, optionnel
        Longueur caratéristique de l'espace.
    dim : int, optionnel
        Dimension de l'espace, peut être 2 ou 3.

    Signature
    ---------
    out : Group
        Groupe contenant les agents générés dans les limites données et avec les paramètres de
        longueur et de dimension donnés.
    """
    agents = [agent_generator(
            position=position,
            speed=speed,
            noise=noise,
            sight=sight,
            field_sight=field_sight,
            fear=fear, dim=dim)
            for _ in range(nb_agents)]
    return Group(agents, length=length, dim=dim)


def norm(vect: np.array):
    """
    Renvoie la norme du vecteur passé en argument.

    Paramètres
    ---------
    vect : np.array
        Vecteur n-dimensionnel.

    Signature
    ---------
    out : float
        Norme du vecteur.
    """
    return math.sqrt(sum(vect ** 2))


def get_colors():
    """Retourne une liste de couleur indexée sur l'angle avec l'horizontale ascendante."""
    color_map = []
    red, green, blue = 255, 0, 0
    for angle in range (360):
        if (angle // 60) == 0:
            green += 4.25
        elif (angle // 60) == 1:
            red -= 4.25
        elif (angle // 60) == 2:
            blue += 4.25
        elif (angle // 60) == 3:
            green -= 4.25
        elif (angle // 60) == 4:
            red += 4.25
        elif (angle // 60) == 5:
            blue -= 4.25
        color_map.append((red / 255, green / 255, blue / 255))
    return color_map


def progress_bar(iteration: int, total: int, finished: str=""):
    """
    Affiche une barre de progression.

    Paramètres
    ---------
    iteration : int
        Itération courante à afficher.
    total: int
        Nombre total d'itération sur la barre
    finished : str, optionnel
        Texte à afficher une fois la barre complète.
    """
    iteration += 1
    completed_length = math.floor(75 * iteration / total)
    track = "#" * completed_length + " " * (75 - completed_length)
    print(f"[{track}] {math.floor(100 * iteration / total)}%", end="\r")

    if iteration == total:
        if finished:
            print("\n" + finished)
        else:
            print()


def stat(agents):
    """
    Affiche des statistiques sur une liste d'agents.
     – bruit moyen
     – peur moyenne
     – vitese moyenne
    """
    noise, fear, velocity = 0, 0, 0
    nb_agents = 0

    for agent in agents:
        if agent.agent_type in (0, 2):
            nb_agents += 1
            noise += agent.noise
            fear += agent.fear
            velocity += agent.max_velocity

    noise /= nb_agents
    fear /= nb_agents
    velocity /= nb_agents

    print("noise   :", noise)
    print("fear    :", fear)
    print("vitesse :", velocity)


# Constantes
COLOR_MAP = get_colors()
