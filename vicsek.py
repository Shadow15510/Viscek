# ┌──────────────────────────────────┐ #
# │          Vicsek — 1.7.2          │ #
# │ Alexis Peyroutet & Antoine Royer │ #
# │ GNU General Public Licence v3.0+ │ #
# └──────────────────────────────────┘ #

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import random


__name__ = "Vicsek"
__version__ = "1.7.2"


# ┌─────────┐ #
# │ Classes │ #
# └─────────┘ #
class Agent:
    """
    Simule un agent avec sa position, sa vitesse, et le bruit associé qui traduit sa tendance naturelle à suivre le groupe ou pas.
    @arguments :
        position    : vecteur position (sous forme d'un tableau numpy de trois entiers)
        speed       : direction du vecteur vitesse (tableau de trois entiers)
        velocity    : norme de la vitesse
        noise       : taux de bruit qui traduit une déviation aléatoire sur la direction de la vitesse
        sight       : distance à laquelle l'agent voit les autres
        field_sight : angle du cône de vision de l'agent en radian                                     [optionnel, default = math.pi/4]
        agent_type  : type d'agent                                                                     [optionnel, default = 0]
            0 : agent normal
            1 : agent répulsif
            2 : agent leader
        fear        : sensibilité aux agents répulsif (entre 0 et 1)                                          [optionnel, default = 1]
    """

    def __init__(self, position: np.array, speed: np.array, velocity: int, noise: int, sight: int, field_sight: float=math.pi/2, agent_type: int=0, fear: float=1):
        self.position = position.copy()
        self.speed = speed.copy()
        self.velocity = velocity
        self.noise = noise
        self.sight = sight
        self.field_sight = field_sight / 2
        self.agent_type = agent_type
        self.fear = fear

        self.max_velocity = self.velocity

    def __str__(self):
        """Affiche l'agent avec ses paramètres."""
        return f"Agent(\n\tposition={list(self.position)},\n\tspeed={list(self.speed)},\n\tvelocity={self.velocity},\n\tnoise={self.noise},\n\tsight={self.sight},\n\tfield_sight={self.field_sight * 360 / math.pi},\n\tagent_type={self.agent_type}\n\tfear={self.fear}\n)"
    
    __repr__ = __str__

    def __sub__(self, agent):
        """
        Retourne la distance entre les deux agents.
        @arguments :
            agent : un autre agent
        """
        return norm(self.position - agent.position)

    __rsub__ = __sub__

    def __eq__(self, agent):
        """
        Retourne True si les agents ont les mêmes positions, False sinon.
        @arguments :
            agent : un autre agent
        """
        return list(self.position) == list(agent.position)

    def copy(self):
        """Renvoie une copie profonde de l'agent."""
        return Agent(self.position.copy(), self.speed.copy(), self.velocity, self.noise, self.sight, self.field_sight, self.agent_type, self.fear)

    def get_color(self):
        """Renvoie la couleur de l'agent en fonction de son orientation."""
        angle = np.angle(self.speed[0] + 1j * self.speed[1]) % (2 * math.pi)
        angle = (180 * angle) / math.pi
        return COLOR_MAP[math.floor(angle) % 360], angle 

    def next_step(self, neighbours: list, dim: int, length: int, dt: float=0.5):
        """
        Fait évoluer l'agent d'un pas temporel en fonction de ses voisins.
        @arguments :
            neighbours : liste des agents voisins
            dim        : dimension de l'espace
            length     : longueur caractéristique de l'espace
            dt         : pas de temps                         [optionnel, default = 0.5]
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
                if self.agent_type != 1: average_speed += agent.speed
                else:
                    average_speed += agent.position - self.position
                    average_velocity += agent.velocity / 4

            elif agent.agent_type == 1:
                if self.agent_type != 1: average_speed += self.fear * len(neighbours) * (self.position - agent.position)
                else: average_speed += agent.speed

            elif agent.agent_type == 2:
                if self.agent_type != 1: average_speed += 5 * agent.speed
                else: average_speed += (agent.position - self.position)

            elif agent.agent_type == 3:
                average_speed += 100 * (self.position - agent.position)
        
        average_speed /= nb_neighbours
        average_velocity /= nb_neighbours

        self.position += self.velocity * dt * self.speed
        self.speed = average_speed + (2 * self.noise * np.random.random(dim) - self.noise)
        self.speed /= norm(self.speed)

        if average_velocity > self.max_velocity: average_velocity = self.max_velocity
        self.velocity = average_velocity

        for i in range(dim):
            if not (-length < self.position[i] < length): self.position = length * np.random.random(dim) - length // 2


class Group:
    """
    Simule un groupe d'agents, permet de le faire évoluer et de l'afficher.
    @arguments :
        agents  : liste des agents du groupe
        length  : longueur caractéristique de l'espace     [optionnel, default = 50]
        dim     : dimension de l'espace considéré (2 ou 3) [optionnel, default = 2]
    @attributs :
        density     : densité d'agents dans l'espace (nombre agent / longueur ** dimension)
        dead_agents : liste des agents touchés par un agent répulsif
    """
    def __init__(self, agents: list, length: int=50, dim: int=2):
        
        self.agents = agents
        self.dead_agents = []
        self.nb_agents = len(agents)
        self.length = length
        
        if not dim in (2, 3):
            raise DimensionError("dim must be 2 or 3")
        self.dimension = dim

        for agent in agents:
            if len(agent.position) != dim or len(agent.speed) != dim:
                raise DimensionError("dimension of agents don't match")
        
        self.density = self.nb_agents / (self.length ** self.dimension)
            
    def __getitem__(self, index: int):
        """
        Renvoie l'agent d'indice index.
        @arguments :
            index : indice de l'agent
        """
        return self.agents[index]
    
    def copy(self):
        """Renvoie une copie profonde du groupe."""
        return Group([agent.copy() for agent in self.agents], self.length, self.dimension)

    def add_agent(self, agent: Agent):
        if len(agent.position) != self.dimension or len(agent.speed) != self.dimension:
                raise DimensionError("dimension of agent doesn't match")
        self.agents.append(agent.copy())
        self.nb_agents += 1

    def get_neighbours(self, targeted_agent: Agent, dmin: int, check_field: bool=True):
        """
        Retourne une liste d'agents appartenant au groupe et étant à une distance inférieure ou égale à dmin.
        @arguments :
            targeted_agent : agent servant de référence pour le calcul de distance
            dmin           : distance minimale à considérer
            check_field    : prise en compte de l'angle de vue de l'agent          [optionnel, default = True]
        """
        length = self.length // 2
        wall_agents = [
            Agent(position=np.array([targeted_agent.position[0], -length]), speed=np.array([0, 0]), velocity=0, noise=0, sight=0, field_sight=0, agent_type=3),
            Agent(position=np.array([targeted_agent.position[0], length]), speed=np.array([0, 0]), velocity=0, noise=0, sight=0, field_sight=0, agent_type=3),
            Agent(position=np.array([length, targeted_agent.position[1]]), speed=np.array([0, 0]), velocity=0, noise=0, sight=0, field_sight=0, agent_type=3),
            Agent(position=np.array([-length, targeted_agent.position[1]]), speed=np.array([0, 0]), velocity=0, noise=0, sight=0, field_sight=0, agent_type=3),
        ]
        agents = []

        if not check_field:
            agents = [agent for agent in (self.agents + wall_agents) if (targeted_agent - agent) <= dmin]
       
        else:
            dead_index = []
            for index, agent in enumerate(self.agents + wall_agents):
                if targeted_agent.agent_type == 1 and agent.agent_type != 1 and (targeted_agent - agent) < 1 and index < self.nb_agents:
                    self.dead_agents.append(agent.copy())
                    dead_index.append(index)
                    self.nb_agents -= 1

                if agent != targeted_agent:
                    pos = agent.position - targeted_agent.position
                    angle_spd = np.angle(targeted_agent.speed[0] + 1j * targeted_agent.speed[1]) % (2 * math.pi)
                    angle_pos = np.angle(pos[0] + 1j * pos[1]) % (2 * math.pi)
                    if (targeted_agent - agent) <= dmin and (agent.agent_type in (1, 3) or abs(angle_spd - angle_pos) <= targeted_agent.field_sight): agents.append(agent)        
                else: agents.append(agent)

            for index in dead_index: self.agents.pop(index)

        return agents

    def get_agents_parameters(self):
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
            ax = plt.axes()
            sight_wedges = []

            for agent in self.agents:
                _, dir_angle = agent.get_color()
                sight_angle = (180 * agent.field_sight) / math.pi

                if agent.agent_type: size = 7
                else: size = 5
                plt.scatter(agent.position[0], agent.position[1], s=size, color="black") #agent_color)
                
                wedge = mpatches.Wedge((agent.position[0], agent.position[1]), agent.sight, dir_angle + 360 - sight_angle, dir_angle + sight_angle, ec="black")
                sight_wedges.append(wedge)

            ax.add_collection(PatchCollection(sight_wedges, alpha=0.3))
                
            ax.axes.set_xlim(-self.length // 2, self.length // 2)
            ax.axes.set_ylim(-self.length // 2, self.length // 2)

        # else:
        #     ax = plt.axes(projection="3d")
        #     ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=5, c="black")
        #     for agent_index in range(self.nb_agents):
        #         ax.quiver(positions[agent_index, 0], positions[agent_index, 1], positions[agent_index, 2], speeds[agent_index, 0], speeds[agent_index, 1], speeds[agent_index, 2], color="black")
        #     ax.axes.set_xlim3d(-self.length // 2, self.length // 2)
        #     ax.axes.set_ylim3d(-self.length // 2, self.length // 2)
        #     ax.axes.set_zlim3d(-self.length // 2, self.length // 2)

        return fig

    def show(self):
        """Affiche le groupe d'agent."""
        self.compute_figure()
        plt.show()

    def compute_animation(self, frames: int=20, interval: int=100, filename: str="vicsek", check_field: bool=True, sight: bool=False, dt: float=0.5):
        """
        Génère une animation.
        @arguments :
            frames      : nombre d'images voulues dans l'animation         [optionnel, default = 20]
            interval    : intervale entre deux images de l'animation en ms [optionnel, default = 100]
            filename    : nom du fichier de sortie GIF                     [optionnel, default = "vicsek"]
            check_field : vérification de l'angle de vue                   [optionnel, default = True]
            sight       : affichage du cône de vision                      [optionnel, default = False]
            dt          : pas de temps                                     [optionnel, default = 0.5]
        """
        def aux(frame_index, ax, sight: bool=True):
            progress_bar(frame_index, frames, finished="exportation GIF en cours")

            if self.dimension == 2:
                sight_wedges = []
                pl = []
                size = 5

                for index, agent in enumerate(self.agents):
                    agent.next_step(self.get_neighbours(agent, agent.sight, check_field), self.dimension, self.length, dt)
                    color, dir_angle = agent.get_color()
                    sight_angle = (180 * agent.field_sight) / math.pi

                    if agent.agent_type: size, color = 7, (0, 0, 0)
                    else: size = 5

                    pl.append(ax.scatter(agent.position[0], agent.position[1], s=size, color=color))

                    wedge = mpatches.Wedge((agent.position[0], agent.position[1]), agent.sight, dir_angle + 360 - sight_angle, dir_angle + sight_angle, ec="none")
                    sight_wedges.append(wedge)
                
                if sight: pl.append(ax.add_collection(PatchCollection(sight_wedges, alpha=0.3)))
                
            # else:
            #     ax = plt.axes(projection="3d")
            #     ax.axes.set_xlim3d(-self.length // 2, self.length // 2)
            #     ax.axes.set_ylim3d(-self.length // 2, self.length // 2)
            #     ax.axes.set_zlim3d(-self.length // 2, self.length // 2)

            #     for index, agent in enumerate(self.agents):
            #         if agent.agent_type == 0: agent.next_step(self.get_neighbours(agent, agent.sight), self.dimension)
            #         positions[index] = agent.position
            #         ax.quiver(agent.position[0], agent.position[1], agent.position[2], agent.speed[0], agent.speed[1], agent.speed[2], color="black")

            return pl

        images = []

        fig = plt.figure()
        ax = plt.axes()
        ax.axes.set_xlim(-self.length // 2, self.length // 2)
        ax.axes.set_ylim(-self.length // 2, self.length // 2)
        
        for findex in range(frames):
            pl = aux(findex, ax, sight)
            images.append(pl)

        ani = animation.ArtistAnimation(fig, images, interval=interval)
        ani.save(filename + ".gif")

    def run(self, steps: int=20, check_field: bool=True, dt: float=0.5):
        """
        Fait avancer le groupe d'agent sans gérérer d'animations.
        @arguments :
            steps       : nombre de pas                  [optionnel, defaut = 20]
            check_field : vérification de l'angle de vue [optionnel, defaut = True]
            dt          : pas de temps                   [optionnel, defaut = 0.5]
        """
        for index in range(steps):
            #progress_bar(index, steps)
            for agent in self.agents:
                agent.next_step(self.get_neighbours(agent, agent.sight, check_field), self.dimension, self.length, dt)


class DimensionError(Exception):
    pass


# ┌───────────┐ #
# │ Fonctions │ #
# └───────────┘ #
rarray = lambda dim, minimum, maximum: minimum + (maximum - minimum) * np.random.random(dim)


def agent_generator(position: tuple=(-25, 25), speed: tuple=(-2, 2), noise: float=-1, sight: tuple=(5, 10), field_sight: tuple=(math.pi/4, math.pi/2), agent_type: int=0, fear: float=-1, dim: int=2):
    """
    Retourne un agent généré aléatoirement.
    @arguments :
        position    : valeurs limites de la position                    [optionnel, defaut = (-25, 25)]
        speed       : valeurs limites de la vitesse                     [optionnel, defaut = (-2, 2)]
        noise       : bruit de l'agent                                  [optionnel, default = random.random()]
        sight       : valeurs limites de la portée de la vue des agents [optionnel, defaut = (5, 10)]
        field_sight : valeurs limites du champ de vision des agents     [optionnel, defaut = (math.pi/4, math.pi/2)]
        agent_type  : type de l'agent                                   [optionnel, defaut = 0]
            0 : agent normal
            1 : agent répulsif
            2 : agent leader
        fear        : sensibilité de l'agent aux agents répulsifs       [optionnel, defaut = random.random()]
        dim         : dimension de l'espace                             [optionnel, defaut = 2]
    """
    if noise == -1: noise = random.random()
    if fear == -1: fear = random.random()
    agent = Agent(
        position=rarray(dim, position[0], position[1]),
        speed=np.zeros(dim),
        velocity=0,
        noise=noise,
        sight=random.randint(sight[0], sight[1]),
        field_sight=(field_sight[1] - field_sight[0]) * random.random() + field_sight[0],
        agent_type=agent_type,
        fear=fear
    )

    velocity = 0
    while velocity == 0:
        agent.speed = rarray(dim, speed[0], speed[1])
        velocity = norm(agent.speed)

    agent.speed /= velocity
    agent.velocity = velocity
    return agent.copy()


def group_generator(nb: int, position: tuple=(-25, 25), speed: tuple=(-2, 2), noise: float=-1, sight: tuple=(5, 10), field_sight: tuple=(math.pi/4, math.pi/2), fear: float=-1, length: int=50, dim: int=2):
    """
    Retourne un groupe d'agents normaux générés aléatoirement dans les limites données.
    @arguments :
        nb          : nombre d'agents à générer
        position    : valeurs limites de la position                        [optionnel, defaut = (-25, 25)]
        speed       : valeurs limites de la vitesse                         [optionnel, defaut = (-2, 2)]
        noise       : bruit de l'agent                                      [optionnel, default = random.random()]
        sight       : valeurs limites de la portée de la vue des agents     [optionnel, defaut = (5, 10)]
        field_sight : valeurs limites du champ de vision des agents         [optionnel, defaut = (math.pi/4, math.pi/2)]
        fear        : sensibilité de l'agent aux agents répulsifs           [optionnel, defaut = random.random()]
        length      : taille de l'arête du cube d'espace considéré          [optionnel, defaut = 50]
        dim         : dimension de l'espace dans lequel les agents évoluent [optionnel, defaut = 2]
    """
    agents = [agent_generator(position=position, speed=speed, noise=noise, sight=sight, field_sight=field_sight, fear=fear, dim=dim) for _ in range(nb)]
    return Group(agents, length=length, dim=dim)


def norm(vect: np.array):
    """
    Renvoie la norme du vecteur passé en argument.
    @arguments
        vect : tableau numpy
    """
    return math.sqrt(sum(vect ** 2))


def get_colors():
    """Retourne une liste de couleur indexée sur l'angle avec l'horizontale ascendante."""
    color_map = []
    r, g, b = 255, 0, 0
    for angle in range (360):
        if (angle // 60) == 0: g += 4.25
        elif (angle // 60) == 1: r -= 4.25
        elif (angle // 60) == 2: b += 4.25
        elif (angle // 60) == 3: g -= 4.25
        elif (angle // 60) == 4: r += 4.25
        elif (angle // 60) == 5: b -= 4.25
        color_map.append((r / 255, g / 255, b / 255))
    return color_map


def progress_bar(iteration: int, total: int, finished: str=""):
    """
    Affiche une barre de progression.
    @arguments :
        iteration : itération courante
        total     : nombre totale d'itération
        finished  : texte à afficher une fois la barre complète [optionnel, defaut = ""]
    """
    iteration += 1
    completed_length = math.floor(80 * iteration / total)
    bar = "#" * completed_length + " " * (80 - completed_length)
    print(f"\r[{bar}] {math.floor(100 * iteration / total)}%", end="\r")

    if iteration == total:
        if finished: print("\n" + finished)
        else: print()


def stat(agents):
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


# ┌─────────┐ #
# │ Données │ #
# └─────────┘ #
COLOR_MAP = get_colors()

group_10 = group_generator(10)

group_20 = group_generator(19, position=(-1, 1), speed=(-1, 1), length=4)

group_40 = group_generator(40, noise=1, fear=0)
for _ in range(2):
    group_40.add_agent(agent_generator(speed=(-3, 3), noise=0.25, agent_type=1))
group_40_bis = group_40.copy()
for i in range(40):
    group_40_bis[i].noise, group_40_bis[i].fear = 0, 1

group_100 = group_generator(100, position=(-25, 25), speed=(-1, 1))

group_200 = group_generator(200, position=(-1, 1), speed=(-1, 1), length=4)



def test():
    group_1 = group_generator(50, noise=0, fear=0)
    for _ in range(2):
        group_1.add_agent(agent_generator(speed=(-3, 3), noise=0.25, agent_type=1))

    group_2 = group_1.copy()
    group_3 = group_1.copy()
    group_4 = group_1.copy()
    for i in range(50):
        group_1[i].noise, group_1[i].fear = 1, 0
        group_2[i].noise, group_2[i].fear = 0, 1
        group_3[i].noise, group_3[i].fear = 1, 1
        group_4[i].noise, group_4[i].fear = 0, 0

    group_1.run(500)
    group_2.run(500)
    group_3.run(500)
    group_4.run(500)

    return len(group_1.dead_agents), len(group_2.dead_agents), len(group_3.dead_agents), len(group_4.dead_agents)
