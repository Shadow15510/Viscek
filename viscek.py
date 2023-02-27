# ┌──────────────────────────────────┐ #
# │          Viscek — 1.6.0          │ #
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


__name__ = "Viscek"
__version__ = "1.6.0"


# ┌─────────┐ #
# │ Classes │ #
# └─────────┘ #

class Agent:
    """Simule un agent avec sa position, sa vitesse, et le bruit associé qui traduit sa tendance naturelle à suivre le groupe ou pas."""

    def __init__(self, position: np.array, speed: np.array, velocity: int, noise: int, sight: int, field_sight: float=math.pi/2, agent_type: int=0):
        """
        position    : vecteur position (sous forme d'un tableau numpy de trois entiers)
        speed       : direction du vecteur vitesse (tableau de trois entiers)
        velocity    : norme de la vitesse
        noise       : taux de bruit qui traduit une déviation aléatoire sur la direction de la vitesse
        sight       : distance à laquelle l'agent voit les autres
        field_sight : angle du cône de vision de l'agent en radian
        agent_type  : type d'agent (0 : agent normal ; 1 : agent répulsif)
        """
        self.position = position.copy()
        self.speed = speed.copy()
        self.velocity = velocity
        self.noise = noise / 2
        self.sight = sight
        self.field_sight = field_sight / 2
        self.agent_type = agent_type

    def __str__(self):
        """Affiche l'agent avec ses paramètres."""
        return f"Agent(\n\tposition={list(self.position)},\n\tspeed={list(self.speed)},\n\tvelocity={self.velocity},\n\tnoise={self.noise},\n\tsight={self.sight},\n\tfield_sight={self.field_sight * 360 / math.pi},\n\ttype={self.agent_type}\n)"
    
    __repr__ = __str__

    def __sub__(self, agent):
        """
        agent : un autre agent
        Retourne la distance entre les deux agents.
        """
        return norm(self.position - agent.position)

    __rsub__ = __sub__

    def __eq__(self, agent):
        """
        agent : un autre agent
        Retourne True si les agents ont les mêmes positions, False sinon.
        """
        return list(self.position) == list(agent.position)

    def copy(self):
        """Renvoie une copie profonde de l'agent."""
        return Agent(self.position.copy(), self.speed.copy(), self.velocity, self.noise, self.sight, self.field_sight, self.agent_type)

    def get_color(self):
        """Renvoie la couleur de l'agent en fonction de son orientation."""
        angle = np.angle(self.speed[0] + 1j * self.speed[1]) % (2 * math.pi)
        angle = (180 * angle) / math.pi
        return COLOR_MAP[math.floor(angle)], angle 

    def next_step(self, neighbours: list, dim: int):
        """
        neighbours : liste des agents voisins
        dim        : dimension de l'espace
        """
        average_speed = np.zeros((dim))
        average_velocity = 0
        for agent in neighbours:
            if agent.agent_type == 1:
                average_speed += (self.position - agent.position)
            average_speed += agent.speed
            average_velocity += agent.velocity

        average_speed /= len(neighbours)
        average_velocity /= len(neighbours)

        self.position += self.velocity * 0.5 * self.speed
        self.speed = average_speed + (2 * self.noise * np.random.random(dim) - self.noise)
        self.speed /= norm(self.speed)
        self.velocity = average_velocity


class Group:
    """Simule un groupe d'agents, permet de le faire évoluer et de l'afficher."""
    
    def __init__(self, agents: list, length: int=50, dim: int=2):
        """
        agents  : liste des agents du groupe
        length  : longueur d'un côté de l'espace en 2 ou 3 dimensions
        dim     : dimension de l'espace considéré (2 ou 3)

        density : densité d'agents dans l'espace (nombre agent / longueur ** dim)
        """
        self.agents = agents
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
        index : indice de l'agent
        Renvoie l'agent d'indice index.
        """
        return self.agents[index]
    
    def copy(self):
        """Renvoie une copie profonde du groupe."""
        return Group([agent.copy() for agent in self.agents], self.dimension)

    def add_agent(self, agent: Agent):
        if len(agent.position) != self.dimension or len(agent.speed) != self.dimension:
                raise DimensionError("dimension of agent doesn't match")
        self.agents.append(agent.copy())
        self.nb_agents += 1

    def get_neighbours(self, targeted_agent: Agent, dmin: int, check_field: bool=True):
        """
        targeted_agent : agent servant de référence pour le calcul de distance
        dmin           : distance minimale à considérer
        check_field    : prise en compte de l'angle de vue de l'agent (True par défaut)
        Retourne une liste d'agents appartenant au groupe et étant à une distance inférieure ou égale à dmin.
        """
        if not check_field:
            return [agent for agent in self.agents if (targeted_agent - agent) <= dmin]
        else:
            agents = []
            for agent in self.agents:
                if agent == targeted_agent: agents.append(agent)
                else:
                    pos = agent.position - targeted_agent.position
                    angle_spd = np.angle(targeted_agent.speed[0] + 1j * targeted_agent.speed[1]) % (2 * math.pi)
                    angle_pos = np.angle(pos[0] + 1j * pos[1]) % (2 * math.pi)
                    if (targeted_agent - agent) <= dmin and (agent.agent_type == 1 or abs(angle_spd - angle_pos) <= targeted_agent.field_sight): agents.append(agent)

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

                plt.scatter(agent.position[0], agent.position[1], s=5, color="black") #agent_color)
                
                wedge = mpatches.Wedge((agent.position[0], agent.position[1]), agent.sight, dir_angle + 360 - sight_angle, dir_angle + sight_angle, ec="black")
                sight_wedges.append(wedge)

            ax.add_collection(PatchCollection(sight_wedges, alpha=0.3))
                
            ax.axes.set_xlim(-self.length, self.length)
            ax.axes.set_ylim(-self.length, self.length)

        # else:
        #     ax = plt.axes(projection="3d")
        #     ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=5, c="black")
        #     for agent_index in range(self.nb_agents):
        #         ax.quiver(positions[agent_index, 0], positions[agent_index, 1], positions[agent_index, 2], speeds[agent_index, 0], speeds[agent_index, 1], speeds[agent_index, 2], color="black")
        #     ax.axes.set_xlim3d(-self.length, self.length)
        #     ax.axes.set_ylim3d(-self.length, self.length)
        #     ax.axes.set_zlim3d(-self.length, self.length)

        return fig

    def show(self):
        """Affiche le groupe d'agent."""
        self.compute_figure()
        plt.show()

    def run(self, frames: int=20, interval: int=100, filename: str="viscek", check_field: bool=True):
        """
        frames      : nombre d'image voulues dans l'animation (20 par défaut)
        interval    : intervale entre deux images de l'animation en ms (100 ms par défaut)
        filename    : nom du GIF enregistré ("viscek" par défaut)
        check_field : vérification de l'angle de vue (True par defaut)
        Génère une animation.
        """
        def compute_animation(frame_index):
            print(f"{math.floor(100 * frame_index/frames)} %")

            if self.dimension == 2:
                ax = plt.axes()
                ax.axes.set_xlim(-self.length, self.length)
                ax.axes.set_ylim(-self.length, self.length)
                sight_wedges = []

                for index, agent in enumerate(self.agents):
                    if agent.agent_type == 0: agent.next_step(self.get_neighbours(agent, agent.sight, check_field), self.dimension)
                    color, dir_angle = agent.get_color()
                    sight_angle = (180 * agent.field_sight) / math.pi

                    plt.scatter(agent.position[0], agent.position[1], s=5, color=color)

                    wedge = mpatches.Wedge((agent.position[0], agent.position[1]), agent.sight, dir_angle + 360 - sight_angle, dir_angle + sight_angle, ec="none")
                    sight_wedges.append(wedge)
                
                ax.add_collection(PatchCollection(sight_wedges, alpha=0.3))
                
            else:
                ax = plt.axes(projection="3d")
                ax.axes.set_xlim3d(-self.length, self.length)
                ax.axes.set_ylim3d(-self.length, self.length)
                ax.axes.set_zlim3d(-self.length, self.length)

                for index, agent in enumerate(self.agents):
                    if agent.agent_type == 0: agent.next_step(self.get_neighbours(agent, agent.sight), self.dimension)
                    positions[index] = agent.position
                    ax.quiver(agent.position[0], agent.position[1], agent.position[2], agent.speed[0], agent.speed[1], agent.speed[2], color="black")

            return ax

        fig = self.compute_figure()
        ani = animation.FuncAnimation(fig, compute_animation, frames=frames, interval=interval)
        ani.save(filename + ".gif")
        plt.close()


class DimensionError(Exception):
    pass


# ┌───────────┐ #
# │ Fonctions │ #
# └───────────┘ #

def group_generator(nb: int, position: tuple=(-25, 25), speed: tuple=(-2, 2), sight: tuple=(5, 10), field_sight: int=math.pi/2, length: int=50, dim: int=2):
    """
    nb          : nombre d'agents à générer
    position    : valeurs limites de la position (xlim ; ylim)
    speed       : valeurs limites de la vitesse (xlim ; ylim)
    sight       : portée de la vue des agents
    field_sight : champ de vision des agents
    length      : taille de l'arête du cube d'espace considéré
    dim         : dimension de l'espace dans lequel les agents évoluent
    Retourne un groupe d'agents générés aléatoirement.
    """
    rlist = lambda minimum, maximum: minimum + (maximum - minimum) * np.random.random(dim)
    agents = []
    
    for _ in range(nb):
        agent = Agent(
            position=rlist(position[0], position[1]),
            speed=np.zeros(dim),
            velocity=0,
            noise=random.random(),
            sight=random.randint(sight[0], sight[1]),
            field_sight=field_sight)
        
        velocity = 0
        while velocity == 0:
            agent.speed = rlist(speed[0], speed[1])
            velocity = norm(agent.speed)
        agent.speed /= velocity
        agent.velocity = velocity
        
        agents.append(agent)
    
    return Group(agents, length=length, dim=dim)


def norm(vect: np.array):
    """
    vect : tableau numpy
    Renvoie la norme du vecteur passé en argument.
    """
    return math.sqrt(sum(vect ** 2))


def get_colors():
    """Retourne une liste de couleur indexée sur l'angle avec la verticale ascendante."""
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


# ┌─────────┐ #
# │ Données │ #
# └─────────┘ #

COLOR_MAP = get_colors()

group_10 = group_generator(10)

group_20 = group_generator(19)
group_20.add_agent(Agent(
    np.array([0., 0.]),
    np.array([0., 0.]),
    0,
    0,
    0,
    0,
    1
))

group_40 = group_generator(40)
