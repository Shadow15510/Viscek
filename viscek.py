# ┌──────────────────────────────────┐ #
# │          Viscek — 1.4.0          │ #
# │ Alexis Peyroutet & Antoine Royer │ #
# │   GNU General Public Licence v3  │ #
# └──────────────────────────────────┘ #

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random


__name__ = "Viscek"
__version__ = "1.4.0"


# ┌─────────┐ #
# │ Classes │ #
# └─────────┘ #

class Agent:
    """Simule un agent avec sa position, sa vitesse, et le bruit associé qui traduit sa tendance naturelle à suivre le groupe ou pas."""

    def __init__(self, position: np.array, speed: np.array, velocity: int, noise: int, sight: int):
        """position : vecteur position (sous forme d'un tableau numpy de trois entiers)
        speed    : direction du vecteur vitesse (tableau de trois entiers)
        velocity : norme de la vitesse
        noise    : taux de bruit qui traduit le comportement de l'agent par rapport aux autres (0 suit les autres, 1 ne suit pas les autres)
        sight    : distance à laquelle l'agent voit les autres"""
        self.position = position[:]
        self.speed = speed[:]
        self.velocity = velocity
        self.noise = noise
        self.sight = sight

    def __str__(self):
        """affiche l'agent"""
        return f"Agent(\n\tposition={list(self.position)},\n\tspeed={list(self.speed)},\n\tvelocity={self.velocity},\n\tnoise={self.noise},\n\tsight={self.sight}\n)"
    
    def __sub__(self, agent):
        """agent : un autre agent
        Retourne la distance entre les deux agents."""
        return math.sqrt(sum((self.position - agent.position) ** 2))

    def __eq__(self, agent):
        """agent : un autre agent
        Retourne True si les agents ont les mêmes positions, False sinon."""
        return list(self.position) == list(agent.position)
    
    __repr__ = __str__


class Group:
    """Simule un groupe d'agents, permet de le faire évoluer et de l'afficher"""
    
    def __init__(self, agents: list, dim: int=2):
        """agents : liste des agents du groupe
        dim    : dimension de l'espace considéré (2 ou 3)"""
        self.agents = agents
        self.nb_agents = len(agents)
        
        if not dim in (2, 3):
            raise DimensionError("dim must be 2 or 3")
        self.dimension = dim

        for agent in agents:
            if len(agent.position) != dim or len(agent.speed) != dim:
                raise DimensionError("dimension of agent didn't match")
    
    def get_neighbours(self, targeted_agent: Agent, dmin: int):
        """targeted_agent : agent servant de référence pour le calcul de distance
        dmin           : distance minimale à considérer
        Retourne une liste d'agents appartenant au groupe et étant à une distance inférieure ou égale à dmin."""
        return [agent for agent in self.agents if (targeted_agent - agent) <= dmin]

    def get_agents_parameters(self):
        """Retourne un tuple de tableaux numpy contenant les positions et les vitesses de tous les agents du groupe."""
        positions = np.zeros((self.nb_agents, self.dimension))
        speeds = np.zeros((self.nb_agents, self.dimension))

        for index, agent in enumerate(self.agents):
            positions[index] = agent.position
            speeds[index] = agent.velocity * agent.speed

        return positions, speeds

    def next_step(self):
        """Fait évoluer le groupe d'agents d'un pas temporel"""
        for agent in self.agents:
            agent.position += agent.velocity * 0.5 * agent.speed
            agent.speed = average_speed(self.get_neighbours(agent, agent.sight), self.dimension) + (agent.noise * np.random.randint(0, math.floor(agent.velocity) + 1, self.dimension))

    def compute_figure(self):
        """Génère une figure matplotlib avec le groupe d'agents sous forme d'un nuage de points en deux ou trois dimensions avec les vecteurs vitesses."""
        positions, speeds = self.get_agents_parameters()

        if self.dimension == 2:
            ax = plt.axes()
            ax.scatter(positions[:, 0], positions[:, 1], s=5, c="black")
            for agent_index in range(self.nb_agents):
                ax.quiver(positions[agent_index, 0], positions[agent_index, 1], speeds[agent_index, 0], speeds[agent_index, 1], color="black", width=0.002, headwidth=0, headaxislength=0, headlength=0)
            ax.axes.set_xlim(-50, 50)
            ax.axes.set_ylim(-50, 50)
        else:
            ax = plt.axes(projection="3d")
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=5, c="black")
            for agent_index in range(self.nb_agents):
                ax.quiver(positions[agent_index, 0], positions[agent_index, 1], positions[agent_index, 2], speeds[agent_index, 0], speeds[agent_index, 1], speeds[agent_index, 2], color="black")
            ax.axes.set_xlim3d(-20, 20)
            ax.axes.set_ylim3d(-20, 20)
            ax.axes.set_zlim3d(-20, 20)

        return ax
        
    def show(self):
        """Affiche le groupe d'agent"""
        self.compute_figure()
        plt.show()

    def run(self, steps: int=20):
        """steps : nombre d'image voulues
        Génère une animation."""
        def animation_function(*args):
            self.next_step()
            return self.compute_figure()

        fig = plt.figure()
        ani = animation.FuncAnimation(fig, animation_function, init_func=self.compute_figure, frames=steps, interval=100)
        ani.save("viscek.gif")


class DimensionError(Exception):
    pass


# ┌───────────┐ #
# │ Fonctions │ #
# └───────────┘ #

def group_generator(nb: int, dim: int=2):
    """nb: nombre d'agents à générer
    Retourne un groupe d'agents."""

    rlist = lambda minimum, maximum: (minimum + (maximum - minimum) * np.random.random(dim))
    agents = [Agent(rlist(-25, 25), np.zeros(dim), 0, random.random(), random.randint(1, 5)) for _ in range(nb)]
    for agent in agents:
        velocity = 0
        while velocity == 0:
            agent.speed = rlist(-2, 2)
            velocity = math.sqrt(sum(agent.speed ** 2))

        agent.speed /= velocity
        agent.velocity = velocity
  
    return Group(agents, dim)


def average_speed(agents: list, dim: int=2):
    """agents : liste d'agents
    dim    : dimension de l'espace considéré
    Renvoie un vecteur normalisé qui correspond à la direction moyenne des agents passés en entrée."""
    speed = np.zeros((dim))
    for agent in agents:
        speed += agent.speed
    return (1 / math.sqrt(sum(speed ** 2))) * speed


# ┌──────┐ #
# │ Data │ #
# └──────┘ #
group_20 = group_generator(20, 2)
group_40 = group_generator(40, 2)