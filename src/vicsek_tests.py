import matplotlib.pyplot as plt
import numpy as np
import vicsek as vi


def op_noise(check_field: bool=False, check_wall: bool=False):
    """Calcule le paramètre d'alignement pour différentes valeurs de bruits et renvoie un tuple de la forme (bruit, paramètre d'alignement)."""
    order_p = []
    noises = np.arange(0, 5.1, 0.1)
    for noise in noises:
        op_temp = 0
        for _ in range(2):
            grp = vi.group_generator(40, position=(-1.5, 1.5), speed=(-1, 1), noise=(noise, noise), length=3.1)
            grp.run(50, check_field=check_field, check_wall=check_wall, dt=0.25)
            op_temp += grp.order_parameter()
        
        order_p.append(op_temp / 5)
        print()
    return list(noises), order_p


def op_density():
    """Calcule le paramètre d'alignement pour différentes densités et renvoie un tuple de la forme (densité, paramètre d'alignement)."""
    order_p = []
    density = [] 
    for nb in np.arange(25, 100, 5):
        op_temp = 0
        density_temp = 0
        for _ in range(5):
            grp = vi.group_generator(nb, position=(-10, 10), speed=(-1, 1), noise=(1.5, 1.5), length=20)
            grp.run(10, check_field=False, check_wall=False, dt=0.5)
            op_temp += grp.order_parameter()
            density_temp += grp.density

        order_p.append(op_temp / 5)
        density.append(density_temp / 5)
        print()
    return density, order_p


def neutral_alignment():
    """Retourne le paramètre d'alignement en fonction de la densité sans itérer le modèle."""
    def get_op():
        order_p = []
        density = [] 
        for nb in np.arange(5, 100, 5):
            op_temp = 0
            density_temp = 0
            for _ in range(5):
                grp = vi.group_generator(nb, position=(-10, 10), speed=(-1, 1), noise=(1.5, 1.5), length=20)
                op_temp += grp.order_parameter()
                density_temp += grp.density

            order_p.append(op_temp / 5)
            density.append(density_temp / 5)
        return density, order_p

    runs = np.array([get_op() for _ in range(50)])
    density = runs[:, 0]
    order_p = runs[:, 1]

    avg_op = []
    for index in range(19):
        avg_op.append(sum(order_p[:, index]) / 50)

    return density[0], avg_op


def stat(fct, iteration: int=10, *args):
    runs = []
    for i in range(iteration):
        print(f"{i+1} / {iteration}") 
        runs.append(fct(*args))
    
    runs = np.array(runs)
    x_axis = runs[:, 0][0]
    y_axis = runs[:, 1]

    avg_y = []
    for index in range(len(x_axis)):
        avg_y.append(sum(y_axis[:, index]) / len(y_axis[:, index]))

    return x_axis, avg_y


def test():
    group_1 = vi.group_generator(50, noise=(0, 0), fear=(0, 0))
    for _ in range(2):
        group_1.add_agent(vi.agent_generator(speed=(-3, 3), noise=0.25, agent_type=1))

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
