"""
Vicsek stat
===========

Description
-----------
Permet de faire des statistiques à partir du modèle de Vicsek et avec différents paramètres.
"""
import numpy as np
import vicsek as vi


def op_noise(check_field: bool=False, check_wall: bool=False):
    """
    Calcule le paramètre d'alignement pour différentes valeurs de bruits et renvoie un tuple de la forme (bruit, paramètre d'alignement).

    Paramètres
    ----------
    check_field : bool, optionnel
        Prise en compte de l'angle de vue des agents.
    check_wall : bool, optionnel
        Prise en compte des murs.

    Signature
    ---------
    out : tuple
        Tuple de deux listes contenant respectivement les valeurs de bruit et du paramètres d'alignement. 
    """
    order_p = []
    noises = np.arange(0, 5.5, 0.10)
    for noise in noises:
        op_temp = 0
        # for _ in range(5):
        grp = vi.group_generator(40, position=(-1.5, 1.5), speed=(-1, 1), noise=(noise, noise), length=3.1)
        grp.run(100, check_field=check_field, check_wall=check_wall, step=0.5)
        op_temp += grp.order_parameter()

        order_p.append(op_temp)
        print()
    return list(noises), order_p


def op_density():
    """
    Calcule le paramètre d'alignement pour différentes densités et renvoie un tuple de la forme (densité, paramètre d'alignement).

    Signature
    ---------
    out : tuple
        Tuple de deux listes contenant respectivement les valeurs de densité et du paramètres d'alignement.
    """
    order_p = []
    density = []
    grp = vi.group_generator(2, position=(-0.5, 0.5), speed=(-0.5, 0.5), noise=(1.5, 1.5), length=1)

    for _ in range(100):
        for _ in range(1):
            agent = vi.agent_generator(position=(-0.5, 0.5), speed=(-0.5, 0.5), noise=(1.5, 1.5))
            grp.add_agent(agent)

        grp.run(10, check_field=False, check_wall=False, step=0.25)

        order_p.append(grp.order_parameter())
        density.append(grp.density)
        print()
    return density, order_p


def neutral_alignment():
    """
    Retourne le paramètre d'alignement en fonction de la densité sans itérer le modèle.

    Signature
    ---------
    out : tuple
        Tuple de deux listes contenant respectivement les valeurs de densité et du paramètres d'alignement. """
    def get_op():
        order_p = []
        density = []
        for numb in np.arange(5, 100, 5):
            op_temp = 0
            density_temp = 0
            for _ in range(5):
                grp = vi.group_generator(numb, position=(-10, 10), speed=(-1, 1), noise=(1.5, 1.5), length=20)
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


def stat(fct, *args, iteration: int=10):
    """
    Permet de faire des essais sur un plus grand nombre de cas et de moyenner les résultats.

    Paramètres
    ----------
    fct : builtin_function_or_method
        Fonction sur laquelle il faut faire les essais.
        Cette fonction doit renvoyer un tuple de deux listes.
    *args : optionnel
        Arguments de `fct`.
    iteration : int, optionnel
            Nobmre de tests à effectuer.
    Signature
    ---------
    out : tuple
        Tuple de deux listes correspondants aux moyennes des résultats renvoyés par `fct`.
    """
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


def predation():
    """Calcule le pourcentage de survivant dans des cas extrêmes de bruit et de sensibilité avec 500 itérations."""
    group_1 = vi.group_generator(50, noise=(0, 0), fear=(0, 0))
    for _ in range(2):
        group_1.add_agent(vi.agent_generator(speed=(-3, 3), noise=(0.25, 0.25), agent_type=1))

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

    return len(group_1.dead_agents) / 50, len(group_2.dead_agents) / 50, len(group_3.dead_agents) / 50, len(group_4.dead_agents) / 50


def predation_stat():
    """Statistiques sur 10 lancement de `predation`."""
    rslt = [-1, -1, -1, -1]
    for i in range(10):
        print(f"{i + 1}/10")
        grps = predation()
        for i in range(4):
            if rslt[i] != -1:
                rslt[i] = (rslt[i] + grps[i]) / 2
            else:
                rslt[i] = grps[i]

    print("noise | fear | % survivants")
    print("1     | 0    |", round(1 - rslt[0], 2))
    print("0     | 1    |", round(1 - rslt[1], 2))
    print("1     | 1    |", round(1 - rslt[2], 2))
    print("0     | 0    |", round(1 - rslt[3], 2))
