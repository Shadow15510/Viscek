import matplotlib.pyplot as plt
import numpy as np
import vicsek as vi


def op_noise():
    order_p = []
    noises = np.arange(0, 5.1, 0.1)
    for noise in noises:
        op_temp = 0
        for _ in range(5):
            grp = vi.group_generator(40, position=(-1.5, 1.5), speed=(-1, 1), noise=(noise, noise), length=3.1)
            grp.run(200, check_field=False, check_wall=False, dt=0.25)
            op_temp += grp.order_parameter()
        
        order_p.append(op_temp / 5)
        print()
    return list(noises), order_p


def op_density():
    order_p = []
    density = [] 
    for nb in np.arange(5, 100, 5):
        op_temp = 0
        density_temp = 0
        for _ in range(5):
            grp = vi.group_generator(nb, position=(-1.5, 1.5), speed=(-1, 1), noise=(1.5, 1.5), length=3)
            grp.run(200, check_field=False, check_wall=False, dt=0.25)
            op_temp += grp.order_parameter()
            density_temp += grp.density

        order_p.append(op_temp / 5)
        density.append(density_temp / 5)
        print()
    return density, order_p


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
