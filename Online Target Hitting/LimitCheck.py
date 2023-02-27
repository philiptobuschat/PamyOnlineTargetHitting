import numpy as np
# %% add the limits
"""
max_pressure_ago = [22000, 25000, 22000, 22000]
max_pressure_ant = [22000, 23000, 22000, 22000]

min_pressure_ago = [13000, 13500, 10000, 8000]
min_pressure_ant = [13000, 14500, 10000, 8000]
"""

"""
anchor = [17500 18500 16000 15000]
"""
limit_max = np.array([4500, 6500, 6000, 7000])
limit_min = np.array([-4500, -5000, -6000, -7000])

def LimitCheck( u, dof, up_margin=0, low_margin=0 ):

    u[ u>limit_max[dof]+up_margin ] = limit_max[dof] + up_margin
    u[ u<limit_min[dof]+low_margin ] = limit_min[dof] + low_margin

    return u

if __name__ == '__main__':
    u = np.array([[7000], [-8000], [9000]])
    print(LimitCheck(u, 0))