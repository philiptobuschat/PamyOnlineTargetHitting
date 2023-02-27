import numpy as np
import math
from math import cos, sin, sqrt
import matplotlib.pyplot as plt
import time
import numba as nb
# %%
@nb.jit(nopython=True)
def CalJerkTerm(t, m, n):
  E = math.exp(1)
  
  l1 = (-(1/(E**((n*t)/m)*(6*m**4*n**2))) \
    -E**((n*t)/m)/(6*m**4*n**2) \
    +cos((sqrt(3)*n*t)/(2*m))/(E**((n*t)/(2*m))*(6*m**4*n**2)) \
    +(E**((n*t)/(2*m))*cos((sqrt(3)*n*t)/(2*m)))/(6*m**4*n**2) \
    -sin((sqrt(3)*n*t)/(2*m))/(E**((n*t)/(2*m))*(16*sqrt(3)*m**4*n**2)) \
    +(3*sqrt(3)*sin((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(16*m**4*n**2)) \
    +(E**((n*t)/(2*m))*sin((sqrt(3)*n*t)/(2*m)))/(16*sqrt(3)*m**4*n**2) \
    -(3*sqrt(3)*E**((n*t)/(2*m))*sin((sqrt(3)*n*t)/(2*m)))/(16*m**4*n**2))

  l2 = (-(1/(E**((n*t)/m)*(6*m**5*n))) \
    +E**((n*t)/m)/(6*m**5*n) \
    -cos((sqrt(3)*n*t)/(2*m))/(E**((n*t)/(2*m))*(6*m**5*n)) \
    +(E**((n*t)/(2*m))*cos((sqrt(3)*n*t)/(2*m)))/(6*m**5*n) \
    -sin((sqrt(3)*n*t)/(2*m))/(E**((n*t)/(2*m))*(16*sqrt(3)*m**5*n)) \
    +(3*sqrt(3)*sin((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(16*m**5*n)) \
    -(E**((n*t)/(2*m))*sin((sqrt(3)*n*t)/(2*m)))/(16*sqrt(3)*m**5*n) \
    +(3*sqrt(3)*E**((n*t)/(2*m))*sin((sqrt(3)*n*t)/(2*m)))/(16*m**5*n))

  l3 = (-(1/(E**((n*t)/m)*(6*m**6))) \
    -E**((n*t)/m)/(6*m**6) \
    -cos((sqrt(3)*n*t)/(2*m))/(E**((n*t)/(2*m))*(3*m**6)) \
    -(E**((n*t)/(2*m))*cos((sqrt(3)*n*t)/(2*m)))/(3*m**6))

  p0 = (-(n**3/(E**((n*t)/m)*(6*m**3))) \
    +(E**((n*t)/m)*n**3)/(6*m**3) \
    +(n**3*cos((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(3*m**3)) \
    -(E**((n*t)/(2*m))*n**3*cos((sqrt(3)*n*t)/(2*m)))/(3*m**3))

  v0 = (n**2/(E**((n*t)/m)*(6*m**2)) \
    +(E**((n*t)/m)*n**2)/(6*m**2) \
    -(n**2*cos((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(6*m**2)) \
    -(E**((n*t)/(2*m))*n**2*cos((sqrt(3)*n*t)/(2*m)))/(6*m**2 ) \
    -(n**2*sin((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(16*sqrt(3)*m**2)) \
    +(3*sqrt(3)*n**2*sin((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(16*m**2)) \
    +(E**((n*t)/(2*m))*n**2*sin((sqrt(3)*n*t)/(2*m)))/(16*sqrt(3)*m**2) \
    -(3*sqrt(3)*E**((n*t)/(2*m))*n**2*sin((sqrt(3)*n*t)/(2*m)))/(16*m**2))

  a0 = (-(n/(E**((n*t)/m)*(6*m))) \
    +(E**((n*t)/m)*n)/(6*m) \
    -(n*cos((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(6*m)) \
    +(E**((n*t)/(2*m))*n*cos((sqrt(3)*n*t)/(2*m)))/(6*m) \
    +(n*sin((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(16*sqrt(3)*m)) \
    -(3*sqrt(3)*n*sin((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(16*m)) \
    +(E**((n*t)/(2*m))*n*sin((sqrt(3)*n*t)/(2*m)))/(16*sqrt(3)*m) \
    -(3*sqrt(3)*E**((n*t)/(2*m))*n*sin((sqrt(3)*n*t)/(2*m)))/(16*m))

  vec = [l1, l2, l3, p0, v0, a0]
  
  return vec

@nb.jit(nopython=True)
def CalAccelerationTerm(t, m, n):
  E = math.exp(1)
  a0 = (1/6/E**((n*t)/m) \
    +(1/6)*E**((n*t)/m) \
    +((1/3)*cos((sqrt(3)*n*t)/(2*m)))/E**((n*t)/(2*m))\
    +(1/3)*E**((n*t)/(2*m))*cos((sqrt(3)*n*t)/(2*m)))

  v0 = (-(n/(E**((n*t)/m)*(6*m))) + (E**((n*t)/m)*n)/(6*m) \
    - (n*cos((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(6*m)) \
    + (E**((n*t)/(2*m))*n*cos((sqrt(3)*n*t)/(2*m)))/(6*m) \
    - (n*sin((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(8*sqrt(3)*m)) \
    - (sqrt(3)*n*sin((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(8*m)) \
    - (E**((n*t)/(2*m))*n*sin((sqrt(3)*n*t)/(2*m)))/(8*sqrt(3)*m) \
    - (sqrt(3)*E**((n*t)/(2*m))*n*sin((sqrt(3)*n*t)/(2*m)))/(8*m))

  p0 = (n**2/(E**((n*t)/m)*(6*m**2)) + (E**((n*t)/m)*n**2)/(6*m**2) - \
    (n**2*cos((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(6*m**2)) - \
    (E**((n*t)/(2*m))*n**2*cos((sqrt(3)*n*t)/(2*m)))/(6*m**2) + \
    (n**2*sin((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(2*sqrt(3)*m**2)) - \
    (E**((n*t)/(2*m))*n**2*sin((sqrt(3)*n*t)/(2*m)))/(2*sqrt(3)*m**2)) 
  
  l1= (1/(E**((n*t)/m)*(6*m**3*n**3)) \
    - E**((n*t)/m)/(6*m**3*n**3) \
    - cos((sqrt(3)*n*t)/(2*m))/(E**((n*t)/(2*m))*(3*m**3*n**3)) \
    + (E**((n*t)/(2*m))*cos((sqrt(3)*n*t)/(2*m)))/(3*m**3*n**3))

  l2 = (1/(E**((n*t)/m)*(6*m**4*n**2)) \
    + E**((n*t)/m)/(6*m**4*n**2) \
    - cos((sqrt(3)*n*t)/(2*m))/(E**((n*t)/(2*m))*(6*m**4*n**2)) \
    - (E**((n*t)/(2*m))*cos((sqrt(3)*n*t)/(2*m)))/(6*m**4*n**2) \
    - sin((sqrt(3)*n*t)/(2*m))/(E**((n*t)/(2*m))*(8*sqrt(3)*m**4*n**2)) \
    - (sqrt(3)*sin((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(8*m**4*n**2)) \
    + (E**((n*t)/(2*m))*sin((sqrt(3)*n*t)/(2*m)))/(8*sqrt(3)*m**4*n**2) \
    + (sqrt(3)*E**((n*t)/(2*m))*sin((sqrt(3)*n*t)/(2*m)))/(8*m**4*n**2))


  l3 = (1/(E**((n*t)/m)*(6*m**5*n)) \
    - E**((n*t)/m)/(6*m**5*n) \
    + cos((sqrt(3)*n*t)/(2*m))/(E**((n*t)/(2*m))*(6*m**5*n)) \
    - (E**((n*t)/(2*m))*cos((sqrt(3)*n*t)/(2*m)))/(6*m**5*n) \
    - sin((sqrt(3)*n*t)/(2*m))/(E**((n*t)/(2*m))*(2*sqrt(3)*m**5*n)) \
    - (E**((n*t)/(2*m))*sin((sqrt(3)*n*t)/(2*m)))/(2*sqrt(3)*m**5*n))

  vec = [l1, l2, l3, p0, v0, a0]
  
  return vec

@nb.jit(nopython=True)
def CalVelocityTerm(t, m, n):
  E = math.exp(1)
  
  v0 = (1/6/E**((n*t)/m) \
    +(1/6)*E**((n*t)/m) \
    +((1/3)*cos((sqrt(3)*n*t)/(2*m)))/E**((n*t)/(2*m)) \
    +(1/3)*E**((n*t)/(2*m))*cos((sqrt(3)*n*t)/(2*m)))

  l2 = (-(1/(E**((n*t)/m)*(6*m**3*n**3))) \
    +E**((n*t)/m)/(6*m**3*n**3) \
    +cos((sqrt(3)*n*t)/(2*m))/(E**((n*t)/(2*m))*(3*m**3*n**3)) \
    -(E**((n*t)/(2*m))*cos((sqrt(3)*n*t)/(2*m)))/(3*m**3*n**3))

  l1 = (-(1/(E**((n*t)/m)*(6*m**2*n**4))) \
    -E**((n*t)/m)/(6*m**2*n**4) \
    +cos((sqrt(3)*n*t)/(2*m))/(E**((n*t)/(2*m))*(6*m**2*n**4)) \
    +(E**((n*t)/(2*m))*cos((sqrt(3)*n*t)/(2*m)))/(6*m**2*n**4) \
    -sin((sqrt(3)*n*t)/(2*m))/(E**((n*t)/(2*m))*(2*sqrt(3)*m**2*n**4)) \
    +(E**((n*t)/(2*m))*sin((sqrt(3)*n*t)/(2*m)))/(2*sqrt(3)*m**2*n**4))

  l3 = (-(1/(E**((n*t)/m)*(6*m**4*n**2))) \
    -E**((n*t)/m)/(6*m**4*n**2) \
    +cos((sqrt(3)*n*t)/(2*m))/(E**((n*t)/(2*m))*(6*m**4*n**2)) \
    +(E**((n*t)/(2*m))*cos((sqrt(3)*n*t)/(2*m)))/(6*m**4*n**2) \
    +sin((sqrt(3)*n*t)/(2*m))/(E**((n*t)/(2*m))*(2*sqrt(3)*m**4*n**2)) \
    -(E**((n*t)/(2*m))*sin((sqrt(3)*n*t)/(2*m)))/(2*sqrt(3)*m**4*n**2))

  a0 = (-(m/(E**((n*t)/m)*(6*n))) \
    +(E**((n*t)/m)*m)/(6*n) \
    -(m*cos((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(6*n)) \
    +(E**((n*t)/(2*m))*m*cos((sqrt(3)*n*t)/(2*m)))/(6*n) \
    +(m*sin((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(2*sqrt(3)*n)) \
    +(E**((n*t)/(2*m))*m*sin((sqrt(3)*n*t)/(2*m)))/(2*sqrt(3)*n))

  p0 = (-(n/(E**((n*t)/m)*(6*m))) \
    +(E**((n*t)/m)*n)/(6*m) \
    -(n*cos((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(6*m)) \
    +(E**((n*t)/(2*m))*n*cos((sqrt(3)*n*t)/(2*m)))/(6*m) \
    -(n*sin((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(2*sqrt(3)*m)) \
    -(E**((n*t)/(2*m))*n*sin((sqrt(3)*n*t)/(2*m)))/(2*sqrt(3)*m))

  vec = [l1, l2, l3, p0, v0, a0]
  
  return vec

@nb.jit(nopython=True)
def CalPositionTerm(t, m, n):
  E = math.exp(1)
  
  a0 = (m**2/(E**((n*t)/m)*(6*n**2)) \
    +(E**((n*t)/(2*m))*m**2*sin((sqrt(3)*n*t)/(2*m)))/(2*sqrt(3)*n**2) \
    +(E**((n*t)/m)*m**2)/(6*n**2) \
    -(E**((n*t)/(2*m))*m**2*cos((sqrt(3)*n*t)/(2*m)))/(6*n**2) \
    -(m**2*cos((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(6*n**2)) \
    -(m**2*sin((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(2*sqrt(3)*n**2)))

  l3 = (-(cos((sqrt(3)*n*t)/(2*m))/(E**((n*t)/(2*m))*(3*m**3*n**3))) \
    +(E**((n*t)/(2*m))*cos((sqrt(3)*n*t)/(2*m)))/(3*m**3*n**3) \
    -E**((n*t)/m)/(6*m**3*n**3)+1/(E**((n*t)/m)*(6*m**3*n**3)))

  l2 = (-(cos((sqrt(3)*n*t)/(2*m))/(E**((n*t)/(2*m))*(6*m**2*n**4))) \
    +sin((sqrt(3)*n*t)/(2*m))/(E**((n*t)/(2*m))*(2*sqrt(3)*m**2*n**4)) \
    +E**((n*t)/m)/(6*m**2*n**4)-(E**((n*t)/(2*m))*cos((sqrt(3)*n*t)/(2*m)))/(6*m**2*n**4) \
    -(E**((n*t)/(2*m))*sin((sqrt(3)*n*t)/(2*m)))/(2*sqrt(3)*m**2*n**4) \
    +1/(E**((n*t)/m)*(6*m**2*n**4)))

  l1 = (cos((sqrt(3)*n*t)/(2*m))/(E**((n*t)/(2*m))*(6*m*n**5)) \
    +sin((sqrt(3)*n*t)/(2*m))/(E**((n*t)/(2*m))*(2*sqrt(3)*m*n**5)) \
    +(E**((n*t)/(2*m))*sin((sqrt(3)*n*t)/(2*m)))/(2*sqrt(3)*m*n**5) \
    -E**((n*t)/m)/(6*m*n**5)-(E**((n*t)/(2*m))*cos((sqrt(3)*n*t)/(2*m)))/(6*m*n**5) \
    +1/(E**((n*t)/m)*(6*m*n**5)))

  v0 = (-(m/(E**((n*t)/m)*(6*n))) \
    +(E**((n*t)/(2*m))*m*cos((sqrt(3)*n*t)/(2*m)))/(6*n) \
    +(m*sin((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(2*sqrt(3)*n)) \
    +(E**((n*t)/(2*m))*m*sin((sqrt(3)*n*t)/(2*m)))/(2*sqrt(3)*n) \
    +(E**((n*t)/m)*m)/(6*n)-(m*cos((sqrt(3)*n*t)/(2*m)))/(E**((n*t)/(2*m))*(6*n)))

  p0 = ((1/3)*E**((n*t)/(2*m))*cos((sqrt(3)*n*t)/(2*m)) \
    +(1/6)*E**((n*t)/m) \
    +cos((sqrt(3)*n*t)/(2*m))/(E**((n*t)/(2*m))*3) \
    +1/(6*E**((n*t)/m)))

  vec = [l1, l2, l3, p0, v0, a0]
  
  return vec

def CalParameter(x_0, x_f, t_0, t_f, m, n):
  term_inital = [CalPositionTerm(t_0, m, n),
                 CalVelocityTerm(t_0, m, n),
                 CalAccelerationTerm(t_0, m, n)]
  
  term_final = [CalPositionTerm(t_f, m, n),
                CalVelocityTerm(t_f, m, n),
                CalAccelerationTerm(t_f, m, n)]

  M = np.vstack((term_final, term_inital))
  vec = np.vstack((x_f.reshape(-1, 1), x_0.reshape(-1, 1)))
  parameter = np.dot(np.linalg.inv(M), vec).reshape(1, -1)

  return parameter

def GetPositionTerm(t, m, n):
  x = np.zeros((len(m), 6))
  for i in range(len(m)):
    x[i, :] = np.array(CalPositionTerm(t, m[i], n[i]))
  return x

def GetVelocityTerm(t, m, n):
  x = np.zeros((len(m), 6))
  for i in range(len(m)):
    x[i, :] = np.array(CalVelocityTerm(t, m[i], n[i]))
  return x

def GetAccelerationTerm(t, m, n):
  x = np.zeros((len(m), 6))
  for i in range(len(m)):
    x[i, :] = np.array(CalAccelerationTerm(t, m[i], n[i]))
  return x

def CalPath(x_0, x_f, t_0, t_f, step, m, n):
  
  nr_dof    = x_0.shape[0]
  nr_point  = int( np.round((t_f-t_0)/step) ) + 1
  parameter = np.zeros((nr_dof, 6))
  
  p = np.zeros((nr_dof, nr_point))
  v = np.zeros((nr_dof, nr_point))
  a = np.zeros((nr_dof, nr_point))

  # j = np.zeros( (nr_dof, nr_point) )

  for i_dof in range( nr_dof ):
    parameter[i_dof, :] = CalParameter(x_0[i_dof, :], x_f[i_dof, :], 0, t_f-t_0, m[i_dof], n[i_dof])

  for i_point in range(nr_point):
    p[:, i_point] = np.sum(parameter*GetPositionTerm(i_point*step, m, n), axis=1)
    v[:, i_point] = np.sum(parameter*GetVelocityTerm(i_point*step, m, n), axis=1)
    a[:, i_point] = np.sum(parameter*GetAccelerationTerm(i_point*step, m, n), axis=1)

  return (p, v, a)

def GetSteady(x, t_0, t_f, step):
  nr_point = int((t_f-t_0)/step) + 1
  p = np.ones(nr_point) * x[0]
  v = np.ones(nr_point) * x[1]
  a = np.ones(nr_point) * x[2]
  j = np.ones(nr_point) * 0
  return (p, v, a, j)

def GetPath(x_temp, v_temp, a_temp, t_temp, m, n, step):
  '''
  i_dof * [position, velocity, acceleration]
  '''
  x_0_list = np.hstack((x_temp[:, 0].reshape(-1, 1), v_temp[:, 0].reshape(-1, 1), a_temp[:, 0].reshape(-1, 1)))
  x_f_list = np.hstack((x_temp[:, 1].reshape(-1, 1), v_temp[:, 1].reshape(-1, 1), a_temp[:, 1].reshape(-1, 1)))
  [p, v, a] = CalPath(x_0_list, x_f_list, t_temp[0], t_temp[1], step, m, n)

  return (p, v, a)

def PathPlanning(x_list, v_list, a_list, t_list, step, m_list, n_list):
  nr_dof = x_list.shape[0]
  nr_point = x_list.shape[1]
  t_stamp = np.linspace(t_list[0], t_list[-1], round((t_list[-1]-t_list[0])/step)+1, endpoint=True)

  position     = np.zeros((nr_dof, len(t_stamp)))
  velocity     = np.zeros((nr_dof, len(t_stamp)))
  acceleration = np.zeros((nr_dof, len(t_stamp)))
  # jerk         = np.zeros((nr_dof, len(t_stamp)))
  
  position[:, 0] = x_list[:, 0]
  velocity[:, 0] = v_list[:, 0]
  acceleration[:, 0] = a_list[:, 0]
  # jerk[:, 0] = np.zeros(nr_dof)

  idx_1 = 0
  idx_2 = 1
  
  for i_point in range(1, nr_point):
    
    idx_1 = idx_2
    idx_2 = int(round((t_list[i_point]-t_list[i_point-1])/step)) + idx_1
    
    [p, v, a] = GetPath(x_list[:, i_point-1:i_point+1], 
                        v_list[:, i_point-1:i_point+1], 
                        a_list[:, i_point-1:i_point+1], 
                        t_list[i_point-1:i_point+1], 
                        m_list[:, i_point-1], 
                        n_list[:, i_point-1], step)

    position[:, idx_1:idx_2] = p[:, 1:]
    velocity[:, idx_1:idx_2] = v[:, 1:]
    acceleration[:, idx_1:idx_2] = a[:, 1:]
    # jerk[:, idx_1:idx_2] = j[:, 1:]
  
  return (position, velocity, acceleration, t_stamp)

if __name__ == '__main__':
  # step size 
  step = 0.01
  # penalty for all DoFs
  m_list = [[1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]]
  m_list = np.array(m_list)
  
  n_list = [[0.1, 0.1, 0.1],
            [0.1, 5.0, 0.1],
            [0.1, 10.0, 0.1]]
  n_list = np.array(n_list)
  '''
  position/velocity/acceleration/duration for all dofs at different time points
  dimension = nr_dof * nr_point 
  '''
  x_list = [[0.0, math.pi/4, 0.0, 0.0],
            [0.0, math.pi/4, 0.0, 0.0],
            [0.0, math.pi/4, 0.0, 0.0]]
  x_list = np.array(x_list)
  v_list = [[0.0, 5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0, 0.0]]
  v_list = np.array(v_list)
  a_list = [[0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]]
  a_list = np.array(a_list)
  t_list = [0.0, 1.0, 2.0, 2.2]
  t_list = np.array(t_list)
  [position, velocity, acceleration, jerk, t_stamp] = PathPlanning(x_list, v_list, a_list, t_list, step, m_list, n_list)
  
  nr_dof = x_list.shape[0]
  legend_position = 'lower right'
  
  fig = plt.figure(figsize=(16, 16))
  ax_position = fig.add_subplot(411)
  plt.xlabel(r'Time $t$ in s')
  plt.ylabel(r'Angle $\theta$ in degree')
  line = []
  for i in range( nr_dof ):
      line_temp, = ax_position.plot(t_stamp, position[i, :] * 180 / math.pi, linewidth=2, label=r'Pos. dof {}'.format(i+1))
      line.append( line_temp )
  plt.legend(handles=line, loc=legend_position, shadow=True)
      
  ax_velocity = fig.add_subplot(412)
  plt.xlabel(r'Time $t$ in s')
  plt.ylabel(r'Velocity $v$ in rad/s')
  line = []
  for i in range( nr_dof ):
      line_temp, = ax_velocity.plot(t_stamp, velocity[i, :], linewidth=2, label=r'Vel. dof {}'.format(i+1))
      line.append( line_temp )
  plt.legend(handles=line, loc=legend_position, shadow=True)
  
  ax_acceleration = fig.add_subplot(413)
  plt.xlabel(r'Time $t$ in s')
  plt.ylabel(r'Acceleration $a$ in rad/$s^2$')
  line = []
  for i in range( nr_dof ):
      line_temp, = ax_acceleration.plot(t_stamp, acceleration[i, :], linewidth=2, label=r'Acc. dof {}'.format(i+1))
      line.append( line_temp )
  plt.legend(handles=line, loc=legend_position, shadow=True)
  
  ax_jerk = fig.add_subplot(414)
  plt.xlabel(r'Time $t$ in s')
  plt.ylabel(r'Jerk $j$ in rad/$s^3$')
  line = []
  for i in range( nr_dof ):
      line_temp, = ax_jerk.plot(t_stamp, jerk[i, :], linewidth=2, label=r'Jerk. dof {}'.format(i+1))
      line.append( line_temp )
  plt.legend(handles=line, loc=legend_position, shadow=True)
  
  plt.show()
    
    



