# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 16:56:59 2024

@author: iioge
"""

#%%
import numpy as np
import random 
import copy
from numpy.random import choice
import matplotlib.pyplot as plt
import math

#create a matrix m by n and fill with x vehicles
def generate_matrix_positions(m, n, wp):
    pos = []
    for i in range(0, m):
        for j in range(0, n):
          if [i,j] not in wp:
              gen_position = [i, j] 
              pos.append(gen_position)

    return pos

#m - rows, n - columns - v - vehicles 
#e.g vehicles - 100, m - 50, n - 50 
# rows to fill => 2*(vehicles/columns)
def create_matrix(m, n, v, wp):
    matrixA = np.zeros((m,n))
    special_index = int(n/2)
    matrix_positions = generate_matrix_positions(math.floor(2*(v/n)),n, wp)
    
    random_places = random.sample(range(0, len(matrix_positions)), v)

    for position in range(0, len(random_places)):
        pos = matrix_positions[random_places[position]]
        # matrixA[pos[0], pos[1]] = random.randint(1,4)
        matrixA[pos[0], pos[1]] = random.choices([1,2,3,4], weights=[0.4,0.4,0.1,0.1])[0]
    
    for w in wp:
      matrixA[w[0], w[1]] = 5
    
    mid_positions = np.array([-1]*m)
    return matrixA


def weighting(matrix, i, j, m, n):
    total = 0.0
    
    if i==0 and j+2 < n and i+2 <m:
        total = (matrix[i, j+2] + matrix[i+1, j+2] + matrix[i+2, j+2] + matrix[i+2, j+1] + matrix[i+2, j]) /5
    elif j == n-1 and i+2 < m:
        total = (matrix[i+2, j] + matrix[i+2, j-1] + matrix[i+2, j-2] + matrix[i+1, j-2] + matrix[i, j-2])/5
    elif i != 0 and j != n-1 and i+2 < m and j+2 < n:
        total = (matrix[i, j+2] + matrix[i+1, j+2] + matrix[i+2, j+2] + matrix[i+2, j+1] + matrix[i+2, j] + matrix[i+2, j-1] + matrix[i+2, j-2] + matrix[i+1, j-2] + matrix[i, j-2])/9
    
    total = total * 10

    
    prob = 0.0
    
    if total < 4:
        prob = 0.8
    elif total >= 4 and total <=6:
        prob = 0.55
    elif total > 6:
        prob = 0.3
    
    return prob

#determine if it is boundary 
#pos - position to move to 
#ep - exit point
#layers - layers of boundaries
def in_boundary(pos, ep, layers):
  # print('--exit: ', ep)
  # print('--pos: ', pos)
  i = ep[0]
  j = ep[1]

  inBoundary = False;

  if pos == ep:
    inBoundary = True

  for x in range(1, layers+1):
    boundaries = [[i,j-x], [i-x, j-x], [i-x,j], [i-x, j+x], [i, j+x], [i+x, j+x], [i+x, j], [i+1, j-x]]
    if pos in boundaries:
      inBoundary = True
      break
      
  return inBoundary


#exit_point_type
#KR - corner-right
#KL - corner-left
#C - center
#B - Base
def rentry(exit_point_type, m , n ):
  #ls - rand(m-1),0, rs - rand(m-1),n-1, top - 0, rand(n-1)
  if exit_point_type == 'C':
    choices = ['ls', 'rs', 'top']
    side = choice(choices, 1, p=[0.4, 0.4, 0.2])
  
  elif exit_point_type == 'KR':
    choices = ['ls', 'top']
    side = choice(choices, 1, p=[0.35, 0.35, 0.3])
  
  elif exit_point_type == 'KL':
    choices = ['rs', 'top']
    side = choice(choices, 1, p=[0.35, 0.35, 0.3])
  
  elif exit_point_type == 'B':
    side = 'top'

  return side

def create_divisions(total_div, total_rows):
  div = math.floor(total_rows/total_div)
  results = []

  start = 0

  for i in range(total_div):
    start +=div
    results.append(start)

  return results

def direction_to_go(current_pos, wpt, matrix):
  i = current_pos[0]
  j = current_pos[1]

  result = [0,0]

  min = 1000

  way_points = []
  for wpx in wpt:
    if wpx[0] > i:
      way_points.append(wpx)
    

  neighbors = [[i+1, j-1], [i+1,j+1], [i+1, j+1]]

  for nx in neighbors:
    for wp in way_points:

      inBoundary = in_boundary(nx, way_points[-1],1)

      if inBoundary == False:
        if nx[0] >= 0 and nx[1] >= 0 and nx[0] < matrix.shape[0] and nx[1] < matrix.shape[1]:
          if matrix[nx[0], nx[1]] == 0:
            p1 = np.array((nx[0], nx[1]))
            p2 = np.array((wp[0], wp[1]))

            temp = p1 - p2

            dist = np.sqrt(np.dot(temp.T, temp))

            if dist < min:
              min = dist
              result = nx

  return result

def translate_matrix(matrix, way_points, velocities, use_vel, exit_point_type='B', layers=2):
  m = matrix.shape[0]
  n = matrix.shape[1]
    
  matrix_copy = copy.deepcopy(matrix)
  explored_sets = []
  dist = 0

  for i in range(0,m):
    for j in range(0,n):

      val = matrix_copy[i,j]
      if (val ==1 or val == 2 or val ==3) and [i,j] not in explored_sets:
        at_boundary = in_boundary([i,j], way_points[-1],layers)

        way_togo = direction_to_go([i,j], way_points,matrix_copy)
        move_prob = weighting(matrix, i, j, m, n)

        if at_boundary == False and way_togo != [0,0]:

            if use_vel == True:
              tomove_vel = 0
              
              for vel in enumerate(velocities):
                if way_togo[0] < vel[1]:
                  tomove_vel = len(velocities) - (vel[0] + 1) + 1
                  break

              if tomove_vel > 1:
                for dest in range(tomove_vel, 0, -1):
                  dest = dest - 1
                  if way_togo[0] + dest < m and matrix_copy[way_togo[0]+dest, way_togo[1]] == 0:
                    way_togo[0] = way_togo[0]+dest
                    break

            if np.random.rand() < move_prob:
              matrix_copy[i,j] = 0
              matrix_copy[way_togo[0], way_togo[1]] = val
              dist +=1
              explored_sets.append(way_togo)

        else:
          #re-enter
          reentry_side = rentry(exit_point_type, m, n)

          if reentry_side == 'top':
            for a in range(0,n):
              r = random.randint(0, n-1)
              if matrix_copy[0,r] == 0:
                matrix_copy[i,j] = 0
                matrix_copy[0,r] = val
                break

          elif reentry_side == 'ls':
            for b in range(0,m):
              p = random.randint(0, m-1)
              if matrix_copy[p,0] == 0:
                matrix_copy[i,j] = 0
                matrix_copy[p,0] = val
                break
          
          elif reentry_side == 'rs':
            for c in range(0,m):
              q = random.randint(0, m-1)
              if matrix_copy[q, n-1] == 0:
                matrix_copy[i,j] = 0
                matrix_copy[q,n-1] = val
                break
          
        
  return matrix_copy, dist

#%%

def run(vehicles=20, time=400, rows=42, columns=14):
    result = []
    distance = []

    #R1
    way_points =[[14,10], [14,18], [20,6], [20,14], [20,22],
                 [26,2], [26,10], [26,18], [26,26],
                 [32,6], [32,22],
                 [38,14]] 

    #[[25,10], [25,40], [40,25]] # [[10,10], [10,40], [40,25]] 
    #[[2,3],[4,2]] #last element [i,j] in this index is the exit point 

    #m = 35 #rows
    #n = 12 #columns

    m = rows #rows
    n = columns #columns

    #velocity - specify number of divisions divide rows to list of [10, 30, 50] so 10 would be 3, 30 would be 2 be 50 1
    velocity_divisions = 5

    #exit_point_type
    #KR - corner-right
    #KL - corner-left
    #C - center
    #B - Base
    exit_point_type = 'B'

    turnoff_velocity = False

    mx = create_matrix(m, n, vehicles, way_points)
    vel_col = create_divisions(velocity_divisions, m)

    result.append(mx)
    # print(mx)

    for i in range(0, time):
    #print('before--')
    #print(mx)
        mx_tr, dist = translate_matrix(mx, way_points, vel_col, turnoff_velocity, exit_point_type)

        # print('after--')
        # print(mx_tr)
        result.append(mx_tr)
        distance.append(dist)

        mx = mx_tr
    
    return result, distance
  
##############

#%%

def cal_flow(result, row1, row2, iterations):
    sum = 0.0
    
    for i in range(0, len(result)-1):
        current_row = result[i][row1: row2]
        next_row = result[i+1][row1: row2]
        
        cost = current_row == next_row
       
        
        #print(cost)
        cost_count= np.count_nonzero(cost== False)/2
        
        cost_count = math.ceil(cost_count)
        
        sum = sum + cost_count
        #print(cost_count)
    
    return sum / iterations


def landing_rate(result, exit_row, iterations):
  sum = 0.0
  layers = 1

  for i in range(0, len(result)):
    sum = sum + result[i][exit_row-1].sum()
    sum = sum + result[i][exit_row+1].sum()

  return sum/iterations

#%%%%%%%

vehicles_list = [l for l in range(20,320,20)]
itr = 20
time = 10000
rows = 42
columns = 28 

distance_dict = {}

flow_dict = {}
landr_dict = {}
vel1_dict = {}
vel2_dict = {}
vel3_dict = {}
vel4_dict = {}
vel5_dict = {}
vel6_dict = {}

flow_avg_dict = {}
landr_avg_dict = {}
vel1_avg_dict = {}
vel2_avg_dict = {}
vel3_avg_dict = {}
vel4_avg_dict = {}
vel5_avg_dict = {}
vel6_avg_dict = {}

for vehicles in vehicles_list:
    flow = 0.0
    landr = 0.0
    vel1 = 0.0
    vel2 = 0.0
    vel3 = 0.0 
    vel4 = 0.0
    vel5 = 0.0
    vel6 = 0.0
    
    
    flow_arr = []
    landr_arr = []
    vel1_arr = []
    vel2_arr = []
    vel3_arr = []
    vel4_arr = []
    vel5_arr = []
    vel6_arr = []
    
    for i in range(0, itr):
        result,dist = run(vehicles, time, rows, columns)
        
        distance_dict[str(i)+ '_'+str(vehicles)] = dist
        
        pflow = cal_flow(result, 12, 34, time)
        plandr = cal_flow(result, 36, 38, time)
        flow += pflow
        landr +=  plandr
        flow_arr.append(pflow)
        landr_arr.append(plandr)
        
        pvel1 = cal_flow(result, 14, 16, time)
        pvel2 = cal_flow(result, 18, 19, time)
        pvel3 = cal_flow(result, 20, 22, time)
        pvel4 = cal_flow(result, 25, 26, time)
        pvel5 = cal_flow(result, 30, 32, time)
        pvel6 = cal_flow(result, 30, 32, time)
        vel1 += pvel1
        vel2 += pvel2
        vel3 += pvel3
        vel4 += pvel4
        vel5 += pvel5
        vel6 += pvel6
        vel1_arr.append(pvel1)
        vel2_arr.append(pvel2)
        vel3_arr.append(pvel3)
        vel4_arr.append(pvel4)
        vel5_arr.append(pvel5)
        vel6_arr.append(pvel6)
        
    landr_dict[vehicles] = landr/itr
    flow_dict[vehicles] = flow/itr
    vel1_dict[vehicles] = vel1/itr
    vel2_dict[vehicles] = vel2/itr
    vel3_dict[vehicles] = vel3/itr
    vel4_dict[vehicles] = vel4/itr
    vel5_dict[vehicles] = vel5/itr
    vel6_dict[vehicles] = vel6/itr
    
    
    flow_avg_dict[vehicles] = flow_arr
    landr_avg_dict[vehicles] = landr_arr
    vel1_avg_dict[vehicles] = vel1_arr
    vel2_avg_dict[vehicles] = vel2_arr
    vel3_avg_dict[vehicles] = vel3_arr
    vel4_avg_dict[vehicles] = vel4_arr
    vel5_avg_dict[vehicles] = vel5_arr
    vel6_avg_dict[vehicles] = vel6_arr

print('****')

for li in range(len(list(flow_avg_dict.values()))):
    print(np.mean(list(flow_avg_dict.values())[li]), ',', np.var(list(flow_avg_dict.values())[li]))

print('****')

for li in range(len(list(landr_avg_dict.values()))):
    print(np.mean(list(landr_avg_dict.values())[li]), ',', np.var(list(landr_avg_dict.values())[li]))

print('****')

for li in range(len(list(vel1_avg_dict.values()))):
    print(np.mean(list(vel1_avg_dict.values())[li]), ',', np.var(list(vel1_avg_dict.values())[li]))

print('****')

for li in range(len(list(vel2_avg_dict.values()))):
    print(np.mean(list(vel2_avg_dict.values())[li]), ',', np.var(list(vel2_avg_dict.values())[li]))

print('****')

for li in range(len(list(vel3_avg_dict.values()))):
    print(np.mean(list(vel3_avg_dict.values())[li]), ',', np.var(list(vel3_avg_dict.values())[li]))

print('****')

for li in range(len(list(vel4_avg_dict.values()))):
    print(np.mean(list(vel4_avg_dict.values())[li]), ',', np.var(list(vel4_avg_dict.values())[li]))

print('****')

for li in range(len(list(vel5_avg_dict.values()))):
    print(np.mean(list(vel5_avg_dict.values())[li]), ',', np.var(list(vel5_avg_dict.values())[li]))

print('****')

for li in range(len(list(vel6_avg_dict.values()))):
    print(np.mean(list(vel6_avg_dict.values())[li]), ',', np.var(list(vel6_avg_dict.values())[li]))

################################
print('****')
################################

k = list(distance_dict.values())

t = []

for i in k:
    s = sum(i)
    t.append(s)
    
avs = [sum(t[i:i + itr])/itr for i in range(0, len(t), itr)]

for i in avs:
    print(i)

print('****')


#%%
