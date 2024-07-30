"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
2024-1 Exercises in Mechanics [001] Project: Three-body Problem

Author : Gwanyoung Ko
E-mail : clevonkk23@snu.ac.kr
Student ID : 2023-14444
Last Modified : June 6, 2024

References : 

[1] Thornton, Stephen T., and Jerry B. Marion. "Chapter 5.
    Gravitation" Classical Dynamics of Particles and Systems,
    6th ed., Cencage Learning, Andover, 2024, pp. 183.

[2] Wikipedia contributors. (2024, May 7). Runge-Kutta methods. 
    In Wikipedia, The Free Encyclopedia. Retrieved 04:10, May 
    15, 2024, from https://en.wikipedia.org/w/index.php?title=
    Runge%E2%80%93Kutta_methods&oldid=1222729994
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# [ Importing Packages & Libraries ]

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# [ Reading Input & Setting Up Variables ]
# This section reads parameters from User Inputs

input1_list = input().split()
x10, y10, vx10, vy10 = map(float, input1_list)

input2_list = input().split()
x20, y20, vx20, vy20 = map(float, input2_list)

input3_list = input().split()
x30, y30, vx30, vy30 = map(float, input3_list)

input4_list = input().split()
mode = int(input4_list[0])

if mode == 1:
    num = int(input())

    tlist = []
    for i in range(num):
        tval = float(input())
        tlist.append(tval)
    
    t_i = tlist[0]
    t_f = tlist[num-1]
    delt = 0.001 # You may adjust the time step

elif mode == 2:
    input5_list = input().split()
    t_i = 0.00
    t_f, dt = map(float, input5_list)
    delt = dt

else :
    print("Error: Insert ONLY 1 or 2")
    exit()

m1 = 1; m2 = 1; m3 = 1 # You may adjust the constants
G = 1 # You may adjust the constants

t = np.arange(0, t_f + delt, delt)
tsize = len(t)

y0 = np.array([x10, y10, vx10, vy10, x20, y20, vx20, vy20, x30, y30, vx30, vy30])
y = np.zeros((tsize, len(y0)))
y[0] = y0

# [ Functions ]
# This section defines several functions used by DE solver.

def norm(vec):
    norm = np.sqrt((vec[0])**2 + vec[1]**2)
    return norm

def dist(pt1, pt2):
    vec = [0, 0]
    vec[0] = pt1[0] - pt2[0]
    vec[1] = pt1[1] - pt2[1]
    dist = norm(vec)
    return dist

def unit(vec):
    unit = [0, 0]
    unit[0] = vec[0] / norm(vec)
    unit[1] = vec[1] / norm(vec)
    return unit

def gravityOn1(x1, y1, x2, y2, x3, y3):
    gravityOn1 = [0, 0]
    by2 = [0, 0]
    by3 = [0, 0]

    by2[0] =  G * m1 * m2 * unit([x2 - x1, y2 - y1])[0] / (dist([x1, y1], [x2, y2]))**2
    by2[1] =  G * m1 * m2 * unit([x2 - x1, y2 - y1])[1] / (dist([x1, y1], [x2, y2]))**2

    by3[0] =  G * m1 * m3 * unit([x3 - x1, y3 - y1])[0] / (dist([x1, y1], [x3, y3]))**2
    by3[1] =  G * m1 * m3 * unit([x3 - x1, y3 - y1])[1] / (dist([x1, y1], [x3, y3]))**2
    
    gravityOn1[0] = by2[0] + by3[0]
    gravityOn1[1] = by2[1] + by3[1]

    return gravityOn1

def gravityOn2(x1, y1, x2, y2, x3, y3):
    gravityOn2 = [0, 0]
    by1 = [0, 0]
    by3 = [0, 0]

    by1[0] =  G * m2 * m1 * unit([x1 - x2, y1 - y2])[0] / (dist([x1, y1], [x2, y2]))**2
    by1[1] =  G * m2 * m1 * unit([x1 - x2, y1 - y2])[1] / (dist([x1, y1], [x2, y2]))**2

    by3[0] =  G * m2 * m3 * unit([x3 - x2, y3 - y2])[0] / (dist([x2, y2], [x3, y3]))**2
    by3[1] =  G * m2 * m3 * unit([x3 - x2, y3 - y2])[1] / (dist([x2, y2], [x3, y3]))**2
    
    gravityOn2[0] = by1[0] + by3[0]
    gravityOn2[1] = by1[1] + by3[1]

    return gravityOn2

def gravityOn3(x1, y1, x2, y2, x3, y3):
    gravityOn3 = [0, 0]
    by1 = [0 ,0]
    by2 = [0, 0]

    by1[0] =  G * m3 * m1 * unit([x1 - x3, y1 - y3])[0] / (dist([x1, y1], [x3, y3]))**2
    by1[1] =  G * m3 * m1 * unit([x1 - x3, y1 - y3])[1] / (dist([x1, y1], [x3, y3]))**2

    by2[0] =  G * m2 * m3 * unit([x2 - x3, y2 - y3])[0] / (dist([x2, y2], [x3, y3]))**2
    by2[1] =  G * m2 * m3 * unit([x2 - x3, y2 - y3])[1] / (dist([x2, y2], [x3, y3]))**2
    
    gravityOn3[0] = by1[0] + by2[0]
    gravityOn3[1] = by1[1] + by2[1]

    return gravityOn3

def f(y, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11):
    res = np.array([
        y[2] + c0 * delt,
        y[3] + c1 * delt,
        gravityOn1(y[0] + c2 * delt, y[1] + c3 * delt, y[4] + c6 * delt, y[5] + c7 * delt, y[8] + c10 * delt, y[9] + c11 * delt)[0] / m1,
        gravityOn1(y[0] + c2 * delt, y[1] + c3 * delt, y[4] + c6 * delt, y[5] + c7 * delt, y[8] + c10 * delt, y[9] + c11 * delt)[1] / m1,
        y[6] + c4 * delt,
        y[7] + c5 * delt,
        gravityOn2(y[0] + c2 * delt, y[1] + c3 * delt, y[4] + c6 * delt, y[5] + c7 * delt, y[8] + c10 * delt, y[9] + c11 * delt)[0] / m2,
        gravityOn2(y[0] + c2 * delt, y[1] + c3 * delt, y[4] + c6 * delt, y[5] + c7 * delt, y[8] + c10 * delt, y[9] + c11 * delt)[1] / m2,
        y[10] + c8 * delt,
        y[11] + c9 * delt,
        gravityOn3(y[0] + c2 * delt, y[1] + c3 * delt, y[4] + c6 * delt, y[5] + c7 * delt, y[8] + c10 * delt, y[9] + c11 * delt)[0] / m3,
        gravityOn3(y[0] + c2 * delt, y[1] + c3 * delt, y[4] + c6 * delt, y[5] + c7 * delt, y[8] + c10 * delt, y[9] + c11 * delt)[1] / m3,
        ])
    return res

# [ Differential Equation Solver : Runge-Kutta Method Order 4 ]
# This section solves given differential equation using Runge-Kutta method order 4.

for i in range(tsize - 1):

    kx11, ky11, kvx11, kvy11, kx21, ky21, kvx21, kvy21, kx31, ky31, kvx31, kvy31 = f(y[i], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    kx12, ky12, kvx12, kvy12, kx22, ky22, kvx22, kvy22, kx32, ky32, kvx32, kvy32 = f(y[i], 0.5 * kvx11, 0.5 * kvy11, 0.5 * kx11, 0.5 * ky11, 0.5 * kvx21, 0.5 * kvy21, 0.5 * kx21, 0.5 * ky21, 0.5 * kvx31, 0.5 * kvy31, 0.5 * kx31, 0.5 * ky31)
    kx13, ky13, kvx13, kvy13, kx23, ky23, kvx23, kvy23, kx33, ky33, kvx33, kvy33 = f(y[i], 0.5 * kvx12, 0.5 * kvy12, 0.5 * kx12, 0.5 * ky12, 0.5 * kvx22, 0.5 * kvy22, 0.5 * kx22, 0.5 * ky22, 0.5 * kvx32, 0.5 * kvy32, 0.5 * kx32, 0.5 * ky32)
    kx14, ky14, kvx14, kvy14, kx24, ky24, kvx24, kvy24, kx34, ky34, kvx34, kvy34 = f(y[i], kvx13, kvy13, kx13, ky13, kvx23, kvy23, kx23, ky23, kvx33, kvy33, kx33, ky33)
    
    dx1 = delt * (kx11 + 2 * kx12 + 2 * kx13 + kx14) / 6
    dy1 = delt * (ky11 + 2 * ky12 + 2 * ky13 + ky14) / 6
    dvx1 = delt * (kvx11 + 2 * kvx12 + 2 * kvx13 + kvx14) / 6
    dvy1 = delt * (kvy11 + 2 * kvy12 + 2 * kvy13 + kvy14) / 6
    
    dx2 = delt * (kx21 + 2 * kx22 + 2 * kx23 + kx24) / 6
    dy2 = delt * (ky21 + 2 * ky22 + 2 * ky23 + ky24) / 6
    dvx2 = delt * (kvx21 + 2 * kvx22 + 2 * kvx23 + kvx24) / 6
    dvy2 = delt * (kvy21 + 2 * kvy22 + 2 * kvy23 + kvy24) / 6
    
    dx3 = delt * (kx31 + 2 * kx32 + 2 * kx33 + kx34) / 6
    dy3 = delt * (ky31 + 2 * ky32 + 2 * ky33 + ky34) / 6
    dvx3 = delt * (kvx31 + 2 * kvx32 + 2 * kvx33 + kvx34) / 6
    dvy3 = delt * (kvy31 + 2 * kvy32 + 2 * kvy33 + kvy34) / 6

    y[i+1][0] = y[i][0] + dx1
    y[i+1][1] = y[i][1] + dy1
    y[i+1][2] = y[i][2] + dvx1
    y[i+1][3] = y[i][3] + dvy1

    y[i+1][4] = y[i][4] + dx2
    y[i+1][5] = y[i][5] + dy2
    y[i+1][6] = y[i][6] + dvx2
    y[i+1][7] = y[i][7] + dvy2

    y[i+1][8] = y[i][8] + dx3
    y[i+1][9] = y[i][9] + dy3
    y[i+1][10] = y[i][10] + dvx3
    y[i+1][11] = y[i][11] + dvy3

# [ Output ]
# This section generates proper output(s) according to the chosen mode.

if mode == 1:

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""[ READ : MODE 1 DESCRIPTION]
    Mode 1 prints the values of x1, y1, vx1, vy1, x2, y2, vx2, vy2,
    x3, y3, vx3, vy3.

    All values are given in *float* type, to the accuracy of 
    thousandths place value.
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    for item in tlist:
        data = np.round(y[int(item / delt)], 3)

        count = 0
        for j in range(int(len(data))):
            print('%.3f'%float(y[int(item / delt)][j]), end = " ")
            count = count + 1

            if count % 4 == 0 and count != 12:
                print("")

        print("\n")

elif mode == 2:

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""[ READ : MODE 2 DESCRIPTION]
    Mode 2 results a graphic simulation of three-body motion in
    given time interval. A window will be automatically showed up.
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    x1_data = np.zeros(tsize); y1_data = np.zeros(tsize)
    vx1_data = np.zeros(tsize); vy1_data = np.zeros(tsize)

    x2_data = np.zeros(tsize); y2_data = np.zeros(tsize)
    vx2_data = np.zeros(tsize); vy2_data = np.zeros(tsize)

    x3_data = np.zeros(tsize); y3_data = np.zeros(tsize)
    vx3_data = np.zeros(tsize); vy3_data = np.zeros(tsize)

    for i in range(tsize):
        x1_data[i] = float(y[i][0]); y1_data[i] = float(y[i][1])
        vx1_data[i] = float(y[i][2]); vy1_data[i] = float(y[i][3])

        x2_data[i] = float(y[i][4]); y2_data[i] = float(y[i][5])
        vx2_data[i] = float(y[i][6]); vy2_data[i] = float(y[i][7])

        x3_data[i] = float(y[i][8]); y3_data[i] = float(y[i][9])
        vx3_data[i] = float(y[i][10]); vy3_data[i] = float(y[i][11])

    def update(frame):

        x1 = x1_data[frame] + vx1_data[frame] * dt
        y1 = y1_data[frame] + vy1_data[frame] * dt

        x2 = x2_data[frame] + vx2_data[frame] * dt
        y2 = y2_data[frame] + vy2_data[frame] * dt

        x3 = x3_data[frame] + vx3_data[frame] * dt
        y3 = y3_data[frame] + vy3_data[frame] * dt 
    
        line1.set_data(x1_data[:frame + 1], y1_data[:frame + 1])
        line2.set_data(x2_data[:frame + 1], y2_data[:frame + 1])
        line3.set_data(x3_data[:frame + 1], y3_data[:frame + 1])

        ptl1.set_data(x1, y1)
        ptl2.set_data(x2, y2)
        ptl3.set_data(x3, y3)

        time_legend.set_text(f'Time: {frame * dt: .2f}')
    
        return line1, line2, line3, ptl1, ptl2, ptl3, time_legend

    fig, ax = plt.subplots()
        
    xplotmax = max(np.max(x1_data), np.max(x2_data), np.max(x3_data))
    xplotmin = min(np.min(x1_data), np.min(x2_data), np.min(x3_data))
    yplotmax = max(np.max(y1_data), np.max(y2_data), np.max(y3_data))
    yplotmin = min(np.min(y1_data), np.min(y2_data), np.min(y3_data))
    ax.set_xlim(xplotmin - 0.1, xplotmax + 0.1)
    ax.set_ylim(yplotmin - 0.1, yplotmax + 0.1)

    line1, = ax.plot([], [], linestyle = 'dotted', color = 'red', lw = 2)
    line2, = ax.plot([], [], linestyle = 'dotted', color = 'green', lw = 2)
    line3, = ax.plot([], [], linestyle = 'dotted', color = 'blue', lw = 2)

    ptl1, = ax.plot([], [], marker='o', color='red', markersize=5)
    ptl2, = ax.plot([], [], marker='o', color='green', markersize=5)
    ptl3, = ax.plot([], [], marker='o', color='blue', markersize=5)

    time_legend = ax.text(0.05, 0.95, '', transform = ax.transAxes, verticalalignment = 'top')

    red_line = plt.Line2D((0,1), (0,0), color='red', linewidth=0.5)
    green_line = plt.Line2D((0,1), (0,0), color='green', linewidth=0.5)
    blue_line = plt.Line2D((0,1), (0,0), color='blue', linewidth=0.5)

    ax.legend([red_line, green_line, blue_line], ['Particle 1', 'Particle 2', 'Particle 3'], loc='upper right')

    ani = FuncAnimation(fig, update, frames = tsize, interval = 50)

    plt.show() 