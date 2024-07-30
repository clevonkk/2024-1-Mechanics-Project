"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
2024-1 Exercises in Mechanics [001] Project: Coupled Pendulum

Author : Gwanyoung Ko
E-mail : clevonkk23@snu.ac.kr
Student ID : 2023-14444
Last Modified : May 29, 2024

References : 

[1] Thornton, Stephen T., and Jerry B. Marion. "Chapter 7.
    Hamilton's Principle - Lagrangian and Hamiltonian 
    Dynamics." Classical Dynamics of Particles and Systems,
    6th ed., Cencage Learning, Andover, 2024, pp. 238.

[2] Wikipedia contributors. (2024, May 7). Runge-Kutta methods. 
    In Wikipedia, The Free Encyclopedia. Retrieved 04:10, May 
    15, 2024, from https://en.wikipedia.org/w/index.php?title=
    Runge%E2%80%93Kutta_methods&oldid=1222729994
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""[ READ : INSTURCTIONS ]
///////////////////////[ Instructions ]////////////////////////

1. Variables
    a. Default
        1) m1 : mass of pendulum 1
        2) l1 : length of massless string connected to m1
        3) m2 : mass of pendulum 2
        4) l2 : length of massless string connected to m2
        5) delx : displacement between two ends of l1, l2
         - 1)~5) *float* type, 1.0 to 10.0, to tenths.

        6) theta1 : angle of m1, measured from y axis
        7) omega1 : angular velocity of m1
         - 6)~7) *float* type, -1.57 to 1.57, to hundredths.

        8) theta2 : angle of m2, measured from y axis
        9) omega2 : anglular velocity of m2
         - 8)~9) *float* type, Automatically set as 0.00.

        10) mode : mode of output
         - 10) *int* type, 1 or 2. O.w., program not work.

        11) t : list of total computing times.
         - 11) *np.array* type, t goes from 0.00 to t_f

        12) tsize : length of t
         - 12) *int* type, for sure.

        13) y : solution of given DE system
         - 13) *np.array* type. y[i] returns values of 
               theta1 (y[i][0]), omega1 (y[i][1]), theta2
               (y[i][2]), and omega2 (y[i][3]) for ith
               frame of time. (i seconds)

    b. Mode 1
        1) num : number of times to be outputted
         - 1) *int* type, for sure.

        2) tlist : list of times to be outputted
         - 2) *list* type, for sure.

        3) t_i : first value of tlist
        4) t_f : final value of tlist
         - 3)~4) *float* type, 1.00 to 40.00, to hundredths.

        5) delt : time step of DE solver
         - 5) *float* type, automatically set as 0.001

    c. Mode 2
        1) t_i : first value of computing times.
        2) t_f : final value of computing times.
         - 1)~2) *float* type, t_i automatically set as 0.00
                 t_f for 1.00 to 40.00, to hundredths.

        3) delt : time step of DE solver
        4) dt : frame interval of time for display
         - 3)~4) *float* type, delt == dt. 0.01 to 1.00, to
                 hundredths.

2. Functions
    a. d
        Compute the distance d between two pendulums.

        Args :
            theta1 : *float* type, angle of m1
            theta2 : *float* type, angle of m2
        
        Returns :
            *float* type, distance d between two pendulums

    b. deld
        Compute the deformed length deld of massless spring.

        Args :
            theta1 : *float* type, angle of m1
            theta2 : *float* type, angle of m2

        Returns :
            *float* type, deformed length deld of massless
            spring
    
    d. ddeld1
        Compute derivative of deld with respect to theta1.

        Args :
            theta1 : *float* type, angle of m1
            theta2 : *float* type, angle of m2

        Returns :
            *float* type, derivative of deld with respect to
            theta1

    e. ddeld2
        Compute derivative of deld with respect to theta2.

        Args :
            theta1 : *float* type, angle of m1
            theta2 : *float* type, angle of m2

        Returns :
            *float* type, derivative of deld with respect to
            theta1

    f. de1
        Compute the value of second time derivative of theta1
        based on given system of DEs.

        Args :
            theta1 : *float* type, angle of m1
            theta2 : *float* type, angle of m2

        Returns :
            *float* type, second time derivative of theta1
            based on given system of DEs

    g. de2
        Compute the value of second time derivative of theta2
        based on given system of DEs.

        Args :
            theta1 : *float* type, angle of m1
            theta2 : *float* type, angle of m2

        Returns :
            *float* type, second time derivative of theta2
            based on given system of DEs

    h. f
        Compute y for next time step, y[i+1] based on y[i].

        Args :
            y : *np.array* type, the solution set
            c0 : *float* type, weight for omega1
            c1 : *float* type, weight for omega2
            c2 : *float* type, weight for theta1
            c3 : *float* type, weight for theta2
        
        Returns :
            *np.array* type, y[i+1] for next time step

>>>>>>>>>>> If you have any questions, contact to my e-mail.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# [ Importing Packages & Libraries ]

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# [ Reading Input & Setting Up Variables ]
# This section reads parameters from User Inputs

input1_list = input().split()
m1, l1, m2, l2, delx = map(float, input1_list)

input2_list = input().split()
theta1, omega1 = map(float, input2_list)

input3_list = input().split()
mode = int(input3_list[0])

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
    input4_list = input().split()
    t_i = 0.00
    t_f, dt = map(float, input4_list)
    delt = dt

else :
    print("Error: Insert ONLY 1 or 2")
    exit()

m1, l1, m2, l2, delx = list(map(float, input1_list))
theta1, omega1 = list(map(float, input2_list))
theta2 = 0; omega2 = 0 # You may adjust the initial conditions for m2
g = 9.8 # You may adjust the constants
k = 1.0 # You may adjust the constants

t = np.arange(0, t_f + delt, delt)
tsize = len(t)

y0 = np.array([theta1, omega1, theta2, omega2])
y = np.zeros((tsize, len(y0)))
y[0] = y0

# [ Functions ]
# This section defines several functions used by DE solver.

def d(theta1, theta2):
    d = np.sqrt((delx + l2 * np.sin(theta2) - l1 * np.sin(theta1))**2 + (l2 * np.cos(theta2) - l1 * np.cos(theta1))**2)
    return d

def deld(theta1, theta2):
    deld = d(theta1, theta2) - np.sqrt((delx)**2 + (l1 - l2)**2)
    return deld

def ddeld1(theta1, theta2):
    ddeld1 = (l1 * l2 * np.sin(theta1 - theta2) - delx * l1 * np.cos(theta1)) / d(theta1, theta2)
    return ddeld1

def ddeld2(theta1, theta2):
    ddeld2 = (delx * l2 * np.cos(theta2) - l1 * l2 * np.sin(theta1 - theta2)) / d(theta1, theta2)
    return ddeld2

def de1(theta1, theta2):
    res1 = - g * np.sin(theta1) / l1 - k * deld(theta1, theta2) * ddeld1(theta1, theta2) / (m1 * l1**2)
    return res1

def de2(theta1, theta2):
    res2 = - g * np.sin(theta2) / l2 - k * deld(theta1, theta2) * ddeld2(theta1, theta2) / (m2 * l2**2)
    return res2

def f(y, c0, c1, c2, c3):
    res = np.array([
        y[1] + c0 * delt,
        de1(y[0] + c2 * delt, y[2] + c3 * delt),
        y[3] + c1 * delt,
        de2(y[0] + c2 * delt, y[2] + c3 * delt)
        ])
    return res

# [ Differential Equation Solver : Runge-Kutta Method Order 4 ]
# This section solves given differential equation using Runge-Kutta method order 4.

for i in range(tsize - 1):

    kx11, kv11, kx21, kv21 = f(y[i], 0, 0, 0, 0)
    kx12, kv12, kx22, kv22 = f(y[i], 0.5 * kv11, 0.5 * kv21, 0.5 * kx11, 0.5 * kx21)
    kx13, kv13, kx23, kv23 = f(y[i], 0.5 * kv12, 0.5 * kv22, 0.5 * kx12, 0.5 * kx22)
    kx14, kv14, kx24, kv24 = f(y[i], kv13, kv23, kx13, kx23)
    
    dx1 = delt * (kx11 + 2 * kx12 + 2 * kx13 + kx14) / 6
    dv1 = delt * (kv11 + 2 * kv12 + 2 * kv13 + kv14) / 6
    dx2 = delt * (kx21 + 2 * kx22 + 2 * kx23 + kx24) / 6
    dv2 = delt * (kv21 + 2 * kv22 + 2 * kv23 + kv24) / 6

    y[i+1][0] = y[i][0] + dx1
    y[i+1][1] = y[i][1] + dv1
    y[i+1][2] = y[i][2] + dx2
    y[i+1][3] = y[i][3] + dv2

# [ Output ]
# This section generates proper output(s) according to the chosen mode.

if mode == 1:

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""[ READ : MODE 1 DESCRIPTION]
    Mode 1 prints the values of theta1, omega1, theta2, omega2.
    All values are given in *float* type, to the accuracy of 
    thousandths place value.
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    for i in range(len(tlist)):
        data = np.round(y[int(tlist[i] / delt)], 3)

        for j in range(len(data)):
            print(float(data[j]), end = " ")

        print("")
        
elif mode == 2:

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""[ READ : MODE 2 DESCRIPTION]
    Mode 2 results a graphic simulation of coupled pendulum in
    given time interval. A window will be automatically showed up.
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    theta1_data = np.zeros(tsize); omega1_data = np.zeros(tsize)
    theta2_data = np.zeros(tsize); omega2_data = np.zeros(tsize)

    for i in range(tsize):
        theta1_data[i] = float(y[i][0])
        omega1_data[i] = float(y[i][1])
        theta2_data[i] = float(y[i][2])
        omega2_data[i] = float(y[i][3])

    def update(frame):

        theta1 = theta1_data[frame] + omega1_data[frame] * dt
        theta2 = theta2_data[frame] + omega2_data[frame] * dt

        x1 = l1 * np.sin(theta1)
        y1 = - l1 * np.cos(theta1)
        x2 = delx + l2 * np.sin(theta2)
        y2 = - l2 * np.cos(theta2)
    
        line1.set_data([0, x1], [0, y1])
        line2.set_data([delx, x2], [0, y2])
        line3.set_data([x1, x2], [y1, y2])

        ax.legend(['Time: {:.2f}'.format(frame * dt)])
    
        return line1, line2, line3

    fig, ax = plt.subplots()

    line1, = ax.plot([], [], lw = 4)
    line2, = ax.plot([], [], lw = 4)
    line3, = ax.plot([], [], lw = 10, linestyle = 'dotted', color = 'gray')
        
    plotmax = delx + np.max(l2 * np.sin(theta2_data))
    plotmin = np.min(l1 * np.sin(theta1_data))
    ax.set_xlim(plotmin - 0.1, plotmax + 0.1)
    ax.set_ylim(- np.maximum(l1, l2) - 0.1, 0.1)

    ani = FuncAnimation(fig, update, frames = tsize, interval = 50)

    plt.show() 