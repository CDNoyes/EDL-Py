import numpy as np 
import matplotlib.pyplot as plt 


def Plots(df_list, labels, show=True, figsize=(16, 7)):
        for label, df in zip(labels, df_list):
            Plot(df, show=False, figsize=figsize, label=label)

        if show:
            plt.show()


def Plot(df, show=False, figsize=(16, 7), label=""):

    x = df['x']
    y = df['y']
    z = df['z']

    hor = np.linalg.norm([x,y], axis=0)
    
    t = df.index 
    m = df['mass']  # not all versions have mass ?

    u = df['vx']
    v = df['vy']
    w = df['vz']

    Tx = df['Tx']
    Ty = df['Ty']
    Tz = df['Tz']

    V = np.linalg.norm((u,v,w), axis=0)
    T = np.linalg.norm((Tx,Ty,Tz), axis=0)

    plt.figure(1, figsize=figsize)

    plt.subplot(2,2,1)
    plt.plot(t, np.array((x,y,z)).T)
    plt.xlabel('Time (s)')
    plt.ylabel("Distance (m)")

    plt.subplot(2,2,2)
    plt.plot(t, m, label=label)
    plt.xlabel('Time (s)')
    plt.ylabel("Mass (kg)")
    plt.legend()

    plt.subplot(2,2,3)
    plt.plot(t, np.array((u,v,w)).T)
    plt.xlabel('Time (s)')
    plt.ylabel("Velocity (m/s)")

    plt.subplot(2,2,4)
    plt.plot(t, T/1000, label="")
#     plt.plot(t, Tx/1000)
#     plt.plot(t, Ty/1000)
#     plt.plot(t, Tz/1000)
    plt.xlabel('Time (s)')
    plt.ylabel("Thrust (kN)")

    # Plot the control unit vector 

#     plt.figure()
#     plt.plot(hor, z, label="Trajectory")
#     plt.plot(hor, hor*np.tan(15*np.pi/180), 'k--', label="15 deg glide-slope constraint")
#     plt.xlabel("Horizontal Distance (m)")
#     plt.ylabel("Altitude (m)")
#     plt.legend()
    if show:
        plt.show()