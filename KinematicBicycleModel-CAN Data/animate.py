import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rand

from kinematic_model import KinematicBicycleModel
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from libs.stanley_controller import StanleyController
from libs.car_description import Description
from libs.cubic_spline_interpolator import generate_cubic_spline

y = []
x = []
v = []
yaw = []
y_ddot=[]
delta=[]
class Simulation:

    def __init__(self):

        fps = 50.0

        self.dt = 1/fps
        self.map_size = 40
        self.frames = 300
        self.loop = False

class Path:

    def __init__(self):

        # Get path to waypoints.csv
        dir_path = 'data/waypoints.csv'
        #dir_path = 'data/waypoints_straight.csv'
        df = pd.read_csv(dir_path)

        x = df['X-axis'].values
        y = df['Y-axis'].values
        ds = 0.05


        self.px, self.py, self.pyaw, _ = generate_cubic_spline(x, y, ds)

class Car:

    def __init__(self, init_x, init_y, init_yaw, px, py, pyaw, dt):

        # Model parameters
        self.x = init_x
        self.y = init_y
        self.yaw = init_yaw
        self.v = 0
        self.delta = 0.0
        self.omega = 0.0
        self.y_ddot=0.0
        self.L = 2.5
        self.max_steer = np.deg2rad(33)
        self.dt = dt
        self.c_r = 0.01
        self.c_a = 2.0

        # Tracker parameters
        self.px = px
        self.py = py
        self.pyaw = pyaw
        self.k = 8.0
        self.ksoft = 1.0
        self.kyaw = 0.01
        self.ksteer = 0.0
        self.crosstrack_error = None
        self.target_id = None

        # Description parameters
        self.length = 4.5
        self.width = 2.0
        self.rear2wheel = 1.0
        self.wheel_dia = 0.15 * 2
        self.wheel_width = 0.2
        self.tread = 0.7
        self.colour = 'black'

        self.tracker = StanleyController(self.k, self.ksoft, self.kyaw, self.ksteer, self.max_steer, self.L, self.px, self.py, self.pyaw)
        self.kbm = KinematicBicycleModel(self.L, self.max_steer, self.dt, self.c_r, self.c_a)

    def drive(self,frame):
        if frame>=100:
            throttle=150# rand.uniform(150, 200)
            self.delta=0 #0.001745   #0
        else:
            throttle = 150 #rand.uniform(150, 200)
            self.delta, self.target_id, self.crosstrack_error = self.tracker.stanley_control(self.x, self.y, self.yaw, self.v, self.delta)
        self.x, self.y, self.yaw, self.v, _, _,self.y_ddot = self.kbm.kinematic_model(self.x, self.y, self.yaw, self.v, throttle, self.delta, self.y_ddot)
        x.append(self.x)
        y.append(self.y)
        yaw.append(self.yaw)
        v.append(self.v)
        delta.append(self.delta)
        y_ddot.append(self.y_ddot)
        os.system('cls' if os.name=='nt' else 'clear')
        print(f"Cross-track term: {self.crosstrack_error}")

def main():
    
    sim = Simulation()
    path = Path()
    car = Car(path.px[0], path.py[0], path.pyaw[0], path.px, path.py, path.pyaw, sim.dt)
    desc = Description(car.length, car.width, car.rear2wheel, car.wheel_dia, car.wheel_width, car.tread, car.L)

    interval = sim.dt * 10**3

    fig = plt.figure()
    ax = plt.axes()
    ax.set_aspect('equal')

    road = plt.Circle((0, 0), 50, color='gray', fill=False, linewidth=30)
    #road = patches.Rectangle((45, 0), 10, 120, facecolor='gray', linewidth=30)
    ax.add_patch(road)
    ax.plot(path.px, path.py, '--', color='gold')

    annotation = ax.annotate(f'{car.x:.1f}, {car.y:.1f}', xy=(car.x, car.y + 5), color='black', annotation_clip=False)
    target, = ax.plot([], [], '+r')

    outline, = ax.plot([], [], color=car.colour)
    fr, = ax.plot([], [], color=car.colour)
    rr, = ax.plot([], [], color=car.colour)
    fl, = ax.plot([], [], color=car.colour)
    rl, = ax.plot([], [], color=car.colour)
    rear_axle, = ax.plot(car.x, car.y, '+', color=car.colour, markersize=2)

    plt.grid()
    
    def animate(frame):
        

        # Camera tracks car
        ax.set_xlim(car.x - sim.map_size, car.x + sim.map_size)
        ax.set_ylim(car.y - sim.map_size, car.y + sim.map_size)

        # Drive and draw car
        car.drive(frame)
        outline_plot, fr_plot, rr_plot, fl_plot, rl_plot = desc.plot_car(car.x, car.y, car.yaw, car.delta)
        outline.set_data(outline_plot[0], outline_plot[1])
        fr.set_data(fr_plot[0], fr_plot[1])
        rr.set_data(rr_plot[0], rr_plot[1])
        fl.set_data(fl_plot[0], fl_plot[1])
        rl.set_data(rl_plot[0], rl_plot[1])
        rear_axle.set_data(car.x, car.y)

        # Show car's target
        target.set_data(path.px[car.target_id], path.py[car.target_id])

        # Annotate car's coordinate above car
        annotation.set_text(f'{car.x:.1f}, {car.y:.1f}')
        annotation.set_position((car.x, car.y + 5))

        plt.title(f'{sim.dt*frame:.2f}s', loc='right')
        plt.xlabel(f'Speed: {car.v:.2f} m/s', loc='left')

        if frame>100 & frame<125:
            ax.text(3, 8, 'Check LKABuzzer_A',style='italic',
            bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10})
        elif frame>125 & frame<150:
            ax.text(3, 8, 'Check LKAAssist_A',style='italic',
            bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10})
        elif frame>150 & frame<165:
            ax.text(3, 8, 'Check DashCam for lane departure',style='italic',
            bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10})
        elif frame>165 & frame<175:
            ax.text(3, 8, 'Check Headpose',style='italic',
            bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10})
        elif frame>165 & frame<175:
            ax.text(3, 8, 'Do you want to press the SOS button?',style='italic',
            bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
        return outline, fr, rr, fl, rl, rear_axle, target,

    _ = FuncAnimation(fig, animate, frames=sim.frames, interval=interval, repeat=sim.loop)
    # anim.save('animation.gif', writer='imagemagick', fps=50)
    plt.show()

if __name__ == '__main__':
    main()
    final=np.column_stack((x, y, y_ddot, delta, yaw, v))
    final[0]
    #np.savetxt('test_vals_nostanley_straight_roadrunoff.csv',final,delimiter=',')
    np.savetxt('test_vals_turn_roadrunoff.csv',final,delimiter=',')

