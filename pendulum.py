import RobotDART as rd
import numpy as np
import numpy as np
from os import path
import random
import dartpy


class PendulumEnv():

    def __init__(self):
        ########## Create simulator object ##########
        self.time_step = 0.001
        self.simu = rd.RobotDARTSimu(self.time_step)
        self.max_torque = 2.5
        
        ########## Create Graphics ##########
        #self.graphics = rd.gui.Graphics()
        #self.simu.set_graphics(graphics)
        #self.graphics.look_at([0., 2.5, 0.5], [0., 0., 0.])
        
        ########## Create robot ##########
        robot = rd.Robot("pendulum.urdf")
        robot.fix_to_world()
        robot.set_actuator_types("torque")
        
        self.simu.add_robot(robot)

        self.dofs = robot.dof_names()
        ###################################################################


    def step(self, action):

        th, speed = self.state  # th := theta

        action = self.max_torque*random.uniform(0,1) if action == 1 else -self.max_torque*random.uniform(0,1)
        #print(action)
        
        i = 0
        action = np.clip(action, -self.max_torque, self.max_torque) # Whitout it i have problem with reward form
        #set new torque
        cmd = np.ndarray([1])
        cmd[0] = action
        self.simu.robots()[0].set_commands(cmd)        
        #advance simulation
        for i in range(100):
            self.simu.step_world()
        #calc reward, new th and speed
        reward = angle_normalize(self.simu.robots()[0].positions()) * 4 - 0.05 * abs(speed) - 0.01 * abs(action)
        newth = self.simu.robots()[0].positions() % (2*np.pi)
        newspeed = self.simu.robots()[0].velocities(self.dofs)
        self.state = np.array([newth, newspeed])

        return self._get_obs(), reward, False, {}


    def reset(self):

        self.simu.robots()[0].reset()

        self.state = np.array([self.simu.robots()[0].positions() % (2*np.pi), 0])
        
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state

        #return np.array([theta, thetadot], dtype=np.float32)
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)
        

def angle_normalize(x):
    y = abs(x % (2*np.pi))
    return -abs(y - np.pi) 