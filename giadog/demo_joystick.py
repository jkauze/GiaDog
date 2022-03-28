"""
    Authors: Amin Arriaga, Eduardo Lopez
    Project: Graduation Thesis: GIAdog

    This file contains a demo for the controller module
"""
import pybullet as p

from src.joystick import XboxController
import pybullet as p
import pybullet_utils.bullet_client as bc
import pybullet_data
from time import sleep
import numpy as np

def main():
    
    
    joy = XboxController()
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    plane = p.loadURDF('plane.urdf')
    

    visualShift = [0,0,0]
    meshScale = [0.1, 0.1, 0.1]
    inertiaShift = [0,0,-0.5]

    #the visual shape and collision shape can be re-used by all createMultiBody instances (instancing)
    
    # Create a visual shape of a cube
    meshScale=[1,1,1]
    visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,
                    fileName="giadog/assets/arrow.obj", rgbaColor=[0.5,1,1,1], 
                    specularColor=[0.4,.4,0], visualFramePosition=visualShift,
                    meshScale=meshScale)

                                
    orientation = p.getQuaternionFromEuler([0,np.pi/2,0])
    arrow = p.createMultiBody(baseMass=1,
                                baseInertialFramePosition=inertiaShift,
                                baseOrientation=orientation, 
                                baseVisualShapeIndex = visualShapeId, 
                                basePosition = [0,0,1], 
                                useMaximalCoordinates=False)
    cam_dist = 15
    cam_yaw = 0
    cam_pitch = -30
    
    
    speed = 0.5
    camera_speed = 1.5
    while True:
        x = joy.LeftJoystickX
        y = joy.LeftJoystickY
        cam_dir, intensity_cam = joy.get_right_joystick()

        cam_pitch = cam_pitch - intensity_cam*camera_speed*np.sin(cam_dir)
        cam_yaw = cam_yaw - intensity_cam*camera_speed*np.cos(cam_dir)
        
        # Print the camera yaw with 2 decimals
        theta = cam_yaw*np.pi/180 +  np.pi/2 
        orientation_b = p.getQuaternionFromEuler([0,np.pi/2,theta])
        position, _ = p.getBasePositionAndOrientation(arrow)

        
        
        new_x = position[0] + y * speed*np.cos(theta) + x * speed*np.sin(theta)
        new_y = position[1] + y * speed*np.sin(theta) - x * speed*np.cos(theta)
        
        position = [new_x, new_y, position[2]]
        p.resetBasePositionAndOrientation(arrow,position,orientation_b)
        p.resetDebugVisualizerCamera(cam_dist, cam_yaw,
                                cam_pitch, position)
        p.stepSimulation()
        sleep(1/240)


if __name__ == '__main__':
    main()