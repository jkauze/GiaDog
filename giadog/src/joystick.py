"""
    Authors: Amin Arriaga, Eduardo Lopez
    Project: Graduation Thesis: GIAdog

    This file contains the code for the use of a gamepad (Xbox 360 controller)
    for the control of the robot.
    
    Reference:
    ----------
    https://stackoverflow.com/questions/46506850/how-can-i-get-input-from-an-xbox-one-controller-in-python
"""
import numpy as np
from inputs import get_gamepad
import math
import threading

class XboxController(object):
    """
    Xbox controller class.

    """
    MAX_TRIG_VAL = math.pow(2, 8)
    MAX_JOY_VAL = math.pow(2, 15)

    def __init__(self):
        """
        Initializes the controller.

        Arguments:
        ----------
        None
        """

        self.LeftJoystickY = 0
        self.LeftJoystickX = 0
        self.RightJoystickY = 0
        self.RightJoystickX = 0
        self.LeftTrigger = 0
        self.RightTrigger = 0
        self.LeftBumper = 0
        self.RightBumper = 0
        self.A = 0
        self.X = 0
        self.Y = 0
        self.B = 0
        self.LeftThumb = 0
        self.RightThumb = 0
        self.Back = 0
        self.Start = 0
        self.LeftDPad = 0
        self.RightDPad = 0
        self.UpDPad = 0
        self.DownDPad = 0

        # We need to start a new thread to monitor the controller
        self._monitor_thread = threading.Thread(target=self._monitor_controller, 
                                args=())
        self._monitor_thread.daemon = True
        self._monitor_thread.start()


    def read(self): 
        """
        Simple read function for testing purposes.

        Arguments:
        ----------
        None
        """
        x = self.LeftJoystickX
        y = self.LeftJoystickY
        a = self.A
        b = self.X # b=1, x=2
        rb = self.RightBumper
        return [x, y, a, b, rb]

    def get_left_joystick(self):
        """
        Returns the left joystick's angle and intensity

        Arguments:
        ----------
        None
        """
        y = self.LeftJoystickY
        x = self.LeftJoystickX
        intensity = x**2 + y**2
        if intensity > 1:
            intensity = 1
        if intensity < 0.1:
            return np.pi/2, 0
        theta = np.arctan2(y,x)
        return theta, intensity
    
    def get_right_joystick(self):
        """
        Returns the right joystick's angle and intensity

        Arguments:
        ----------
        None
        """
        y = self.RightJoystickY
        x = self.RightJoystickX
        intensity = x**2 + y**2
        if intensity > 1:
            intensity = 1
        if intensity < 0.1:
            return np.pi/2, 0
        theta = np.arctan2(y,x)
        return theta, intensity

    def _monitor_controller(self):
        """
        Monitor controller and update the controller's state.

        Arguments:
        ----------
        None
        """
        while True:
            events = get_gamepad()
            for event in events:

                if event.code == 'ABS_Y':
                    self.LeftJoystickY = event.state /\
                         XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_X':
                    self.LeftJoystickX = event.state /\
                         XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_RY':
                    self.RightJoystickY = event.state / \
                        XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_RX':
                    self.RightJoystickX = event.state / \
                        XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_Z':
                    self.LeftTrigger = event.state / \
                        XboxController.MAX_TRIG_VAL # normalize between 0 and 1
                elif event.code == 'ABS_RZ':
                    self.RightTrigger = event.state / \
                        XboxController.MAX_TRIG_VAL # normalize between 0 and 1
                elif event.code == 'BTN_TL':
                    self.LeftBumper = event.state
                elif event.code == 'BTN_TR':
                    self.RightBumper = event.state
                elif event.code == 'BTN_SOUTH':
                    self.A = event.state
                elif event.code == 'BTN_NORTH':
                    self.X = event.state
                elif event.code == 'BTN_WEST':
                    self.Y = event.state
                elif event.code == 'BTN_EAST':
                    self.B = event.state
                elif event.code == 'BTN_THUMBL':
                    self.LeftThumb = event.state
                elif event.code == 'BTN_THUMBR':
                    self.RightThumb = event.state
                elif event.code == 'BTN_SELECT':
                    self.Back = event.state
                elif event.code == 'BTN_START':
                    self.Start = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY1':
                    self.LeftDPad = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY2':
                    self.RightDPad = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY3':
                    self.UpDPad = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY4':
                    self.DownDPad = event.state




if __name__ == '__main__':
    joy = XboxController()
    while True:
        print(joy.read())